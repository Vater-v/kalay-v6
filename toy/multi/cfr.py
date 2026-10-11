"""Vectorised tree passes over the public tree, CFR+ and exact exploitability.

Notation (all arrays carry a trailing private-card fiber axis of length D):
  sigma[n, c]   prob. that owner of parent(n) takes the edge into n holding c (decision edges)
  reach[p,n,c]  product of player p's own edge probabilities on the path root -> n, times
                [c != board] below the chance node (holding the board card is impossible)
  val_i[n, c]   counterfactual value of player i at n holding c:
                sum over opponents' cards of (prod_j reach_j) * (chance prob) * E[u_i | below n]
                with chance prob. expressed RELATIVE to the uniform private deal. Concretely:
                  chance edge weight = 1/(D-N)  (each remaining card equally likely as board)
                  EV_i = sum_c val_i[root, c] / n_deals,  n_deals = D!/(D-N)!
Counterfactual regret of action a at (n, c) owned by i:  val_i[child_a, c] - val_i[n, c].

Exploitability: delta_i = BR_i(sigma_-i) - EV_i(sigma) >= 0 for every player i, computed
exactly by one backward pass that takes max over i's actions per (node, card). In 2-player
zero-sum, sum_i delta_i is the usual exploitability*2; in N-player it is "NashConv".
"""
from __future__ import annotations

import time

import numpy as np

from tree import PublicTree, DECISION, CHANCE
from terminal import TerminalEvaluator


# ---------------------------------------------------------------------------- strategy helpers
def uniform_strategy(tree: PublicTree) -> np.ndarray:
    sig = np.ones((tree.M, tree.cfg.deck))
    nch = tree.n_children[tree.parent[tree.dec_edges]]
    sig[tree.dec_edges] = 1.0 / nch[:, None]
    return sig


def normalize(tree: PublicTree, weights: np.ndarray, edges=None) -> np.ndarray:
    """Turn non-negative per-edge weights into a behaviour strategy (uniform where all zero).

    weights: [M, D]; only decision edges are read. Returns a full [M, D] sigma.
    """
    dec = tree.dec_edges
    w = np.maximum(weights[dec], 0.0)
    gsum = np.add.reduceat(w, tree.dec_gstart, axis=0)               # [groups, D]
    s = gsum[tree.dec_group_of_edge]
    nch = tree.n_children[tree.parent[dec]][:, None]
    out = np.ones_like(weights)
    out[dec] = np.where(s > 0, w / np.where(s > 0, s, 1.0), 1.0 / nch)
    return out


# ---------------------------------------------------------------------------- tree passes
def reach_all(tree: PublicTree, sigma: np.ndarray) -> np.ndarray:
    """Forward pass: reach[p, n, c] for every player p (own probabilities only)."""
    N, D = tree.cfg.n_players, tree.cfg.deck
    reach = np.ones((N, tree.M, D))
    for lev in tree.levels:
        lo, hi = lev["lo"], lev["hi"]
        reach[:, lo:hi] = reach[:, lev["parent"]]
        own = lev["owner"]
        dec = np.flatnonzero(own >= 0)
        # multiply the owner's reach by its action probability (one (player,node) pair per edge)
        reach[own[dec], lo + dec] *= sigma[lo + dec]
        # board card dealt: nobody can hold it
        reach[:, lev["chance_nodes"], lev["chance_boards"]] = 0.0
    return reach


def backward(tree: PublicTree, sigma: np.ndarray, term_val: np.ndarray, i: int,
             best_response: bool = False) -> np.ndarray:
    """Backward pass for player i given terminal cfvs term_val [T, D].

    Decision node of i: sum_a sigma(a) v(child_a)  (or max_a for best response).
    Decision node of another player: sum_a v(child_a) (their policy is already in the reach
    used to build term_val). Chance node: mean over boards = sum / (D - N).
    """
    D, N = tree.cfg.deck, tree.cfg.n_players
    val = np.zeros((tree.M, D))
    val[tree.terminals] = term_val
    chance_w = 1.0 / (D - N)
    for lev in reversed(tree.levels):
        lo, hi = lev["lo"], lev["hi"]
        v = val[lo:hi]
        own = lev["owner"]
        w = np.where((own == i)[:, None], sigma[lo:hi], 1.0)
        w = np.where((own == -1)[:, None], chance_w, w)
        s = np.add.reduceat(w * v, lev["gstart"], axis=0)
        if best_response:
            mx = np.maximum.reduceat(v, lev["gstart"], axis=0)
            mine = (lev["gkind"] == DECISION) & (lev["gplayer"] == i)
            s = np.where(mine[:, None], mx, s)
        val[lev["gparent"]] = s
    return val


# ---------------------------------------------------------------------------- evaluation
def evaluate(tree: PublicTree, sigma: np.ndarray, ev_exact: TerminalEvaluator | None = None):
    """Exact EV_i, BR_i and delta_i = BR_i - EV_i for every player (antes per hand)."""
    ev_exact = ev_exact or TerminalEvaluator(tree, "exact")
    N, nd = tree.cfg.n_players, tree.cfg.n_deals
    reach = reach_all(tree, sigma)
    ev, br = np.zeros(N), np.zeros(N)
    for i in range(N):
        tv = ev_exact.cfv(i, reach)
        ev[i] = backward(tree, sigma, tv, i)[0].sum() / nd
        br[i] = backward(tree, sigma, tv, i, best_response=True)[0].sum() / nd
    return dict(ev=ev, br=br, delta=br - ev, nashconv=float((br - ev).sum()))


# ---------------------------------------------------------------------------- CFR+
class CFRPlus:
    """Vectorised CFR+ : regret matching+, alternating updates, linear averaging.

    Each iteration t = 1, 2, ... updates players 0..N-1 in turn. For player p:
      1. reach of everybody under the current profile (players < p already updated at t),
      2. terminal cfvs of p via `evaluator` (exact or approximate card removal),
      3. backward pass -> instantaneous regrets r = v(child) - v(node) on p's edges,
      4. R <- max(R + r, 0)   (RM+), sigma_p <- R / sum R,
      5. avg <- avg + t * reach_p(parent) * sigma_p^t   (linear averaging of the played sigma).
    `players` restricts which seats learn (others keep `sigma` fixed) - used for M3.
    """

    def __init__(self, tree: PublicTree, evaluator: TerminalEvaluator, sigma0=None,
                 players=None, track_external_regret: bool = False):
        self.tree, self.ev = tree, evaluator
        self.N, self.D = tree.cfg.n_players, tree.cfg.deck
        self.sigma = uniform_strategy(tree) if sigma0 is None else sigma0.copy()
        self.R = np.zeros((tree.M, self.D))
        self.avg = np.zeros((tree.M, self.D))
        self.t = 0
        self.players = list(range(self.N)) if players is None else list(players)
        self.track = track_external_regret
        if self.track:  # sum over iterations of terminal cfvs and of realised values
            self.term_sum = np.zeros((self.N, len(tree.terminals), self.D))
            self.val_sum = np.zeros(self.N)

    def iterate(self, n_iter: int = 1):
        tree = self.tree
        dec = tree.dec_edges
        for _ in range(n_iter):
            self.t += 1
            for p in self.players:
                reach = reach_all(tree, self.sigma)
                tv = self.ev.cfv(p, reach)
                val = backward(tree, self.sigma, tv, p)
                if self.track:
                    self.term_sum[p] += tv
                    self.val_sum[p] += val[0].sum()
                mine = dec[tree.dec_owner == p]
                par = tree.parent[mine]
                # linear averaging of the strategy actually played this iteration
                self.avg[mine] += self.t * reach[p, par] * self.sigma[mine]
                # RM+ update
                self.R[mine] = np.maximum(self.R[mine] + val[mine] - val[par], 0.0)
                new = normalize(tree, self.R)
                self.sigma[mine] = new[mine]

    def average_strategy(self) -> np.ndarray:
        """Average strategy for learning seats, current (fixed) strategy for the others."""
        avg = normalize(self.tree, self.avg)
        out = self.sigma.copy()
        mine = self.tree.dec_edges[np.isin(self.tree.dec_owner, self.players)]
        out[mine] = avg[mine]
        return out

    def external_regret(self) -> np.ndarray:
        """Total external regret of each learning player w.r.t. the realised sequence.

        R_i^T = max_{pure s_i} sum_t u_i(s_i, sigma^t_-i) - sum_t u_i(sigma^t)  (antes, summed
        over iterations). u_i(s_i, sigma^t_-i) is linear in the terminal cfvs, so the max over
        s_i is ONE best-response pass on the accumulated terminal values.
        """
        assert self.track
        out = np.zeros(self.N)
        nd = self.tree.cfg.n_deals
        for p in self.players:
            best = backward(self.tree, self.sigma, self.term_sum[p], p, best_response=True)[0].sum()
            out[p] = (best - self.val_sum[p]) / nd
        return out


def run_cfr(tree, evaluator, iters, checkpoints, exact=None, track=False, verbose=True):
    """Run CFR+ and evaluate the average strategy (exactly) at checkpoints.

    Wall-clock reported excludes evaluation time."""
    exact = exact or TerminalEvaluator(tree, "exact")
    solver = CFRPlus(tree, evaluator, track_external_regret=track)
    log, clock = [], 0.0
    for t in range(1, iters + 1):
        t0 = time.perf_counter()
        solver.iterate(1)
        clock += time.perf_counter() - t0
        if t in checkpoints:
            e = evaluate(tree, solver.average_strategy(), exact)
            ec = evaluate(tree, solver.sigma, exact)
            row = dict(iter=t, seconds=clock, delta=e["delta"].tolist(), ev=e["ev"].tolist(),
                       nashconv=e["nashconv"], cur_delta=ec["delta"].tolist(),
                       cur_nashconv=ec["nashconv"])
            if track:
                row["ext_regret_avg"] = (solver.external_regret() / t).tolist()
            log.append(row)
            if verbose:
                print(f"  it {t:6d} {clock:7.1f}s  delta_avg(mA)={np.round(1000*e['delta'],2)}"
                      f"  nashconv_cur(mA)={1000*ec['nashconv']:.2f}", flush=True)
    return solver, log


def best_response_strategy(tree: PublicTree, sigma: np.ndarray, i: int,
                           ev_exact: TerminalEvaluator | None = None) -> np.ndarray:
    """sigma with player i's edges replaced by a best response (ties split uniformly)."""
    ev_exact = ev_exact or TerminalEvaluator(tree, "exact")
    reach = reach_all(tree, sigma)
    val = backward(tree, sigma, ev_exact.cfv(i, reach), i, best_response=True)
    dec = tree.dec_edges
    v = val[dec]
    gmax = np.maximum.reduceat(v, tree.dec_gstart, axis=0)[tree.dec_group_of_edge]
    w = np.zeros_like(sigma)
    w[dec] = (v >= gmax - 1e-12 * (1 + np.abs(gmax))).astype(float)
    br = normalize(tree, w)
    out = sigma.copy()
    mine = dec[tree.dec_owner == i]
    out[mine] = br[mine]
    return out
