"""Vectorised CFR / CFR+ on a public tree with private-hand fibers.

Notation (per player p, per node n, per private hand h):
  pi_p[n, h]  reach of p: product of p's own action probabilities on the path
              (times 0/1 card-removal masks at chance nodes)
  v_i[n, h]   counterfactual value of player i holding h at n:
              v_i[n,h] = sum_{terminal z under n} cscale(z) * sum_{h'} pi_o[z,h'] u_i(z,h,h') * pi_i(n->z | h)
  regret      r_i[n.a, h] += v_i[n.a, h] - v_i[n, h]
CFR+ = regret matching+ (clip regrets at 0), linear averaging (iteration t
weighs t), alternating updates (player 0 then player 1 inside one iteration).
Vanilla CFR = plain regret matching, uniform averaging.

Two optional hooks implement the other experiments with tiny diffs:
  * gadget (safe re-solving, DeepStack-style): the opponent g may, per hand,
    "terminate" at the subgame root and receive a fixed counterfactual value
    w[root, h] instead of entering the subgame.  It changes only g's root reach.
  * rnr (restricted Nash response): when the exploiter traverses, the
    opponent's reach is  p * reach(fixed model) + (1-p) * reach(free CFR player).
"""
from __future__ import annotations

import numpy as np

from tree import PublicTree, DEC, LEAF


def forward(t: PublicTree, eff, root):
    """Reach pi_p for every node given p's edge multipliers `eff` and the
    root ranges: pi[child] = pi[parent] * eff[child], one numpy op per level."""
    r = np.empty((t.M, t.N))
    r[: t.n_roots] = root
    for lv in t.levels:
        par = np.repeat(r[lv["nt"]], lv["cnt"], axis=0)
        np.multiply(par, eff[lv["lo"]: lv["hi"]], out=r[lv["lo"]: lv["hi"]])
    return r


class Solver:
    def __init__(self, tree: PublicTree, root_reach, *, plus=True, linear=True,
                 leaf_fn=None, gadget=None, rnr=None):
        """root_reach: [r0, r1], each [n_roots, N] initial ranges (already
        masked by the root board).  leaf_fn(i, reach0, reach1) -> CFV of
        player i at the LEAF nodes (tree.L).  gadget = dict(player=g, w=[n_roots,N]).
        rnr = dict(player=g, p=float, eff=eff array of the fixed model for g)."""
        self.t = t = tree
        self.N = N = tree.N
        self.root_reach = [np.asarray(root_reach[0], float), np.asarray(root_reach[1], float)]
        self.plus, self.linear = plus, linear
        self.leaf_fn, self.gadget, self.rnr = leaf_fn, gadget, rnr
        self.eff = [t.eff_template.copy(), t.eff_template.copy()]
        self.R = [np.zeros((len(t.E[p]), N)) for p in (0, 1)]
        self.Ssum = [np.zeros((len(t.E[p]), N)) for p in (0, 1)]
        for p in (0, 1):
            self.eff[p][t.E[p]] = self._match(p, self.R[p])
        if gadget is not None:
            self.gR = np.zeros((t.n_roots, N, 2))       # regrets of [follow, terminate]
            self.gS = np.zeros((t.n_roots, N, 2))
            self.g_follow = np.full((t.n_roots, N), 0.5)
        if rnr is not None:
            self.fix_reach = self.reach(rnr["player"], eff=rnr["eff"])
        self.it = 0

    # ---------------- strategy from regrets ---------------------------
    def _match(self, p, R):
        """Regret matching on every decision node of p at once:
        sigma(a|h) = R+(a,h) / sum_b R+(b,h), uniform if the sum is 0."""
        t = self.t
        if len(R) == 0:
            return R
        Rp = np.maximum(R, 0.0)
        tot = np.add.reduceat(Rp, t.E_starts[p], axis=0)
        cnt = t.E_cnt[p]
        tot_e = np.repeat(tot, cnt, axis=0)
        uni = np.repeat(1.0 / cnt, cnt)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot_e > 0, Rp / np.where(tot_e > 0, tot_e, 1.0), uni)

    # ---------------- forward pass: reach -----------------------------
    def reach(self, p, eff=None, root=None):
        eff = self.eff[p] if eff is None else eff
        return forward(self.t, eff, self.root_reach[p] if root is None else root)

    def opp_reach(self, i, eff=None):
        """Reach of the opponent o = 1-i as seen by traverser i (hooks here)."""
        o = 1 - i
        root = None
        if self.gadget is not None and self.gadget["player"] == o:
            root = self.root_reach[o] * self.g_follow
        q = self.reach(o, eff=eff, root=root)
        if self.rnr is not None and self.rnr["player"] == o:      # <- RNR: the whole diff
            p = self.rnr["p"]
            q = p * self.fix_reach + (1.0 - p) * q
        return q

    # ---------------- terminal values ---------------------------------
    def terminal_values(self, i, q, reach_both=None):
        """CFVs of player i at FOLD/SHOW/LEAF nodes given opponent reach q.
        Card removal is explicit: the opponent cannot hold my card h, so
        fold value uses (sum q - q[h]); showdown uses W[board] with zero diagonal."""
        t = self.t
        v = np.zeros((t.M, self.N))
        F = t.F
        if len(F):
            qF = q[F]
            v[F] = t.F_u[i][:, None] * (qF.sum(1, keepdims=True) - qF) * t.bmask[F]
        for idx, WT, scale in t.S_groups:
            v[idx] = (q[idx] @ WT) * scale
        if len(t.L):
            r0, r1 = reach_both
            v[t.L] = self.leaf_fn(i, r0[t.L], r1[t.L])
        return v

    # ---------------- backward pass -----------------------------------
    def backward(self, i, v, eff=None, best_response=False):
        """v[n] = sum_a eff_i[n.a] * v[n.a]  (eff_i = sigma_i on i's edges,
        1 on opponent edges, card mask on chance edges).  With
        best_response=True, i's decision nodes take the max over actions."""
        t = self.t
        eff = self.eff[i] if eff is None else eff
        for lv in reversed(t.levels):
            lo, hi = lv["lo"], lv["hi"]
            if best_response:
                vc = v[lo:hi] * t.eff_template[lo:hi]
                s = np.add.reduceat(vc, lv["starts"], axis=0)
                m = np.maximum.reduceat(v[lo:hi], lv["starts"], axis=0)
                dec = lv["is_dec"][i]
                s[dec] = m[dec]
                v[lv["nt"]] = s
            else:
                v[lv["nt"]] = np.add.reduceat(v[lo:hi] * eff[lo:hi], lv["starts"], axis=0)
        return v

    # ---------------- one CFR iteration --------------------------------
    def iterate(self, n=1):
        for _ in range(n):
            self.it += 1
            for i in (0, 1):
                self._traverse(i)

    def _traverse(self, i):
        t = self.t
        w = float(self.it) if self.linear else 1.0
        q = self.opp_reach(i)
        r = self.reach(i)
        both = (r, q) if i == 0 else (q, r)
        v = self.backward(i, self.terminal_values(i, q, both))
        E = t.E[i]
        # average strategy numerator: sum_t w_t * pi_i(parent) * sigma_t(a) = w_t * pi_i(child)
        self.Ssum[i] += w * r[E]
        inst = v[E] - v[t.E_parent[i]]
        if self.plus:
            np.maximum(self.R[i] + inst, 0.0, out=self.R[i])
        else:
            self.R[i] += inst
        self.eff[i][E] = self._match(i, self.R[i])
        g = self.gadget
        if g is not None and g["player"] == i:
            vf = v[: t.n_roots]
            vt = g["w"]
            sf = self.g_follow
            vg = sf * vf + (1 - sf) * vt
            self.gS[..., 0] += w * sf
            self.gS[..., 1] += w * (1 - sf)
            self.gR[..., 0] = np.maximum(self.gR[..., 0] + vf - vg, 0)
            self.gR[..., 1] = np.maximum(self.gR[..., 1] + vt - vg, 0)
            tot = self.gR.sum(-1)
            self.g_follow = np.where(tot > 0, self.gR[..., 0] / np.where(tot > 0, tot, 1), 0.5)

    # ---------------- average strategy & evaluation --------------------
    def avg_eff(self, p):
        t = self.t
        e = t.eff_template.copy()
        S = self.Ssum[p]
        if len(S) == 0:
            return e
        tot = np.repeat(np.add.reduceat(S, t.E_starts[p], axis=0), t.E_cnt[p], axis=0)
        uni = np.repeat(1.0 / t.E_cnt[p], t.E_cnt[p])[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            e[t.E[p]] = np.where(tot > 0, S / np.where(tot > 0, tot, 1), uni)
        return e

    def avg_follow(self):
        tot = self.gS.sum(-1)
        return np.where(tot > 0, self.gS[..., 0] / np.where(tot > 0, tot, 1), 0.5)


def values(solver: Solver, eff0, eff1, best_response_for=None, g_follow=None):
    """Counterfactual values of both players for the profile (eff0, eff1).
    If best_response_for = i, player i best-responds instead of using eff_i.
    Returns (v0, v1, reach0, reach1) full [M, N] arrays."""
    t = solver.t
    effs = [eff0, eff1]
    root = [solver.root_reach[0], solver.root_reach[1]]
    if g_follow is not None:
        gp = solver.gadget["player"]
        root[gp] = root[gp] * g_follow
    r = [solver.reach(p, eff=effs[p], root=root[p]) for p in (0, 1)]
    out = []
    for i in (0, 1):
        q = r[1 - i]
        v = solver.terminal_values(i, q, (r[0], r[1]))
        v = solver.backward(i, v, eff=effs[i], best_response=(best_response_for == i))
        out.append(v)
    return out[0], out[1], r[0], r[1]


def root_value(solver, v, i):
    """Expected value of player i per root: sum_h pi_i(root,h) * v_i(root,h)."""
    return (v[: solver.t.n_roots] * solver.root_reach[i]).sum(1)


def exploitability(solver: Solver, eff0=None, eff1=None):
    """Exact best-response values. Returns dict with BR0 = max_s0 u0(s0, eff1),
    BR1 = max_s1 u1(eff0, s1), expl = (BR0 + BR1)/2 (game value cancels in a
    zero-sum game).  Units: chips (antes) per hand, root(s) summed.  A value is
    sum_h pi_i(root,h) * v_i(root,h): CFVs exclude the player's own reach."""
    eff0 = solver.avg_eff(0) if eff0 is None else eff0
    eff1 = solver.avg_eff(1) if eff1 is None else eff1
    t = solver.t
    r0 = solver.reach(0, eff=eff0)
    r1 = solver.reach(1, eff=eff1)
    res = {}
    for i, q in ((0, r1), (1, r0)):
        v = solver.terminal_values(i, q, (r0, r1))
        v = solver.backward(i, v, eff=[eff0, eff1][i], best_response=True)
        res[f"BR{i}"] = float((v[: t.n_roots] * solver.root_reach[i]).sum())
    v = solver.terminal_values(0, r1, (r0, r1))
    v = solver.backward(0, v, eff=eff0)
    res["EV0"] = float((v[: t.n_roots] * solver.root_reach[0]).sum())
    res["expl"] = 0.5 * (res["BR0"] + res["BR1"])
    return res


def best_response_eff(solver: Solver, i, eff_opp):
    """A pure best response of player i to eff_opp (as an eff array).
    At every decision node of i and every hand, pick the action whose
    best-response CFV is maximal (first one on ties)."""
    t = solver.t
    q = solver.reach(1 - i, eff=eff_opp)
    assert len(t.L) == 0, "best response needs a tree without depth-limit leaves"
    v = solver.terminal_values(i, q)
    v = solver.backward(i, v, best_response=True)
    E, starts, cnt = t.E[i], t.E_starts[i], t.E_cnt[i]
    ve = v[E]
    mx = np.repeat(np.maximum.reduceat(ve, starts, axis=0), cnt, axis=0)
    is_max = ve >= mx - 1e-12
    # keep only the first maximiser in each segment
    csum = np.cumsum(is_max, axis=0)
    seg_base = np.repeat(np.concatenate([np.zeros((1, t.N)), csum[starts[1:] - 1]]), cnt, axis=0)
    first = is_max & ((csum - seg_base) == 1)
    e = t.eff_template.copy()
    e[E] = first.astype(float)
    return e
