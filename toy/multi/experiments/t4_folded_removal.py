"""T4 (targeted): what does ignoring the FOLDED players' card removal cost?

Exact multiway terminals use every dealt player's reach, including players who
folded earlier (their cards are dead but their fold was informative).  A solver
that hands a pot "after the folds" to a k-live-player value map / HU solver
treats the deck as uniform apart from the visible cards: the folded players'
reach vectors are replaced by a constant (their mean over non-board cards).
That is the 'drop nu' approximation.  Live opponents keep exact card removal.

Measured (same CFR+ budget as M1 part A, scored by EXACT NashConv):
  N=3, D=12 (main game), 2000 iterations: exact vs drop-nu
  N=4, D=8, raise cap 1, 400 iterations:  mobius (exact) vs drop-nu(mobius)
Variant drop_nu_post applies the approximation only at street-2 terminals (after the
board), i.e. only where the live-count principle hands the pot to an L-live value map;
street-1 (preflop-like) terminals, incl. all-fold terminals, stay exact.
plus, at the exact solution, the terminal relative L1 error and the max error in a
player's conditional EV given its card (same metrics as M1 part A).
"""
import os
import time
import multiprocessing as mp

import numpy as np

from common import MAIN, save, table, ma
from game import GameConfig
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import CFRPlus, evaluate
from m1_card_removal import cfv_errors


class DropFolded:
    """Wraps an evaluator; folded OPPONENTS of the hero get a flat reach vector."""

    def __init__(self, base: TerminalEvaluator, post_only: bool = False):
        self.base = base
        self.nodes = base.nodes
        D = base.D
        self.mask = np.ones((len(self.nodes), D))
        hb = base.board >= 0
        self.mask[np.flatnonzero(hb), base.board[hb]] = 0.0
        self.dead = ~base.alive                                  # [T, N]
        if post_only:
            self.dead = self.dead & (base.board >= 0)[:, None]
        self.method = "drop_nu"

    def cfv(self, i, reach):
        R = reach[:, self.nodes, :].copy() if reach.shape[1] != len(self.nodes) else reach.copy()
        for f in range(self.base.n):
            if f == i:
                continue
            rows = np.flatnonzero(self.dead[:, f])
            if len(rows) == 0:
                continue
            m = self.mask[rows]
            mean = (R[f, rows] * m).sum(1, keepdims=True) / m.sum(1, keepdims=True)
            R[f, rows] = mean * m
        return self.base.cfv(i, R)


CFGS = {"N3": (MAIN, 2000, "exact"),
        "N4": (GameConfig(n_players=4, n_ranks=4, n_suits=2, stack=15, raise_cap=1), 400, "mobius")}


def job(args):
    name, variant = args
    cfg, iters, exact_m = CFGS[name]
    tree = PublicTree(cfg)
    ex = TerminalEvaluator(tree, exact_m)
    ev = ex if variant == "exact" else DropFolded(TerminalEvaluator(tree, exact_m),
                                                  post_only=(variant == "drop_nu_post"))
    s = CFRPlus(tree, ev)
    t0 = time.perf_counter()
    s.iterate(iters)
    sec = time.perf_counter() - t0
    avg = s.average_strategy()
    e = evaluate(tree, avg, ex)
    return name, variant, avg, e["delta"].tolist(), e["nashconv"], e["ev"].tolist(), sec


if __name__ == "__main__":
    VARIANTS = os.environ.get("T4_VARIANTS", "exact,drop_nu,drop_nu_post").split(",")
    jobs = [(n, v) for n in CFGS for v in VARIANTS]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(job, jobs)
    out, rows = {}, []
    for name in CFGS:
        cfg, iters, exact_m = CFGS[name]
        tree = PublicTree(cfg)
        ex = TerminalEvaluator(tree, exact_m)
        r = {x[1]: x for x in res if x[0] == name}
        sig_exact = r["exact"][2]
        # share of terminal reach mass that has at least one folder (at the exact solution)
        from cfr import reach_all
        rc = reach_all(tree, sig_exact)[:, ex.nodes]
        mass = np.prod(rc.sum(2), axis=0)            # crude product-of-marginals weight per terminal
        folded = (~ex.alive).any(1) & (ex.alive.sum(1) >= 2)
        out[name] = dict(config=cfg.describe(), iters=iters, exact_method=exact_m,
                         share_terminal_weight_multiway_after_fold=float(mass[folded].sum() / mass.sum()))
        for v in VARIANTS:
            _, _, _, d, nc, evs, sec = r[v]
            out[name][v] = dict(delta=d, nashconv=nc, ev=evs, seconds=sec)
            if v != "exact":
                dn = DropFolded(TerminalEvaluator(tree, exact_m), post_only=(v == "drop_nu_post"))
                at_eq = cfv_errors(tree, sig_exact, dn, ex)
                own = cfv_errors(tree, r[v][2], dn, ex)
                out[name][v].update(err_at_exact_eq=at_eq, err_own=own)
            d = d + [""] * (4 - len(d))
            rows.append([name, v, iters, round(1000 * sec / iters, 1), *[ma(x) if x != "" else "" for x in d],
                         ma(nc),
                         "" if v == "exact" else f"{at_eq['term_rel']:.2e}",
                         "" if v == "exact" else at_eq["root_err_mA"],
                         "" if v == "exact" else own["ev_err_mA"]])
    print(table(["game", "variant", "iters", "ms/iter", "d0 mA", "d1 mA", "d2 mA", "d3 mA", "NashConv mA",
                 "terminal rel L1 err (at exact eq)", "max root cond-EV err mA (at exact eq)",
                 "self-assessed EV err mA"], rows))
    for k, v in out.items():
        print(k, "share of (product-weight) terminal mass at live>=2 with a folder:",
              round(v["share_terminal_weight_multiway_after_fold"], 3))
    save("t4", out)
