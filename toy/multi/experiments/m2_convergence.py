"""M2: CFR+ self-play convergence in the 3-player game (2-player reduction as control).

Measures at log-spaced checkpoints:
  delta_i(avg)  = BR_i - EV_i of the linear-averaged profile (exact, antes/hand -> mA/hand)
  NashConv(avg) = sum_i delta_i(avg);  NashConv(cur) for the current RM+ iterate
  R_i^T / T     = average external regret of player i against the realised self-play sequence
                  (the no-regret guarantee: R_i^T/T -> 0 at rate O(1/sqrt T) regardless of N)
Saves the final 3-player average strategy for M3 (results/m2_sigma_3p.npy).
"""
import time

import numpy as np

from common import MAIN, save, table, ma, RESULTS
from game import GameConfig
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import run_cfr

ITERS = 10000
CHECKS = sorted({1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, ITERS})


def run(cfg, label):
    tree = PublicTree(cfg)
    print(f"\n== {label}: {cfg.describe()}  public nodes={tree.M}")
    ex = TerminalEvaluator(tree, "exact")
    solver, log = run_cfr(tree, ex, ITERS, set(CHECKS), exact=ex, track=True)
    rows = []
    prev = None
    for r in log:
        slope = ""
        if prev is not None:
            slope = round(float(np.log(r["nashconv"] / prev["nashconv"]) /
                                np.log(r["iter"] / prev["iter"])), 2)
        rows.append([r["iter"], round(r["seconds"], 1)] + [ma(d) for d in r["delta"]] +
                    [ma(r["nashconv"]), slope, ma(r["cur_nashconv"])] +
                    [ma(x) for x in r["ext_regret_avg"]] +
                    [round(1000 * max(r["ext_regret_avg"]) * np.sqrt(r["iter"]), 1)])
        prev = r
    n = cfg.n_players
    hdr = (["iter", "sec"] + [f"d{i}(avg) mA" for i in range(n)] +
           ["NashConv(avg) mA", "loglog slope", "NashConv(cur) mA"] +
           [f"R{i}/T mA" for i in range(n)] + ["max R/T*sqrt(T)"])
    print(table(hdr, rows))
    final = log[-1]
    print("final EV per seat (antes/hand):", np.round(final["ev"], 5))
    return solver, log, hdr, rows, tree


if __name__ == "__main__":
    out = {}
    s3, log3, h3, r3, tree3 = run(MAIN, "3 players")
    np.save(f"{RESULTS}/m2_sigma_3p.npy", s3.average_strategy())
    np.save(f"{RESULTS}/m2_regrets_3p.npy", s3.R)          # for warm-started play in M3
    cfg2 = GameConfig(n_players=2, n_ranks=MAIN.n_ranks, n_suits=MAIN.n_suits, stack=MAIN.stack,
                      raise_cap=MAIN.raise_cap)
    s2, log2, h2, r2, _ = run(cfg2, "2 players (control)")
    save("m2", dict(log3=log3, log2=log2, table3=dict(h=h3, rows=r3), table2=dict(h=h2, rows=r2)))
