"""T3b: price of paranoia.  Solve the card-sharing coalition maxmin for each seat
(team.py, 1000 CFR+ iterations) and play that maxmin strategy against the
self-play population (M2's sigma* for the other seats).  Reports the EV change
vs self-play and the security level of each, for every seat."""
import multiprocessing as mp
import numpy as np
from common import MAIN, save, table, ma, RESULTS
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import evaluate
from team import TeamGame

TREE = PublicTree(MAIN)


def solve(h):
    g = TeamGame(TREE, h)
    avg, log = g.solve(1000, checkpoints=(1000,))
    return h, avg["H"], log


if __name__ == "__main__":
    ex = TerminalEvaluator(TREE, "exact")
    sig = np.load(f"{RESULTS}/m2_sigma_3p.npy")
    base = evaluate(TREE, sig, ex)
    with mp.get_context("fork").Pool(3) as pool:
        res = pool.map(solve, range(3))
    rows, out = [], []
    for h, sH, log in res:
        g = TeamGame(TREE, h)
        m = sig.copy()
        e = TREE.dec_edges[TREE.dec_owner == h]
        m[e] = sH[e]
        ev = evaluate(TREE, m, ex)
        sec_mm, up = log[-1][1], log[-1][2]
        sec_sp = g.value(sig, np.ones((TREE.M, g.F)), br="C")
        np.save(f"{RESULTS}/t3_maxmin_sigma_seat{h}.npy", sH)
        out.append(dict(seat=h, maxmin_bracket=[sec_mm, up], selfplay_ev=float(base["ev"][h]),
                        maxmin_ev_vs_selfplay=float(ev["ev"][h]), selfplay_security=sec_sp,
                        others_gain=[float(ev["ev"][j] - base["ev"][j]) for j in range(3)],
                        delta_of_mixed_profile=ev["delta"].tolist()))
        rows.append([h, round(float(base["ev"][h]), 4), round(sec_sp, 4), round(float(ev["ev"][h]), 4),
                     round(sec_mm, 4), ma(ev["ev"][h] - base["ev"][h]), ma(ev["delta"][h])])
    print(table(["seat", "self-play EV", "self-play security", "maxmin EV vs self-play others",
                 "maxmin security", "EV change mA", "seat's BR gain in mixed profile mA"], rows))
    save("t3b", out)
