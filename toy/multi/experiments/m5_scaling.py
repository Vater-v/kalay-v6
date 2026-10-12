"""M5: cost scaling with the number of players.

1. Measured: public-tree size and seconds per CFR+ iteration for N = 2, 3, 4 (+5 with the
   Moebius evaluator only) of the SAME game (same deck, stacks, bet menu; the N-player game is
   obtained by adding/removing seats).
2. Counted (no fiber arrays needed): public-tree size for N = 2..9 with a memoised counter.
3. Fitted cost model, extrapolated to N = 6 and 9:
      t_iter(N) = N * [ a * M(N) * D * N  +  b * T_sd(N) * W(N, D) ]
   N alternating passes; each pass = forward/backward over M public nodes x D cards for N
   reach arrays, plus terminal evaluation over T_sd showdown terminals of work W per terminal:
      W_exact  = D^N                                   (brute force over disjoint tuples)
      W_mobius = (2^m - 1) * D^2 + Bell(m) * m^2 * D   (m = N-1; blocks + partition polynomials)
      W_indep  = m * D^2 + m^2 * D
"""
import time
from functools import lru_cache

import numpy as np

from common import MAIN, save, table
from game import GameConfig, initial_state, legal_actions, apply_action, round_over, start_street2
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import CFRPlus


def bell(n):
    row = [1]
    for _ in range(n):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[0]


def count_tree(cfg: GameConfig):
    """(nodes, decision nodes, terminals, showdown terminals) of the public tree, memoised."""
    D = cfg.deck

    @lru_cache(maxsize=None)
    def rec(street, contrib, folded, pending, raises):
        from game import BetState
        st = BetState(street, contrib, folded, pending, raises, -1, "")
        if st.n_alive() <= 1:
            return (1, 0, 1, 0)
        if round_over(st):
            if street == 2:
                return (1, 0, 1, 1)
            # chance node: D boards, each followed by an identical street-2 subtree
            s2 = start_street2(cfg, st, 0)
            if len(s2.pending) == 0:
                sub = (1, 0, 1, 1)
            else:
                sub = rec(2, s2.contrib, s2.folded, s2.pending, 0)
            return (1 + D * sub[0], D * sub[1], D * sub[2], D * sub[3])
        tot = [1, 1, 0, 0]
        for a in legal_actions(cfg, st):
            c = apply_action(cfg, st, a)
            r = rec(c.street, c.contrib, c.folded, c.pending, c.raises)
            for k in range(4):
                tot[k] += r[k]
        return tuple(tot)

    s0 = initial_state(cfg)
    return rec(s0.street, s0.contrib, s0.folded, s0.pending, s0.raises)


def time_iter(cfg, method, reps):
    tree = PublicTree(cfg)
    ev = TerminalEvaluator(tree, method)
    s = CFRPlus(tree, ev)
    s.iterate(1)                                   # warm-up (kernel caches etc.)
    t0 = time.perf_counter()
    s.iterate(reps)
    return (time.perf_counter() - t0) / reps, tree


def w_eval(method, n, D):
    m = n - 1
    if method == "exact":
        return float(D) ** n
    if method == "mobius":
        return (2 ** m - 1) * D ** 2 + bell(m) * m * m * D
    return m * D ** 2 + m * m * D


if __name__ == "__main__":
    D = MAIN.deck
    meas = []
    for n, methods in ((2, ("exact", "mobius", "independent")), (3, ("exact", "mobius", "independent")),
                       (4, ("exact", "mobius", "independent")), (5, ("mobius", "independent"))):
        cfg = GameConfig(n, MAIN.n_ranks, MAIN.n_suits, MAIN.stack, MAIN.raise_cap)
        for m in methods:
            reps = {2: 50, 3: 20, 4: 4, 5: 2}[n]
            sec, tree = time_iter(cfg, m, reps)
            st = tree.stats()
            mem = (cfg.n_players + 4) * tree.M * D * 8      # reach(N) + R, avg, sigma, values
            meas.append(dict(N=n, method=m, sec=sec, M=tree.M, T_sd=st["showdown"],
                             infosets=st["infosets"], mem_MB=mem / 2 ** 20))
            print(meas[-1], flush=True)
    # counted tree sizes up to 9 players
    counts = []
    for n in range(2, 10):
        cfg = GameConfig(n, MAIN.n_ranks, MAIN.n_suits, MAIN.stack, MAIN.raise_cap)
        t0 = time.perf_counter()
        c = count_tree(cfg)
        counts.append(dict(N=n, M=c[0], decision=c[1], terminals=c[2], T_sd=c[3],
                           infosets=c[1] * D, count_sec=time.perf_counter() - t0))
        print(counts[-1], flush=True)
    for m in meas:   # counter must agree with the materialised tree
        assert counts[m["N"] - 2]["M"] == m["M"], (m, counts[m["N"] - 2])
    # fit a, b per method by least squares on the measured points
    fits = {}
    for method in ("exact", "mobius", "independent"):
        pts = [x for x in meas if x["method"] == method]
        A = np.array([[p["N"] * p["M"] * D * p["N"], p["N"] * p["T_sd"] * w_eval(method, p["N"], D)]
                      for p in pts])
        y = np.array([p["sec"] for p in pts])
        # relative least squares (each point weighted by 1/y) so small N is not ignored
        coef, *_ = np.linalg.lstsq(A / y[:, None], np.ones_like(y), rcond=None)
        coef = np.maximum(coef, 0)
        fits[method] = coef.tolist()
        for p in pts:
            p["fit_sec"] = float(A[pts.index(p)] @ coef)
    pred = []
    for c in counts:
        n = c["N"]
        row = dict(N=n, M=c["M"], T_sd=c["T_sd"], infosets=c["infosets"],
                   mem_GB=(n + 4) * c["M"] * D * 8 / 2 ** 30)
        for method, (a, b) in fits.items():
            row[method] = n * (a * c["M"] * D * n + b * c["T_sd"] * w_eval(method, n, D))
        pred.append(row)
    rows1 = [[p["N"], p["method"], p["M"], p["T_sd"], p["infosets"], round(1000 * p["sec"], 1),
              round(1000 * p["fit_sec"], 1), round(p["mem_MB"], 1)] for p in meas]
    h1 = ["N", "evaluator", "public nodes M", "showdown terminals", "infosets", "measured ms/iter",
          "model ms/iter", "fiber memory MB"]
    rows2 = [[p["N"], p["M"], p["T_sd"], p["infosets"], f"{p['mem_GB']:.3g}",
              f"{p['exact']:.3g}", f"{p['mobius']:.3g}", f"{p['independent']:.3g}",
              f"{1000 * p['mobius'] / 3600:.3g}"] for p in pred]
    h2 = ["N", "public nodes M", "showdown terminals", "infosets", "fiber memory GB",
          "s/iter exact (model)", "s/iter mobius (model)", "s/iter independent (model)",
          "hours for 1000 iters (mobius)"]
    print(table(h1, rows1))
    print(table(h2, rows2))
    print("fitted (a, b):", fits)
    save("m5", dict(measured=meas, counts=counts, fits=fits, predictions=pred,
                    t1=dict(h=h1, rows=rows1), t2=dict(h=h2, rows=rows2)))
