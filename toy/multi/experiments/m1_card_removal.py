"""M1: exact vs approximate card removal in the multiway terminal counterfactual values.

Part A (in-game): run CFR+ with the EXACT, INDEPENDENT and PAIRWISE terminal evaluators on the
main 3-player game (and a 4-player variant, where PAIRWISE is no longer exact), same iteration
budget. Score every resulting strategy with the EXACT evaluator: per-player exploitability
delta_i, NashConv, plus the CFV error of each approximate evaluator, and wall-clock.

Part B (isolated terminal): random opponent ranges at a single all-alive showdown, varying deck
size D and player count N, to measure how the error of INDEPENDENT and PAIRWISE scales and what
each method costs. Reference = brute-force sum over disjoint tuples where D^N is small enough,
MOBIUS (exact, verified against brute force) otherwise.
"""
import time
from math import comb, factorial

import numpy as np

from common import MAIN, save, table, ma
from game import GameConfig, strength_table, showdown_share
from tree import PublicTree
from terminal import TerminalEvaluator, partitions_for, partition_mass_win
from cfr import CFRPlus, evaluate, reach_all, backward

rng = np.random.default_rng(1)


# ----------------------------------------------------------------------------- Part A
def cfv_errors(tree, sigma, approx: TerminalEvaluator, exact: TerminalEvaluator):
    """Errors of `approx` vs `exact` under one fixed profile `sigma`.

    root_err_mA : max over players and cards of |E[u_i | c_i = c]| difference, in milli-antes
    term_rel    : sum |cfv_approx - cfv_exact| / sum |cfv_exact| over all terminals/cards/players
    ev_err_mA   : max over players of |EV_i(approx terminal values) - EV_i(exact)|
    """
    cfg = tree.cfg
    reach = reach_all(tree, sigma)
    n_opp_deals = factorial(cfg.deck - 1) // factorial(cfg.deck - cfg.n_players)
    root_err, num, den, ev_err = 0.0, 0.0, 0.0, 0.0
    for i in range(cfg.n_players):
        te, ta = exact.cfv(i, reach), approx.cfv(i, reach)
        num += np.abs(ta - te).sum(); den += np.abs(te).sum()
        ve, va = backward(tree, sigma, te, i)[0], backward(tree, sigma, ta, i)[0]
        root_err = max(root_err, np.abs(va - ve).max() / n_opp_deals)
        ev_err = max(ev_err, abs(va.sum() - ve.sum()) / cfg.n_deals)
    return dict(root_err_mA=ma(root_err), term_rel=float(num / den), ev_err_mA=ma(ev_err))


def part_a(cfg, iters, methods):
    tree = PublicTree(cfg)
    exact = TerminalEvaluator(tree, "exact")
    print(f"\n== Part A: {cfg.describe()}, public nodes {tree.M}, {iters} CFR+ iterations")
    res, sig_exact = {}, None
    for m in methods:
        ev = exact if m == "exact" else TerminalEvaluator(tree, m)
        s = CFRPlus(tree, ev)
        t0 = time.perf_counter()
        s.iterate(iters)
        sec = time.perf_counter() - t0
        avg = s.average_strategy()
        e = evaluate(tree, avg, exact)
        res[m] = dict(seconds=sec, ms_per_iter=1000 * sec / iters, delta=e["delta"].tolist(),
                      nashconv=e["nashconv"], ev=e["ev"].tolist(), sigma=avg)
        if m == "exact":
            sig_exact = avg
        print(f"  {m:12s} {sec:6.1f}s  delta(mA)={np.round(1000 * e['delta'], 2)} "
              f"NashConv={1000 * e['nashconv']:.2f} mA", flush=True)
    rows = []
    for m in methods:
        r = res[m]
        own = cfv_errors(tree, r["sigma"], TerminalEvaluator(tree, m), exact) if m != "exact" \
            else dict(root_err_mA=0.0, term_rel=0.0, ev_err_mA=0.0)
        at_eq = cfv_errors(tree, sig_exact, TerminalEvaluator(tree, m), exact) if m != "exact" \
            else dict(root_err_mA=0.0, term_rel=0.0, ev_err_mA=0.0)
        r.update(err_own=own, err_at_exact_eq=at_eq)
        rows.append([m, round(r["ms_per_iter"], 1), round(r["seconds"], 1)] +
                    [ma(d) for d in r["delta"]] + [ma(r["nashconv"]),
                     f"{at_eq['term_rel']:.2e}", at_eq["root_err_mA"], own["ev_err_mA"]])
        del r["sigma"]
    hdr = (["method", "ms/iter", "total s"] + [f"d{i} mA" for i in range(cfg.n_players)] +
           ["NashConv mA", "terminal rel. L1 err (at exact eq.)",
            "max root cond.-EV err mA (at exact eq.)", "self-assessed EV err mA (own sol.)"])
    print(table(hdr, rows))
    return dict(config=cfg.describe(), nodes=tree.M, iters=iters, results=res, h=hdr, rows=rows)


# ----------------------------------------------------------------------------- Part B
def brute_mass_win(R, boards, strength, n):
    """Reference: explicit sum over disjoint opponent tuples (hero = player 0, all alive)."""
    T, D = R.shape[1], R.shape[2]
    grids = np.indices((D,) * n)
    mass, win = np.zeros((T, D)), np.zeros((T, D))
    for t in range(T):
        b = boards[t]
        valid = np.ones((D,) * n, dtype=bool)
        for p in range(n):
            valid &= grids[p] != b
            for q in range(p + 1, n):
                valid &= grids[p] != grids[q]
        share = showdown_share(None, np.moveaxis(strength[b][grids], 0, -1), np.ones(n, bool))[..., 0]
        P = np.ones((D,) * n)
        for j in range(1, n):
            shape = [1] * n; shape[j] = D
            P = P * R[j, t].reshape(shape)
        W = P * valid
        mass[t] = W.reshape(D, -1).sum(1)
        win[t] = (W * share).reshape(D, -1).sum(1)
    return mass, win


def part_b_case(n, n_ranks, family, trials=40):
    cfg = GameConfig(n_players=n, n_ranks=n_ranks, n_suits=2)
    D = cfg.deck
    s = strength_table(cfg)
    boards = rng.integers(0, D, trials)
    if family == "uniform":
        R = np.ones((n, trials, D))
    else:  # concentrated: Gamma(0.5) weights, i.e. ranges after aggressive betting
        R = rng.gamma(0.5, 1.0, (n, trials, D))
    R[:, np.arange(trials), boards] = 0.0
    LESS = (s[:, :, None] < s[:, None, :]).astype(float)[boards]
    EQ = ((s[:, :, None] == s[:, None, :]) & ~np.eye(D, dtype=bool)[None]).astype(float)[boards]
    sd = np.arange(trials)
    alive = np.ones((trials, n), bool)
    opps = list(range(1, n))
    out, timing = {}, {}
    for m in ("independent", "pairwise", "mobius"):
        parts = partitions_for(m, opps)
        t0 = time.perf_counter()
        mass, win = partition_mass_win(R, 0, parts, sd, alive, LESS, EQ)
        mass[np.arange(trials), boards] = 0.0          # the hero cannot hold the board card
        win[np.arange(trials), boards] = 0.0
        out[m] = (mass, win)
        timing[m] = (time.perf_counter() - t0) / trials
    if D ** n <= 4e5:
        t0 = time.perf_counter()
        out["brute"] = brute_mass_win(R, boards, s, n)
        timing["brute"] = (time.perf_counter() - t0) / trials
        assert np.allclose(out["brute"][0], out["mobius"][0], rtol=1e-9, atol=1e-9)
        assert np.allclose(out["brute"][1], out["mobius"][1], rtol=1e-9, atol=1e-9)
    me, we = out["mobius"]
    ok = np.ones((trials, D), bool)
    ok[np.arange(trials), boards] = False
    ok &= me > 1e-12 * me.max()
    eq_exact = we[ok] / me[ok]
    # effective opponent range size (participation ratio), averaged
    p = R[1:] / R[1:].sum(2, keepdims=True)                       # [m, T, D] normalised ranges
    d_eff = float(np.mean(1.0 / (p[0] ** 2).sum(1)))
    row = dict(N=n, D=D, family=family, D_eff=round(d_eff, 1),
               timing_us={k: 1e6 * v for k, v in timing.items()})
    for m in ("independent", "pairwise"):
        ma_, wa = out[m]
        good = ok & (ma_ > 0)            # PAIRWISE can over-subtract to a non-positive mass
        eq_a = wa[good] / ma_[good]
        eq_e = we[good] / me[good]
        row[m] = dict(mass_rel=float(np.mean(np.abs(ma_[ok] / me[ok] - 1))),
                      nonpos_frac=float(1 - good.sum() / ok.sum()),
                      equity_abs_mean=float(np.mean(np.abs(eq_a - eq_e))) if good.any() else float("nan"),
                      equity_abs_max=float(np.max(np.abs(eq_a - eq_e))) if good.any() else float("nan"))
    # first-order predictions from the actual ranges (size of the omitted Moebius terms):
    #   INDEPENDENT omits  sum_{pairs j<k} P(c_j = c_k)
    #   PAIRWISE omits     2 sum_{triples} P(c_j = c_k = c_l) + sum_{2 disjoint pairs} P(..)P(..)
    m_ = n - 1
    pair = {(j, k): (p[j] * p[k]).sum(1) for j in range(m_) for k in range(j + 1, m_)}
    row["pred_indep_mass"] = float(np.mean(sum(pair.values())))
    trip = sum((p[j] * p[k] * p[l]).sum(1) for j in range(m_) for k in range(j + 1, m_)
               for l in range(k + 1, m_)) if m_ >= 3 else np.zeros(trials)
    quad = np.zeros(trials)
    for (a, b_), v in pair.items():
        for (c_, d), w in pair.items():
            if len({a, b_, c_, d}) == 4 and (a, b_) < (c_, d):
                quad = quad + v * w
    row["pred_pair_mass"] = float(np.mean(2 * trip + quad))
    return row


def part_b():
    print("\n== Part B: isolated showdown, error/cost scaling (hero = seat 0, all alive)")
    rows, raw = [], []
    for family in ("uniform", "concentrated"):
        for n in (3, 4, 5, 6):
            for n_ranks in (4, 6, 8, 12, 16, 24, 32):
                if 2 * n_ranks < n + 2:
                    continue
                r = part_b_case(n, n_ranks, family)
                raw.append(r)
                tm = r["timing_us"]
                rows.append([family, n, r["D"], r["D_eff"],
                             f"{r['independent']['mass_rel']:.2e}", f"{r['pred_indep_mass']:.2e}",
                             f"{r['independent']['equity_abs_mean']:.2e}",
                             f"{r['independent']['equity_abs_max']:.2e}",
                             f"{r['pairwise']['mass_rel']:.2e}", f"{r['pred_pair_mass']:.2e}",
                             f"{r['pairwise']['equity_abs_mean']:.2e}",
                             f"{r['pairwise']['nonpos_frac']:.2f}",
                             round(tm["independent"], 1), round(tm["pairwise"], 1),
                             round(tm["mobius"], 1), round(tm["brute"], 1) if "brute" in tm else "-"])
                print(rows[-1], flush=True)
    hdr = ["ranges", "N", "D", "D_eff", "indep mass rel err", "pred sum_pairs P(c_j=c_k)",
           "indep equity err mean", "indep equity err max", "pairwise mass rel err",
           "pred (omitted 3-way + 2-pair terms)", "pairwise equity err mean",
           "pairwise frac. mass<=0",
           "us/terminal indep", "us pairwise", "us mobius(exact)", "us brute(exact)"]
    print(table(hdr, rows))
    return dict(h=hdr, rows=rows, raw=raw)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "a" in which:
        a3 = part_a(MAIN, 2000, ["exact", "independent", "pairwise"])
        cfg4 = GameConfig(n_players=4, n_ranks=4, n_suits=2, stack=15, raise_cap=1)
        a4 = part_a(cfg4, 400, ["exact", "independent", "pairwise", "mobius"])
        save("m1a", dict(a3=a3, a4=a4))
    if "b" in which:
        save("m1b", part_b())
