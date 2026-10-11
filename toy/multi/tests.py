"""Correctness checks. Run: python3 tests.py   (takes ~20 s)

1. The vectorised fiber machinery (reach x terminal cfv x backward) reproduces the brute-force
   EV obtained by enumerating every concrete deal and walking the tree (scalar.py).
2. MOBIUS == EXACT for N = 2, 3, 4 on random reaches; PAIRWISE == EXACT for N = 3 only.
3. Zero-sum: sum_i EV_i = 0. Best response >= EV.
4. CFR+ on the 2-player game drives exploitability towards 0.
"""
import itertools
import time

import numpy as np

from game import GameConfig
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import evaluate, normalize, reach_all, CFRPlus, best_response_strategy
from scalar import ScalarTree

rng = np.random.default_rng(0)


def random_strategy(tree):
    return normalize(tree, rng.random((tree.M, tree.cfg.deck)) ** 3)


def test_bruteforce_ev():
    cfg = GameConfig(n_players=3, n_ranks=3, n_suits=2, stack=6, raise_cap=1)
    tree = PublicTree(cfg)
    sig = random_strategy(tree)
    st = ScalarTree(tree)
    D, N = cfg.deck, cfg.n_players
    tot, cnt = np.zeros(N), 0
    for deal in itertools.permutations(range(D), N + 1):   # N private cards + board
        tot += st.expected_value(sig, deal[:N], deal[N])
        cnt += 1
    brute = tot / cnt
    vec = evaluate(tree, sig)
    print("  brute-force EV", brute, "\n  vectorised  EV", vec["ev"])
    assert np.allclose(brute, vec["ev"], atol=1e-12)
    assert abs(vec["ev"].sum()) < 1e-12
    assert np.all(vec["delta"] >= -1e-12)
    # the best-response VALUE from the max-backward pass is achieved by an explicit policy
    for i in range(N):
        br = best_response_strategy(tree, sig, i)
        tot = np.zeros(N)
        for deal in itertools.permutations(range(D), N + 1):
            tot += st.expected_value(br, deal[:N], deal[N])
        print(f"  player {i}: BR value {vec['br'][i]:.12f}  brute-force EV of BR policy {tot[i]/cnt:.12f}")
        assert abs(vec["br"][i] - tot[i] / cnt) < 1e-12


def test_methods_agree():
    for n, ranks in ((2, 4), (3, 4), (4, 4)):
        cfg = GameConfig(n_players=n, n_ranks=ranks, n_suits=2, stack=8, raise_cap=1)
        tree = PublicTree(cfg)
        reach = reach_all(tree, random_strategy(tree))
        ex = TerminalEvaluator(tree, "exact")
        for m in ("mobius", "pairwise", "independent"):
            ap = TerminalEvaluator(tree, m)
            err = max(np.abs(ap.cfv(i, reach) - ex.cfv(i, reach)).max() for i in range(n))
            print(f"  N={n} D={cfg.deck} {m:12s} max|cfv - exact| = {err:.3e}")
            if m == "mobius" or (m == "pairwise" and n <= 3):
                assert err < 1e-9
            if m == "pairwise" and n == 4:
                assert err > 1e-6
            if m == "independent" and n >= 3:
                assert err > 1e-6


def test_cfr_2p():
    cfg = GameConfig(n_players=2, n_ranks=3, n_suits=2, stack=10, raise_cap=2)
    tree = PublicTree(cfg)
    s = CFRPlus(tree, TerminalEvaluator(tree, "exact"))
    s.iterate(1000)
    e = evaluate(tree, s.average_strategy())
    print("  2p Leduc-like, 1000 CFR+ iters: delta (mA) =", 1000 * e["delta"], " EV", e["ev"])
    assert e["nashconv"] < 2e-3


if __name__ == "__main__":
    for f in (test_bruteforce_ev, test_methods_agree, test_cfr_2p):
        t0 = time.time()
        print(f.__name__)
        f()
        print(f"  ok ({time.time() - t0:.1f}s)")
