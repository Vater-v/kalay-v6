"""Sanity checks of the public-tree/fiber implementation (run: python3 tests.py).

1. W[board] is antisymmetric and zero on impossible pairs.
2. The vectorised expected value of a RANDOM strategy profile equals a
   brute-force enumeration over every deal (h0, h1, flop, turn) that walks the
   tree with the real hidden cards -> validates card removal and chance weights.
3. Exploitability is >= 0 and BR values bound the profile value.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import permutations
import numpy as np
from game import GameConfig, Boards, strength
from tree import full_game_tree, DEC, CHANCE, FOLD, SHOW
from cfr import Solver, exploitability, values


def brute_force_ev0(T, eff0, eff1):
    cfg = T.cfg
    n = cfg.n_cards
    eff = [eff0, eff1]
    total, cnt = 0.0, 0
    for h0, h1, c1, c2 in permutations(range(n), 4):
        hands = (h0, h1)
        board_full = (c1, c2)

        def val(node):
            k = T.kind[node]
            if k == FOLD:
                f = T.player[node]
                c = T.contrib[node]
                return -c[0] if f == 0 else c[1]
            if k == SHOW:
                s0, s1 = strength(cfg, h0, board_full), strength(cfg, h1, board_full)
                return T.contrib[node][0] * np.sign(s0 - s1)
            cs, cc = T.child_start[node], T.n_child[node]
            if k == CHANCE:
                want = board_full[len(T.states[node].board)]
                for j in range(cs, cs + cc):
                    if T.card[j] == want:
                        return val(j)
                raise RuntimeError
            p = T.player[node]
            return sum(eff[p][j, hands[p]] * val(j) for j in range(cs, cs + cc))
        total += val(0)
        cnt += 1
    return total / cnt


if __name__ == "__main__":
    cfg = GameConfig(n_ranks=3, n_suits=2, stack=6.0)
    B = Boards(cfg)
    for W, m in zip(B.W, B.mask):
        assert np.allclose(W, -W.T)
        assert np.all(W[m == 0] == 0) and np.all(W[:, m == 0] == 0)
    print("1. W antisymmetric, zero on impossible pairs: ok")
    T = full_game_tree(cfg, B)
    S = Solver(T, [T.bmask[:1].copy()] * 2)
    rng = np.random.default_rng(0)
    for p in (0, 1):  # random behavioural strategies
        E = T.E[p]
        x = rng.random((len(E), T.N))
        tot = np.repeat(np.add.reduceat(x, T.E_starts[p], axis=0), T.E_cnt[p], axis=0)
        S.eff[p][E] = x / tot
    v0, v1, r0, r1 = values(S, S.eff[0], S.eff[1])
    ev_vec = float(v0[0].sum())
    ev_bf = brute_force_ev0(T, S.eff[0], S.eff[1])
    print(f"2. EV0 vectorised {ev_vec:.12f}  brute force {ev_bf:.12f}  "
          f"zero-sum check {v0[0].sum() + v1[0].sum():.2e}")
    assert abs(ev_vec - ev_bf) < 1e-10
    e = exploitability(S, S.eff[0], S.eff[1])
    assert e["BR0"] >= e["EV0"] - 1e-12 and e["BR1"] >= -e["EV0"] - 1e-12 and e["expl"] >= 0
    S2 = Solver(T, [T.bmask[:1].copy()] * 2); S2.iterate(300)
    e2 = exploitability(S2)
    print(f"3. random profile expl {e['expl']:.4f}; CFR+ 300 it expl {e2['expl']:.6f}: ok")
