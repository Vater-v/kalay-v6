"""Scalar (one hidden deal at a time) view of the same public tree.

Used by (1) the brute-force reference in tests.py and (2) the sampling algorithms in mccfr.py.
A "history" = public node + the full deal (c_1..c_N, board). Nothing here is vectorised;
it is the classic way poker AIs walk a game tree.
"""
from __future__ import annotations

import numpy as np

from tree import PublicTree, DECISION, CHANCE, TERMINAL
from game import strength_table


class ScalarTree:
    def __init__(self, tree: PublicTree):
        self.tree = tree
        self.N, self.D = tree.cfg.n_players, tree.cfg.deck
        self.kind = tree.kind.tolist()
        self.player = tree.player.tolist()
        fc, nc = tree.first_child.tolist(), tree.n_children.tolist()
        self.children = [list(range(f, f + k)) if k else [] for f, k in zip(fc, nc)]
        self.contrib = tree.contrib.tolist()
        self.alive = tree.alive.tolist()
        self.pot = tree.contrib.sum(1).tolist()
        self.showdown = tree.showdown.tolist()
        self.strength = strength_table(tree.cfg).tolist()

    def utility(self, node: int, cards, board: int) -> list[float]:
        """Payoff vector (antes) at a terminal node for a concrete deal."""
        x, alive, pot = self.contrib[node], self.alive[node], self.pot[node]
        u = [-xi for xi in x]
        if not self.showdown[node]:
            u[alive.index(True)] += pot
            return u
        s = self.strength[board]
        st = [s[cards[p]] if alive[p] else -1 for p in range(self.N)]
        best = max(st)
        winners = [p for p in range(self.N) if alive[p] and st[p] == best]
        for p in winners:
            u[p] += pot / len(winners)
        return u

    def expected_value(self, sigma: np.ndarray, cards, board: int) -> np.ndarray:
        """Exact EV vector of a behaviour profile for ONE deal (recursive enumeration)."""
        sig = sigma

        def rec(n):
            k = self.kind[n]
            if k == TERMINAL:
                return np.array(self.utility(n, cards, board))
            if k == CHANCE:
                return rec(self.children[n][board])
            p = self.player[n]
            return sum(sig[ch, cards[p]] * rec(ch) for ch in self.children[n])
        return rec(0)
