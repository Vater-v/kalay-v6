"""Terminal counterfactual values with explicit card removal.

For hero i holding card c at terminal t (board b, contributions x, alive set A):

    cfv_i(t, c) = sum over opponent cards (c_j)_{j != i}, all distinct, != c, != b of
                  prod_j r_j(t, c_j) * u_i(c, c_-i, b)
    u_i = -x_i + pot * share_i        (share_i = 0 if i folded / lost; fold pot to the survivor)

r_j(t, .) is opponent j's own reach (product of ITS action probabilities) - chance is uniform and
is folded into a constant (see cfr.py). Methods:

EXACT (brute force): materialise the joint opponent reach  P = r_j (x) r_k (x) ...  [T, D^(N-1)]
    and multiply by precomputed kernels valid(c, c_-i, b) and valid*share  [D^(N-1), D].
    Cost O(T * D^N). This is the reference.

PARTITION family (cheap): every opponent sum factorises over opponents if we IGNORE the
    "distinct cards" constraint among opponents. Inclusion-exclusion on the lattice of set
    partitions of the opponents (Moebius inversion) restores it exactly:

      sum_{distinct} f = sum_{partitions pi} mu(pi) * sum_{c constant on every block of pi} f
      mu(pi) = prod_{blocks B} (-1)^(|B|-1) (|B|-1)!

    For a fixed partition the sum factorises over blocks: block B behaves like ONE opponent with
    range rho_B(d) = prod_{j in B} r_j(d) whose |B cap A| alive members all hold card d.
    For a showdown the hero's expected share is read off the generating polynomial
        prod_B G_B(z),  G_B(z) = L_B + E_B z^{a_B}   (a_B = #alive in B > 0)   or  T_B  (a_B = 0)
    with L_B(c) = mass of d weaker than c, E_B(c) = mass of d != c tying c, T_B(c) = mass d != c.
    Hero share = sum_k coef_k / (k + 1)   (k = number of opponents tying the hero).
      INDEPENDENT = finest partition only (opponents treated as independent; only the hero's
                    own card and the board are removed).            error ~ C(m,2)/D_eff
      PAIRWISE    = finest partition + every single-pair partition (mu = -1): subtracts the
                    mass where two opponents hold the same card.    error ~ (C(m,3)+3C(m,4))/D_eff^2
      MOBIUS      = all partitions -> exact. Bell(m) partitions, m = N - 1 opponents.
    For N = 3 (m = 2) PAIRWISE == MOBIUS == EXACT: the only collision is c_j == c_k.
"""
from __future__ import annotations

import itertools

import numpy as np

from game import GameConfig, strength_table, showdown_share
from tree import PublicTree

METHODS = ("exact", "independent", "pairwise", "mobius")


def set_partitions(items):
    """All set partitions of the list `items` (as lists of tuples)."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in set_partitions(rest):
        # put `first` in its own block, or into one of the existing blocks
        yield [(first,)] + part
        for k in range(len(part)):
            yield part[:k] + [(first,) + part[k]] + part[k + 1:]


def mobius_coefficient(partition) -> int:
    """mu(0, pi) on the partition lattice = prod_B (-1)^(|B|-1) (|B|-1)!"""
    mu = 1
    for B in partition:
        k = len(B)
        mu *= (-1) ** (k - 1) * int(np.prod(np.arange(1, k)))
    return mu


def partitions_for(method: str, opps: list[int]):
    """[(mu, partition)] used by a PARTITION-family method."""
    out = []
    for part in set_partitions(list(opps)):
        sizes = sorted(len(B) for B in part)
        if method == "independent" and max(sizes, default=1) > 1:
            continue
        if method == "pairwise" and not (max(sizes, default=1) == 1 or
                                         (sizes.count(2) == 1 and max(sizes) == 2)):
            continue
        out.append((mobius_coefficient(part), part))
    return out


class TerminalEvaluator:
    """Computes cfv_i[t, c] for all terminals t of a PublicTree."""

    def __init__(self, tree: PublicTree, method: str = "exact"):
        assert method in METHODS
        self.method = method
        self.cfg: GameConfig = tree.cfg
        n, D = self.cfg.n_players, self.cfg.deck
        self.n, self.D = n, D
        self.nodes = tree.terminals
        self.board = tree.board[self.nodes]
        self.contrib = tree.contrib[self.nodes]                 # [T, N]
        self.pot = self.contrib.sum(1)
        self.alive = tree.alive[self.nodes]                      # [T, N]
        self.showdown = tree.showdown[self.nodes]
        self.strength = strength_table(self.cfg)                 # [b, c]
        # fold terminals: the unique survivor takes the pot
        self.fold_winner = np.where(self.showdown, -1, self.alive.argmax(1))
        # groups by board (-1 = no board, only street-1 fold terminals)
        self.by_board = {b: np.flatnonzero(self.board == b) for b in range(-1, D)}
        self.by_board = {b: ix for b, ix in self.by_board.items() if len(ix)}
        if method == "exact":
            self._kernels = {}
        else:
            # LESS[b][d, c] = s_b(d) < s_b(c); EQ[b][d, c] = s_b(d) == s_b(c), d != c
            s = self.strength
            self.LESS = (s[:, :, None] < s[:, None, :]).astype(float)
            self.EQ = ((s[:, :, None] == s[:, None, :]) & ~np.eye(D, dtype=bool)[None]).astype(float)
            self._parts = {i: partitions_for(method, [j for j in range(n) if j != i])
                           for i in range(n)}

    # ------------------------------------------------------------------ public API
    def cfv(self, i: int, reach: np.ndarray) -> np.ndarray:
        """reach: [N, M, D] (all nodes) or [N, T, D] (terminals only). Returns [T, D]."""
        R = reach[:, self.nodes, :] if reach.shape[1] != len(self.nodes) else reach
        if self.method == "exact":
            return self._cfv_exact(i, R)
        return self._cfv_partition(i, R)

    def base_utility(self, i: int):
        """cfv = A * mass + pot * win, with A_t = -x_i + pot*[i is the fold winner]."""
        return -self.contrib[:, i] + self.pot * (self.fold_winner == i)

    # ------------------------------------------------------------------ EXACT
    def _kernel(self, i: int, b: int, alive: tuple | None):
        """[D^(N-1), D] matrix: valid(c, c_-i, b) (alive None) or valid*share_i (alive set)."""
        key = (i, b, alive)
        if key in self._kernels:
            return self._kernels[key]
        n, D = self.n, self.D
        grids = np.indices((D,) * n)                             # grids[p] = card of player p
        valid = np.ones((D,) * n, dtype=bool)
        for p in range(n):
            for q in range(p + 1, n):
                valid &= grids[p] != grids[q]
            if b >= 0:
                valid &= grids[p] != b
        K = valid.astype(float)
        if alive is not None:
            st = self.strength[b][grids]                          # [N, D, ..., D]
            share = showdown_share(self.cfg, np.moveaxis(st, 0, -1), np.array(alive))[..., i]
            K = K * share
        # axes: hero first, then opponents in increasing seat order
        order = [i] + [j for j in range(n) if j != i]
        K = np.transpose(K, order).reshape(D, -1).T.copy()       # [D^(N-1), D]
        self._kernels[key] = K
        return K

    def _exact_groups(self, i):
        """Per board: (terminal ids, [(alive key, local ids, global ids)]) for hero i. Cached."""
        if not hasattr(self, "_groups"):
            self._groups = {}
        if i not in self._groups:
            g = []
            for b, ix in self.by_board.items():
                sd_local = np.flatnonzero(self.showdown[ix] & self.alive[ix, i])
                keys = {}
                for loc in sd_local:
                    keys.setdefault(tuple(self.alive[ix[loc]]), []).append(loc)
                g.append((b, ix, [(k, np.array(v), ix[np.array(v)]) for k, v in keys.items()]))
            self._groups[i] = g
        return self._groups[i]

    def _cfv_exact(self, i, R):
        T, D = R.shape[1], self.D
        out = np.zeros((T, D))
        opps = [j for j in range(self.n) if j != i]
        A = self.base_utility(i)
        for b, ix, sd_groups in self._exact_groups(i):
            P = R[opps[0], ix]
            for j in opps[1:]:                                   # joint opponent reach
                P = (P[:, :, None] * R[j, ix][:, None, :]).reshape(len(ix), -1)
            out[ix] = A[ix, None] * (P @ self._kernel(i, b, None))
            for key, loc, glob in sd_groups:                     # showdowns, by alive set
                out[glob] += self.pot[glob, None] * (P[loc] @ self._kernel(i, b, key))
        return out

    # ------------------------------------------------------------------ PARTITION family
    def _cfv_partition(self, i, R):
        T, D = R.shape[1], self.D
        A = self.base_utility(i)
        # masses of every block B (subset of opponents) that some partition uses
        blocks = sorted({B for _, part in self._parts[i] for B in part})
        rho = {}
        for B in blocks:
            r = R[B[0]].copy()
            for j in B[1:]:
                r *= R[j]                                        # all members hold the same card
            rho[B] = r
        Tm = {B: rho[B].sum(1, keepdims=True) - rho[B] for B in blocks}   # mass on d != c
        sd = self.showdown & self.alive[:, i]
        Lm, Em = {}, {}
        for B in blocks:
            Lm[B] = np.zeros((T, D)); Em[B] = np.zeros((T, D))
            for b, ix in self.by_board.items():
                ix = ix[sd[ix]]
                if b < 0 or not len(ix):
                    continue
                Lm[B][ix] = rho[B][ix] @ self.LESS[b]
                Em[B][ix] = rho[B][ix] @ self.EQ[b]
        mass = np.zeros((T, D))
        win = np.zeros((T, D))
        alive = self.alive
        for mu, part in self._parts[i]:
            m = np.ones((T, D))
            for B in part:
                m = m * Tm[B]
            mass += mu * m
            # generating polynomial in z (z counts opponents tying the hero)
            coef = np.ones((T, D, 1))
            for B in part:
                a_B = alive[:, list(B)].sum(1)                   # [T] alive members of B
                deg = coef.shape[2]
                new = np.zeros((T, D, deg + len(B)))
                folded_blk = (a_B == 0)[:, None, None]
                new[:, :, :deg] += coef * np.where(folded_blk, Tm[B][:, :, None], Lm[B][:, :, None])
                for a in range(1, len(B) + 1):
                    sel = (a_B == a)[:, None, None]
                    new[:, :, a:a + deg] += np.where(sel, coef * Em[B][:, :, None], 0.0)
                coef = new
            share = (coef / np.arange(1, coef.shape[2] + 1)).sum(2)
            win += mu * share
        out = A[:, None] * mass + np.where(sd[:, None], self.pot[:, None] * win, 0.0)
        # the hero cannot hold the board card
        has_b = self.board >= 0
        out[np.flatnonzero(has_b), self.board[has_b]] = 0.0
        return out
