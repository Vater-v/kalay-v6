"""Street-boundary value functions V(public state, ranges) and leaf evaluation.

Normalised interface (used by every value source):
    u0, u1 = V(board_ids, x, p0, p1)
  board_ids : public board at the NEXT street root (after the card is dealt)
  x         : chips each player has in the pot (equal at a boundary)
  p0, p1    : NORMALISED ranges (sum 1, zero on board cards)
  u_i[h]    : value of player i holding h, in units of x, against the
              normalised opponent range, such that the absolute CFV is
              v_i(S, h) = cscale(S) * x * M_opp * u_i[h],  M_opp = sum of opp reach.
Because CFVs are homogeneous of degree 1 in the opponent's reach and the
equilibrium of a subgame does not change when a range is rescaled, this
normalisation loses nothing.

Value sources:
  checkdown_V : both players check to showdown; u0 = W p1, u1 = W p0
                (exact, bilinear, zero strategic content)
  oracle labels: solve the remaining game exactly with CFR+ (oracle_label)
  learned     : MLP trained on oracle (or bootstrapped) labels; it predicts the
                RESIDUAL u - u_checkdown (or the raw u, for the ablation)
"""
from __future__ import annotations

import numpy as np

from game import PState
from tree import PublicTree
from cfr import Solver, values


# ---------------------------------------------------------------------------
class LeafEvaluator:
    """Turns a normalised V into the LEAF counterfactual values a Solver needs.
    A LEAF L sits where the street closed; its value is the sum over the next
    public card c of the value at the next street root S_c = L + card c."""

    def __init__(self, tree: PublicTree, V):
        N = tree.N
        B = tree.B
        self.V = V
        pl, pc, bid = [], [], []
        for j, n in enumerate(tree.L):
            st = tree.states[n]
            for c in range(N):
                if c not in st.board:
                    pl.append(j); pc.append(c); bid.append(B.id(st.board + (c,)))
        self.pl, self.pc, self.bid = np.array(pl), np.array(pc), np.array(bid)
        Lp = tree.L[self.pl]
        nb = np.array([len(tree.states[n].board) for n in Lp])
        self.x = tree.contrib[Lp, 0]
        self.cs = tree.cscale[Lp] / (N - 2 - nb)
        self.cmask = (np.arange(N)[None, :] != self.pc[:, None]).astype(float)
        self.valid = self.cmask * B.mask[self.bid]
        self.starts = np.nonzero(np.concatenate([[True], self.pl[1:] != self.pl[:-1]]))[0]
        self.next_keys = [tree.states[tree.L[j]].key + f"/d{c}" for j, c in zip(pl, pc)]

    def cfv_pairs(self, r0L, r1L):
        """Absolute CFVs at every next-street root, shape [pairs, N] x 2."""
        r0 = r0L[self.pl] * self.cmask
        r1 = r1L[self.pl] * self.cmask
        M0, M1 = r0.sum(1), r1.sum(1)
        p0 = r0 / np.where(M0 > 0, M0, 1)[:, None]
        p1 = r1 / np.where(M1 > 0, M1, 1)[:, None]
        u0, u1 = self.V(self.bid, self.x, p0, p1)
        c0 = (self.cs * self.x * M1)[:, None] * u0 * self.valid
        c1 = (self.cs * self.x * M0)[:, None] * u1 * self.valid
        return c0, c1, r0, r1

    def __call__(self, i, r0L, r1L):
        c = self.cfv_pairs(r0L, r1L)[i]
        return np.add.reduceat(c, self.starts, axis=0)


class CheckdownV:
    """u0[h] = sum_h' W[h,h'] p1[h'] : exact equity x pot share if nobody bets."""

    def __init__(self, boards):
        self.W = boards.W

    def __call__(self, bid, x, p0, p1):
        W = self.W[bid]
        return np.einsum("pij,pj->pi", W, p1), np.einsum("pij,pj->pi", W, p0)


class ByStreetV:
    """Dispatch on the board length: 1 card -> flop value fn, 2 -> turn."""

    def __init__(self, boards, v_flop, v_turn):
        self.blen = np.array([len(b) for b in boards.boards])
        self.vf, self.vt = v_flop, v_turn

    def __call__(self, bid, x, p0, p1):
        u0 = np.zeros_like(p0); u1 = np.zeros_like(p1)
        for L, V in ((1, self.vf), (2, self.vt)):
            m = self.blen[bid] == L
            if m.any():
                a, b = V(bid[m], x[m], p0[m], p1[m])
                u0[m], u1[m] = a, b
        return u0, u1


# ---------------------------------------------------------------------------
def reachable_x(full: PublicTree, street: int):
    """Pot contributions x at which `street` closes without an all-in."""
    from tree import CHANCE
    m = (full.kind == CHANCE) & (full.street == street)
    return np.unique(full.contrib[m, 0])


def sample_situations(cfg, boards, n, board_len, xs, rng):
    """Random boundary situations: board, pot share x, two random ranges.
    Ranges: logits = a * (hand strength z-score) + s * noise, a random subset of
    hands removed, normalised.  a ~ U(-3,3) gives both value-heavy and
    capped/condensed ranges; this is the only 'prior' about ranges we use."""
    from game import strength
    N = cfg.n_cards
    bids = np.empty(n, dtype=np.int64)
    P = np.zeros((2, n, N))
    X = rng.choice(xs, size=n)
    for j in range(n):
        b = tuple(int(c) for c in rng.choice(N, size=board_len, replace=False))
        bids[j] = boards.id(b)
        mask = boards.mask[bids[j]]
        st = np.array([strength(cfg, h, b) for h in range(N)], float)
        z = (st - st[mask > 0].mean()) / (st[mask > 0].std() + 1e-9)
        for p in (0, 1):
            a = rng.uniform(-3, 3)
            s = rng.uniform(0.0, 2.0)
            lg = a * z + s * rng.standard_normal(N)
            keep = rng.random(N) > rng.uniform(0.0, 0.6)
            w = np.exp(lg - lg.max()) * mask * keep
            if w.sum() <= 0:
                w = mask.copy()
            P[p, j] = w / w.sum()
    return bids, X, P[0], P[1]


def solve_situations(cfg, boards, bids, X, p0, p1, iters, V_leaf=None, chunk=4000):
    """Exact (V_leaf=None: solve to the end of the game) or bootstrapped
    (V_leaf given: solve only the current street, leaves valued by V_leaf)
    counterfactual values at the root of each situation, normalised.
    Returns u0, u1 [n, N] and the mean exploitability of the solves (x units)."""
    n = len(bids)
    U0 = np.zeros((n, cfg.n_cards)); U1 = np.zeros_like(U0)
    expl = []
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        roots = []
        for j in range(sl.start, sl.stop):
            b = boards.boards[bids[j]]
            roots.append(PState(len(b), b, (float(X[j]), float(X[j])), 0, 0, 0, f"s{j}"))
        T = PublicTree(cfg, boards, roots, [1.0] * len(roots), depth_limited=V_leaf is not None)
        leaf = LeafEvaluator(T, V_leaf) if (V_leaf is not None and len(T.L)) else None
        s = Solver(T, [p0[sl], p1[sl]], leaf_fn=leaf)
        s.iterate(iters)
        e0, e1 = s.avg_eff(0), s.avg_eff(1)
        v0, v1, _, _ = values(s, e0, e1)
        x = X[sl][:, None]
        U0[sl] = v0[: T.n_roots] / x
        U1[sl] = v1[: T.n_roots] / x
        b0, _, _, _ = values(s, e0, e1, best_response_for=0)
        _, b1, _, _ = values(s, e0, e1, best_response_for=1)
        g = ((b0[: T.n_roots] * p0[sl]).sum(1) + (b1[: T.n_roots] * p1[sl]).sum(1)) / 2 / X[sl]
        expl.append(g)
    return U0, U1, float(np.concatenate(expl).mean())


# ---------------------------------------------------------------------------
# Tiny MLP with manual backprop (numpy only)
# ---------------------------------------------------------------------------
def features(boards, bid, x, p0, p1, stack, residual=True):
    """[N*p0, N*p1, board multi-hot, x/stack, (checkdown u0, u1 if residual)]"""
    N = p0.shape[1]
    bm = 1.0 - boards.mask[bid]
    cols = [N * p0, N * p1, bm, (x / stack)[:, None]]
    if residual:
        cd0, cd1 = CheckdownV(boards)(bid, x, p0, p1)
        cols += [cd0, cd1]
    return np.concatenate(cols, 1)


class MLP:
    """x -> ReLU(W1 x) -> ReLU(W2 .) -> W3 . -> zero-sum projection.
    Zero-sum projection (DeepStack): e = p0.y0 + p1.y1 ; y_i -= e/2, so that the
    predicted values satisfy p0.u0 + p1.u1 = 0 exactly (a zero-sum game)."""

    def __init__(self, d_in, n_out, hidden=(128, 128), seed=0):
        rng = np.random.default_rng(seed)
        dims = [d_in, *hidden, n_out]
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]
        self.W[-1] *= 0.1

    def _fwd(self, X):
        hs = [X]
        h = X
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if k < len(self.W) - 1:
                h = np.maximum(h, 0)
            hs.append(h)
        return hs

    @staticmethod
    def _zero_sum(y, p0, p1):
        N = p0.shape[1]
        e = (p0 * y[:, :N]).sum(1) + (p1 * y[:, N:]).sum(1)
        return y - 0.5 * e[:, None]

    def predict(self, X, p0, p1):
        """Inference in float32 (2x faster matmuls); training stays float64."""
        h = X.astype(np.float32)
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W.astype(np.float32) + b.astype(np.float32)
            if k < len(self.W) - 1:
                h = np.maximum(h, 0)
        return self._zero_sum(h.astype(np.float64), p0, p1)

    def fit(self, X, Y, M, p0, p1, Xv, Yv, Mv, p0v, p1v, steps=4000, lr=1e-3,
            batch=256, wd=1e-4, seed=0, eval_every=200):
        """Adam on masked MSE; keeps the parameters with the best validation loss."""
        rng = np.random.default_rng(seed)
        N = p0.shape[1]
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        best = (np.inf, None)
        n = len(X)
        for step in range(1, steps + 1):
            idx = rng.integers(0, n, size=min(batch, n))
            xb, yb, mb, a0, a1 = X[idx], Y[idx], M[idx], p0[idx], p1[idx]
            hs = self._fwd(xb)
            out = self._zero_sum(hs[-1], a0, a1)
            g = 2 * (out - yb) * mb / mb.sum()
            # backprop through the zero-sum projection
            s = g.sum(1, keepdims=True)
            g = g - 0.5 * s * np.concatenate([a0, a1], 1)
            gW, gb = [None] * len(self.W), [None] * len(self.b)
            for k in range(len(self.W) - 1, -1, -1):
                gW[k] = hs[k].T @ g + wd * self.W[k]
                gb[k] = g.sum(0)
                if k > 0:
                    g = (g @ self.W[k].T) * (hs[k] > 0)
            grads = gW + gb
            for j, (p, gr) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                p -= lr * (m[j] / (1 - 0.9 ** step)) / (np.sqrt(v[j] / (1 - 0.999 ** step)) + 1e-8)
            if step % eval_every == 0 or step == steps:
                pv = self.predict(Xv, p0v, p1v)
                lv = float((((pv - Yv) ** 2) * Mv).sum() / Mv.sum())
                if lv < best[0]:
                    best = (lv, ([w.copy() for w in self.W], [b.copy() for b in self.b]))
        self.W, self.b = best[1]
        return best[0]


class LearnedV:
    """V(board, x, p0, p1) = checkdown + MLP residual  (or raw MLP)."""

    def __init__(self, boards, mlp, stack, residual=True):
        self.boards, self.mlp, self.stack, self.residual = boards, mlp, stack, residual
        self.cd = CheckdownV(boards)

    def __call__(self, bid, x, p0, p1):
        N = p0.shape[1]
        X = features(self.boards, bid, x, p0, p1, self.stack, self.residual)
        y = self.mlp.predict(X, p0, p1)
        u0, u1 = y[:, :N], y[:, N:]
        if self.residual:
            c0, c1 = self.cd(bid, x, p0, p1)
            u0, u1 = u0 + c0, u1 + c1
        m = self.boards.mask[bid]
        return u0 * m, u1 * m


def train_value_net(boards, cfg, data, val, residual=True, steps=4000, seed=0, hidden=(128, 128),
                    range_weighted=False):
    """data/val = (bid, x, p0, p1, u0, u1). Returns LearnedV and val loss.
    range_weighted=True weighs the squared error of u_i[h] by 0.1 + N*p_i[h]:
    the value of a hand the player does not hold is not pinned down by the
    equilibrium (non-unique, noisy label), the value of held hands is."""
    def prep(d):
        bid, x, p0, p1, u0, u1 = d
        X = features(boards, bid, x, p0, p1, cfg.stack, residual)
        Y = np.concatenate([u0, u1], 1)
        if residual:
            c0, c1 = CheckdownV(boards)(bid, x, p0, p1)
            Y = Y - np.concatenate([c0, c1], 1)
        m = boards.mask[bid]
        Mk = np.concatenate([m, m], 1)
        if range_weighted:
            N = p0.shape[1]
            Mk = Mk * (0.1 + N * np.concatenate([p0, p1], 1))
        return X, Y, Mk, p0, p1
    X, Y, Mk, p0, p1 = prep(data)
    Xv, Yv, Mv, p0v, p1v = prep(val)
    mlp = MLP(X.shape[1], Y.shape[1], hidden=hidden, seed=seed)
    batch = min(256, len(X))
    vl = mlp.fit(X, Y, Mk, p0, p1, Xv, Yv, Mv, p0v, p1v, steps=steps, batch=batch, seed=seed)
    return LearnedV(boards, mlp, cfg.stack, residual), vl


def rmse(boards, V, d):
    """Root-mean-square error of V on labelled data d, in units of x (pot share)."""
    bid, x, p0, p1, u0, u1 = d
    a0, a1 = V(bid, x, p0, p1)
    m = boards.mask[bid]
    err = ((a0 - u0) ** 2 * m).sum() + ((a1 - u1) ** 2 * m).sum()
    return float(np.sqrt(err / (2 * m.sum())))
