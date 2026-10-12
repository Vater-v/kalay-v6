"""Independent references for the solver audit (no poker.solver code used for the math).

- eval7(cards [N, 7]) -> int64 score: an independent 7-card evaluator (best of 21 five-card subsets,
  category * 13^5 + ordered tie-break ranks). Card = suit * 13 + rank, rank 0 = deuce (engine convention).
- hands(): the 1326 two-card combos in lexicographic order (same order as poker.solver.cards.HANDS,
  rebuilt here with itertools).
- disjoint matrix and dense score matrices built from explicit card sets.
"""
import itertools
import numpy as np

HANDS = np.asarray(list(itertools.combinations(range(52), 2)), np.int64)
F = len(HANDS)
SUBSETS = np.asarray(list(itertools.combinations(range(7), 5)), np.int64)   # [21, 5]


def _eval5(c):
    """c [N, 5] cards -> int64 score."""
    r = c % 13
    s = c // 13
    flush = np.all(s == s[:, :1], axis=1)
    counts = np.zeros((len(c), 13), np.int64)
    np.add.at(counts, (np.repeat(np.arange(len(c)), 5), r.ravel()), 1)
    # order ranks by (count desc, rank desc)
    key = counts * 16 + np.arange(13)[None, :]
    order = np.argsort(-key, axis=1)
    cnt_sorted = np.take_along_axis(counts, order, axis=1)
    distinct = (counts > 0).sum(1)
    rs = np.sort(r, axis=1)
    wheel = np.all(rs == np.array([0, 1, 2, 3, 12]), axis=1)
    straight = (distinct == 5) & ((rs[:, 4] - rs[:, 0] == 4) | wheel)
    top = np.where(wheel, 3, rs[:, 4])
    # tie-break: expand ranks by multiplicity in (count desc, rank desc) order
    tb = np.zeros((len(c), 5), np.int64)
    # vectorised expansion: repeat each ordered rank by its count, take first 5
    cum = np.cumsum(cnt_sorted, axis=1)
    for k in range(5):
        idx = np.argmax(cum > k, axis=1)
        tb[:, k] = np.take_along_axis(order, idx[:, None], axis=1)[:, 0]
    c0, c1 = cnt_sorted[:, 0], cnt_sorted[:, 1]
    cat = np.zeros(len(c), np.int64)
    cat = np.where(c0 == 2, 1, cat)
    cat = np.where((c0 == 2) & (c1 == 2), 2, cat)
    cat = np.where(c0 == 3, 3, cat)
    cat = np.where(straight, 4, cat)
    cat = np.where(flush, 5, cat)
    cat = np.where((c0 == 3) & (c1 == 2), 6, cat)
    cat = np.where(c0 == 4, 7, cat)
    cat = np.where(straight & flush, 8, cat)
    tie = (((tb[:, 0] * 13 + tb[:, 1]) * 13 + tb[:, 2]) * 13 + tb[:, 3]) * 13 + tb[:, 4]
    tie = np.where(straight, top, tie)
    return cat * 13 ** 5 + tie


def eval7(cards):
    cards = np.asarray(cards, np.int64)
    n = len(cards)
    sub = cards[:, SUBSETS]                                   # [N, 21, 5]
    sc = _eval5(sub.reshape(-1, 5)).reshape(n, 21)
    return sc.max(axis=1)


def ranks_on_board(board):
    """[F] independent score of every hand on a 5-card board; -1 for board-blocked hands."""
    board = np.asarray(board, np.int64)
    blocked = np.isin(HANDS, board).any(axis=1)
    cards = np.concatenate([HANDS, np.broadcast_to(board, (F, 5))], axis=1)
    out = np.full(F, -1, np.int64)
    out[~blocked] = eval7(cards[~blocked])
    return out


def disjoint():
    m = np.zeros((F, 52), bool)
    m[np.arange(F), HANDS[:, 0]] = True
    m[np.arange(F), HANDS[:, 1]] = True
    return (m.astype(np.int64) @ m.T.astype(np.int64)) == 0, m


def score_matrix(ranks):
    """S[h, o] = 1 win, .5 tie, 0 loss (both valid), else 0."""
    ok = ranks >= 0
    rh, ro = ranks[:, None], ranks[None, :]
    s = (rh > ro) + 0.5 * (rh == ro)
    return np.where(ok[:, None] & ok[None, :], s, 0.0)
