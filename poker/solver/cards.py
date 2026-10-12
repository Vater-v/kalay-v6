"""Hole-card combinations, preflop classes and per-board rank tables.

Engine convention (poker/envs/evaluator.py): card = suit * 13 + rank, rank 0 = deuce .. 12 = ace.
A hand index h in 0..1325 is the lexicographic pair (c1 < c2). Index 1326 is a sentinel
whose weight is always zero (used by PAIR_INDEX for impossible pairs).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import guards
from poker.envs.evaluator import evaluate_hand

NUM_HANDS = 1326
NUM_CLASSES = 169
HANDS = np.asarray(list(itertools.combinations(range(52), 2)), dtype=np.int32)    # [1326, 2]
PAIR_INDEX = np.full((52, 52), NUM_HANDS, dtype=np.int32)                         # {a,b} -> h
PAIR_INDEX[HANDS[:, 0], HANDS[:, 1]] = np.arange(NUM_HANDS)
PAIR_INDEX[HANDS[:, 1], HANDS[:, 0]] = np.arange(NUM_HANDS)
MEMBER = np.zeros((52, NUM_HANDS), dtype=bool)                                    # card c in hand h
MEMBER[HANDS[:, 0], np.arange(NUM_HANDS)] = True
MEMBER[HANDS[:, 1], np.arange(NUM_HANDS)] = True
CARD_HANDS = np.stack([np.flatnonzero(MEMBER[c]) for c in range(52)]).astype(np.int32)  # [52, 51]


def hand_index(cards):
    """[2] card ids -> hand index 0..1325. Concrete (host) cards are validated first (poker.envs.guards.cards: two
    distinct ids in 0..51, ValueError): a pair gives the sentinel 1326, which a jax gather clamps to hand 1325,
    and -1 wraps to card 51 (edge_cases C1). Traced cards are the engine's own deal."""
    if guards.concrete(cards):
        guards.cards(cards, "hand_index cards", count=2)
    a, b = jnp.minimum(cards[0], cards[1]), jnp.maximum(cards[0], cards[1])
    return jnp.asarray(PAIR_INDEX)[a, b]


def _class_of(c1: int, c2: int) -> int:
    """169 classes: 13 pairs, 78 suited, 78 offsuit; index 13*hi+lo layout like a grid."""
    r1, r2, s1, s2 = c1 % 13, c2 % 13, c1 // 13, c2 // 13
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return 13 * hi + hi
    return 13 * hi + lo if s1 == s2 else 13 * lo + hi


CLASS_OF = np.asarray([_class_of(a, b) for a, b in HANDS], dtype=np.int32)       # [1326] in 0..168
CLASS_SIZE = np.bincount(CLASS_OF, minlength=NUM_CLASSES)                         # 6 / 4 / 12


def board_valid(board: np.ndarray) -> np.ndarray:
    """bool[..., 1326]: hands that do not use any of the given board cards (dealt cards only: an id outside
    0..51 or a card twice on one board is a ValueError; -1 would index card 51, edge_cases C2)."""
    board = np.asarray(board)
    if board.size:
        rows = board.reshape(-1, board.shape[-1]).astype(np.int64)
        bad = np.any((rows < 0) | (rows >= 52), axis=1) | np.any(np.diff(np.sort(rows, axis=1), axis=1) == 0, axis=1)
        for i in np.flatnonzero(bad)[:1]:
            guards.cards(rows[i], f"board_valid board (row {i})")
    return ~MEMBER[board].any(axis=-2)


def hand_ranks(board: jax.Array) -> jax.Array:
    """int32[1326] seven-card scores for one 5-card river board; blocked hands get -1. A concrete board is
    validated first (five distinct ids in 0..51, ValueError): -1 would be scored as an ace (edge_cases C2)."""
    if guards.concrete(board):
        guards.cards(board, "hand_ranks board", count=5)
    hands = jnp.asarray(HANDS)
    cards = jnp.concatenate((hands, jnp.broadcast_to(board, (NUM_HANDS, 5))), axis=1)
    scores = jax.vmap(evaluate_hand)(cards)
    blocked = jnp.any(hands[:, :, None] == board[None, None, :], axis=(1, 2))
    return jnp.where(blocked, jnp.int32(-1), scores)


def rank_tables(ranks: jax.Array) -> dict:
    """Sorted positions used by showdown.py, for one board. All shapes are static.

    lt/le: number of hands with rank < / <= rank(h) among all 1326 (blocked hands have
    rank -1 and zero weight, so counting them is harmless).
    card_order[c]: CARD_HANDS[c] sorted by rank; klt/kle[c, h]: number of hands containing c
    with rank < / <= rank(h).
    """
    order = jnp.argsort(ranks, stable=True)
    sorted_ranks = ranks[order]
    lt = jnp.searchsorted(sorted_ranks, ranks, side="left")
    le = jnp.searchsorted(sorted_ranks, ranks, side="right")
    card_hands = jnp.asarray(CARD_HANDS)
    card_ranks = ranks[card_hands]                                                   # [52, 51]
    card_sort = jnp.argsort(card_ranks, axis=1, stable=True)
    card_order = jnp.take_along_axis(card_hands, card_sort, axis=1)
    card_sorted = jnp.take_along_axis(card_ranks, card_sort, axis=1)
    klt = jax.vmap(lambda row: jnp.searchsorted(row, ranks, side="left"))(card_sorted)   # [52, 1326]
    kle = jax.vmap(lambda row: jnp.searchsorted(row, ranks, side="right"))(card_sorted)
    return {"ranks": ranks, "order": order, "lt": lt.astype(jnp.int32), "le": le.astype(jnp.int32),
            "card_order": card_order, "klt": klt.astype(jnp.int32), "kle": kle.astype(jnp.int32)}
