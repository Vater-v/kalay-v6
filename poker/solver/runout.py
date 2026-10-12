"""Exact all-in runouts for two players: equity matrices over the board cards still to come.

W[h, o] = P(h beats o) + P(tie) / 2 over every runout of the missing board cards that avoids
h, o and the known board; 0 when h and o share a card or either is blocked by the board. Every
such pair has the same number of runouts (turn known: 52 - 4 - 4 = 44 river cards; flop known:
C(45, 2) = 990 turn-river pairs), so W is a plain average. At an all-in node the hero's value is
pot_net * (W @ w_opp)(h) - contribution * mass(h), exactly (shared by the solver and the checks).
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.solver import rank7
from poker.solver.cards import HANDS, MEMBER, NUM_HANDS, hand_ranks

_DISJOINT = ~(MEMBER.T.astype(np.int32) @ MEMBER.astype(np.int32)).astype(bool)        # [1326, 1326]


def _ranks(board5, table):
    """Hand classes on a full board: table lookup when the rank7 table exists, else the engine's
    evaluator (same order, ties shared)."""
    return hand_ranks(board5) if table is None else rank7.hand_ranks_fast(table, board5)


def _pair_score(ranks):
    """[1326, 1326] win + tie/2 of row hand against column hand on one full board (blocked = 0)."""
    ok = ranks >= 0
    r_h, r_o = ranks[:, None], ranks[None, :]
    score = (r_h > r_o).astype(jnp.float32) + 0.5 * (r_h == r_o).astype(jnp.float32)
    return jnp.where(ok[:, None] & ok[None, :], score, 0.0)


def equity_matrix_turn(board4, table="auto"):
    """board4 int32[4] (turn dealt) -> W [1326, 1326] over the 44 river cards."""
    return _turn(board4, rank7.cached_table() if table == "auto" else table)


@jax.jit
def _turn(board4, table):
    def body(c, acc):
        known = jnp.any(board4 == c)
        ranks = _ranks(jnp.concatenate([board4, c[None]]).astype(jnp.int32), table)
        return acc + jnp.where(known, 0.0, _pair_score(ranks))

    total = jax.lax.fori_loop(0, 52, body, jnp.zeros((NUM_HANDS, NUM_HANDS), jnp.float32))
    return jnp.where(jnp.asarray(_DISJOINT), total / 44.0, 0.0)


def equity_matrix_flop(board3, table="auto"):
    """board3 int32[3] (flop dealt) -> W [1326, 1326] over the 990 turn-river pairs that avoid
    both hands (every pair of the 1326 two-card combinations is a candidate runout)."""
    return _flop(board3, rank7.cached_table() if table == "auto" else table)


@jax.jit
def _flop(board3, table):
    runouts = jnp.asarray(HANDS)

    def body(r, acc):
        pair = runouts[r]
        known = jnp.any(board3[:, None] == pair[None, :])
        ranks = _ranks(jnp.concatenate([board3, pair]).astype(jnp.int32), table)
        return acc + jnp.where(known, 0.0, _pair_score(ranks))

    total = jax.lax.fori_loop(0, NUM_HANDS, body, jnp.zeros((NUM_HANDS, NUM_HANDS), jnp.float32))
    return jnp.where(jnp.asarray(_DISJOINT), total / 990.0, 0.0)


FLOP_CACHE = Path(__file__).resolve().parents[2] / "results" / "cache" / "flop_equity_v1"     # absolute: any cwd


def flop_equity_cached(flop, cache=FLOP_CACHE):
    """The exact all-in matrix of one flop (equity_matrix_flop), cached on disk per flop: the
    flop pool repeats boards across batches, stages and iterations. Atomic writes,
    so workers sharing the cache may race harmlessly."""
    flop = sorted(int(c) for c in np.asarray(flop))
    path = Path(cache) / f"{flop[0]}_{flop[1]}_{flop[2]}.npy"
    if path.exists():
        return jnp.asarray(np.load(path))
    w = np.asarray(equity_matrix_flop(jnp.asarray(flop, jnp.int32)), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.save(fh, w)
    os.replace(tmp, path)
    return jnp.asarray(w)


def disjoint_mask():
    return _DISJOINT


def check_turn_matrix(board4, samples=20, seed=0):
    """Host oracle: brute-force equity for random disjoint pairs (used by checks)."""
    from poker.envs.evaluator import evaluate_hand
    evaluate = jax.jit(evaluate_hand)
    rng = np.random.default_rng(seed)
    w = np.asarray(equity_matrix_turn(jnp.asarray(board4, jnp.int32)))
    board4 = [int(c) for c in board4]
    out = []
    while len(out) < samples:
        h, o = rng.integers(0, NUM_HANDS, 2)
        cards_h, cards_o = set(HANDS[h].tolist()), set(HANDS[o].tolist())
        if cards_h & cards_o or cards_h & set(board4) or cards_o & set(board4):
            continue
        score = 0.0
        rivers = [c for c in range(52) if c not in board4 and c not in cards_h and c not in cards_o]
        for c in rivers:
            full = board4 + [c]
            a = int(evaluate(jnp.asarray(list(HANDS[h]) + full, jnp.int32)))
            b = int(evaluate(jnp.asarray(list(HANDS[o]) + full, jnp.int32)))
            score += 1.0 if a > b else 0.5 if a == b else 0.0
        out.append((float(w[h, o]), score / len(rivers), len(rivers)))
    return out


def check_flop_matrix(board3, samples=6, seed=0):
    """Host oracle for the flop matrix: brute force over every turn-river pair for random pairs."""
    from poker.envs.evaluator import evaluate_hand
    evaluate = jax.jit(jax.vmap(evaluate_hand))
    rng = np.random.default_rng(seed)
    w = np.asarray(equity_matrix_flop(jnp.asarray(board3, jnp.int32)))
    board3 = [int(c) for c in board3]
    out = []
    while len(out) < samples:
        h, o = rng.integers(0, NUM_HANDS, 2)
        cards_h, cards_o = set(HANDS[h].tolist()), set(HANDS[o].tolist())
        if cards_h & cards_o or cards_h & set(board3) or cards_o & set(board3):
            continue
        rest = [c for c in range(52) if c not in board3 and c not in cards_h and c not in cards_o]
        runs = np.asarray([(a, b) for i, a in enumerate(rest) for b in rest[i + 1:]], np.int32)
        boards = np.concatenate([np.tile(board3, (len(runs), 1)), runs], axis=1)
        a = np.asarray(evaluate(jnp.asarray(np.concatenate([np.tile(HANDS[h], (len(runs), 1)), boards], 1))))
        b = np.asarray(evaluate(jnp.asarray(np.concatenate([np.tile(HANDS[o], (len(runs), 1)), boards], 1))))
        score = float(np.mean((a > b) + 0.5 * (a == b)))
        out.append((float(w[h, o]), score, len(runs)))
    return out


BOARDS_PER_PAIR_PREFLOP = 1712304             # C(48, 5): boards that avoid both hands


def all_boards() -> np.ndarray:
    """int8 [2598960, 5]: every 5-card board, lexicographic."""
    import itertools
    return np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(52), 5)),
                       dtype=np.int8).reshape(-1, 5)


@jax.jit
def _chunk_ranks(boards, table):
    return jax.vmap(lambda b: _ranks(b, table))(boards.astype(jnp.int32))        # [chunk, 1326]


def _score_sum(boards, table):
    """Sum over a chunk of full boards of win + tie/2 for every (row hand, column hand) pair. The
    ranks are materialised first (fused into the reduction XLA would recompute them per pair);
    the compare and the reduction over boards then fuse without storing [chunk, 1326, 1326]."""
    return _score_from_ranks(_chunk_ranks(boards, table))


@jax.jit
def _score_from_ranks(ranks):
    ok = ranks >= 0
    r_h, r_o = ranks[:, :, None], ranks[:, None, :]
    both = ok[:, :, None] & ok[:, None, :]
    return jnp.sum(jnp.where(both, (r_h > r_o) + 0.5 * (r_h == r_o), 0.0), axis=0)


def equity_matrix_preflop(chunk=256, boards=None, log=None, table="auto"):
    """W [1326, 1326] exact: average over the C(48, 5) boards that avoid both hands (0 for pairs
    that share a card). `boards` restricts the sum (tests); the normalisation assumes all boards."""
    boards = all_boards() if boards is None else np.asarray(boards, np.int8)
    table = rank7.cached_table() if table == "auto" else table
    total = jnp.zeros((NUM_HANDS, NUM_HANDS), jnp.float32)
    full = len(boards) // chunk * chunk
    for i in range(0, full, chunk):
        total = total + _score_sum(jnp.asarray(boards[i:i + chunk]), table)
        if log is not None and (i // chunk) % 500 == 0:
            log(f"preflop equity: {i}/{len(boards)} boards")
    if full < len(boards):
        rest = boards[full:]
        pad = np.repeat(rest[:1], chunk - len(rest), axis=0)
        part = _score_sum(jnp.asarray(np.concatenate([rest, pad])), table)
        dup = _score_sum(jnp.asarray(np.repeat(rest[:1], chunk, axis=0)), table) * ((chunk - len(rest)) / chunk)
        total = total + part - dup
    return jnp.where(jnp.asarray(_DISJOINT), total / BOARDS_PER_PAIR_PREFLOP, 0.0)


def preflop_equity_cached(path: Path, log=None):
    """The exact preflop matrix from `path` (.npy, float32), computed and saved on first use."""
    path = Path(path)
    if path.exists():
        return jnp.asarray(np.load(path))
    w = np.asarray(equity_matrix_preflop(log=log), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npy")     # one name per writer (static sniff S3)
    np.save(tmp, w)
    tmp.replace(path)
    return jnp.asarray(w)
