"""Seven-card hand strength by table lookup: one gather instead of poker.envs.evaluator.evaluate_hand.

The table holds, for every 7-card set in combinadic order (C(52, 7) = 133,784,560 entries), the
class of its best five cards: 0..K-1 in increasing strength (ties share a class), int16. It is
built once from evaluate_hand itself (the engine's evaluator stays the definition) and cached;
build_table checks a sample against evaluate_hand. Index of sorted cards c0 < ... < c6:
sum_k C(c_k, k + 1). Board-blocked hands get -1 like poker.solver.cards.hand_ranks.
"""

import os
import time
from math import comb
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs.evaluator import evaluate_hand
from poker.solver.cards import HANDS, NUM_HANDS

N7 = comb(52, 7)
BINOM = np.asarray([[comb(c, k) for c in range(52)] for k in range(8)], np.int64)        # [8, 52]
# Absolute (from the repository root): a relative path made cached_table() return None from any other
# working directory, silently (the slow evaluator then, or a 10-minute rebuild in poker.solver.loop).
TABLE_PATH = Path(__file__).resolve().parents[2] / "results" / "cache" / "rank7_v1.npy"


def combo_index(cards):
    """[..., 7] sorted ascending card ids -> int32 combinadic index."""
    b = jnp.asarray(BINOM[1:8], jnp.int32)                                                 # [7, 52]
    return jnp.sum(b[jnp.arange(7), cards], axis=-1)


def unrank(idx):
    """[N] int32 combinadic indices -> [N, 7] sorted card ids (vectorised greedy)."""
    out = []
    rest = jnp.asarray(idx, jnp.int32)
    for k in range(7, 0, -1):
        row = jnp.asarray(BINOM[k], jnp.int32)                                             # C(c, k), c = 0..51
        c = jnp.searchsorted(row, rest, side="right") - 1
        rest = rest - row[c]
        out.append(c)
    return jnp.stack(out[::-1], axis=-1).astype(jnp.int32)


@jax.jit
def _scores(idx):
    return jax.vmap(evaluate_hand)(unrank(idx))


def build_table(chunk=1 << 20, log=print):
    """All C(52, 7) scores -> int16 classes (sorted unique scores). ~10 min on a GTX 1070."""
    scores = np.empty(N7, np.int32)
    t0 = time.time()
    for start in range(0, N7, chunk):
        stop = min(N7, start + chunk)
        idx = np.arange(start, start + chunk, dtype=np.int64)
        idx = np.minimum(idx, N7 - 1).astype(np.int32)
        scores[start:stop] = np.asarray(_scores(jnp.asarray(idx)))[:stop - start]
        if log is not None and (start // chunk) % 16 == 0:
            log(f"rank7: {stop}/{N7} ({time.time() - t0:.0f} s)")
    values = np.unique(scores)
    table = np.searchsorted(values, scores).astype(np.int16)
    # Spot check against the evaluator itself.
    rng = np.random.default_rng(7)
    sample = np.sort(np.stack([rng.choice(52, 7, replace=False) for _ in range(20000)]), axis=1).astype(np.int32)
    got = table[np.asarray(combo_index(jnp.asarray(sample)))]
    want = np.searchsorted(values, np.asarray(jax.jit(jax.vmap(evaluate_hand))(jnp.asarray(sample))))
    if not np.array_equal(got, want):
        raise AssertionError("rank7 table disagrees with evaluate_hand")
    return table, len(values)


def load_table(path=TABLE_PATH, log=print):
    """The cached table (built and saved on first use)."""
    path = Path(path)
    if not path.exists():
        table, classes = build_table(log=log)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}.npy")     # one name per writer: concurrent first builds
        np.save(tmp, table)                                             # never share a temporary file (static sniff S3)
        tmp.replace(path)
        if log is not None:
            log(f"rank7: {classes} classes, saved {path}")
    return jnp.asarray(np.load(path, mmap_mode="r"))


_CACHED = {}


def cached_table():
    """The device table if it has been built (results/cache), else None (callers fall back to
    evaluate_hand); loaded once per process. The file is memory-mapped: the host side is the page
    cache, not a 268 MB copy in the process (CPU backend: RSS +524 MB right after np.load + device_put,
    +0.2 MB with the map in the recorded timing measurements)."""
    if "t" not in _CACHED:
        # Concrete even when first asked inside a trace (lax.map in situations.prepare): a cached
        # tracer would leak into the next batch.
        with jax.ensure_compile_time_eval():
            _CACHED["t"] = jax.device_put(np.load(TABLE_PATH, mmap_mode="r")) if TABLE_PATH.exists() else None
    return _CACHED["t"]


def hand_ranks_fast(table, board):
    """int32 [1326] class of every hand on one 5-card board; blocked hands get -1."""
    hands = jnp.asarray(HANDS)
    cards = jnp.concatenate((hands, jnp.broadcast_to(board, (NUM_HANDS, 5))), axis=1)
    blocked = jnp.any(hands[:, :, None] == board[None, None, :], axis=(1, 2))
    # Combinadic index without a sort: the position of a card in the sorted set is the number of
    # smaller cards in it (blocked hands have duplicates and are overwritten below).
    pos = jnp.sum(cards[:, None, :] < cards[:, :, None], axis=2)
    idx = jnp.sum(jnp.asarray(BINOM, jnp.int32)[pos + 1, cards], axis=1)
    ranks = table[jnp.clip(idx, 0, N7 - 1)].astype(jnp.int32)
    return jnp.where(blocked, jnp.int32(-1), ranks)
