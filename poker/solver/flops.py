"""The 1755 strategically distinct flops: the classes of the
22,100 flops under the 24 suit permutations. FLOPS [1755, 3] holds each class's representative
(its smallest sorted card triple; cards are suit * 13 + rank), SIZES [1755] the number of flops
in the class (they sum to 22,100). A suit-symmetric quantity averaged over all flops equals the
SIZES-weighted average over FLOPS; for per-hand quantities the class sum is the sum over the
24 suit permutations of the representative's values (critics.end_of_preflop_classes).
"""

import itertools

import numpy as np

SUIT_PERMS = np.asarray(list(itertools.permutations(range(4))), np.int32)                        # [24, 4]
CARD_PERM = np.asarray([[p[c // 13] * 13 + c % 13 for c in range(52)] for p in SUIT_PERMS], np.int32)


def _code(cards):
    return (cards[..., 0] * 52 + cards[..., 1]) * 52 + cards[..., 2]


def _classes():
    flops = np.asarray(list(itertools.combinations(range(52), 3)), np.int32)                     # [22100, 3]
    images = np.sort(CARD_PERM[:, flops], axis=-1)                                               # [24, 22100, 3]
    canon = _code(images).min(axis=0)
    codes, sizes = np.unique(canon, return_counts=True)
    reps = np.stack([codes // 2704, codes // 52 % 52, codes % 52], axis=1).astype(np.int32)
    return reps, sizes.astype(np.int64)


FLOPS, SIZES = _classes()
if len(FLOPS) != 1755 or int(SIZES.sum()) != 22100:
    raise RuntimeError(f"flop classes: {len(FLOPS)} classes over {int(SIZES.sum())} flops, expected 1755 over 22100")


def canonical(flop):
    """Representative of the class of a flop (int[3], any order)."""
    images = np.sort(CARD_PERM[:, np.asarray(flop)], axis=-1)
    return images[np.argmin(_code(images))].astype(np.int32)


def sample_classes(rng, count, distinct=True):
    """`count` flop classes drawn with probability proportional to class size (a uniformly random
    flop, canonicalised); distinct=True draws without replacement (a pool of distinct flops)."""
    return rng.choice(len(FLOPS), size=count, replace=not distinct, p=SIZES / SIZES.sum())
