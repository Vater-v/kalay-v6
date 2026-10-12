"""Seed-deterministic spots: board, pot, chips behind and a range per player.

Current HU training uses make_training_spots and one depth-derived physical domain on every
street, including continuation critics. Its low/core/deep allocation is an initial calibration
choice, and its smooth synthetic ranges do not represent the states real play reaches.

make_spots with explicit SPR/pot bounds preserves historical fixtures: pot and SPR drawn
independently, a one-BB stack floor and no starting-depth restriction. Those draws reproduce
old measurements; they are not the current production sampling contract. Ranges are smooth
mixtures of top, polar, random-class and narrow weights, with blocked hands removed.
"""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config, guards
from poker.solver import cards as C
from poker.solver import flops as FL
from poker.solver import runout
from poker.solver.tree import Topology, replay


def _chen(c1, c2):
    r1, r2, s1, s2 = c1 % 13, c2 % 13, c1 // 13, c2 // 13
    hi, lo = max(r1, r2), min(r1, r2)
    points = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(hi, (hi + 2) / 2.0)
    if r1 == r2:
        return max(5.0, points * 2)
    gap = hi - lo - 1
    points += 2.0 if s1 == s2 else 0.0
    points -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 10:
        points += 1
    return points


_SCORE = np.asarray([_chen(a, b) for a, b in C.HANDS])
_ORDER = np.argsort(-_SCORE, kind="stable")
PERCENTILE = np.empty(C.NUM_HANDS)
PERCENTILE[_ORDER] = (np.arange(C.NUM_HANDS) + 0.5) / C.NUM_HANDS   # 0 = strongest preflop
CLASS_PERCENTILE = (np.bincount(C.CLASS_OF, weights=PERCENTILE, minlength=C.NUM_CLASSES)
                    / np.bincount(C.CLASS_OF, minlength=C.NUM_CLASSES))


def random_range(rng, valid, narrow_prob=0.2):
    kind = rng.choice(3, p=[0.5, 0.25, 0.25])
    if rng.random() < narrow_prob:
        kind = 3
    if kind == 0:
        top = rng.uniform(0.05, 0.8)
        w = 1.0 / (1.0 + np.exp((PERCENTILE - top) / (0.08 * top)))
    elif kind == 1:
        top, bottom = rng.uniform(0.03, 0.3), rng.uniform(0.05, 0.4)
        w = (1.0 / (1.0 + np.exp((PERCENTILE - top) / (0.08 * top)))
             + 1.0 / (1.0 + np.exp(((1 - PERCENTILE) - bottom) / (0.08 * bottom))))
    elif kind == 2:
        w = rng.random(C.NUM_CLASSES)[C.CLASS_OF] ** 2
    else:
        pool = np.flatnonzero(valid)
        chosen = rng.choice(pool, size=int(rng.integers(20, 151)), replace=False)
        w = np.zeros(C.NUM_HANDS)
        w[chosen] = rng.random(len(chosen)) + 0.2
    w = w * np.exp(rng.normal(0.0, 0.3, C.NUM_HANDS)) * valid
    if w.sum() <= 0:
        w = valid.astype(float)
    return w / w.sum()


def class_range(rng, narrow_prob=0.2):
    """A suit-symmetric range: one weight per 169-class, every combo of a class equal. Ranges
    that reach the flop from a preflop solved in class space are of this kind, and the 1755-class
    flop average is exact only for them. Kinds as random_range, with
    per-class instead of per-combo noise."""
    kind = rng.choice(3, p=[0.5, 0.25, 0.25])
    if rng.random() < narrow_prob:
        kind = 3
    pc = CLASS_PERCENTILE
    if kind == 0:
        top = rng.uniform(0.05, 0.8)
        w = 1.0 / (1.0 + np.exp((pc - top) / (0.08 * top)))
    elif kind == 1:
        top, bottom = rng.uniform(0.03, 0.3), rng.uniform(0.05, 0.4)
        w = (1.0 / (1.0 + np.exp((pc - top) / (0.08 * top)))
             + 1.0 / (1.0 + np.exp(((1 - pc) - bottom) / (0.08 * bottom))))
    elif kind == 2:
        w = rng.random(C.NUM_CLASSES) ** 2
    else:
        chosen = rng.choice(C.NUM_CLASSES, size=int(rng.integers(5, 41)), replace=False)
        w = np.zeros(C.NUM_CLASSES)
        w[chosen] = rng.random(len(chosen)) + 0.2
    w = (w * np.exp(rng.normal(0.0, 0.3, C.NUM_CLASSES)))[C.CLASS_OF]
    return w / w.sum()


# Table conditions of every spot before the regime toggle (2026-10-10), kept as fixtures for the checks
# and fixed measurements (independent of POKER_REGIME, the spots they were written on bit for bit): NL5
# rake on every hand - with an explicit mixture sample_rake draws the players dealt from poker.envs.table's
# sizes 2-9 as before - and preflop depths 70% 100 BB, otherwise log-uniform 20-500 BB.
NL5 = ((1.0, 1.0, 1.0),)
PRE_TOGGLE_DEPTHS = ((100, 100, 0.7), (20, 500, 0.3))


def sample_rake(seed: int, count: int, n: int, rake="none", whole_table=False):
    """Rake of each spot's hand: level, players dealt in, and the engine's terms (bps, cap chips).
    "mix": the regime's - level from config.RAKE_MIX, players dealt from the table-size mix of
    poker.envs.table restricted to the regime's table sizes (config.PLAYERS) with >= n players (the others
    folded before the spot; ValueError when the regime has none: a 3-player spot under heads-up
    regime 1); whole_table: exactly n dealt (a preflop root is the whole table). A mixture of (low,
    high, probability) rows instead (a fixture, e.g. NL5): level from it, players dealt from
    poker.envs.table's sizes >= n whatever the regime and whole_table (the draw of every spot before the
    toggle). "none": no rake (River Lab M1). A separate stream, so rake never moves the other draws."""
    from poker.envs.table import DEFAULT_SIZES
    if isinstance(rake, str) and rake == "none":
        zeros = np.zeros(count, np.int32)
        return {"rake_level": np.zeros(count, np.float32), "n_dealt": np.full(count, n, np.int32),
                "rake_bps": zeros, "rake_cap": zeros.copy()}
    if isinstance(rake, str) and rake == "mix":
        mix = config.RAKE_MIX
        sizes = [(k, p) for k, p in DEFAULT_SIZES if (k == n if whole_table else k >= n) and k in config.PLAYERS]
        if not sizes:
            raise ValueError(f"{n}-player spots: {config.describe_regime(config.regime_record())} deals no table "
                             f"with {n}{'' if whole_table else ' or more'} players (POKER_REGIME, or an explicit "
                             f"rake mixture as a fixture)")
    elif isinstance(rake, (tuple, list)):
        mix, sizes = rake, [(k, p) for k, p in DEFAULT_SIZES if k >= n]
    else:
        raise ValueError(f"rake must be 'none', 'mix' or a mixture of (low, high, probability) rows, got {rake!r}")
    rng = np.random.default_rng([seed, n, 17])
    mix = np.asarray(mix, np.float64).reshape(-1, 3)
    rows = rng.choice(len(mix), count, p=mix[:, 2])
    level = rng.uniform(mix[rows, 0], mix[rows, 1]).astype(np.float32)
    p = np.asarray([q for _, q in sizes])
    dealt = np.asarray([k for k, _ in sizes])[rng.choice(len(sizes), count, p=p / p.sum())].astype(np.int32)
    bps, cap = jax.vmap(config.rake_terms)(jnp.asarray(level), jnp.asarray(dealt))
    return {"rake_level": level, "n_dealt": dealt, "rake_bps": np.asarray(bps), "rake_cap": np.asarray(cap)}


# One HU training domain for street starts and continuation critics. The allocation is an
# initial coverage/cost choice, not a measured optimum. The old make_spots ranges below remain
# explicit historical fixtures; production callers use make_training_spots.
TRAINING_SPR_BANDS = ((0.0, 0.25, 0.2), (0.25, 16.0, 0.6), (16.0, None, 0.2))
TRAINING_ALLIN_FRACTION = 0.1


def training_domain_record(depth_mix=None):
    """Versioned, separately hashed sampling protocol; independent of the game's tree rules.

    This is an equal-stack HU synthetic envelope, not an empirical reach distribution. The
    same bounds apply before and after every postflop street. Real states are never clipped by
    this sampler, and this record supplies no quality claim beyond the covered domain.
    """
    mix = np.asarray(config.DEPTH_MIX if depth_mix is None else depth_mix, np.float64).reshape(-1, 3).copy()
    if (not len(mix) or not np.all(np.isfinite(mix)) or np.any(mix[:, 0] < 2)
            or np.any(mix[:, 1] < mix[:, 0]) or np.any(mix[:, 2] < 0) or mix[:, 2].sum() <= 0):
        raise ValueError("training depth mix needs finite (2 <= low <= high, probability >= 0) rows")
    mix[:, 2] /= mix[:, 2].sum()
    deepest = int(round(float(mix[mix[:, 2] > 0, 1].max())))
    if 2 * deepest * config.BB > np.iinfo(np.int32).max:
        raise ValueError("training pots must fit signed int32 chips")
    record = {
        "version": 1,
        "kind": "equal_stack_hu_synthetic",
        "players": 2,
        "bb_chips": config.BB,
        "depth_mix_bb": mix.tolist(),
        "minimum_pot_bb": 2.0,
        "maximum_pot_bb": 2.0 * deepest,
        "maximum_spr": (deepest - 1.0) / 2.0,
        "physical_constraint": "pot / 2 + behind = sampled starting depth (gross pot before terminal rake)",
        "spr_bands": [{"low": lo, "high": hi, "probability": probability}
                      for lo, hi, probability in TRAINING_SPR_BANDS],
        "empty_bands": "redistribute probabilities over chip-feasible bands at the sampled depth",
        "draw": "log1p SPR in the first band; log SPR otherwise; derive pot and behind conserving sampled depth",
        "chip_rounding": "round pot / 2 to a chip within its SPR band; behind is depth minus pot / 2",
        "minimum_positive_behind_chips": 1,
        "street_start_allin_fraction": 0.0,
        "street_end_allin_fraction": TRAINING_ALLIN_FRACTION,
        "river_allin_rows": "separate exact check-down copies on the same board split",
        "calibration": "unmeasured initial coverage/cost allocation; validate quality and cost by band",
        "beyond_domain": "do not clip real states; quality outside the sampled envelope is unvalidated",
    }
    record["protocol_hash"] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"))
                                             .encode()).hexdigest()
    return record


def sample_training_chips(rng, count, *, depth_mix=None, allin_fraction=0.0):
    """One shared HU draw in chips, with a physical bound derived from each sampled depth.

    Positive rows allocate 20% / 60% / 20% to SPR < .25 / .25..16 / 16+, renormalising
    when a shallow depth cannot reach a chip-feasible band. Pot >= 2 BB, both seats have at
    least one chip behind and pot / 2 + behind equals the sampled starting stack. This is the
    conserved gross-pot accounting at a HU equal-stack street start before terminal rake.
    The upper SPR follows from the 2 BB pot: (depth_bb - 1) / 2, without a per-street or
    hand-tuned depth cap. Pot is derived from SPR and depth, so there is no unused independent
    pot dimension. Synthetic rounding stays within the chosen band and conserves chips;
    it never changes a live state. Exact-zero rows are only for continuation critics.
    """
    if count < 0 or int(count) != count:
        raise ValueError("training spot count must be a nonnegative integer")
    if not np.isfinite(allin_fraction) or not 0 <= allin_fraction <= 1:
        raise ValueError("allin_fraction must be finite and in [0, 1]")
    protocol = training_domain_record(depth_mix)
    depth = config.sample_depths_bb(rng, count, protocol["depth_mix_bb"])
    chips = depth.astype(np.int64) * config.BB
    maximum_spr = (depth.astype(np.float64) - 1.0) / 2.0
    lo = np.asarray([band[0] for band in TRAINING_SPR_BANDS])
    upper_spr = np.asarray([np.inf if band[1] is None else band[1] for band in TRAINING_SPR_BANDS])
    hi = np.minimum(upper_spr[None, :], maximum_spr[:, None])
    # With x = pot / 2, SPR = depth / (2*x) - .5. Integer x bounds enforce the
    # inclusive lower and exclusive upper SPR edges independently of floating rounding.
    x_min = np.maximum(config.BB, np.floor(chips[:, None] / (2 * upper_spr[None, :] + 1)).astype(np.int64) + 1)
    x_max = np.minimum(chips[:, None] - 1, np.floor(chips[:, None] / (2 * lo[None, :] + 1)).astype(np.int64))
    weights = (x_min <= x_max) * np.asarray([band[2] for band in TRAINING_SPR_BANDS])
    weights /= weights.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(weights, axis=1)
    cumulative[:, -1] = 1.0
    band = np.sum(rng.random(count)[:, None] >= cumulative, axis=1)
    high = hi[np.arange(count), band]
    low = lo[band]
    u = rng.random(count)
    spr = np.expm1(u * np.log1p(high))
    positive = low > 0
    spr[positive] = np.exp(np.log(low[positive]) + u[positive] * np.log(high[positive] / low[positive]))
    allin = rng.random(count) < allin_fraction
    spr[allin] = 0.0
    half_pot = np.rint(chips / (2 * spr + 1)).astype(np.int64)
    lower, upper = x_min[np.arange(count), band], x_max[np.arange(count), band]
    half_pot = np.minimum(np.maximum(half_pot, lower), upper)
    half_pot[allin] = chips[allin]
    pot, stack = 2 * half_pot, chips - half_pot
    band[allin] = -1
    return {"pot": pot.astype(np.int32), "stack": stack.astype(np.int32),
            "depth_bb": depth, "spr_band": band.astype(np.int32)}


POT_BB = {3: (4.0, 60.0), 2: (3.0, 40.0), 1: (2.0, 25.0)}     # street -> log-uniform pot range
BOARD_CARDS = {3: 5, 2: 4, 1: 3}


def make_spots(seed: int, count: int, n: int, narrow_prob: float = 0.2, spr_range=(0.5, 6.0),
               rake: str = "none", street: int = 3, pot_bb=None, board_pool: int = 0, *,
               depth_mix=None, allin_fraction=0.0) -> dict:
    """spr_range: stack-to-pot ratio bounds (log-uniform). (0.5, 6) is the first M1 lab;
    later solves use (0.5, 50): 100 BB single-raised pots reach the river at SPR 10-20. Pot and SPR
    are drawn directly, not a depth: the regime's depth mix does not apply (poker/solver/loop.py,
    depth_report: the share of spots deeper than the regime's table).
    rake: "none" (River Lab M1), "mix" (the regime's training mixture) or a fixture mixture (sample_rake).
    street: 3 river (5 board cards), 2 turn (4), 1 flop (3); missing cards are -1. pot_bb: (low,
    high) log-uniform pot in BB (default POT_BB[street]). board_pool > 0 (flop only): a pool of
    that many distinct canonical flop classes drawn in proportion to class size (poker/solver/flops.py),
    each spot on one of them (all-in equity matrices are cached per flop, critics split by flop).
    The random stream is the same for every street and pot range (river spots of M1 are
    unchanged); the pool has its own stream. spr_range=None selects the current shared training
    domain (make_training_spots), using its own chip stream; pot_bb then must be absent.
    """
    if spr_range is None:
        if n != 2:
            raise NotImplementedError("the shared training domain currently supports equal-stack HU only")
        if pot_bb is not None:
            raise ValueError("the shared training domain derives its pot bounds from depth; pot_bb must be absent")
    elif depth_mix is not None or allin_fraction:
        raise ValueError("depth_mix/allin_fraction apply only to the shared training domain (spr_range=None)")
    rng = np.random.default_rng([seed, n])
    boards = np.stack([rng.choice(52, 5, replace=False) for _ in range(count)]).astype(np.int32)
    k = BOARD_CARDS[street]
    if board_pool:
        if street != 1:
            raise ValueError("board pools are flop pools (canonical flop classes)")
        pool_rng = np.random.default_rng([seed, n, 23])
        pool = FL.FLOPS[FL.sample_classes(pool_rng, min(board_pool, count))]
        boards[:, :3] = pool[pool_rng.integers(0, len(pool), count)]
    boards[:, k:] = -1
    domain = {}
    if spr_range is None:
        domain = sample_training_chips(np.random.default_rng([seed, n, 43]), count,
                                       depth_mix=depth_mix, allin_fraction=allin_fraction)
        pot, stack = domain["pot"], domain["stack"]
    else:
        lo, hi = pot_bb or POT_BB[street]
        pot_bb = np.exp(rng.uniform(np.log(lo), np.log(hi), count))
        pot = (np.round(pot_bb * config.BB / n) * n).astype(np.int32)
        spr = np.exp(rng.uniform(np.log(spr_range[0]), np.log(spr_range[1]), count))
        stack = np.maximum(np.round(spr * pot), config.BB).astype(np.int32)
    valid = C.board_valid(boards[:, :k])                                    # [count, 1326]
    ranges = np.stack([[random_range(rng, valid[i], narrow_prob) for _ in range(n)]
                       for i in range(count)]).astype(np.float32)
    return {"board": boards, "pot": pot, "stack": stack, "valid": valid, "ranges": ranges,
            "street": np.full(count, street, np.int32), **sample_rake(seed, count, n, rake), **domain}


def make_training_spots(seed: int, count: int, n: int = 2, *, street: int = 3, rake="mix", board_pool=0,
                        allin_fraction=0.0, depth_mix=None) -> dict:
    """Current synthetic HU spots: shared chips/domain on all streets, board-specific ranges.

    allin_fraction=0 for decision roots; continuation critics may request the protocol's .1.
    depth_mix is an explicit fixture override; production uses the regime's declared mixture.
    """
    return make_spots(seed, count, n, spr_range=None, rake=rake, street=street, board_pool=board_pool,
                      depth_mix=depth_mix, allin_fraction=allin_fraction)


def make_preflop_spots(seed: int, count: int, n: int = 2, rake="mix", depth_mix=None) -> dict:
    """Preflop configurations: every seat the same depth in whole BB from depth_mix (default the
    regime's, config.DEPTH_MIX: regime 1 200 BB, regime 2 70% 100 BB, otherwise log-uniform 20-500
    BB; config.sample_depths_bb, the draw of this function before the toggle), full uniform ranges,
    rake of the hand (sample_rake; with the regime's "mix" the root is the whole table: n players
    dealt - before the toggle, and with a fixture mixture, the players dealt were drawn from 2-9 as
    for a postflop spot). The pot is the blinds (the pot unit of preflop solves)."""
    rng = np.random.default_rng([seed, n, 31])
    stack = (config.sample_depths_bb(rng, count, depth_mix).astype(np.int64) * config.BB).astype(np.int32)
    ranges = np.full((count, n, C.NUM_HANDS), 1.0 / C.NUM_HANDS, np.float32)
    return {"board": np.full((count, 5), -1, np.int32), "pot": np.full(count, config.SB + config.BB, np.int32),
            "stack": stack, "valid": np.ones((count, C.NUM_HANDS), bool), "ranges": ranges,
            "street": np.zeros(count, np.int32), **sample_rake(seed, count, n, rake, whole_table=True)}


@jax.jit
def _tables(boards):
    return jax.vmap(lambda b: C.rank_tables(C.hand_ranks(b)))(boards)


FLOP_PAIRS = 48                       # sampled turn-river pairs of a 3-player flop spot


def flop_pairs(board3, seed, k=FLOP_PAIRS):
    """k distinct turn-river pairs drawn uniformly from the 1176 pairs of cards off the flop, seeded
    from the spot (seed: ints, e.g. the flop cards and the pot in chips); int32 [k, 2], ascending."""
    rest = np.setdiff1d(np.arange(52), np.asarray(board3))
    pairs = np.asarray([(a, b) for i, a in enumerate(rest) for b in rest[i + 1:]], np.int32)       # [1176, 2]
    pick = np.random.default_rng([int(x) for x in np.ravel(seed)]).choice(len(pairs), size=k, replace=False)
    return pairs[np.sort(pick)]


@jax.jit
def pair_tables(board3, pairs):
    """Rank tables of the boards flop + pair for every pair (leading axis k)."""
    return jax.vmap(lambda pr: C.rank_tables(C.hand_ranks(jnp.concatenate([board3, pr]).astype(jnp.int32))))(pairs)


@jax.jit
def turn_river_tables(board4):
    """The 48 river cards of a turn board (int32 [48], ascending) and the rank tables of the 48
    river boards (leading axis 48): the three-player turn runouts (pcfr.turn_runout_three)."""
    on = jnp.zeros(52, bool).at[board4].set(True)
    cards = jnp.nonzero(~on, size=48)[0].astype(jnp.int32)
    tables = jax.vmap(lambda c: C.rank_tables(C.hand_ranks(jnp.concatenate([board4, c[None]]).astype(jnp.int32))))(cards)
    return cards, tables


def check_spots(spots: dict, topo: Topology) -> None:
    """Host guard of the solver's spots (ValueError naming the field, the spot and
    the value): every array one entry per spot; the board shows exactly the street's cards (BOARD_CARDS, distinct ids
    in 0..51), then -1; valid = the hands the shown cards do not block; ranges [count, n, 1326] finite, >= 0, some
    mass per seat and none on a blocked hand; rake terms >= 0; postflop a pot that n seats split equally (the street
    root gives each seat pot // n: an odd chip would be lost against the pot unit), preflop the blinds as the pot
    unit; the chips of tree.replay (poker.envs.guards.spot_chips)."""
    n, street = topo.n, topo.street
    count = len(spots["pot"])
    for key in ("board", "pot", "stack", "valid", "ranges", "rake_bps", "rake_cap", "rake_level"):
        if key in spots and len(spots[key]) != count:
            raise ValueError(f"spots[{key!r}] has {len(spots[key])} entries, spots['pot'] {count}")
    board = np.asarray(spots["board"])
    if board.shape != (count, 5):
        raise ValueError(f"spots['board'] must be [count, 5] card ids (-1 = not dealt), got shape {board.shape}")
    shown = BOARD_CARDS.get(street, 0)
    rest = np.flatnonzero(np.any(board[:, shown:] != -1, axis=1))
    for i in rest[:1]:
        raise ValueError(f"spots['board'][{i}] = {board[i].tolist()}: a street-{street} spot shows {shown} cards, "
                         f"the others must be -1")
    dealt = board[:, :shown].astype(np.int64)
    bad = np.any((dealt < 0) | (dealt >= 52), axis=1) | np.any(np.diff(np.sort(dealt, axis=1), axis=1) == 0, axis=1)
    for i in np.flatnonzero(bad)[:1]:
        guards.cards(board[i, :shown], f"spots['board'][{i}] (street {street}: {shown} cards)")
    pot = np.asarray(spots["pot"]).astype(np.int64)
    if street == 0:
        for i in np.flatnonzero(pot != config.SB + config.BB)[:1]:
            raise ValueError(f"spots['pot'][{i}] = {pot[i]}: a preflop spot's pot unit is the blinds, "
                             f"{config.SB + config.BB}")
    else:
        for i in np.flatnonzero((pot <= 0) | (pot % n != 0))[:1]:
            raise ValueError(f"spots['pot'][{i}] = {pot[i]}: a postflop pot is positive and a multiple of n = {n} "
                             f"(the street root splits it equally)")
    guards.spot_chips(n, street, pot, spots["stack"], "situations.prepare")
    valid = np.asarray(spots["valid"]).astype(bool)
    want = C.board_valid(board[:, :shown]) if shown else np.ones((count, C.NUM_HANDS), bool)
    for i in np.flatnonzero(np.any(valid != want, axis=1))[:1]:
        raise ValueError(f"spots['valid'][{i}] differs from the hands its board {board[i].tolist()} leaves "
                         f"({int(valid[i].sum())} valid, {int(want[i].sum())} expected)")
    ranges = np.asarray(spots["ranges"])
    if ranges.shape != (count, n, C.NUM_HANDS):
        raise ValueError(f"spots['ranges'] must be [count, {n}, {C.NUM_HANDS}], got shape {ranges.shape}")
    bad = ~np.isfinite(ranges) | (ranges < 0)
    for i, p, h in np.argwhere(bad)[:1]:
        raise ValueError(f"spots['ranges'][{i}, {p}, {h}] = {ranges[i, p, h]}: weights are finite and >= 0")
    for i, p in np.argwhere(ranges.sum(axis=-1) <= 0)[:1]:
        raise ValueError(f"spots['ranges'][{i}, {p}] has no mass")
    for i, p, h in np.argwhere((ranges > 0) & ~want[:, None, :])[:1]:
        raise ValueError(f"spots['ranges'][{i}, {p}, {h}] = {ranges[i, p, h]} on hand {C.HANDS[h].tolist()}, "
                         f"blocked by the board {board[i, :shown].tolist()}")
    for key in ("rake_bps", "rake_cap"):
        if key in spots:
            for i in np.flatnonzero(np.asarray(spots[key]) < 0)[:1]:
                raise ValueError(f"spots[{key!r}][{i}] = {spots[key][i]} < 0")


def prepare(spots: dict, topo: Topology, dtype=jnp.float32, preflop_equity=None) -> dict:
    """Device-ready batch for poker/solver/pcfr.py: tree data in pot units of the street start, rank
    tables on the river, the exact two-player all-in equity matrix on the turn, and what the
    next street's critic needs at street ends (board one-hot, pot in BB, chips behind, rake).
    The spots are validated first (check_spots, ValueError)."""
    check_spots(spots, topo)
    count = len(spots["pot"])
    bps = spots.get("rake_bps", np.zeros(count, np.int32))
    cap = spots.get("rake_cap", np.zeros(count, np.int32))
    data = replay(topo, spots["pot"], spots["stack"], bps, cap)
    if not (bool(jnp.all(data["actor_ok"])) and bool(jnp.all(data["terminal_ok"]))):
        raise AssertionError("engine replay disagrees with the topology")
    scale = jnp.asarray(spots["pot"], dtype)[:, None]
    board = jnp.asarray(spots["board"])
    out = {"legal": data["legal"], "alive": data["alive"],
           "contrib": data["contrib"].astype(dtype) / scale[:, :, None],
           "pot": data["pot"].astype(dtype) / scale,
           "behind": data["stack"].astype(dtype) / scale[:, :, None],
           "street_bet": data["street_bet"].astype(dtype) / scale[:, :, None],
           "ranges": jnp.asarray(spots["ranges"], dtype),
           "valid": jnp.asarray(spots["valid"], dtype),
           "board": board,
           "board1h": jax.vmap(lambda b: jnp.sum(jax.nn.one_hot(b, 52, dtype=dtype), axis=0))(board),
           "pot_bb": jnp.asarray(np.asarray(spots["pot"]) / config.BB, dtype),
           "stack_pot": jnp.asarray(spots["stack"] / spots["pot"], dtype),
           "rake": data["rake"].astype(dtype) / scale,
           "rake_level": jnp.asarray(spots.get("rake_level", np.zeros(count)), dtype),
           "rake_rate": jnp.asarray(np.asarray(bps) / 500.0, dtype),
           "rake_cap": jnp.asarray(np.asarray(cap) / np.asarray(spots["pot"]), dtype),
           # whole chips for the check-down values' rake (poker/solver/checkdown.py net_fraction)
           "pot_chips": jnp.asarray(data["pot"], jnp.int32),
           "rake_bps": jnp.asarray(bps, jnp.int32),
           "rake_cap_chips": jnp.asarray(cap, jnp.int32)}
    street = topo.street
    if street == 3:
        out["tables"] = _tables(board)
    elif street == 2 and topo.n == 2:
        table = runout.rank7.cached_table()
        out["equity"] = jax.lax.map(lambda b: runout.equity_matrix_turn(b, table), board[:, :4]).astype(dtype)
    elif street == 2 and topo.n == 3:
        out["river_cards"], out["river_tables"] = jax.lax.map(turn_river_tables, board[:, :4].astype(jnp.int32))
    elif street == 1 and topo.n == 3:              # sampled turn-river pairs, seeded from the spot (flop, pot)
        flops, pots = np.asarray(board[:, :3]), np.asarray(spots["pot"])
        pairs = jnp.asarray(np.stack([flop_pairs(flops[i], [7, *flops[i], int(pots[i])]) for i in range(count)]))
        out["flop_pairs"] = pairs
        out["pair_tables"] = jax.lax.map(lambda a: pair_tables(a[0], a[1]), (board[:, :3].astype(jnp.int32), pairs))
    elif street == 0 and topo.n == 2:
        if preflop_equity is None:
            raise ValueError("preflop all-ins need the exact preflop equity matrix (runout.preflop_equity_cached)")
        out["equity"] = jnp.broadcast_to(jnp.asarray(preflop_equity, dtype), (count,) + preflop_equity.shape)
    elif street == 1 and topo.n == 2:
        flops = np.asarray(board[:, :3])
        unique, index = np.unique(flops, axis=0, return_inverse=True)
        mats = jnp.stack([runout.flop_equity_cached(f) for f in unique])
        out["equity"] = mats[jnp.asarray(index.reshape(-1))].astype(dtype)
    return out
