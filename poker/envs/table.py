"""One fixed-capacity table for every size (2..9 occupied seats) and stack depth.

Seat 0 is the button and is always occupied; the other occupied seats are a random subset,
so relative order among occupied seats - not seat numbers - defines positions. Stacks: with
probability p_standard every occupied seat starts with standard_bb; otherwise every occupied
seat draws its own depth log-uniformly in [min_bb, max_bb] BB (20 and 500 are symmetric
around 100 on a log scale, so the median non-standard stack is 100 BB). Depths are whole BB.

One engine path for every depth: hands record their history with the capacity proven for
max_stack (config.history_bound(n, start=max_stack)) and use the pure rule functions below.
Everything is jit/vmap friendly; host code only fixes the static configuration.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.envs.actions import action_table
from poker.envs.betting import init as _init
from poker.envs.evaluator import get_rewards
from poker.envs.state import State
from poker.envs.transition import step as _step

SEATS = 9
# Table-size mix of the 9-seat engine table. poker/solver/situations.sample_rake draws the players dealt of a
# spot from it, restricted to the regime's table sizes (config.PLAYERS) for the "mix" rake. The
# TableSampler defaults below (sizes 2-9, a depth per seat, 70% 100 BB / 20-500 BB) are the engine's
# 9-seat fixture (checks/s5_0b.py); the pipeline does not deal with it (it deals the regime: match,
# LBR, the loop's spots). Its rake_mix default is the regime's (config.RAKE_MIX at import).
DEFAULT_SIZES = ((2, 0.12), (3, 0.06), (4, 0.08), (5, 0.12), (6, 0.34), (7, 0.08), (8, 0.08), (9, 0.12))


@dataclass(frozen=True)
class TableSampler:
    seats: int = SEATS
    sizes: tuple = DEFAULT_SIZES          # (occupied seats, probability); probabilities sum to 1
    p_standard: float = 0.70
    standard_bb: int = 100
    min_bb: int = 20
    max_bb: int = 500
    rake_mix: tuple = config.RAKE_MIX     # (low, high, probability): rake level ~ U[low, high]

    def __post_init__(self):
        config.validate_player_count(self.seats)
        counts = [k for k, _ in self.sizes]
        probs = np.asarray([p for _, p in self.sizes], dtype=np.float64)
        if (len(set(counts)) != len(counts) or min(counts) < 2 or max(counts) > self.seats
                or np.any(probs < 0) or abs(probs.sum() - 1.0) > 1e-9):
            raise ValueError("sizes must be distinct counts in 2..seats with probabilities summing to 1")
        if not (2 <= self.min_bb <= self.standard_bb <= self.max_bb) or not 0 <= self.p_standard <= 1:
            raise ValueError("need 2 <= min_bb <= standard_bb <= max_bb and 0 <= p_standard <= 1")
        mix = np.asarray(self.rake_mix, dtype=np.float64).reshape(-1, 3)
        if (np.any(mix[:, :2] < 0) or np.any(mix[:, :2] > 1) or np.any(mix[:, 0] > mix[:, 1])
                or np.any(mix[:, 2] < 0) or abs(mix[:, 2].sum() - 1.0) > 1e-9):
            raise ValueError("rake_mix must be (low <= high in [0, 1], probability) rows summing to 1")

    @property
    def max_stack(self) -> int:
        return self.max_bb * config.BB

    def sample(self, key):
        """(stacks int32[seats], active bool[seats]) for one hand."""
        k_size, k_seats, k_mode, k_depth = jax.random.split(key, 4)
        counts = jnp.asarray([k for k, _ in self.sizes], jnp.int32)
        probs = jnp.asarray([p for _, p in self.sizes], jnp.float32)
        occupied = counts[jax.random.choice(k_size, len(self.sizes), p=probs)]
        # Seat 0 always; then the first occupied-1 seats of a random order of seats 1..S-1.
        order = jax.random.permutation(k_seats, jnp.arange(1, self.seats, dtype=jnp.int32))
        rank = jnp.zeros(self.seats, jnp.int32).at[order].set(jnp.arange(1, self.seats, dtype=jnp.int32))
        active = (jnp.arange(self.seats) == 0) | (rank < occupied)
        lo, hi = np.log(self.min_bb), np.log(self.max_bb)
        depth = jnp.round(jnp.exp(jax.random.uniform(k_depth, (self.seats,), minval=lo, maxval=hi)))
        depth = jnp.clip(depth, self.min_bb, self.max_bb)
        standard = jax.random.uniform(k_mode) < self.p_standard
        depth = jnp.where(standard, jnp.float32(self.standard_bb), depth)
        stacks = jnp.where(active, depth.astype(jnp.int32) * config.BB, 0)
        return stacks.astype(jnp.int32), active

    def sample_rake(self, key):
        """Rake level of one hand from rake_mix (float32 in [0, 1])."""
        k_row, k_level = jax.random.split(key)
        mix = jnp.asarray(self.rake_mix, jnp.float32).reshape(-1, 3)
        row = jax.random.choice(k_row, mix.shape[0], p=mix[:, 2])
        return jax.random.uniform(k_level, minval=mix[row, 0], maxval=mix[row, 1])

    def deal(self, key):
        """Full hand (cards dealt, blinds posted, history on, rake fixed) on the sampled table."""
        k_table, k_cards = jax.random.split(key)
        stacks, active = self.sample(k_table)
        level = self.sample_rake(jax.random.fold_in(k_table, 7))
        return _init(k_cards, self.seats, stacks, active, history=True, max_stack=self.max_stack,
                     rake_level=level)


def legal_mask(state: State):
    """Rule mask of the acting seat (deep stacks are legal; nothing is clipped to START)."""
    return action_table(state)[0] & ~state.terminal


def step(state: State, action):
    """Pure rules step; the caller guarantees legality (as everywhere in the engine)."""
    return _step(state, jnp.asarray(action, jnp.int32))


def rewards(state: State):
    """Net chips per seat at a terminal (side pots, splits, returns, rake); zeros otherwise."""
    return get_rewards(state)


def occupied(state: State):
    """Seats that were dealt into the hand (empty seats start folded with nothing invested)."""
    return ~(state.folded & (state.bet_total == 0) & (state.stack == 0))
