"""Check-down values CD: the exact baseline of the residual critics and their
input F.

CD of a hand h of one slot = share(h) * net. share = win + tie / 2 of h at showdown against the
other slot's range when every remaining street is checked down: card removal between the hands and
the board, every completion of the board weighted equally (river: the board itself; turn: the 44
river cards; flop: the 990 turn-river pairs; preflop: all C(48, 5) boards). net = (pot - rake) / pot
with the engine's rake of that pot once the flop is dealt (config.rake_from_terms; at a street end
every bet is matched, so the whole pot is contested). In the critics' relative accounting nothing
more goes in, so CD is the hand's EV of checking down, in pot units of the situation.
Two players; check-down values for three or more players are not implemented here.

Two exact ways to the same share:
- an all-in equity matrix W of the board (poker/solver/runout.py; the spot carries it for its all-in
  nodes): share(h) = (W w)(h) / J(w)(h), J = compatible opponent mass;
- the card-removal average of the next street's shares over the next card (river_cards,
  turn_cards): every compatible pair of hands sees the same number of completions, so
      sum_c m_c(h) share_c(h) / sum_c m_c(h) = share(h),   m_c(h) = J(w with c removed)(h).
The second is the identity behind the auxiliary critics: the average of the next critic's CD over
the next card is this street's CD, so their targets are the average of the next critic's network
part plus their own CD (checks/s5_1.py, check_checkdown_identity; the loop measures it on every
end-of-street situation).
"""

import jax
import jax.numpy as jnp

from poker.envs import config
from poker.solver import cards as C
from poker.solver import rank7
from poker.solver import showdown as sd
from poker.solver.cards import HANDS, MEMBER



def net_fraction(pot, bps, cap, dtype=jnp.float32):
    """(pot - rake) / pot for pots in whole chips: the share of a checked-down pot that is paid out."""
    pot = jnp.asarray(pot, jnp.int32)
    rake = config.rake_from_terms(bps, cap, pot, 1)
    return (pot - rake).astype(dtype) / jnp.maximum(pot, 1).astype(dtype)


def j_mass(w):
    """Compatible mass of one range for every hand (hand-hand card removal; board-blocked hands
    must already carry zero weight)."""
    c = sd.card_sums(w)
    return jnp.sum(w) - c[HANDS[:, 0]] - c[HANDS[:, 1]] + w


def ratio(num, den):
    """num / den where the compatible mass is positive, else 0; clipped to [0, 1] (a share)."""
    ok = den > 1e-30
    return jnp.where(ok, jnp.clip(num / jnp.where(ok, den, 1.0), 0.0, 1.0), 0.0)


def share_river(w_opp, tables):
    """[H] showdown share of every hand against one range on a full board."""
    beats, ties, comp = sd.apply_all(w_opp, tables)
    return ratio(beats + 0.5 * ties, comp)


def share_matrix(w_opp, equity):
    """[H] share against one range from the all-in matrix of the board: (W w)(h) / J(w)(h)."""
    return ratio(equity @ w_opp, j_mass(w_opp))


def two(share, ranges):
    """[2, H]: slot 0 against slot 1's range and slot 1 against slot 0's."""
    return jnp.stack([share(ranges[1]), share(ranges[0])])


def ranks_on(board5, table):
    """Hand classes on a full board: the rank7 table, or the engine's evaluator when table is None
    (explicitly: the float64 checks on the CPU)."""
    return C.hand_ranks(board5) if table is None else rank7.hand_ranks_fast(table, board5)


def _showdown_terms(w, tables):
    """(win + tie / 2, compatible mass) [..., 2, H] of every slot against the other slot's range
    w [..., 2, H] on one full board (rank tables shared by every leading index)."""
    flat = w.reshape((-1,) + w.shape[-2:])
    out = []
    for j in range(2):
        beats, ties, comp = jax.vmap(lambda x: sd.apply_all(x, tables))(flat[:, 1 - j])
        out.append((beats + 0.5 * ties, comp))
    num = jnp.stack([o[0] for o in out], axis=1).reshape(w.shape)
    den = jnp.stack([o[1] for o in out], axis=1).reshape(w.shape)
    return num, den


def river_cards(board4, ranges, table):
    """Every river card c of a turn board (two players; ranges [..., 2, H] zero on board-blocked
    hands, any leading shape on the same board): shares [52, ..., 2, H] on board4 + c with the
    hands holding c removed, and the card-removal weights m_c [52, ..., 2, H] (compatible opponent
    mass without c). Rows of cards on the board are zero and so are hands holding c:
    sum_c m_c share_c / sum_c m_c is the turn share (the identity)."""
    on_board = jnp.zeros(52, bool).at[board4].set(True)

    def one(c):
        keep = ~jnp.asarray(MEMBER)[c]
        tables = C.rank_tables(ranks_on(jnp.concatenate([board4, c[None]]).astype(jnp.int32), table))
        num, den = _showdown_terms(ranges * keep, tables)
        ok = keep & ~on_board[c]
        return jnp.where(ok, ratio(num, den), 0.0), jnp.where(ok, den, 0.0)

    return jax.lax.map(one, jnp.arange(52, dtype=board4.dtype))


def turn_cards(board3, ranges, table, pairs_per_step=51):
    """Every turn card c of a flop board (two players; ranges [..., 2, H] as in river_cards):
    shares [52, ..., 2, H] on board3 + c with the hands holding c removed, each the card-removal
    average of the river shares (== the turn equity matrix share, exact), and the weights m_c
    [52, ..., 2, H]. One river board board3 + {c, r} serves turn c with river r and turn r with
    river c: 1176 rank tables, not 49 x 48; the pairs are scanned in steps and their terms added to
    both cards' rows (memory of 52 rows, not 1326)."""
    on_board = jnp.zeros(52, bool).at[board3].set(True)
    member = jnp.asarray(MEMBER)

    def one(pair):
        a, b = pair[0], pair[1]
        keep = ~(member[a] | member[b])
        tables = C.rank_tables(ranks_on(jnp.concatenate([board3, pair]).astype(jnp.int32), table))
        num, den = _showdown_terms(ranges * keep, tables)
        ok = keep & ~(on_board[a] | on_board[b])
        return jnp.where(ok, num, 0.0), jnp.where(ok, den, 0.0)

    def step(acc, pairs):
        num, den = jax.vmap(one)(pairs)                                                # [P, ..., 2, H]
        for side in range(2):
            acc = (acc[0].at[pairs[:, side]].add(num), acc[1].at[pairs[:, side]].add(den))
        return acc, None

    if len(HANDS) % pairs_per_step:
        raise ValueError("pairs_per_step must divide 1326")
    zero = jnp.zeros((52,) + ranges.shape, ranges.dtype)
    steps = jnp.asarray(HANDS).reshape(-1, pairs_per_step, 2)
    (num, den), _ = jax.lax.scan(step, (zero, zero), steps)
    keep = (~member & ~on_board[:, None]).reshape((52,) + (1,) * (num.ndim - 2) + (-1,))   # hero avoids c
    river_count = 44.0               # rivers per compatible pair of hands on a turn board
    return jnp.where(keep, ratio(num, den), 0.0), jnp.where(keep, den / river_count, 0.0)


def card_average(values, mass):
    """sum_c m_c(h) v_c(h) / sum_c m_c(h) over the leading card axis (0 where no mass)."""
    den = jnp.sum(mass, axis=0)
    num = jnp.sum(mass * values, axis=0)
    ok = den > 1e-30
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)
