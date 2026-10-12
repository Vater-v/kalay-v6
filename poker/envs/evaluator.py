import jax
import jax.numpy as jnp

from poker.envs import config
from poker.envs.state import State


def evaluate_flush_straight(cards: jnp.ndarray) -> tuple:
    """Seven distinct int32 cards in 0..51 -> flush/straight/SF flags and ranks.

    Returns (has_flush, flush_kickers, has_straight, straight_high,
    has_sf, sf_high). Flush kickers are five descending int32 ranks;
    their values are unspecified without a flush. Straight highs use
    rank 3 for a wheel, rank 12 for Broadway, and -1 without a straight.
    """
    suit = cards // 13
    rank = cards % 13

    # At most one suit can have five cards among seven distinct cards.
    # Comparisons and reductions avoid duplicate-index scatter on the GPU.
    suit_counts = jnp.sum(
        suit[:, None] == jnp.arange(4, dtype=jnp.int32), axis=0, dtype=jnp.int32,
    )
    flush_suit = jnp.argmax(suit_counts)
    has_flush = suit_counts[flush_suit] >= 5
    flush_ranks = jnp.where(suit == flush_suit, rank, jnp.int32(-1))
    flush_kickers = jnp.sort(flush_ranks)[::-1][:5]

    # Index 0 duplicates the ace below deuce; indices 1..13 are deuce..ace.
    rank_matches = rank[:, None] == jnp.arange(13, dtype=jnp.int32)
    ranks_present = jnp.any(rank_matches, axis=0)
    present = jnp.concatenate((ranks_present[12:13], ranks_present))
    straight = present[:-4] & present[1:-3] & present[2:-2] & present[3:-1] & present[4:]
    has_straight = jnp.any(straight)
    straight_high = jnp.max(jnp.where(straight, jnp.arange(3, 13, dtype=jnp.int32), jnp.int32(-1)))

    # Build the same rank mask from cards of the selected suit only.
    suited_present = jnp.any(rank_matches & (suit == flush_suit)[:, None], axis=0)
    present_sf = jnp.concatenate((suited_present[12:13], suited_present))
    sf = present_sf[:-4] & present_sf[1:-3] & present_sf[2:-2] & present_sf[3:-1] & present_sf[4:]
    has_sf = jnp.any(sf)
    sf_high = jnp.max(jnp.where(sf, jnp.arange(3, 13, dtype=jnp.int32), jnp.int32(-1)))

    return has_flush, flush_kickers, has_straight, straight_high, has_sf, sf_high


def evaluate_hand(cards: jnp.ndarray) -> jnp.ndarray:
    """Seven distinct int32 cards -> scalar int32 score; larger is stronger.

    Category 0..8 occupies bits 20..23. Five tie-break ranks
    occupy four bits each at 16, 12, 8, 4, 0; unused ranks are zero.
    """
    has_flush, fk, has_straight, straight_high, has_sf, sf_high = evaluate_flush_straight(cards)
    ranks = jnp.arange(13, dtype=jnp.int32)
    counts = jnp.sum(cards[:, None] % 13 == ranks, axis=0, dtype=jnp.int32)

    # Multiplicity first, then rank; two trips use the higher trip for the FH.
    order = jnp.argsort(counts * jnp.int32(16) + ranks)[::-1].astype(jnp.int32)
    c = counts[order]
    present = counts > 0
    quad_kicker = jnp.max(jnp.where(present & (ranks != order[0]), ranks, jnp.int32(0)))
    other_ranks = jnp.sort(jnp.where(present & (ranks != order[0]), ranks, jnp.int32(0)))[::-1]
    two_pair_kicker = jnp.max(jnp.where(
        present & (ranks != order[0]) & (ranks != order[1]), ranks, jnp.int32(0),
    ))
    high_cards = jnp.sort(jnp.where(present, ranks, jnp.int32(0)))[::-1]

    # Each category packs only the ranks that can break ties in that category.
    quad = jnp.where(c[0] == 4,
        jnp.int32(7 << 20) | (order[0] << 16) | (quad_kicker << 12), jnp.int32(0))
    full_house = jnp.where((c[0] == 3) & (c[1] >= 2),
        jnp.int32(6 << 20) | (order[0] << 16) | (order[1] << 12), jnp.int32(0))
    three = jnp.where(c[0] == 3,
        jnp.int32(3 << 20) | (order[0] << 16) | (other_ranks[0] << 12)
        | (other_ranks[1] << 8), jnp.int32(0))
    two_pair = jnp.where((c[0] == 2) & (c[1] == 2),
        jnp.int32(2 << 20) | (order[0] << 16) | (order[1] << 12)
        | (two_pair_kicker << 8), jnp.int32(0))
    pair = jnp.where(c[0] == 2,
        jnp.int32(1 << 20) | (order[0] << 16) | (other_ranks[0] << 12)
        | (other_ranks[1] << 8) | (other_ranks[2] << 4), jnp.int32(0))
    high_card = (
        (high_cards[0] << 16) | (high_cards[1] << 12) | (high_cards[2] << 8)
        | (high_cards[3] << 4) | high_cards[4]
    )
    straight_flush = jnp.where(has_sf,
        jnp.int32(8 << 20) | (sf_high << 16), jnp.int32(0))
    flush = jnp.where(has_flush,
        jnp.int32(5 << 20) | (fk[0] << 16) | (fk[1] << 12) | (fk[2] << 8)
        | (fk[3] << 4) | fk[4], jnp.int32(0))
    straight = jnp.where(has_straight,
        jnp.int32(4 << 20) | (straight_high << 16), jnp.int32(0))
    return jnp.max(jnp.stack((
        straight_flush, quad, full_house, flush, straight, three, two_pair, pair, high_card,
    )))


def rake_split(bets: jnp.ndarray, pot: jnp.ndarray, rake_bps, rake_cap, street_reached):
    """Total rake of the hand and its share of every pot level (config.py, rake rules).

    pot[k] are the level amounts of compute_payouts (ascending live caps; the last level holds
    the top contributor's uncalled excess). Base = all chips minus the uncalled excess (top bet
    minus the second largest bet of anyone); each level pays floor(R * base_k / base), the main
    (lowest non-empty) level also pays the remainder. int32-safe: R <= cap (<= 10 BB) and
    base_k <= sum of bets.
    """
    top = jnp.sort(bets)[-2:]
    uncalled = top[1] - top[0]
    last = jnp.arange(pot.shape[0]) == pot.shape[0] - 1
    base_k = pot - jnp.where(last, uncalled, jnp.int32(0))
    base = jnp.sum(base_k, dtype=jnp.int32)
    total = config.rake_from_terms(rake_bps, rake_cap, base, street_reached)
    share = jnp.where(base > 0, total * base_k // jnp.maximum(base, 1), jnp.int32(0))
    main = jnp.argmax(base_k > 0)
    share = share.at[main].add(total - jnp.sum(share, dtype=jnp.int32))
    return total, share


def compute_payouts(bets: jnp.ndarray, active: jnp.ndarray, scores: jnp.ndarray,
                    rake_bps=0, rake_cap=0, street_reached=0) -> jnp.ndarray:
    """Legal terminal contributions -> gross int32 payouts for one table.

    bets/scores are int32[n], active is bool[n] (~folded), n is static.
    At least one player is active and a maximum contributor is active.
    Contributions are nonnegative; scores are nonnegative evaluate_hand scores.
    Rake (config.py rules) comes out of the pots before they are split; the default is none.
    """
    n = bets.shape[0]
    # Only live contribution caps change who can win a pot. Folded players'
    # lower contributions add dead money without creating additional odd chips.
    caps = jnp.sort(jnp.where(active, bets, jnp.int32(0)))
    previous = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), caps[:-1]))
    pot = jnp.sum(
        jnp.minimum(bets[None, :], caps[:, None])
        - jnp.minimum(bets[None, :], previous[:, None]),
        axis=1, dtype=jnp.int32,
    )
    pot = pot - rake_split(bets, pot, rake_bps, rake_cap, street_reached)[1]
    eligible = active[None, :] & (bets[None, :] >= caps[:, None])
    best = jnp.max(jnp.where(eligible, scores[None, :], jnp.int32(-1)), axis=1)
    winners = eligible & (scores[None, :] == best[:, None])
    count = jnp.sum(winners, axis=1, dtype=jnp.int32)
    safe_count = jnp.maximum(count, jnp.int32(1))
    base = pot // safe_count
    remainder = pot % safe_count

    # Split real pots separately, even when they happen to have the same winners.
    # An unmatched top contribution has one eligible player and returns in full.
    priority = (jnp.arange(n, dtype=jnp.int32) - 1) % n
    earlier_winners = jnp.sum(
        winners[:, None, :] & (priority[None, None, :] < priority[None, :, None]),
        axis=2, dtype=jnp.int32,
    )
    extra = winners & (earlier_winners < remainder[:, None])
    return jnp.sum(
        winners * base[:, None] + extra.astype(jnp.int32), axis=0, dtype=jnp.int32,
    )


def rake_paid(state: State) -> jnp.ndarray:
    """Rake (chips) a terminal state pays: rewards sum to -rake_paid; zero before the end."""
    caps = jnp.sort(jnp.where(~state.folded, state.bet_total, jnp.int32(0)))
    previous = jnp.concatenate((jnp.zeros(1, dtype=jnp.int32), caps[:-1]))
    bets = state.bet_total
    pot = jnp.sum(jnp.minimum(bets[None, :], caps[:, None]) - jnp.minimum(bets[None, :], previous[:, None]),
                  axis=1, dtype=jnp.int32)
    total = rake_split(bets, pot, state.rake_bps, state.rake_cap, state.street)[0]
    return jnp.where(state.terminal, total, jnp.int32(0))


def get_rewards(state: State) -> jnp.ndarray:
    """One table -> net int32 chip profit; nonterminal rewards are zero.

    Showdown requires valid hole/board cards from init. State stays unchanged.
    Batched tables are evaluated through vmap; profit is not normalized here.
    Rake is charged by the hand's rake fields (zero by default); profits sum to -rake_paid.
    """
    board = jnp.broadcast_to(state.board, (state.hole.shape[0], 5))
    cards = jnp.concatenate((state.hole, board), axis=1)
    scores = jax.vmap(evaluate_hand)(cards)
    gross = compute_payouts(state.bet_total, ~state.folded, scores,
                            state.rake_bps, state.rake_cap, state.street)
    profit = gross - state.bet_total
    return jnp.where(state.terminal, profit, jnp.zeros_like(profit))
