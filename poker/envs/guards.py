"""Host guards of the engine's inputs: the engine refuses invalid input loudly at its Python entry points instead of computing on it.

Inside jitted code the engine keeps its semantics: the legality of an action and the validity of its inputs stay the
caller's preconditions (poker/envs/__init__.py), and the checkify.debug_check preconditions of poker/envs/betting.py and
poker/envs/transition.py still fire under checkify. These functions run on the host, where the input is concrete, and raise
ValueError naming the field and the value. Callers: poker.envs.betting.init_betting (concrete stacks and seats),
poker.solver.cards.hand_index / hand_ranks / board_valid (concrete cards), poker.solver.tree.replay and
poker.solver.situations.prepare (spots), poker.arena.match.table_depths / deal (match deals),
poker.arena.slumbot.cards (Slumbot cards), and poker.player.live (exact action amounts). Traced input (inside
jit or vmap) is not read here: its host caller validates it before the trace (the deals above).
"""

import functools

import jax
import numpy as np

from poker.envs import config, derived
from poker.envs.actions import action_table

CARDS = 52
VISIBLE = (0, 3, 4, 5)        # board cards shown on streets 0..3


def concrete(*values) -> bool:
    """True when no value is a JAX tracer: host data or concrete arrays, which the guards can read."""
    return not any(isinstance(v, jax.core.Tracer) for v in values)


def cards(values, field, *, count=None, unknown=False) -> np.ndarray:
    """int64 [k] card ids (suit * 13 + rank) of `field`, validated: integers in 0..51, pairwise distinct, exactly
    `count` of them when given. unknown=True: -1 marks a card not dealt yet, allowed only after the dealt ones."""
    ids = np.asarray(values)
    if ids.size == 0:
        ids = ids.astype(np.int64)                    # an empty board: np.asarray(()) is float64
    if ids.dtype.kind not in "iu":
        raise ValueError(f"{field} must be integer card ids, got {ids.tolist()!r} ({ids.dtype})")
    ids = ids.astype(np.int64).reshape(-1)
    if count is not None and ids.size != count:
        raise ValueError(f"{field} = {ids.tolist()}: {count} cards expected, got {ids.size}")
    dealt = ids
    if unknown:
        k = int(np.argmax(ids < 0)) if np.any(ids < 0) else ids.size
        if np.any(ids[k:] != -1):
            raise ValueError(f"{field} = {ids.tolist()}: -1 (not dealt) is allowed only after the dealt cards")
        dealt = ids[:k]
    bad = dealt[(dealt < 0) | (dealt >= CARDS)]
    if bad.size:
        raise ValueError(f"{field} = {ids.tolist()}: card id {int(bad[0])} outside 0..{CARDS - 1}")
    seen, twice = np.unique(dealt, return_counts=True)
    if np.any(twice > 1):
        raise ValueError(f"{field} = {ids.tolist()}: card {int(seen[np.argmax(twice > 1)])} appears twice")
    return ids


def table(n, stacks, active, max_stack=None, where="init_betting"):
    """Host guard of a table's start (poker.envs.betting.init_betting): seat 0 (the button) occupied - a vacant button
    posts no small blind (edge_cases B3) -, at least two occupied seats, every occupied seat with more than BB chips
    - a stack <= BB posts a partial blind or reaches its first decision with 0 behind, which the engine does not
    normalise (A6) -, at most max_stack chips where the history capacity is proven for max_stack (history on), and
    the engine's int32 proof for the deepest seat (config.history_bound; history off skipped it, A15). No-op for
    traced input."""
    if not concrete(stacks, active):
        return
    occupied = np.ones(n, bool) if active is None else np.asarray(active).reshape(-1)
    if not occupied[0]:
        raise ValueError(f"{where}: active[0] = False: seat 0 is the button and must be occupied (a vacant button "
                         f"posts no small blind; seat the table so that the button is seat 0)")
    if occupied.sum() < 2:
        raise ValueError(f"{where}: active = {occupied.tolist()}: a hand needs at least two occupied seats")
    chips = (np.full(n, config.START, np.int64) if stacks is None
             else np.asarray(stacks).astype(np.int64).reshape(-1))
    for seat in np.flatnonzero(occupied & (chips <= config.BB)):
        raise ValueError(f"{where}: stacks[{seat}] = {chips[seat]} <= BB = {config.BB} at an occupied seat: it would "
                         f"post a partial blind or reach a decision with 0 behind (not modelled); a table starts "
                         f"above the big blind")
    if max_stack is not None:
        for seat in np.flatnonzero(occupied & (chips > max_stack)):
            raise ValueError(f"{where}: stacks[{seat}] = {chips[seat]} > max_stack = {max_stack}: the history "
                             f"capacity is proven for stacks <= max_stack (config.history_capacity)")
    deepest = int(chips[occupied].max())
    try:
        config.history_bound(n, start=deepest)
    except ValueError as err:
        raise ValueError(f"{where}: stacks up to {deepest} chips at {n} seats: {err}") from None


def deal_stacks(stacks, max_stack, where):
    """Host guard of the stacks a batched deal (vmap of poker.envs.betting.init: poker.arena.match.deal)
    traces: every one above BB and at most max_stack (the history capacity of the deal)."""
    chips = np.asarray(stacks).astype(np.int64).reshape(-1)
    for i in np.flatnonzero(chips <= config.BB)[:1]:
        raise ValueError(f"{where}: stacks[{i}] = {chips[i]} <= BB = {config.BB}: a table starts above the big blind")
    for i in np.flatnonzero(chips > max_stack)[:1]:
        raise ValueError(f"{where}: stacks[{i}] = {chips[i]} > max_stack = {max_stack} (the history capacity of the deal)")


def rake_levels(levels, where):
    """Host guard of rake levels (config.rake_terms clips silently): finite, in [0, 1]."""
    lv = np.asarray(levels, np.float64).reshape(-1)
    for i in np.flatnonzero(~np.isfinite(lv) | (lv < 0) | (lv > 1))[:1]:
        raise ValueError(f"{where}: rake level [{i}] = {lv[i]} outside [0, 1]")


def spot_chips(n, street, pot, stack, where):
    """Host guard of solver spots (poker.solver.tree.replay): the pot >= 0; the chips behind every seat at the street start -
    preflop above BB (init_betting's rule, the pot is the blinds), postflop at least 1 (0 leaves no decision) - and the
    engine's int32 proof for the table with stack + pot / n chips per seat (config.history_bound: every action target,
    masked or not, fits int32; static sniff S13)."""
    pot = np.asarray(pot).astype(np.int64).reshape(-1)
    stack = np.asarray(stack).astype(np.int64).reshape(-1)
    if pot.shape != stack.shape:
        raise ValueError(f"{where}: pot {pot.shape} and stack {stack.shape} must have one entry per spot")
    for i in np.flatnonzero(pot < 0)[:1]:
        raise ValueError(f"{where}: pot[{i}] = {pot[i]} < 0")
    floor, why = (config.BB + 1, "preflop the stack starts above the big blind") if street == 0 else \
        (1, "postflop a seat needs chips behind to decide")
    for i in np.flatnonzero(stack < floor)[:1]:
        raise ValueError(f"{where}: stack[{i}] = {stack[i]} < {floor}: {why}")
    if stack.size:
        i = int(np.argmax(stack + pot // n))
        try:
            config.history_bound(n, start=int(stack[i] + pot[i] // n))
        except ValueError as err:
            raise ValueError(f"{where}: spot {i} (pot {pot[i]}, stack {stack[i]}) at {n} seats: {err}") from None


def state(s, where="state"):
    """Host guard of a State that enters from outside the engine (an adapter's replayed hand, a live table): one table
    of 2..MAX_PLAYERS seats, chips non-negative, every contribution at least the seat's street bet, street 0..3, a
    non-terminal state's actor in the hand with chips (no decision for a seat with 0 behind), the hole cards of every
    occupied seat and the board valid and distinct, the board showing at least the street's cards."""
    stack = np.asarray(s.stack).astype(np.int64)
    if stack.ndim != 1:
        raise ValueError(f"{where}: one table expected, stack has shape {stack.shape}")
    n = config.validate_player_count(int(stack.shape[0]))
    bet, total = np.asarray(s.bet_street).astype(np.int64), np.asarray(s.bet_total).astype(np.int64)
    folded = np.asarray(s.folded).astype(bool)
    for name, value in (("stack", stack), ("bet_street", bet)):
        for i in np.flatnonzero(value < 0)[:1]:
            raise ValueError(f"{where}: {name}[{i}] = {value[i]} < 0")
    for i in np.flatnonzero(total < bet)[:1]:
        raise ValueError(f"{where}: bet_total[{i}] = {total[i]} < bet_street[{i}] = {bet[i]}")
    street = int(s.street)
    if not 0 <= street <= 3:
        raise ValueError(f"{where}: street = {street} outside 0..3")
    if not bool(s.terminal):
        me = int(s.to_act)
        if not 0 <= me < n:
            raise ValueError(f"{where}: to_act = {me} outside 0..{n - 1} at a decision")
        if folded[me] or stack[me] <= 0:
            raise ValueError(f"{where}: to_act = {me} cannot act (folded {bool(folded[me])}, stack {stack[me]}): no "
                             f"decision for a seat out of the hand or with 0 behind")
    occupied = ~(folded & (total == 0) & (stack == 0))            # poker.envs.table.occupied
    hole = np.asarray(s.hole).astype(np.int64)
    board = cards(np.asarray(s.board), f"{where}: board", count=5, unknown=True)
    shown = int(np.sum(board >= 0))
    if shown < VISIBLE[street]:
        raise ValueError(f"{where}: board = {board.tolist()}: street {street} shows {VISIBLE[street]} cards, "
                         f"{shown} dealt")
    held = hole[occupied].reshape(-1)
    cards(np.concatenate([held, board[:shown]]), f"{where}: hole cards of the occupied seats + board",
          count=held.size + shown)


@functools.lru_cache(maxsize=None)
def _rules_fn():
    return jax.jit(lambda s: (*action_table(s), derived.bet_level(s), derived.min_raise_to(s)))


def target(s, a, total, where="step_target"):
    """Host guard of an exact-amount action for the shared poker.envs.transition.step_target engine.
    The live path validates its actions in poker/player/live.py check_legal. Action kind `a`
    (FOLD, CHECK_CALL, or any raise index 2..ALLIN) to the street total `total` must be legal at the concrete State s
    by the engine's own table (poker.envs.actions.action_table, poker.envs.derived): a fold faces a bet and keeps the street bet; a
    check/call reaches exactly the call; a raise needs an open raise and a total in [minimum raise, chips] or exactly
    the chips (an all-in below the minimum is a short raise). Returns None; ValueError names the action and the
    total; an amount above the stack or a sub-minimum raise without an all-in is rejected before tracing."""
    a, total = int(a), int(total)
    if bool(s.terminal):
        raise ValueError(f"{where}: action {a} to {total} at a terminal state")
    mask, targets, level, minimum = (np.asarray(x) for x in _rules_fn()(s))
    me = int(s.to_act)
    current = int(np.asarray(s.bet_street)[me])
    chips = current + int(np.asarray(s.stack)[me])
    if a == config.FOLD:
        if not mask[config.FOLD]:
            raise ValueError(f"{where}: fold at seat {me} with nothing to call (street bet {current}, level {int(level)})")
        if total != current:
            raise ValueError(f"{where}: fold to {total}: a fold keeps seat {me}'s street bet {current}")
    elif a == config.CHECK_CALL:
        if total != int(targets[config.CHECK_CALL]):
            raise ValueError(f"{where}: check/call to {total}: the call puts seat {me}'s street total at "
                             f"{int(targets[config.CHECK_CALL])} (level {int(level)}, chips {chips})")
    elif 2 <= a <= config.ALLIN:
        if not mask[config.ALLIN]:
            raise ValueError(f"{where}: raise to {total} at seat {me}: no raise is open (acted and not reopened, no "
                             f"opponent with chips, or chips {chips} not above the level {int(level)})")
        if total > chips:
            raise ValueError(f"{where}: raise to {total} > {chips}, seat {me}'s chips (street bet + stack)")
        if total < int(minimum) and total != chips:
            raise ValueError(f"{where}: raise to {total} below the minimum raise {int(minimum)} and not the all-in {chips}")
    else:
        raise ValueError(f"{where}: action index {a} outside 0..{config.ALLIN}")
