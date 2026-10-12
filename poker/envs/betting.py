import jax
import jax.numpy as jnp
from jax.experimental import checkify
from poker.envs import config, guards
from poker.envs.state import State


def _next_active(active: jnp.ndarray, seat) -> jnp.ndarray:
    """First active seat strictly after `seat`, clockwise (wraps to `seat` if alone)."""
    n = active.shape[0]
    order = (seat + 1 + jnp.arange(n, dtype=jnp.int32)) % n
    return order[jnp.argmax(active[order])].astype(jnp.int32)


def init_betting(n: int = config.N, stacks=None, active=None, history: bool = True,
                 max_stack=None, rake_level=0.0) -> State:
    """Post blinds on an n-seat table; seat 0 is the button and must be active.

    stacks: int32[n] starting chips before blinds (default START everywhere); active stacks
    must exceed BB. active: bool[n] occupied seats (default all). An empty seat is folded with
    zero chips: it never acts, contributes or wins. Blinds go to the next active seats after
    the button (in HU the button posts SB); the first preflop actor is the next active seat
    after BB. max_stack (static int, default START) bounds every starting stack; the history
    arrays get the capacity proven for max(max_stack, START) (config.history_capacity), so deep
    tables (500 BB) record history on the same engine path as standard ones. history=False
    allocates zero-length history arrays: decisions are still counted in history_len, nothing
    is recorded. rake_level in [0, 1] (config.rake_terms) fixes the rake of the hand from the
    number of occupied seats; 0 means no rake.
    Concrete (host) stacks and seats are validated by poker.envs.guards.table (ValueError: a vacant
    button, fewer than two seats, an occupied stack <= BB, a stack above max_stack with history,
    stacks outside the engine's int32 proof); traced input keeps the checkify preconditions below.
    Defaults reproduce the previous fixed-START table bit for bit (plus zero rake fields).
    """
    config.validate_player_count(n)
    if history is not True and history is not False:
        raise ValueError("history must be a static bool")
    if max_stack is None:
        max_stack = config.START
    if not isinstance(max_stack, int) or isinstance(max_stack, bool) or max_stack <= config.BB:
        raise ValueError(f"max_stack must be a static int > BB, got {max_stack!r}")
    # Never below the START capacity: shallower tables keep the standard layout.
    capacity = config.history_capacity(n, start=max(max_stack, config.START)) if history else 0
    # Blind all-in initialization needs actor/terminal normalization first.
    if config.START <= config.BB:
        raise ValueError("init_betting requires START > BB; blind all-in initialization is not supported")
    raw_stacks, raw_active = stacks, active

    if active is None:
        active = jnp.ones(n, dtype=jnp.bool_)
    else:
        active = jnp.asarray(active)
        if active.shape != (n,) or active.dtype != jnp.bool_:
            raise ValueError(f"active must be a bool array of shape ({n},)")
    if stacks is None:
        stacks = jnp.full(n, config.START, dtype=jnp.int32)
    else:
        stacks = jnp.asarray(stacks)
        if stacks.shape != (n,) or not jnp.issubdtype(stacks.dtype, jnp.integer):
            raise ValueError(f"stacks must be an integer array of shape ({n},)")
        stacks = stacks.astype(jnp.int32)
    guards.table(n, raw_stacks, raw_active, max_stack if history else None)     # concrete input: ValueError
    checkify.debug_check(active[0], "Seat 0 (button) must be active")
    checkify.debug_check(jnp.sum(active, dtype=jnp.int32) >= 2, "At least two active seats")
    checkify.debug_check(jnp.all(~active | (stacks > config.BB)), "Active stacks must exceed BB")
    if history:   # the history capacity is proven for stacks <= max_stack
        checkify.debug_check(jnp.all(stacks <= max_stack), "Stacks must not exceed max_stack")
    stacks = jnp.where(active, stacks, jnp.int32(0))
    rake_bps, rake_cap = config.rake_terms(rake_level, jnp.sum(active, dtype=jnp.int32))

    # Правило NLHE: в HU баттон ставит SB и первым ходит на префлопе;
    # при >=3 активных SB и BB — следующие активные места после BTN. Место 0 = BTN всегда.
    heads_up = jnp.sum(active, dtype=jnp.int32) == 2
    sb_seat = jnp.where(heads_up, jnp.int32(0), _next_active(active, 0))
    bb_seat = _next_active(active, sb_seat)

    sb_bet = jnp.minimum(stacks[sb_seat], config.SB)
    bb_bet = jnp.minimum(stacks[bb_seat], config.BB)

    bet_amount = (
        jnp.zeros(n, dtype=jnp.int32)
        .at[sb_seat].set(sb_bet)
        .at[bb_seat].set(bb_bet)
    )

    stack = stacks - bet_amount
    bet_street = bet_amount
    bet_total = bet_amount

    folded = ~active
    acted = jnp.zeros(n, dtype=jnp.bool_)

    last_raise = jnp.array(config.BB, dtype=jnp.int32)
    street = jnp.array(0, dtype=jnp.int32)
    to_act = _next_active(active, bb_seat)
    terminal = jnp.array(False, dtype=jnp.bool_)

    return State(
        stack=stack,
        bet_street=bet_street,
        bet_total=bet_total,
        folded=folded,
        acted=acted,
        last_raise=last_raise,
        street=street,
        to_act=to_act,
        terminal=terminal,
        hole=jnp.full((n, 2), -1, dtype=jnp.int32),
        board=jnp.full(5, -1, dtype=jnp.int32),
        history_player=jnp.zeros(capacity, dtype=jnp.int32),
        history_action=jnp.zeros(capacity, dtype=jnp.int32),
        history_amount=jnp.zeros(capacity, dtype=jnp.int32),
        history_street=jnp.zeros(capacity, dtype=jnp.int32),
        history_len=jnp.array(0, dtype=jnp.int32),
        rake_bps=rake_bps,
        rake_cap=rake_cap,
    )


def init(key: jax.Array, n: int = config.N, stacks=None, active=None, history: bool = True,
         max_stack=None, rake_level=0.0) -> State:
    """Инициализирует торговлю и все карты; n — статическое число мест.

    Полный борд хранится внутри State; observe раскрывает его по улице. Empty seats are
    dealt cards too (never used); the joint law of live cards and board is unchanged.
    """
    state = init_betting(n, stacks, active, history, max_stack, rake_level)
    deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int32))
    hole = deck[: 2 * n].reshape((n, 2))
    board = deck[2 * n : 2 * n + 5]
    return state.replace(hole=hole, board=board)
