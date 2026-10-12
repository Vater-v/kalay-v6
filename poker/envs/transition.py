import jax
import jax.numpy as jnp
from jax.experimental import checkify

from poker.envs import config, derived
from poker.envs.actions import action_table
from poker.envs.action_input import action_input, fits_int32
from poker.envs.state import State


def _can_apply(s: State) -> jnp.ndarray:
    """Safe runtime gate; debug diagnostics activate under external checkify."""
    if s.stack.ndim != 1:
        raise ValueError("transitions require one table; batch through vmap")
    n = s.stack.shape[0]
    # Zero-length history arrays mean "history disabled" (init_betting(history=False)):
    # decisions are only counted. Otherwise the capacity must be a proven one: a multiple of 64
    # at least the bound for START stacks (deeper tables allocate the bound for their max_stack,
    # init_betting); the overflow check below still guards every write.
    disabled = s.history_player.shape == (0,)
    capacity = s.history_player.shape[0] if s.history_player.ndim == 1 else -1
    minimum = 0 if disabled else config.history_capacity(n)
    if capacity < minimum or capacity % 64:
        raise ValueError(f"history arrays must be int32[64 * k >= {minimum}] for n={n}")
    for name in ("history_player", "history_action", "history_amount", "history_street"):
        value = getattr(s, name)
        if value.shape != (capacity,) or value.dtype != jnp.int32:
            raise ValueError(f"{name} must be int32[{capacity}] for n={n}")
    if s.history_len.shape != () or s.history_len.dtype != jnp.int32:
        raise ValueError("history_len must be scalar int32")
    idx = s.history_len
    if disabled:
        length_valid = idx >= 0
        space_valid = jnp.bool_(True)
    else:
        length_valid = (idx >= 0) & (idx <= capacity)
        space_valid = idx < capacity
    actor_valid = (s.to_act >= 0) & (s.to_act < n)
    checkify.debug_check(length_valid, "Invalid history length")
    checkify.debug_check(s.terminal | space_valid, "History overflow")
    checkify.debug_check(s.terminal | actor_valid, "Invalid actor")
    return (~s.terminal) & length_valid & space_valid & actor_valid


def apply_action(s: State, a: jnp.ndarray) -> State:
    """Apply one legal decision and append once, before actor/street changes.

    Active action legality is the caller's precondition (legal in the table with every grid size, or
    with the sizes a tree offers: poker/envs/actions.py action_table). Terminal inputs return
    the complete original State for any int32 action index. Invalid length or
    actor also preserves State; checked callers must reject the reported error.
    """
    a = action_input(a)
    representable = fits_int32(a)
    checkify.debug_check(s.terminal | representable, "Action outside int32 range")
    can_apply = _can_apply(s) & representable
    a = jnp.where(representable, a, 0).astype(jnp.int32)
    n = s.stack.shape[0]
    capacity = s.history_player.shape[0]
    me = jnp.clip(s.to_act, 0, n - 1)
    safe_action = jnp.clip(a, 0, config.NUM_ACTIONS - 1)
    safe_s = s.replace(to_act=me)
    level = derived.bet_level(safe_s)
    current = s.bet_street[me]
    # The chosen action's own chips: the table with `a` the only offered size (poker/envs/actions.py sizes), so a
    # fraction raise that a tree offers keeps its target where the whole grid masks it as a duplicate of a
    # lower size with the same chips; an action legal on the whole grid gets the same target.
    alone = jnp.arange(config.NUM_ACTIONS, dtype=jnp.int32) == safe_action
    target = jnp.where(can_apply, action_table(safe_s, alone)[1][safe_action], current)
    is_fold = a == config.FOLD
    add = jnp.where(is_fold, jnp.int32(0), target - current)

    # Полный рейз обновляет минимум и переоткрывает торговлю остальным.
    is_raise = ~is_fold & (target > level)
    full_raise = is_raise & (target - level >= s.last_raise)
    acted = s.acted.at[me].set(True)
    acted = jnp.where(
        full_raise,
        jnp.arange(n, dtype=jnp.int32) == me,
        acted,
    )

    # Только apply_action пишет историю; step не добавляет вторую запись.
    new_s = s.replace(
        stack=s.stack.at[me].add(-add),
        bet_street=s.bet_street.at[me].add(add),
        bet_total=s.bet_total.at[me].add(add),
        folded=s.folded.at[me].set(s.folded[me] | is_fold),
        acted=acted,
        last_raise=jnp.where(full_raise, target - level, s.last_raise),
        history_len=jnp.where(can_apply, s.history_len + jnp.int32(1), s.history_len),
    )
    if capacity:
        idx = jnp.clip(s.history_len, 0, capacity - 1)
        new_s = new_s.replace(
            history_player=s.history_player.at[idx].set(
                jnp.where(can_apply, me, s.history_player[idx])
            ),
            history_action=s.history_action.at[idx].set(
                jnp.where(can_apply, a, s.history_action[idx])
            ),
            history_amount=s.history_amount.at[idx].set(
                jnp.where(can_apply, add, s.history_amount[idx])
            ),
            history_street=s.history_street.at[idx].set(
                jnp.where(can_apply, s.street, s.history_street[idx])
            ),
        )
    # Guarding the appended slot preserves history without selecting every slot.
    return _sel_physical(can_apply, new_s, s)


def next_to_act(s: State) -> jnp.ndarray:
    """После хода ищет следующее место по часовой; terminal или нет ходов → -1.

    s.to_act — тот, кто только что сходил. Первого игрока новой улицы выбирают step и step_target после сброса acted.
    """
    config.validate_player_count(s.stack.shape[0])

    def find_next(state):
        n = state.stack.shape[0]
        live = derived.can_act(state)
        owes = state.bet_street < derived.bet_level(state)
        live_count = jnp.sum(live, dtype=jnp.int32)
        others_live = live_count - live.astype(jnp.int32) >= 1
        needs = live & (owes | (~state.acted & others_live))

        # Должник ходит даже с acted=True; без долга нужен ещё живой соперник.
        order = (state.to_act + jnp.arange(1, n + 1, dtype=jnp.int32)) % n
        ordered_needs = needs[order]
        first = order[jnp.argmax(ordered_needs)]
        return jnp.where(jnp.any(ordered_needs), first, jnp.int32(-1))

    return jax.lax.cond(
        s.terminal,
        lambda _: jnp.array(-1, dtype=jnp.int32),
        find_next,
        s,
    )


def _sel(condition: jnp.ndarray, x: State, y: State) -> State:
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), x, y)


def _sel_physical(condition: jnp.ndarray, x: State, y: State) -> State:
    """Select physical fields; x already carries the guarded action history."""
    return x.replace(**{
        name: jnp.where(condition, getattr(x, name), getattr(y, name))
        for name in State.__dataclass_fields__
        if not name.startswith("history_")
    })


def step(s: State, a: jnp.ndarray) -> State:
    """Применяет легальный ход до следующего решения или терминала.

    При terminal=False легальность a гарантирует вызывающий.
    При terminal=True возвращает s без изменений для любого индекса действия.
    """
    a = action_input(a)
    s1 = apply_action(s, a)
    n_alive = jnp.sum(~s1.folded, dtype=jnp.int32)
    nxt = next_to_act(s1)

    # Постфлоп первым ходит активное место слева от BTN, в HU это BB.
    fresh = s1.replace(
        street=s1.street + jnp.int32(1),
        bet_street=jnp.zeros_like(s1.bet_street, dtype=jnp.int32),
        acted=jnp.zeros_like(s1.acted, dtype=jnp.bool_),
        last_raise=jnp.array(config.BB, dtype=jnp.int32),
        to_act=jnp.int32(0),
    )
    first = next_to_act(fresh)
    terminal = jnp.array(True, dtype=jnp.bool_)
    no_actor = jnp.int32(-1)

    # Выбор снизу вверх: новая улица, шоудаун, текущая улица, победа фолдом.
    out = fresh.replace(to_act=first)
    out = _sel(
        (s1.street == 3) | (first == -1),
        s1.replace(terminal=terminal, to_act=no_actor, street=jnp.int32(3)),
        out,
    )
    out = _sel(nxt != -1, s1.replace(to_act=nxt), out)
    out = _sel(n_alive == 1, s1.replace(terminal=terminal, to_act=no_actor), out)
    # History in every branch comes from the already guarded apply_action.
    # Do not advance physical fields after an action rejected by the safety gate.
    return _sel_physical(_can_apply(s) & fits_int32(a), out, s)


def apply_target(s: State, a: jnp.ndarray, target: jnp.ndarray) -> State:
    """apply_action with the street total `target` given explicitly instead of the grid's.

    The live play path (poker/solver/subgame.py, poker/player/live.py) applies every action at its exact chip
    amount - an opponent's bet of any size, our own tree sizes that are not grid fractions - so no
    amount is ever rounded to the grid. Same rules as apply_action: a fold adds nothing, a target
    above the bet level is a raise, a full raise (>= last_raise) reopens the betting; the history
    records the index `a` (FOLD, CHECK_CALL, or a raise index >= 2: raises_on_street counts it) and
    the chips actually added. Legality of (a, target) is the caller's precondition, as everywhere
    in the engine: target must be the actor's street bet (fold), the call total, or a legal raise
    total (the all-in or a full raise below it). Host checks before the jitted call: poker/player/live.py
    check_legal (the live path), poker.envs.guards.target (other callers, including poker/arena/slumbot/bot.py). All use this shared engine."""
    a = action_input(a)
    can_apply = _can_apply(s) & fits_int32(a)
    a = jnp.where(fits_int32(a), a, 0).astype(jnp.int32)
    n = s.stack.shape[0]
    capacity = s.history_player.shape[0]
    me = jnp.clip(s.to_act, 0, n - 1)
    level = derived.bet_level(s.replace(to_act=me))
    current = s.bet_street[me]
    target = jnp.where(can_apply, jnp.asarray(target, jnp.int32), current)
    is_fold = a == config.FOLD
    add = jnp.where(is_fold, jnp.int32(0), target - current)
    is_raise = ~is_fold & (target > level)
    full_raise = is_raise & (target - level >= s.last_raise)
    acted = s.acted.at[me].set(True)
    acted = jnp.where(full_raise, jnp.arange(n, dtype=jnp.int32) == me, acted)
    new_s = s.replace(
        stack=s.stack.at[me].add(-add),
        bet_street=s.bet_street.at[me].add(add),
        bet_total=s.bet_total.at[me].add(add),
        folded=s.folded.at[me].set(s.folded[me] | is_fold),
        acted=acted,
        last_raise=jnp.where(full_raise, target - level, s.last_raise),
        history_len=jnp.where(can_apply, s.history_len + jnp.int32(1), s.history_len),
    )
    if capacity:
        idx = jnp.clip(s.history_len, 0, capacity - 1)
        new_s = new_s.replace(
            history_player=s.history_player.at[idx].set(jnp.where(can_apply, me, s.history_player[idx])),
            history_action=s.history_action.at[idx].set(jnp.where(can_apply, a, s.history_action[idx])),
            history_amount=s.history_amount.at[idx].set(jnp.where(can_apply, add, s.history_amount[idx])),
            history_street=s.history_street.at[idx].set(jnp.where(can_apply, s.street, s.history_street[idx])),
        )
    return _sel_physical(can_apply, new_s, s)


def step_target(s: State, a: jnp.ndarray, target: jnp.ndarray) -> State:
    """step with apply_target: the same street, showdown and fold logic as step, the amount explicit.
    With target = action_table(s)[1][a] it is step(s, a) bit for bit."""
    a = action_input(a)
    s1 = apply_target(s, a, target)
    n_alive = jnp.sum(~s1.folded, dtype=jnp.int32)
    nxt = next_to_act(s1)
    fresh = s1.replace(
        street=s1.street + jnp.int32(1),
        bet_street=jnp.zeros_like(s1.bet_street, dtype=jnp.int32),
        acted=jnp.zeros_like(s1.acted, dtype=jnp.bool_),
        last_raise=jnp.array(config.BB, dtype=jnp.int32),
        to_act=jnp.int32(0),
    )
    first = next_to_act(fresh)
    terminal = jnp.array(True, dtype=jnp.bool_)
    no_actor = jnp.int32(-1)
    out = fresh.replace(to_act=first)
    out = _sel((s1.street == 3) | (first == -1), s1.replace(terminal=terminal, to_act=no_actor, street=jnp.int32(3)), out)
    out = _sel(nxt != -1, s1.replace(to_act=nxt), out)
    out = _sel(n_alive == 1, s1.replace(terminal=terminal, to_act=no_actor), out)
    return _sel_physical(_can_apply(s) & fits_int32(a), out, s)


# Internal checked workers retain the legal-action precondition (the mask is not checked).
# Host callers must call err.throw() BEFORE accepting candidate state.
# For a batched scan, wrap the entire rollout with checkify.checkify instead.
checked_apply_action = jax.jit(checkify.checkify(apply_action))
checked_step = jax.jit(checkify.checkify(step))
