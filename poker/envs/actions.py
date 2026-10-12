import jax.numpy as jnp

from poker.envs import config, derived
from poker.envs.state import State


def action_candidates(s: State, sizes=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Маска действий и итоговые ставки на улице для одного стола.

    Canonical targets are retained on masked columns too, for safe replay of dead tree branches.
    Only columns selected by the returned mask may be played. action_table hides the masked targets.

    Предусловие: если s.terminal == False, то can_act(s)[s.to_act] == True.
    Это обеспечивает step; здесь контракт не проверяется и не маскируется.

    sizes: None (every fraction raise of the grid) or bool[NUM_ACTIONS], possibly traced: the fraction
    raises a tree offers at this decision (poker/solver/tree.py; the fold, check-call and all-in entries are
    ignored); the others are masked. Duplicates are resolved within the offered sizes: a fraction raise
    whose target equals that of a lower offered size legal here is masked (identical chips are one
    engine action, the lowest index keeps it), so a size clamped up to the minimum raise stays when the
    lower grid size with the same chips is not offered.
    Under grid 13 (config.COMMIT_MERGE) a fraction raise that leaves the raiser at most
    config.COMMIT_MERGE_POT times the pot after the raise behind is the all-in: its target becomes the
    stack and it is masked like a size at or above the stack (the all-in stays). Grids 5 and 7: no rule.
    Under grid 13 (config.EFFECTIVE_STACK) the stack of these two rules is the effective one: at most what
    the deepest live opponent can have in on this street (street bet + stack); a raise to that or more
    pays like the all-in (the excess comes back), so it is masked and the all-in stays. Equal stacks: the table is unchanged.
    """
    config.validate_player_count(s.stack.shape[0])
    if sizes is not None:
        sizes = jnp.asarray(sizes, dtype=jnp.bool_)
        if sizes.shape != (config.NUM_ACTIONS,):
            raise ValueError(f"sizes must be bool[{config.NUM_ACTIONS}], got shape {sizes.shape}")
    me = s.to_act
    current = s.bet_street[me]
    level = derived.bet_level(s)
    to_call = derived.to_call(s)
    pot = derived.pot(s)
    minimum = derived.min_raise_to(s)
    maximum = derived.max_raise_to(s)

    opponents = jnp.arange(s.stack.shape[0], dtype=jnp.int32) != me
    has_live_opponent = jnp.any(derived.can_act(s) & opponents)
    effective = maximum
    if config.EFFECTIVE_STACK:
        cover = jnp.max(jnp.where(opponents & ~s.folded, s.bet_street + s.stack, jnp.int32(0)))
        effective = jnp.minimum(maximum, cover)
    # Суммарный рост ставки после последнего хода может переоткрыть торговлю.
    reopened = to_call >= s.last_raise
    raise_open = ((~s.acted[me]) | reopened) & has_live_opponent

    mask = jnp.zeros(config.NUM_ACTIONS, dtype=jnp.bool_)
    target = jnp.full(config.NUM_ACTIONS, -1, dtype=jnp.int32)

    # Бесплатный чек исключает фолд; колл ограничен оставшимся стеком.
    mask = mask.at[config.FOLD].set(to_call > 0)
    target = target.at[config.FOLD].set(current)
    mask = mask.at[config.CHECK_CALL].set(True)
    target = target.at[config.CHECK_CALL].set(current + derived.call_amount(s))

    # Статические доли банка после колла; полный рейз, кроме олл-ина и дублей. Дубль - та же цель у
    # меньшего размера, уже разрешённого здесь: mask несёт только предложенные (sizes) размеры.
    for i, (num, den) in enumerate(config.RAISE_FRACS):
        action = 2 + i
        raw = level + (pot + to_call) * num // den
        tgt = jnp.maximum(raw, minimum)
        if config.COMMIT_MERGE is not None:
            # ★ Слияние с олл-ином: за спиной <= COMMIT_MERGE_POT банка после рейза - цель = стек (точно в целых:
            # floor(after * num / den) без переполнения int32).
            merge_num, merge_den = config.COMMIT_MERGE
            after = pot + tgt - current
            limit = after // merge_den * merge_num + after % merge_den * merge_num // merge_den
            tgt = jnp.where(effective - tgt <= limit, effective, tgt)
        duplicate = jnp.any(mask[2:action] & (target[2:action] == tgt))
        allowed = raise_open & (tgt < effective) & ~duplicate
        if sizes is not None:
            allowed = allowed & sizes[action]
        mask = mask.at[action].set(allowed)
        target = target.at[action].set(tgt)

    # Олл-ин выше уровня ставки может быть полным или коротким рейзом.
    mask = mask.at[config.ALLIN].set(raise_open & (maximum > level))
    target = target.at[config.ALLIN].set(maximum)

    return mask & ~s.terminal, target


def action_table(s: State, sizes=None) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Canonical offered actions, with target -1 on every masked column (the engine contract)."""
    mask, target = action_candidates(s, sizes)
    return mask, jnp.where(mask, target, jnp.int32(-1))
