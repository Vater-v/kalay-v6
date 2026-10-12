import jax.numpy as jnp

from poker.envs import config
from poker.envs.state import State


def pot(s: State) -> jnp.ndarray:
    # Банк — все фишки, вложенные за раздачу, включая ставки сбросивших игроков.
    return jnp.sum(s.bet_total, dtype=jnp.int32)


def max_bet(s: State) -> jnp.ndarray:
    # Текущая ставка — наибольший вклад игрока на этой улице.
    return jnp.max(s.bet_street)


def bet_level(s: State) -> jnp.ndarray:
    # На префлопе уровень не ниже полного BB, даже если BB поставлен коротким олл-ином.
    level = max_bet(s)
    return jnp.where(s.street == 0, jnp.maximum(level, config.BB), level)


def to_call(s: State) -> jnp.ndarray:
    # Долг ходящего игрока — разница до уровня ставки, не меньше нуля.
    return jnp.maximum(0, bet_level(s) - s.bet_street[s.to_act])


def call_amount(s: State) -> jnp.ndarray:
    # Реальный колл ограничен стеком; нехватка фишек означает колл олл-ином.
    return jnp.minimum(to_call(s), s.stack[s.to_act])


def min_raise_to(s: State) -> jnp.ndarray:
    # Минимум полного рейза; олл-ин выше bet_level, но ниже минимума — короткий рейз без изменения last_raise; права — s1.3/s1.4 (TDA 45/49).
    return bet_level(s) + s.last_raise


def max_raise_to(s: State) -> jnp.ndarray:
    # Рейз до олл-ина — текущий вклад ходящего игрока плюс весь его оставшийся стек.
    return s.bet_street[s.to_act] + s.stack[s.to_act]


def all_in(s: State) -> jnp.ndarray:
    # Олл-ин — нулевой стек у игрока, который не сбросил карты.
    return (s.stack == 0) & ~s.folded


def can_act(s: State) -> jnp.ndarray:
    # Принимать решения ещё могут игроки с фишками, которые не сбросили карты.
    return ~s.folded & (s.stack > 0)
