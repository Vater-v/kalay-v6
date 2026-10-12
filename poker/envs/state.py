import chex
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class State:
    stack:      jnp.ndarray  # int32[n]  фишки за спиной
    bet_street: jnp.ndarray  # int32[n]  вложено на текущей улице
    bet_total:  jnp.ndarray  # int32[n]  вложено за раздачу
    folded:     jnp.ndarray  # bool[n]
    acted:      jnp.ndarray  # bool[n]   ходил после последней агрессии
    last_raise: jnp.ndarray  # int32[]   последнее полное повышение
    street:     jnp.ndarray  # int32[]   0 префлоп .. 3 ривер
    to_act:     jnp.ndarray  # int32[]   чей ход (номер места)
    terminal:   jnp.ndarray  # bool[]
    hole:       jnp.ndarray  # int32[n, 2]  карты игроков
    board:      jnp.ndarray  # int32[5]  полный борд; видимость определяет observe
    history_player: jnp.ndarray  # int32[H]  место до применения действия
    history_action: jnp.ndarray  # int32[H]  выбранный индекс действия
    history_amount: jnp.ndarray  # int32[H]  фактически добавленные фишки
    history_street: jnp.ndarray  # int32[H]  улица до перехода
    history_len:    jnp.ndarray  # int32[]   число публичных решений (без блайндов)
    rake_bps:       jnp.ndarray  # int32[]   рейк, базисные пункты (config.rake_terms; 0 — без рейка)
    rake_cap:       jnp.ndarray  # int32[]   кэп рейка в фишках по числу сданных игроков
