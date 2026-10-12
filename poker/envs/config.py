from fractions import Fraction
import hashlib
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np

from poker.envs import fingerprint

N = 2
MAX_PLAYERS = 12
SB = 50
BB = 100
START = 20000          # the engine's default stack and history-capacity floor; table depths: DEPTH_MIX

FOLD = 0
CHECK_CALL = 1
# Сетки ставок: доли банка рейзов как целые (числитель, знаменатель); рейз i имеет индекс 2 + i.
# Сетка 13 — текущая игра с наборами STREET_SIZES. Сетка 7 — воспроизведение итерации 0 (k0),
# сетка 5 — прежние проверки движка. Сетка процесса задаётся режимом REGIMES; явный POKER_GRID=5|7
# включает воспроизведение легаси с баннером при импорте. checks/run.py задаёт сетку каждому модулю.
GRIDS = {5: ((1, 2), (1, 1)), 7: ((1, 3), (2, 3), (1, 1), (2, 1)), 13: ((1, 100), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1), (3, 2), (2, 1), (5, 2))}

_INT32_MAX = (1 << 31) - 1

# Рейк. Источник: https://coinpoker.com/rake/ («Last Updated: 8 October 2026»,
# снято 2026-10-08), кэш NLHE/PLO: 5% на всех лимитах, кэп по числу сданных игроков 2 / 3–4 / 5+.
#   NL5  ($0.02/$0.05): $0.13 / $0.30 / $0.50 = 2,6 / 6 / 10 BB
#   NL10 ($0.05/$0.10): $0.25 / $0.60 / $1.00 = 2,5 / 6 / 10 BB
#   NL25 ($0.10/$0.25): $0.50 / $1.20 / $2.00 = 2 / 4,8 / 8 BB
#   NL50 ($0.25/$0.50): $1.00 / $2.40 / $4.00 = 2 / 4,8 / 8 BB (в BB то же, что NL25)
# Отдельные HU-столы сайта (кэп 3 BB на NL5–10, 2,4 BB на NL25–50) не моделируются: два сданных
# игрока за обычным столом берут кэп «2 игрока».
# Правила, которых сайт не публикует (выбор записан, меняется только с RAKE_SCHEME):
# - no flop no drop: рука, закончившаяся до раздачи флопа, без рейка; олл-ин на префлопе с
#   раздачей борда — с рейком;
# - база — банк без неоплаченной части последней ставки (она возвращается);
# - рейк = min(floor(база * bps / 10000), кэп) в фишках движка (1 фишка = 0,01 BB), вниз;
# - один рейк на руку; основной и побочные банки платят пропорционально своей базе (вниз),
#   остаток от округления — основной банк.
# Уровень rake_level в [0, 1]: кусочно-линейная интерполяция процента и кэпов между опорами
# RAKE_ANCHORS (0 — без рейка, 1/3 — NL25 = NL50, 2/3 — NL10, 1 — NL5), округление до целых bps
# и фишек.
RAKE_SOURCE = "https://coinpoker.com/rake/ (Last Updated: 8 October 2026; fetched 2026-10-08)"
RAKE_SCHEME = 1
RAKE_ANCHORS = (            # (level, basis points, caps in BB for 2 / 3-4 / 5+ players dealt)
    (0.0, 0, (0.0, 0.0, 0.0)),
    (1 / 3, 500, (2.0, 4.8, 8.0)),
    (2 / 3, 500, (2.5, 6.0, 10.0)),
    (1.0, 500, (2.6, 6.0, 10.0)),
)
RAKE_LEVELS = {"none": 0.0, "NL50": 1 / 3, "NL25": 1 / 3, "NL10": 2 / 3, "NL5": 1.0}
RAKE_CAP_MAX_CHIPS = int(max(c for a in RAKE_ANCHORS for c in a[2]) * BB)     # the largest cap: evaluator.rake_split's int32 product

# One explicit table regime sets the loop's spot generators and live matches; every
# artifact records it in scheme(). check_scheme refuses a different regime unless POKER_REGIME_FORCE=1.
#   1 "slumbot": heads-up, 200 BB stacks, no rake.
#   2 "nl5_2to6": 2-6 players, 100 BB in 70% of hands, otherwise log-uniform 20-500 BB, NL5 rake.
# depth_mix / rake_mix rows are (low, high, probability): equal bounds mean a fixed value; otherwise
# depth is log-uniform in BB (whole BB, equal stacks) and rake level uniform. players controls
# the table sizes and the rake cap. The loop and live matches currently deal
# two seats only; regime 2's 3-6 player entries describe the intended field, which they do not cover.
# POKER_REGIME=1|2 has no default: an unset regime fails at import. checks/run.py pins checks to 1;
# checks/s6_3.py exercises both. POKER_GRID overrides the regime's grid only for legacy reproduction.
# Artifacts without a regime record predate the toggle: check_regime treats them as regime 1 with
# a note that their data used NL5 rake and preflop depths 70% 100 BB / 30% 20-500 BB.
REGIMES = {
    1: {"name": "slumbot", "players": (2,), "grid": 13, "depth_mix": ((200, 200, 1.0),), "rake_mix": ((0.0, 0.0, 1.0),)},
    2: {"name": "nl5_2to6", "players": (2, 3, 4, 5, 6), "grid": 13, "depth_mix": ((100, 100, 0.7), (20, 500, 0.3)),
        "rake_mix": ((RAKE_LEVELS["NL5"], RAKE_LEVELS["NL5"], 1.0),)},
}
_REGIME_SETTING = os.environ.get("POKER_REGIME", "").strip()
if not _REGIME_SETTING:
    raise ValueError("POKER_REGIME is not set: the table regime has no default (task #39), every process names it - "
                     + "; ".join(f"POKER_REGIME={k}: {r['name']} (players {list(r['players'])}, depth mix "
                                 f"{[list(x) for x in r['depth_mix']]} BB, rake mix {[list(x) for x in r['rake_mix']]}, "
                                 f"grid {r['grid']})" for k, r in REGIMES.items()))
REGIME = int(_REGIME_SETTING) if _REGIME_SETTING.isdigit() else None
if REGIME not in REGIMES:
    raise ValueError(f"POKER_REGIME must be one of {sorted(REGIMES)}, got {_REGIME_SETTING}")
_FORCE = os.environ.get("POKER_REGIME_FORCE", "0")
if _FORCE not in ("", "0", "1"):
    raise ValueError(f"POKER_REGIME_FORCE must be 0 or 1, got {_FORCE!r}")
REGIME_FORCE = _FORCE == "1"
REGIME_NAME = REGIMES[REGIME]["name"]
PLAYERS = REGIMES[REGIME]["players"]
DEPTH_MIX = REGIMES[REGIME]["depth_mix"]          # depth in BB of every hand and preflop spot
RAKE_MIX = REGIMES[REGIME]["rake_mix"]            # rake level of every hand and spot
_GRID_SETTING = os.environ.get("POKER_GRID", "").strip()
GRID = REGIMES[REGIME]["grid"] if not _GRID_SETTING else int(_GRID_SETTING) if _GRID_SETTING.isdigit() else None
if GRID not in GRIDS:
    raise ValueError(f"POKER_GRID must be one of {sorted(GRIDS)} (unset: the regime's grid), got {_GRID_SETTING}")
if GRID != REGIMES[REGIME]["grid"]:              # legacy reproduction only: one loud line per process
    print(f"!!! LEGACY GRID {GRID}: POKER_GRID={GRID} replaces regime {REGIME}'s grid {REGIMES[REGIME]['grid']} - for "
          f"reproducing iteration-0 artifacts (k0) and the grid-5/7 checks only; nothing made in this process is the "
          f"decided game !!!", file=sys.stderr, flush=True)
RAISE_FRACS = GRIDS[GRID]
ALLIN = 2 + len(RAISE_FRACS)
NUM_ACTIONS = ALLIN + 1


def sample_rake_levels(rng, count, mix=None):
    """float32 [count] rake levels from a mixture (default RAKE_MIX) with a numpy Generator: the
    host-side draw of live matches (TableSampler.sample_rake is the jax one)."""
    mix = np.asarray(RAKE_MIX if mix is None else mix, np.float64).reshape(-1, 3)
    rows = rng.choice(len(mix), size=count, p=mix[:, 2] / mix[:, 2].sum())
    return rng.uniform(mix[rows, 0], mix[rows, 1]).astype(np.float32)


def sample_depths_bb(rng, count, mix=None):
    """int32 [count] depths in whole BB from a mixture (default DEPTH_MIX) with a numpy Generator: a
    row by its probability, then the depth log-uniform in [low, high] (low == high: that depth),
    rounded. The host-side draw of training depths and live matches. Uniform for uniform it is the draw of
    situations.make_preflop_spots before the toggle: regime 2's mix gives those depths unchanged."""
    mix = np.asarray(DEPTH_MIX if mix is None else mix, np.float64).reshape(-1, 3)
    if mix[:, 0].min() < 2:
        raise ValueError(f"depths start above the big blind (>= 2 BB): {mix.tolist()}")
    rows = rng.choice(len(mix), size=count, p=mix[:, 2] / mix[:, 2].sum())
    return np.round(np.exp(rng.uniform(np.log(mix[rows, 0]), np.log(mix[rows, 1])))).astype(np.int32)


def depth_range_bb(mix=None):
    """(shallowest, deepest) depth in BB a mixture (default DEPTH_MIX) deals. The deepest sizes the
    history of a run's hands (poker.envs.betting.init max_stack): one compiled shape for every batch."""
    mix = np.asarray(DEPTH_MIX if mix is None else mix, np.float64).reshape(-1, 3)
    live = mix[:, 2] > 0
    return int(mix[live, 0].min()), int(mix[live, 1].max())


def depth_coverage(depths) -> dict:
    """Host report of starting depths against the declared mixture; no quality or validity gate."""
    import math
    actual = sorted({float(depth) for depth in depths})
    if not actual or not all(math.isfinite(depth) for depth in actual):
        raise ValueError("depth coverage needs finite starting depths")
    outside = [depth for depth in actual
               if not any(probability > 0 and low <= depth <= high for low, high, probability in DEPTH_MIX)]
    return {"status": "extrapolation" if outside else "within_declared_depth_range",
            "training_depth_mix_bb": [list(row) for row in DEPTH_MIX],
            "actual_depths_bb": actual, "outside_training_depths_bb": outside}


def regime_record(number=None) -> dict:
    """A regime (default this process's) as artifacts record it: scheme(), layouts, run results."""
    number = REGIME if number is None else number
    r = REGIMES[number]
    return {"number": number, "name": r["name"], "players": list(r["players"]),
            "depth_mix": [list(x) for x in r["depth_mix"]], "rake_mix": [list(x) for x in r["rake_mix"]]}


def describe_regime(record) -> str:
    return (f"regime {record.get('number')} ({record.get('name')}: players {record.get('players')}, "
            f"depth mix {record.get('depth_mix')} BB, rake mix {record.get('rake_mix')})")


def saved_regime(meta) -> dict:
    """The regime record of a saved scheme() dict; regime 1 for one from before the toggle (check_regime
    notes it when such an artifact is loaded)."""
    regime = meta.get("regime") if isinstance(meta, dict) else None
    return regime_record(1) if regime is None else regime


_NOTED = set()


def _note(message):
    """One line on stdout, once per process for the same message (an artifact is often loaded again)."""
    if message not in _NOTED:
        _NOTED.add(message)
        print(time.strftime("%H:%M:%S"), message, flush=True)


def check_regime(saved, what: str) -> None:
    """Raise ValueError unless `saved` (an artifact's regime record, scheme()["regime"]) is this
    process's regime. None: an artifact from before the toggle, taken as regime 1 with a one-line note
    (pre-toggle data and weights were made under NL5 rake). POKER_REGIME_FORCE=1 accepts another
    regime with a WARNING (a deliberate cross-regime evaluation)."""
    mine = regime_record()
    if saved is None:
        saved = regime_record(1)
        _note(f"{what}: no regime recorded (made before the regime toggle of 2026-10-10), taken as regime 1 "
              f"({REGIMES[1]['name']}): trained under NL5 rake; regime 1 has none (and its preflop depths were "
              f"70% 100 BB / 30% 20-500 BB, regime 1 deals 200 BB only)")
    if saved == mine:
        return
    if saved.get("number") == REGIME:
        problem = f"{what}: made under {describe_regime(saved)}; this code defines it as {describe_regime(mine)}"
        remedy = "re-make it under this definition"
    else:
        problem = f"{what}: made under {describe_regime(saved)}; this process is {describe_regime(mine)}"
        remedy = f"set POKER_REGIME={saved.get('number')} to use it"
    if REGIME_FORCE:
        _note(f"WARNING: {problem}; accepted for a deliberate cross-regime evaluation (POKER_REGIME_FORCE=1)")
        return
    raise ValueError(f"{problem}; {remedy}, or POKER_REGIME_FORCE=1 for a deliberate cross-regime evaluation")


# The table size the training loop deals and solves (poker/solver/loop.py PLAYERS; heads_up_part below guards the pipeline):
# The current training pipeline deals and solves two seats; multiway training is not implemented.
LOOP_PLAYERS = 2


def heads_up_part(what: str) -> tuple:
    """Guard of the heads-up pipeline (two seats dealt: poker/solver/loop.py and poker/arena/match.py): ValueError unless the regime deals two players; returns the table
    sizes of the regime it leaves out (regime 2: 3-6 players - multiway play is not implemented)."""
    if 2 not in PLAYERS:
        raise ValueError(f"{what} deals heads-up hands only; {describe_regime(regime_record())} has no "
                         f"two-player table")
    return tuple(n for n in PLAYERS if n != 2)


def rake_terms(level, n_dealt):
    """(basis points, cap in chips), both int32, of the rake at `level` in [0, 1] for `n_dealt`
    players dealt in. Same arithmetic for Python numbers and traced JAX values."""
    xs = jnp.asarray([a[0] for a in RAKE_ANCHORS], jnp.float32)
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 0.0, 1.0)
    bps = jnp.interp(lv, xs, jnp.asarray([a[1] for a in RAKE_ANCHORS], jnp.float32))
    caps = jnp.asarray([[c * BB for c in a[2]] for a in RAKE_ANCHORS], jnp.float32)    # [K, 3]
    n = jnp.asarray(n_dealt, jnp.int32)
    column = jnp.where(n <= 2, 0, jnp.where(n <= 4, 1, 2))
    cap = jnp.interp(lv, xs, jnp.take(caps, column, axis=1))
    return jnp.round(bps).astype(jnp.int32), jnp.round(cap).astype(jnp.int32)


def rake_from_terms(bps, cap, pot, street_reached):
    """Rake in chips: min(floor(pot * bps / 10000), cap) once the flop is dealt
    (street_reached >= 1), else 0. pot is the contested pot (without the uncalled bet).
    The floor is split as q * bps + floor(r * bps / 10000) so int32 never overflows."""
    pot = jnp.asarray(pot, jnp.int32)
    bps = jnp.asarray(bps, jnp.int32)
    q, r = pot // 10000, pot % 10000
    amount = jnp.minimum(q * bps + r * bps // 10000, jnp.asarray(cap, jnp.int32))
    return jnp.where(jnp.asarray(street_reached) >= 1, amount, jnp.int32(0)).astype(jnp.int32)


def scheme() -> dict:
    """Payout scheme of this process: everything that changes solved terminals or the network's
    action space, and the table regime (REGIMES) the data come from. Datasets and weights carry it;
    every load compares it (check_scheme)."""
    return {"grid": GRID, "raise_fracs": [list(f) for f in RAISE_FRACS], "num_actions": NUM_ACTIONS,
            "chip_per_bb": BB, "sb": SB, "rake_scheme": RAKE_SCHEME, "regime": regime_record()}


def check_scheme(meta, what: str) -> None:
    """Raise ValueError unless `meta` (a scheme() dict saved with data or weights) matches this
    process. A missing scheme is rejected too: unlabeled artifacts cannot be trusted. The regime is
    compared by check_regime (a scheme without one is from before the toggle: regime 1)."""
    if not isinstance(meta, dict):
        raise ValueError(f"{what}: no payout scheme recorded; expected {scheme()}")
    mine = scheme()
    diff = {k: (meta.get(k), v) for k, v in mine.items() if k != "regime" and meta.get(k) != v}
    if diff:
        raise ValueError(f"{what}: payout scheme mismatch (saved, this process): {diff}; "
                         f"set POKER_GRID / use matching code, or re-solve")
    check_regime(meta.get("regime"), what)


# Legacy teacher trees under grids 5 and 7: postflop cap 2 + all-in heads-up, cap 1 + all-in
# with 3-6 players; preflop heads-up uses every grid size at cap 2 + all-in. With 3-5 players,
# preflop uses 2/3 and pot at each raise level; with 6 it uses 2/3, pot, then 2/3, at cap 3 + all-in.
# From 3 players there is no open limp: only the SB may complete and the BB may check unopened.
# The last size entry repeats. checks/s5_4.py checks the resulting preflop amounts.
# Grid 13 takes its caps from TREE_CAPS and its heads-up sizes from STREET_SIZES below; multiway
# preflop keeps the sizes and limp rule of PREFLOP_TREES. Layouts record the tree rules separately
# from scheme(): the tree determines the solved game, while scheme() records payouts/action space.
TREE_MAX_PLAYERS = 6
POSTFLOP_CAP = {2: 2, 3: 1, 4: 1, 5: 1, 6: 1}
PREFLOP_TREES = {
    2: {"cap": 2, "sizes": None, "no_open_limp": False},
    3: {"cap": 3, "sizes": (((2, 3), (1, 1)),), "no_open_limp": True},
    4: {"cap": 3, "sizes": (((2, 3), (1, 1)),), "no_open_limp": True},
    5: {"cap": 3, "sizes": (((2, 3), (1, 1)),), "no_open_limp": True},
    6: {"cap": 3, "sizes": (((2, 3),), ((1, 1),), ((2, 3),)), "no_open_limp": True},
}


# Размеры ставок по улицам и глубине — покерная эвристика, не результат замера. Доли банка
# (числитель, знаменатель) заданы по уровням рейза: 0 = первая ставка или открытие, последний уровень
# повторяется до капа. Олл-ин доступен всегда; движок убирает размеры выше стека и дубликаты.
# Наборы действуют только на сетке 13; сетки 5 и 7 воспроизводят прежние деревья.
# Префлоп-меню выбирается по эффективному стеку в BB на старте руки, постфлоп — по SPR на старте
# улицы (стек / банк). Учитель группирует батчи по меню (poker/solver/loop.py stage_batches).
STREET_SIZES = {
    "normal": {
        0: (((1, 100), (1, 1)), ((1, 1),), ((2, 3),)),   # открытие мин-рейз или банк; 3-бет банк; 4-бет 2/3; дальше олл-ин
        1: (((1, 3), (3, 4)), ((2, 3),)),                 # флоп: 33 %, 75 %; рейз 66 %
        2: (((1, 2), (1, 1), (3, 2)), ((2, 3),)),         # терн: 50 %, 100 %, 150 % (оверставка глубоко); рейз 66 %
        3: (((1, 3), (3, 4), (3, 2)), ((3, 4),)),         # ривер: 33 %, 75 %, 150 %; рейз 75 %
    },
    "short": {
        0: (((1, 100), (1, 1)), ((1, 1),)),               # открытие мин-рейз или банк; 3-бет банк; 4-бет = олл-ин
        1: (((1, 2), (1, 1)), ((2, 3),)),                 # 50 %, 100 %; оверставка = олл-ин
        2: (((1, 2), (1, 1)), ((2, 3),)),
        3: (((1, 2), (1, 1)), ((3, 4),)),
    },
}
STREET_SIZES_GRID = 13
# Капы решённых деревьев сетки 13: TREE_CAPS[игроки][улица][меню], читается через _cap / cap_for.
# Два игрока: normal 3 и short 2 + олл-ин на каждой улице. Дополнительный обычный рейз в глубоком
# банке сохраняет ответ меньшего размера перед пушем. 3-6 игроков: префлоп 3, постфлоп 1 во всех меню;
# эти ограничения требуют нового замера при подключении multiway игры. Под сетками 5 и 7 действуют
# прежние PREFLOP_TREES[n]["cap"] и POSTFLOP_CAP[n] независимо от меню.
TREE_CAPS = {
    2: {0: {"normal": 3, "short": 2}, 1: {"normal": 3, "short": 2}, 2: {"normal": 3, "short": 2}, 3: {"normal": 3, "short": 2}},
    3: {0: {"normal": 3, "short": 3}, 1: {"normal": 1, "short": 1}, 2: {"normal": 1, "short": 1}, 3: {"normal": 1, "short": 1}},
    4: {0: {"normal": 3, "short": 3}, 1: {"normal": 1, "short": 1}, 2: {"normal": 1, "short": 1}, 3: {"normal": 1, "short": 1}},
    5: {0: {"normal": 3, "short": 3}, 1: {"normal": 1, "short": 1}, 2: {"normal": 1, "short": 1}, 3: {"normal": 1, "short": 1}},
    6: {0: {"normal": 3, "short": 3}, 1: {"normal": 1, "short": 1}, 2: {"normal": 1, "short": 1}, 3: {"normal": 1, "short": 1}},
}
# Слияние почти-олл-ина с олл-ином на сетке 13: если после рейза долей банка за спиной остаётся
# не больше COMMIT_MERGE_POT банка после рейза, этот размер становится олл-ином и маскируется;
# сам олл-ин остаётся. Это эвристика, общая для учителя и игры (poker/envs/actions.py).
# Под сетками 5 и 7 правило выключено для воспроизведения k0. COMMIT_MERGE хранит ту же долю
# как (числитель, знаменатель); правило входит в game_record для сетки 13.
COMMIT_MERGE_POT = 0.25
if not 0 <= COMMIT_MERGE_POT <= 1:
    raise ValueError(f"COMMIT_MERGE_POT must be in [0, 1], got {COMMIT_MERGE_POT}")
COMMIT_MERGE = Fraction(COMMIT_MERGE_POT).limit_denominator(64).as_integer_ratio() if GRID == STREET_SIZES_GRID else None
# Under grid 13, fraction raises are bounded by what the deepest live opponent can match on
# this street (street bet + stack). Raising above that cover pays like all-in after the excess
# returns, so poker/envs/actions.py masks such fractions and measures commit-merge against the cover.
# All-in stays available. Grids 5 and 7 retain their old tables; equal stacks are unchanged.
EFFECTIVE_STACK = GRID == STREET_SIZES_GRID
# Size menus: the menus are the keys of
# STREET_SIZES; MENU_BOUNDS[menu] = the exclusive upper bounds of the menu - preflop the effective stack in BB at the
# hand start ("depth_bb"), postflop the SPR at the street start ("spr") - or None for the one menu without bounds (deep
# spots). A spot takes the bounded menu with the smallest bound above its value, else the unbounded one (menu_for).
# A new menu is a config entry - its sizes here, its bounds, its caps (TREE_CAPS) - and no code change.
MENU_BOUNDS = {"normal": None, "short": {"depth_bb": 40.0, "spr": 2.0}}
MENUS = tuple(STREET_SIZES)               # the index order of menu_index and rule_tables


def _cap(n: int, street: int, menu: str, sized: bool) -> int:
    """The raise cap of the teacher's tree for n players on `street` in `menu`: TREE_CAPS under the grid of STREET_SIZES
    (sized), else the caps of grids 5 and 7 (PREFLOP_TREES, POSTFLOP_CAP; the same in every menu). The only reader of
    the cap constants (cap_for is the accessor)."""
    if sized:
        return TREE_CAPS[n][street][menu]
    return PREFLOP_TREES[n]["cap"] if street == 0 else POSTFLOP_CAP[n]


for _menu_name, _streets in STREET_SIZES.items():
    if sorted(_streets) != [0, 1, 2, 3] or _menu_name not in MENU_BOUNDS:
        raise ValueError(f"menu {_menu_name!r}: sizes for streets 0-3 and an entry in MENU_BOUNDS are needed")
    for _street, _levels in _streets.items():
        _missing = {tuple(f) for lv in _levels for f in lv} - set(GRIDS[STREET_SIZES_GRID])
        if _missing:
            raise ValueError(f"STREET_SIZES[{_menu_name!r}][{_street}] uses {sorted(_missing)} outside grid {STREET_SIZES_GRID}")
if sorted(TREE_CAPS) != list(range(2, TREE_MAX_PLAYERS + 1)) or any(
        sorted(by_street) != [0, 1, 2, 3] or any(sorted(by_menu) != sorted(STREET_SIZES) or not all(
            isinstance(c, int) and c >= 1 for c in by_menu.values()) for by_menu in by_street.values())
        for by_street in TREE_CAPS.values()):
    raise ValueError(f"TREE_CAPS: a cap >= 1 for every players 2..{TREE_MAX_PLAYERS}, street 0-3 and menu {sorted(STREET_SIZES)}")
if sorted(MENU_BOUNDS) != sorted(MENUS) or [b for b in MENU_BOUNDS.values()].count(None) != 1 or any(
        b is not None and sorted(b) != ["depth_bb", "spr"] for b in MENU_BOUNDS.values()):
    raise ValueError(f"MENU_BOUNDS {MENU_BOUNDS}: one entry per menu {MENUS}, exactly one None, the others "
                     f"{{'depth_bb': ..., 'spr': ...}}")


def _menu(menu):
    """`menu`, or the unbounded menu (the deep spots') for None; ValueError for a name that is not a menu."""
    if menu is None:
        return next(m for m, b in MENU_BOUNDS.items() if b is None)
    if menu not in STREET_SIZES:
        raise ValueError(f"size menu must be one of {list(MENUS)}, got {menu!r}")
    return menu


def _menu_order(key):
    """(the bounds on `key` in increasing order, the menus in that order + the unbounded menu last)."""
    bounded = sorted((b[key], m) for m, b in MENU_BOUNDS.items() if b is not None)
    return [x for x, _ in bounded], [m for _, m in bounded] + [_menu(None)]


def menu_for(street: int, *, depth_bb=None, spr=None) -> str:
    """The size menu of a spot (MENU_BOUNDS): preflop by the effective stack in BB at the hand start, postflop by the
    SPR (effective stack behind / pot) at the street start. A function of the state, the same in every regime (the
    toggle sets players, stacks and rake, not the trees): regime 1's preflop (200 BB) always takes the unbounded menu,
    its postflop street starts below a bound take the bounded menu like anywhere else. menu_index is the traced twin."""
    value, key = (depth_bb, "depth_bb") if street == 0 else (spr, "spr")
    if value is None:
        raise ValueError(f"menu_for(street={street}) needs {'depth_bb' if street == 0 else 'spr'}")
    bounds, menus = _menu_order(key)
    return next((m for bound, m in zip(bounds, menus) if value < bound), menus[-1])


size_bucket = menu_for     # legacy alias for menu_for


def menu_index(street, stack_chips, pot_chips):
    """int32 index into MENUS of menu_for, traceable (jit, vmap): stack_chips = the effective stack in chips at
    the hand start (preflop) or behind at the street start (postflop), pot_chips = the pot at the street start. The
    comparisons are menu_for's on whole chips (depth_bb = stack / BB, spr = stack / pot): exact in float32 below
    2^23 chips for bounds that are multiples of 1/8 (checks/s7_3.py compares both on every bound +- one chip)."""
    stack = jnp.asarray(stack_chips, jnp.float32)
    scale = {"depth_bb": jnp.float32(BB), "spr": jnp.asarray(pot_chips, jnp.float32)}
    picks = {}
    for key in scale:
        bounds, menus = _menu_order(key)
        position = sum((stack >= bound * scale[key]).astype(jnp.int32) for bound in bounds) if bounds else jnp.int32(0)
        picks[key] = jnp.asarray([MENUS.index(m) for m in menus], jnp.int32)[position]
    return jnp.where(jnp.asarray(street) == 0, picks["depth_bb"], picks["spr"]).astype(jnp.int32)


def street_sizes(street: int, menu=None):
    """The bet-size fractions (num, den) per raise level of `street` in `menu` (STREET_SIZES; None: the unbounded
    menu)."""
    return STREET_SIZES[_menu(menu)][street]


def _size_indices(levels):
    """Grid action indices (2 + i for RAISE_FRACS[i]) of fraction levels; the fractions must be in the grid."""
    index = {tuple(f): 2 + i for i, f in enumerate(RAISE_FRACS)}
    missing = {tuple(f) for level in levels for f in level} - set(index)
    if missing:
        raise ValueError(f"sizes {sorted(missing)} are not in the grid {RAISE_FRACS} (POKER_GRID={GRID})")
    return tuple(tuple(index[tuple(f)] for f in level) for level in levels)


def tree_rule(n: int, street: int, menu=None) -> dict:
    """{cap, allin_after_cap, sizes, no_open_limp} of the teacher's tree for `n` players on `street`
    (0 = preflop) in size menu `menu` (None: the unbounded menu): sizes is None (every grid size at every level) or
    a tuple per raise level of action indices (2 + i for RAISE_FRACS[i]), the last level repeating. Under grid 13 the
    sizes come from STREET_SIZES[menu] (postflop for every n; preflop for two players); from 3 players the preflop
    keeps PREFLOP_TREES. The cap is cap_for's. Other grids: the trees of iteration 0 (every grid size postflop,
    cap 2 heads-up), whatever the menu."""
    if not isinstance(n, int) or not 2 <= n <= TREE_MAX_PLAYERS:
        raise ValueError(f"teacher trees exist for 2..{TREE_MAX_PLAYERS} players, got {n!r}")
    menu = _menu(menu)
    sized = GRID == STREET_SIZES_GRID
    cap = cap_for(n, street, menu)
    if street != 0:
        sizes = _size_indices(STREET_SIZES[menu][street]) if sized else None
        return {"cap": cap, "allin_after_cap": True, "sizes": sizes, "no_open_limp": False}
    if sized and n == 2:
        return {"cap": cap, "allin_after_cap": True, "sizes": _size_indices(STREET_SIZES[menu][0]), "no_open_limp": False}
    spec = PREFLOP_TREES[n]
    sizes = None if spec["sizes"] is None else _size_indices(spec["sizes"])
    return {"cap": cap, "allin_after_cap": True, "sizes": sizes, "no_open_limp": spec["no_open_limp"]}


def cap_for(n: int, street: int, menu=None) -> int:
    """The raise cap of the teacher's tree for n players on `street` in `menu` (None: the unbounded menu): the one
    accessor of the caps (tree_rule, the play mask poker.player.tree_mask, matches, LBR, self-play, the Slumbot adapter)."""
    return _cap(n, street, _menu(menu), GRID == STREET_SIZES_GRID)


def _rule_json(rule: dict) -> dict:
    return {"cap": rule["cap"], "allin_after_cap": rule["allin_after_cap"], "no_open_limp": rule["no_open_limp"],
            "sizes": None if rule["sizes"] is None else [list(level) for level in rule["sizes"]]}


def _rule_record(n: int, street: int, menu: str) -> dict:
    """_rule_json of tree_rule, or {"undefined_on_grid": GRID} where the rule's sizes are not in this process's grid
    (grid 5 has no 2/3 pot: the preflop trees of three and more players do not exist there)."""
    try:
        return _rule_json(tree_rule(n, street, menu))
    except ValueError:
        return {"undefined_on_grid": GRID}


def tree_rules() -> dict:
    """The FULL tree rule as a JSON-able dict (run layouts, the game record): the whole field - every table size
    2..TREE_MAX_PLAYERS, street and menu, not the regime's slice: the grid; every menu with its sizes (fractions per
    raise level and street; in effect under the grid of STREET_SIZES only, sizes_in_effect) and its bounds; the
    preflop trees of three and more players (PREFLOP_TREES); the caps (cap_for) per players, street and menu; and,
    under "trees", tree_rule's output itself, so a change of the function shows even with the constants unchanged."""
    players = range(2, TREE_MAX_PLAYERS + 1)
    return {"grid": GRID, "sizes_in_effect": GRID == STREET_SIZES_GRID, "max_players": TREE_MAX_PLAYERS,
            "menus": {m: {"bounds": MENU_BOUNDS[m],
                          "sizes": {str(st): [[list(f) for f in lv] for lv in levels] for st, levels in STREET_SIZES[m].items()}}
                      for m in MENUS},
            "preflop_trees": {str(n): {"cap": s["cap"], "no_open_limp": s["no_open_limp"],
                                       "sizes": None if s["sizes"] is None else [[list(f) for f in lv] for lv in s["sizes"]]}
                              for n, s in PREFLOP_TREES.items()},
            "caps": {str(n): {str(st): {m: cap_for(n, st, m) for m in MENUS} for st in range(4)} for n in players},
            "trees": {str(n): {str(st): {m: _rule_record(n, st, m) for m in MENUS} for st in range(4)}
                      for n in players}}


def rule_tables(n: int) -> dict:
    """tree_rule(n, street, menu) of every street and menu as numpy tables for jitted callers (poker.player.tree_mask):
    cap int32 [4, M]; allin_after_cap bool [4, M]; offered bool [4, M, L, NUM_ACTIONS] = the fraction raises the
    tree offers at raise level l (none from the cap on; L = the largest cap + 1, a level above it reads row L - 1);
    no_open_limp bool [4, M]. M = len(MENUS), in the order of MENUS (menu_index)."""
    rules = [[tree_rule(n, street, m) for m in MENUS] for street in range(4)]
    levels = max(r["cap"] for row in rules for r in row) + 1
    cap = np.zeros((4, len(MENUS)), np.int32)
    allin = np.zeros((4, len(MENUS)), bool)
    offered = np.zeros((4, len(MENUS), levels, NUM_ACTIONS), bool)
    no_limp = np.zeros((4, len(MENUS)), bool)
    for street, row in enumerate(rules):
        for b, r in enumerate(row):
            cap[street, b], allin[street, b], no_limp[street, b] = r["cap"], r["allin_after_cap"], r["no_open_limp"]
            for level in range(r["cap"]):
                sizes = range(2, ALLIN) if r["sizes"] is None else r["sizes"][min(level, len(r["sizes"]) - 1)]
                offered[street, b, level, list(sizes)] = True
    return {"cap": cap, "allin_after_cap": allin, "offered": offered, "no_open_limp": no_limp}


# The game record: which game an artifact was made in - the grid, the table regime
# and the tree rules (tree_rules(): sizes, caps, buckets, and tree_rule's output). Every artifact of the loop (layout,
# critics, stage files) and live matches carries game_record() and its game_hash(); strict loaders
# compare them (check_game) and stops on a difference: a critic's leaf values embed the next street's tree, so a
# tree mismatch is another game, invisible in the payout scheme (scheme(), check_scheme).
GAME_RECORD_VERSION = 1
# The engine's semantics of a tree with its own sizes (poker/envs/actions.py: duplicates within the offered sizes, sizes
# clamped to the minimum raise, the commit-merge rule) change what a sized tree (grid 13) offers at real spots without
# changing tree_rules(): its version is in the record under grid 13 (under grids 5 and 7 every size is offered and the
# engine's whole-grid semantics are the iteration-0 ones). Version 2: duplicates within the
# offered sizes (a size clamped to the minimum raise stays when its twin is not offered). Version 3: the
# commit-merge rule (COMMIT_MERGE_POT). Version 4: the effective stack (EFFECTIVE_STACK; unequal stacks only).
SIZED_TREE_ENGINE_VERSION = 4
# Artifacts from before the record (k0, every run up to 2026-10-11) were made on the iteration-0 heads-up trees:
# cap 2 + all-in after it, every size of the grid, no limp rule - tree_rule's output for two players under grids 5
# and 7.
ITERATION0_TREE = {"cap": 2, "allin_after_cap": True, "sizes": None, "no_open_limp": False}


def game_record() -> dict:
    record = {"version": GAME_RECORD_VERSION, "grid": GRID, "regime": regime_record(), "tree_rules": tree_rules()}
    if GRID == STREET_SIZES_GRID:
        record["sized_tree_engine"] = SIZED_TREE_ENGINE_VERSION
        record["commit_merge_pot"] = COMMIT_MERGE_POT
    return record


def game_hash(record=None) -> str:
    """sha256 of a game record (default this process's), canonical JSON."""
    record = game_record() if record is None else record
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def game_meta() -> dict:
    """What an artifact stores: {"game_hash", "game_record"}, and next to them the provenance (poker/envs/fingerprint.py): "code_fingerprint", printed and compared by the rental gate, never by check_game."""
    record = game_record()
    return {"game_hash": game_hash(record), "game_record": record, "code_fingerprint": fingerprint.code_fingerprint()}


def describe_game(record=None) -> str:
    """One line of a game record: grid, regime, the caps of the regime's table sizes by street and menu."""
    record = game_record() if record is None else record
    rules, regime = record.get("tree_rules", {}), record.get("regime", {})
    caps = {f"{n} players": {("preflop", "flop", "turn", "river")[int(s)]: by_m for s, by_m in
                             rules.get("caps", {}).get(str(n), {}).items()} for n in regime.get("players", [])}
    return (f"grid {record.get('grid')}, regime {regime.get('number')}, caps {caps}"
            + (", street sizes by menu" if rules.get("sizes_in_effect") else ", every grid size"))


def _differences(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            _differences(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}" if path else str(k), out)
    elif a != b:
        out.append(f"{path}: {json.dumps(a)[:80]} != {json.dumps(b)[:80]}")
    return out


def check_game(saved, what: str) -> None:
    """SystemExit unless `saved` (an artifact's {"game_hash", "game_record"}: a layout or critic's meta,
    a stage file's game_hash) is this process's game (game_hash). Without a record (made before it, 2026-10-11):
    accepted with a one-line note only where this process's heads-up trees are the iteration-0 trees those artifacts
    were made on (grids 5 and 7), refused under grid 13. A record that differs only in the regime follows
    check_regime: POKER_REGIME_FORCE=1 accepts it with a WARNING (a deliberate cross-regime evaluation)."""
    saved = saved if isinstance(saved, dict) else {}
    mine = game_record()
    if saved.get("game_hash") is None:
        legacy = GRID != STREET_SIZES_GRID and all(tree_rule(2, street, m) == ITERATION0_TREE
                                                   for street in range(4) for m in MENUS)
        if not legacy:
            raise SystemExit(f"{what}: no game record (made before the tree-rule record of 2026-10-11, on the iteration-0 "
                             f"heads-up trees: cap 2 + all-in, every grid size); this process plays {describe_game(mine)} "
                             f"(game {game_hash(mine)[:12]}): re-make it under this game")
        _note(f"{what}: no game record (made before 2026-10-11): taken as the iteration-0 heads-up trees of grid {GRID} "
              f"(cap 2 + all-in, every grid size), which are this process's tree rule")
        return
    if saved["game_hash"] == game_hash(mine):
        return
    record = saved.get("game_record")
    diff = _differences(record, mine) if isinstance(record, dict) else ["the record itself is missing"]
    if isinstance(record, dict) and record.get("regime") != mine["regime"] and REGIME_FORCE \
            and game_hash({**record, "regime": mine["regime"]}) == game_hash(mine):
        _note(f"WARNING: {what}: made under {describe_regime(record['regime'])}, the same trees; accepted for a "
              f"deliberate cross-regime evaluation (POKER_REGIME_FORCE=1)")
        return
    raise SystemExit(f"{what}: made in game {str(saved['game_hash'])[:12]}, this process plays game "
                     f"{game_hash(mine)[:12]} ({describe_game(mine)}); differences (saved != this process): "
                     + "; ".join(diff[:6]) + (f"; ... {len(diff) - 6} more" if len(diff) > 6 else "")
                     + ". Use the artifact's POKER_GRID / POKER_REGIME and code, or re-make it")


# PCFR+ iterations per postflop training solve. E is measured once at the end;
# spots above TEACHER_E_FLAG times their stage median are flagged. These budgets
# come from earlier narrower-domain measurements, so the new low/deep tails need
# E(T/2), E(T), time and memory calibration before a production run.
# The separate preflop cache builder stops on its explicit local error target.
TEACHER_T = {"river": 800, "turn": 1500, "flop": 1500}
TEACHER_E_FLAG = 2.0
# Broken-solve fuse on stage/batch medians, not a critic or full-game quality goal.
TEACHER_E_BOUND = {"river": 0.01, "turn": 0.01, "flop": 0.01}


# The play path's batch ladder (poker/player/live.py SolveServer, poker/solver/subgame.py solve_batch): the closed set of batch
# shapes compiled at the warm-up, per device. CPU: one at a time - the warm-up of 16 topologies x more steps with the
# critics does not fit the 1.5 GB a CPU process gets on the shared box.
PLAY_LADDER = {"gpu": (1, 2, 4, 8, 16), "cpu": (1,)}
PLAY_ITERATIONS = {0: 300, 1: 400, 2: 400, 3: 300}   # live play: PCFR iterations per street (flop/turn 400 after T5: median E <= 0.01 at T = 200, population-weighted 400-800)
# The persistent JAX compilation cache of the play path: a directory on disk (never tmpfs), one subfolder
# per backend and grid (poker/solver/subgame.py compilation_cache_dir checks it at the ready gate: filesystem type, free space).
JAX_CACHE_DIR = os.environ.get("POKER_JAX_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "poker-huy", "jax_cache"))


def regime_field() -> dict:
    """The field the play path must cover before the first hand (poker/solver/subgame.warm_up, poker/player/live.py): the
    regime's table sizes, every seat of each, the streets, the size menus (STREET_SIZES' keys) and the batch ladder
    per device. Regime 1 declares two players; regime 2 declares 2-6 (multiway live solves are not implemented)."""
    players = tuple(REGIMES[REGIME]["players"])
    return {"players": players, "seats": {n: tuple(range(n)) for n in players}, "streets": (0, 1, 2, 3),
            "menus": tuple(STREET_SIZES), "ladder": {k: tuple(v) for k, v in PLAY_LADDER.items()}}


def rake(level, pot, n_dealt, street_reached):
    """Rake in chips for one hand: rake level in [0, 1], contested pot, players dealt in, and the
    street index the hand reached (0 = preflop only)."""
    bps, cap = rake_terms(level, n_dealt)
    return rake_from_terms(bps, cap, pot, street_reached)


def validate_player_count(n: int) -> int:
    """Validate a static table size before allocating arrays or tracing JIT."""
    if not isinstance(n, int) or isinstance(n, bool) or not 2 <= n <= MAX_PLAYERS:
        raise ValueError(f"n must be a static Python int in 2..{MAX_PLAYERS}, got {n!r}")
    return n


def _positive_int(name, value, *, allow_zero=False):
    lower = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or not lower <= value <= _INT32_MAX:
        raise ValueError(f"{name} must be a Python int in {lower}..{_INT32_MAX}, got {value!r}")
    return value


def history_bound(n: int, *, start=None, bb=None, sb=None, raise_fracs=None,
                  num_actions=None, allin=None) -> int:
    """Sufficient decision bound for four streets and GRID (pot-fraction) raises; NOT for exact amounts.

    Pure host calculation; overrides validate hypothetical configurations without
    mutating module constants. Initial stacks must be <= start. No chip refills.
    From P=0, each fraction raise adds >= max(bb, floor(P * num / den)) chips for the
    smallest fraction num/den (the raiser pays the call plus that share of pot + call, and a
    full raise is never below the minimum raise >= bb); all-in raises
    occur at most n times. Between raises/street starts each seat acts once.
    All computed action targets, including masked ones, must fit signed int32.
    Exact-amount actions (step_target: the live path, the Slumbot replay) can min-raise by ~1 BB per decision, so a
    min-raise war needs up to n * start / BB decisions, more than this bound above ~850 BB heads-up (envs audit
    2026-10-12, F1: 1000 BB HU stalled at 832 of 832 slots; 500 BB holds at 2, 6 and 9 seats with a 7-10 % margin).
    Overflow is a silent no-op inside the jitted engine; the live path checks the capacity on the host before every
    step (poker/player/live.py check_legal) and raises instead.
    """
    validate_player_count(n)
    start = _positive_int("START", START if start is None else start)
    bb = _positive_int("BB", BB if bb is None else bb)
    sb = _positive_int("SB", SB if sb is None else sb, allow_zero=True)
    if sb > bb:
        raise ValueError("SB must not exceed BB")
    custom_fractions = raise_fracs is not None
    fractions = RAISE_FRACS if raise_fracs is None else raise_fracs
    if not isinstance(fractions, (tuple, list)):
        raise ValueError("RAISE_FRACS must contain integer numerator/denominator pairs")
    for pair in fractions:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("RAISE_FRACS must contain integer numerator/denominator pairs")
        _positive_int("raise numerator", pair[0])
        _positive_int("raise denominator", pair[1])
    if fractions:
        min_num, min_den = min(((a, b) for a, b in fractions), key=lambda f: Fraction(*f))

    expected_allin = 2 + len(fractions)
    effective_allin = (expected_allin if custom_fractions else ALLIN) if allin is None else allin
    effective_actions = (expected_allin + 1 if custom_fractions else NUM_ACTIONS) if num_actions is None else num_actions
    if (not isinstance(effective_allin, int) or isinstance(effective_allin, bool)
            or not isinstance(effective_actions, int) or isinstance(effective_actions, bool)
            or effective_allin != expected_allin or effective_actions != expected_allin + 1
            or FOLD != 0 or CHECK_CALL != 1):
        raise ValueError("action indices/count do not match FOLD, CHECK_CALL, fractions, ALLIN")

    total = n * start
    level_max = max(start, bb)  # Nominal preflop BB survives a short blind.
    pot_after_call = total + level_max
    if pot_after_call > _INT32_MAX or 2 * level_max > _INT32_MAX:
        raise ValueError("int32 range exceeded by chip sums or minimum raise target")
    for num, den in fractions:
        product = pot_after_call * num
        if product > _INT32_MAX or level_max + product // den > _INT32_MAX:
            raise ValueError("int32 range exceeded by fraction multiplication or raise target")

    p = count = 0
    if fractions:
        while True:
            candidate = p + max(bb, p * min_num // min_den)
            if candidate > total:
                break
            p = candidate
            count += 1
    return n * (4 + count + n)


def history_capacity(n: int, **kwargs) -> int:
    """Round the proven host bound up to a multiple of 64 decisions."""
    bound = history_bound(n, **kwargs)
    return ((bound + 63) // 64) * 64


MAX_HISTORY = history_capacity(N)  # Default only; each table uses history_capacity(n).
