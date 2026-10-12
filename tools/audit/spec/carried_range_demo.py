"""What the live player carries between decisions, and what it costs when the carried opponent range is wrong.

River, HU, pot 20 BB, 100 BB behind (SPR 5, 'normal' menu). Hero (seat 1, first to act) holds only
bluff-catchers; the opponent's TRUE river range is value (top 10% of hands on the board) plus air (bottom 25%),
half the mass each. The live player (poker.player.live.LivePlayer, real tree with SLOT, real PCFR+ at the
deployed T) is started twice at the river with two different opponent beliefs:
  belief A: value + air at 1e-3 relative weight (what the likelihood floor eps=1e-3 leaves of hands that the
            earlier solves said never reach the river this way);
  belief B: the true range.
Hero checks (his own frozen action), the opponent bets POT (off-menu: river sizes are 1/3, 3/4, 3/2 pot -> SLOT),
and we read the hero's response. The opponent's air then gains EV(bet) - EV(check) = pot*(2 f - 1) per air hand,
f = hero's range-weighted fold(+raise) frequency. A ceiling gadget (spec 2.4) would cap air at its value under the
previous plan; the live player has no such state (module doc item 10).
    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python3 tools/audit/spec/carried_range_demo.py
"""
import json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from poker.envs import config
from poker.solver import tree as TR, subgame as SG
from poker.solver import cards as C
from poker.player import live as LV

OUT = Path("results/audit/spec/carried_range_demo.json")
H = C.NUM_HANDS
board = np.array([12, 13 + 11, 26 + 5, 39 + 2, 0 + 0], np.int32)     # As Kh 7d 4c 2s (suit*13+rank) - rainbow, dry
ranks = np.asarray(C.hand_ranks(jnp.asarray(board)))
valid = ranks >= 0
order = np.argsort(np.where(valid, ranks, -10**9))
vh = order[~np.isin(order, np.flatnonzero(~valid))]                 # valid hands, weakest first
nv = len(vh)
air = vh[: int(0.25 * nv)]
catch = vh[int(0.40 * nv): int(0.75 * nv)]
value = vh[int(0.90 * nv):]

def rng(parts):
    r = np.zeros(H)
    for idx, mass in parts:
        r[idx] += mass / len(idx)
    return r / r.sum()

pot, behind = 2000, 10000
root = TR.street_root(2, 3, pot, behind)
root = root.replace(board=jnp.asarray(board), hole=jnp.full((2, 2), -1, jnp.int32))
hero, villain = 1, 0
assert int(root.to_act) == hero, int(root.to_act)
true_opp = rng([(value, 0.5), (air, 0.5)])
beliefs = {"A_floor_survivor_air": rng([(value, 1.0), (air, 1e-3 * len(air) / len(value))]),
           "B_true_range": true_opp}
hero_r = rng([(catch, 1.0)])
solver = LV.Solver(iterations={3: config.PLAY_ITERATIONS[3]})
out = {"board": board.tolist(), "pot": pot, "behind": behind, "T": config.PLAY_ITERATIONS[3],
       "hands": {"value": len(value), "air": len(air), "bluff_catchers": len(catch)}, "cases": {}}
for name, belief in beliefs.items():
    p = LV.LivePlayer(hero, solver=solver, prefetch_equity=False)
    ranges = np.zeros((2, H)); ranges[hero] = hero_r; ranges[villain] = belief
    t0 = time.perf_counter()
    p.new_hand(root, ranges=ranges)
    c0 = p.candidates(root)
    k_chk = c0.column(LV.CALL, 0)
    check_freq = float((p.ranges[hero] * c0.probs[:, k_chk]).sum())
    p.choose(root, LV.CALL, 0)
    s1 = LV.engine_step(root, LV.CALL, 0)
    p.observe(root, LV.CALL, 0, s1)
    bet = pot                                                           # pot-size bet: off the menu
    s2 = LV.engine_step(s1, LV.RAISE, bet)
    p.observe(s1, LV.RAISE, bet, s2)
    c = p.candidates(s2)
    reach = p.ranges[hero] * c0.probs[:, k_chk]                         # hero's range at the node (after his check)
    reach = reach / reach.sum()
    kinds = [int(k) for k in c.kinds]
    call_cols = [k for k in range(len(kinds)) if kinds[k] == LV.CALL]
    call = c.probs[:, call_cols].sum(1)
    # card removal: weight hero hands by compatibility with the air hands
    comp = np.array([np.mean([not (set(C.HANDS[h]) & set(C.HANDS[a])) for a in air]) for h in range(H)])
    w = reach * comp; w /= w.sum()
    f = float((w * (1 - call)).sum())
    out["cases"][name] = {
        "hero_check_freq_first": check_freq,
        "opp_belief_mass_on_air_at_river_start": float(belief[air].sum()),
        "opp_belief_mass_on_air_after_bet_update": float(p.ranges[villain][air].sum()),
        "hero_response_columns": [[LV.SG.KIND_NAMES.get(k, k), int(t)] for k, t in zip(c.kinds, c.targets)],
        "hero_call_freq_vs_pot_bet": float((w * call).sum()),
        "hero_fold_or_raise_freq": f,
        "mdf_pot_bet": 0.5,
        "air_gain_per_air_hand_pot_units": 2 * f - 1,
        "opp_true_range_gain_from_air_bets_pot_units": 0.5 * (2 * f - 1),
        "stats": {k: (v if not isinstance(v, (set,)) else sorted(v)) for k, v in p.stats.items()
                  if k in ("off_tree", "floor_hits", "opponent_updates", "solves", "slot_overflow", "translated")},
        "solve_seconds": [round(x, 2) for x in p.stats["solve_seconds"]],
        "wall_s": round(time.perf_counter() - t0, 1),
        "player_has_ceiling_state": any("ceil" in a.lower() for a in vars(p)),
        "player_state_attributes": sorted(a for a in vars(p) if not a.startswith("_")),
    }
    print(name, json.dumps(out["cases"][name], default=str)[:900], flush=True)
OUT.write_text(json.dumps(out, indent=1, default=str))
