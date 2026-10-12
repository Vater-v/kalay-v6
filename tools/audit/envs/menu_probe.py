"""What the grid-13 size menus (config.STREET_SIZES / TREE_CAPS / MENU_BOUNDS) and the commit-merge rule
(config.COMMIT_MERGE_POT, poker/envs/actions.py) offer at the regime-1 (HU 200 BB) spots, line by line, and which
sizes the merge turns into the all-in. Also: the merge acts on whichever seat is to act (so the opponent's modelled
tree options too), while an observed exact amount (step_target) is applied unchanged. Usage: menu_probe.py OUT.json"""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
from fractions import Fraction
from poker.envs import config, derived, guards
from poker.envs.betting import init
from poker.envs.actions import action_candidates
from poker.envs.transition import step_target

RT = config.rule_tables(2)
FR = {2 + i: f"{a}/{b}" for i, (a, b) in enumerate(config.RAISE_FRACS)}
jst = jax.jit(step_target)

def menu_of(s, start_pot, start_stack):
    return int(config.menu_index(int(s.street), start_stack, start_pot))

def offered_row(s, level, menu):
    street = int(s.street)
    L = min(level, RT["offered"].shape[2] - 1)
    sizes = RT["offered"][street, menu, L].copy()
    cap = int(RT["cap"][street, menu])
    if level >= cap:
        sizes[:] = False
    mask_all, tgt_all = action_candidates(s, jnp.asarray(sizes))
    # the same sizes without the merge: raw targets
    me = int(s.to_act); lvl = int(derived.bet_level(s)); tc = int(derived.to_call(s)); pot = int(derived.pot(s))
    mn = int(derived.min_raise_to(s)); mx = int(derived.max_raise_to(s))
    rows = []
    for a in np.flatnonzero(sizes[2:config.ALLIN]) + 2:
        num, den = config.RAISE_FRACS[a - 2]
        raw = max(lvl + (pot + tc) * num // den, mn)
        rows.append({"size": FR[int(a)], "raw_target": raw, "engine_target": int(tgt_all[a]), "offered": bool(mask_all[a]),
                     "behind_after": mx - raw, "pot_after": pot + raw - int(s.bet_street[me]),
                     "why_masked": None if mask_all[a] else ("merged into all-in" if raw < mx and int(tgt_all[a]) == mx else
                                                             "at/above stack" if raw >= mx else "duplicate/closed")})
    return {"street": street, "menu": config.MENUS[menu], "level": level, "cap": cap, "pot": pot, "to_call": tc,
            "stack_behind": int(s.stack[me]), "allin_offered": bool(mask_all[config.ALLIN]), "allin_target": int(tgt_all[config.ALLIN]),
            "sizes": rows}

def flop_spot(preflop_total):
    s = init(jax.random.PRNGKey(0), 2, jnp.asarray([20000, 20000], jnp.int32))
    if preflop_total == 100:
        s = jst(s, jnp.int32(1), jnp.int32(100)); s = jst(s, jnp.int32(1), jnp.int32(100))
    else:
        s = jst(s, jnp.int32(2), jnp.int32(preflop_total)); s = jst(s, jnp.int32(1), jnp.int32(preflop_total))
    return s

out = {"commit_merge": config.COMMIT_MERGE, "menu_bounds": config.MENU_BOUNDS, "preflop": [], "flop_lines": []}
# preflop normal menu, raise levels 0..3
s = init(jax.random.PRNGKey(0), 2, jnp.asarray([20000, 20000], jnp.int32))
menu = menu_of(s, 150, 20000)
for level in range(4):
    row = offered_row(s, level, menu); out["preflop"].append(row)
    nxt = [r for r in row["sizes"] if r["offered"]]
    if not nxt: break
    s = jst(s, jnp.int32(2), jnp.int32(nxt[-1]["engine_target"]))
# flop lines: walk the largest offered size at each level (bet, raise, re-raise), and the smallest
for pf in (100, 200, 300, 900, 2100):
    for pick in ("largest", "smallest"):
        s = flop_spot(pf)
        pot0, stack0 = int(derived.pot(s)), int(s.stack[1])
        menu = menu_of(s, pot0, stack0)
        line = {"preflop_to": pf, "flop_pot": pot0, "spr": round(stack0 / pot0, 2), "pick": pick, "levels": []}
        for level in range(4):
            row = offered_row(s, level, menu); line["levels"].append(row)
            nxt = [r for r in row["sizes"] if r["offered"]]
            if not nxt: break
            r = nxt[-1] if pick == "largest" else nxt[0]
            s = jst(s, jnp.int32(2), jnp.int32(r["engine_target"]))
            if bool(s.terminal): break
        out["flop_lines"].append(line)
# sweep: at a flop bet decision (level 0, normal menu sizes 1/3, 3/4), for which SPR does the merge fire?
sweep = []
for spr10 in range(10, 60, 2):
    stack = 20000; pot = int(round(stack / (spr10 / 10)))
    # synthetic HU flop state: pot split evenly, stacks equal
    s = flop_spot(100)
    s = s.replace(stack=jnp.asarray([stack, stack], jnp.int32), bet_total=jnp.asarray([pot // 2, pot - pot // 2], jnp.int32))
    sizes = np.zeros(config.NUM_ACTIONS, bool); sizes[2:config.ALLIN] = True
    m, t = action_candidates(s, jnp.asarray(sizes))
    merged = [FR[a] for a in range(2, config.ALLIN) if not m[a] and int(t[a]) == stack and
              max(int(derived.bet_level(s)) + (pot) * config.RAISE_FRACS[a - 2][0] // config.RAISE_FRACS[a - 2][1], 100) < stack]
    sweep.append({"spr": spr10 / 10, "pot": pot, "merged_grid_sizes": merged})
out["flop_bet_merge_sweep"] = sweep
# opponent seat: the merge also rewrites the opponent's grid options; an exact observed amount is applied as is
s = flop_spot(2100)                    # pot 4200, stacks 17900, seat 1 (BB) to act first postflop
pot = int(derived.pot(s))
s2 = jst(s, jnp.int32(1), jnp.int32(0))  # BB checks; button (seat 0) to act = "the opponent" from BB's view
sizes = np.zeros(config.NUM_ACTIONS, bool); sizes[2:config.ALLIN] = True
m, t = action_candidates(s2, jnp.asarray(sizes))
opp = {"to_act": int(s2.to_act), "grid_targets": {FR[a]: (int(t[a]), bool(m[a])) for a in range(2, config.ALLIN)}}
exact = 16500                           # a bet leaving 1400 behind (1400 <= 0.25 * (4200 + 16500)): inside the merge zone
guards.target(s2, 2, exact)
s3 = jst(s2, jnp.int32(2), jnp.int32(exact))
opp["observed_exact_bet"] = exact
opp["after_step_target_bet_street"] = np.asarray(s3.bet_street).tolist()
opp["after_step_target_stack"] = np.asarray(s3.stack).tolist()
out["opponent_side"] = opp
json.dump(out, open(sys.argv[1], "w"), indent=1)
for row in out["preflop"]: print("PF", row["level"], row["pot"], [(r["size"], r["engine_target"], r["offered"], r["why_masked"]) for r in row["sizes"]], "allin", row["allin_offered"])
for line in out["flop_lines"]:
    print("FLOP pf", line["preflop_to"], "pot", line["flop_pot"], "spr", line["spr"], line["pick"])
    for row in line["levels"]:
        print("   ", row["menu"], "lvl", row["level"], "pot", row["pot"], "call", row["to_call"], "behind", row["stack_behind"],
              [(r["size"], r["raw_target"], r["offered"], r["why_masked"]) for r in row["sizes"]], "allin", row["allin_offered"])
print("sweep", [(x["spr"], x["merged_grid_sizes"]) for x in sweep])
print("opp", json.dumps(opp))
