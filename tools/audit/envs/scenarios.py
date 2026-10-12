"""Hand-written rule scenarios for poker/envs; each prints engine values next to the rule's expected values.
Usage: scenarios.py OUT.json"""
import json, sys
import numpy as np
import jax, jax.numpy as jnp
from poker.envs import config, derived, guards
from poker.envs.betting import init, init_betting
from poker.envs.actions import action_table
from poker.envs.transition import step_target, step
from poker.envs.evaluator import get_rewards, rake_paid, compute_payouts

jst = jax.jit(step_target)
def tbl(s):
    m, t = action_table(s)
    return np.asarray(m), np.asarray(t), int(derived.min_raise_to(s))
def act(s, a, to):
    guards.target(s, a, to)
    return jst(s, jnp.int32(a), jnp.int32(to))
R = {}
def rec(name, got, exp):
    R[name] = {"engine": got, "expected": exp, "ok": got == exp}

key = jax.random.PRNGKey(0)
# HU blinds
s = init(key, 2, jnp.asarray([20000, 20000], jnp.int32))
rec("hu_blinds_button_posts_sb", {"bets": np.asarray(s.bet_street).tolist(), "to_act": int(s.to_act)}, {"bets": [50, 100], "to_act": 0})
s = act(s, 1, 100); s = act(s, 1, 100)       # SB completes, BB checks
rec("hu_postflop_first_actor_is_bb", {"street": int(s.street), "to_act": int(s.to_act)}, {"street": 1, "to_act": 1})
# 3-handed: SB=1, BB=2, first preflop = button 0
s = init(key, 3, jnp.asarray([20000] * 3, jnp.int32))
rec("3h_blinds", {"bets": np.asarray(s.bet_street).tolist(), "to_act": int(s.to_act)}, {"bets": [0, 50, 100], "to_act": 0})
# 5 players, everyone limps, BB checks -> flop first actor seat 1 (SB); if SB folded preflop -> seat 2
s = init(key, 5, jnp.asarray([20000] * 5, jnp.int32))
for _ in range(3): s = act(s, 1, 100)          # seats 3,4,0 call
s = act(s, 0, 50)                              # SB folds
s = act(s, 1, 100)                             # BB checks
rec("5h_postflop_first_actor_after_sb_folded", {"street": int(s.street), "to_act": int(s.to_act)}, {"street": 1, "to_act": 2})
# min-raise after a short all-in (3 players, preflop): 0 raises to 300 (+200), 1 shoves 400 (short +100), 2 min = 600
s = init(key, 3, jnp.asarray([20000, 450, 20000], jnp.int32))
s = act(s, 2, 300)
s = act(s, config.ALLIN, 450)                  # seat 1: 450 total (+150 < 200: short)
m, t, mn = tbl(s)
rec("min_raise_after_short_allin", {"to_act": int(s.to_act), "min_raise_to": mn, "last_raise": int(s.last_raise)}, {"to_act": 2, "min_raise_to": 650, "last_raise": 200})
s = act(s, 1, 450)                              # seat 2 calls
m, t, mn = tbl(s)
rec("short_allin_does_not_reopen_for_original_raiser", {"to_act": int(s.to_act), "raise_open": bool(m[config.ALLIN]), "call_to": int(t[1])}, {"to_act": 0, "raise_open": False, "call_to": 450})
# cumulative short all-ins reopen: 4 players: 0 opens 300 (+200), 1 shoves 450 (+150), 2 shoves 560 (+110) -> 0 faces 260 >= 200
s = init(key, 4, jnp.asarray([20000, 450, 560, 20000], jnp.int32))
s = act(s, 2, 300)            # seat 3 (UTG) raises to 300
s = act(s, 1, 300)            # seat 0 calls
s = act(s, config.ALLIN, 450) # seat 1 short shove
s = act(s, config.ALLIN, 560) # seat 2 short shove
m, t, mn = tbl(s)
rec("cumulative_short_allins_reopen", {"to_act": int(s.to_act), "raise_open": bool(m[config.ALLIN]), "min_raise_to": mn}, {"to_act": 3, "raise_open": True, "min_raise_to": 760})
# all-in for less than a call, HU, and uncalled bet return
s = init(jax.random.PRNGKey(3), 2, jnp.asarray([20000, 250], jnp.int32))
s = act(s, 2, 1000)                             # button raises to 1000
s = act(s, 1, 250)                              # BB calls all-in for 250
r = np.asarray(get_rewards(s)).tolist()
rec("allin_for_less_terminal_runout", {"terminal": bool(s.terminal), "street": int(s.street), "abs_profit": sorted(abs(x) for x in r), "sum": sum(r)}, {"terminal": True, "street": 3, "abs_profit": [250, 250], "sum": 0})
# no flop no drop vs preflop all-in raked (rake level 1.0 = NL5: 500 bps, HU cap 2.6 BB = 260 chips)
s = init(jax.random.PRNGKey(4), 2, jnp.asarray([20000, 20000], jnp.int32), rake_level=1.0)
s1 = act(s, 2, 300); s1 = act(s1, 0, 100)
rec("rake_preflop_fold_no_drop", int(rake_paid(s1)), 0)
s2 = act(s, config.ALLIN, 20000); s2 = act(s2, 1, 20000)
rec("rake_preflop_allin_raked_cap", {"rake": int(rake_paid(s2)), "sum_rewards": int(np.asarray(get_rewards(s2)).sum())}, {"rake": 260, "sum_rewards": -260})
s3 = act(s, 1, 100); s3 = act(s3, 1, 100); s3 = act(s3, 2, 300); s3 = act(s3, 0, 0)   # limp, check, flop bet 300 (uncalled), fold
rec("rake_flop_fold_uncalled_excluded", int(rake_paid(s3)), 200 * 500 // 10000)
# rake terms per anchor and n dealt (expected from the config comment table, in chips)
terms = {}
for lv, exp in ((0.0, (0, 0, 0, 0)), (1/3, (500, 200, 480, 800)), (2/3, (500, 250, 600, 1000)), (1.0, (500, 260, 600, 1000))):
    got = [int(config.rake_terms(lv, 2)[0])] + [int(config.rake_terms(lv, k)[1]) for k in (2, 3, 5)]
    terms[str(round(lv, 3))] = {"engine": got, "expected": list(exp), "ok": got == list(exp)}
mid = [int(x) for x in config.rake_terms(1/6, 2)]
terms["1/6 (interpolated)"] = {"engine": mid, "expected": [250, 100], "ok": mid == [250, 100]}
R["rake_terms"] = terms
# int32: rake_split share = total * base_k // base overflows when total * base_k > 2^31 - 1 (accepted deep table)
n = 3; deep = 3_000_000
guards.table(n, np.asarray([20000, deep, deep]), None, None)          # the host guard accepts these stacks
bets = jnp.asarray([1000, deep, deep], jnp.int32)                      # seat 0 all-in small (main pot), 1 and 2 deep (side pot)
pay = np.asarray(compute_payouts(bets, jnp.ones(n, bool), jnp.asarray([9, 5, 1], jnp.int32), 500, 1000, 3)).tolist()
base = 3000 + 2 * (deep - 1000); rake = 1000; main_share = rake * 3000 // base
rec("rake_split_int32_overflow_deep_sidepot", {"payouts": pay}, {"payouts": [3000 - main_share, 2 * (deep - 1000) - (rake - main_share), 0]})
json.dump(R, open(sys.argv[1], "w"), indent=1)
for k, v in R.items():
    print(k, json.dumps(v))
