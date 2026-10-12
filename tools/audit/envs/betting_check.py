"""Random-play differential test: poker/envs (init, action_table, step, step_target, get_rewards, rake_paid)
against tools/audit/envs/ref_engine.py on 2-6 player hands with unequal (non-round) stacks and rake.

At every decision: actor, street, stacks, street bets, fold legality, raise-open legality, min-raise and all-in
targets are compared; the action is drawn from the reference's legal set (fold / call / min-raise / all-in /
uniform raise in [min, all-in) / an engine grid size), applied to the engine through step (grid sizes) or
step_target (exact chips, accepted by poker.envs.guards.target) and to the reference. At the end net profits
and rake are compared. Usage: betting_check.py HANDS SEED OUT.json
"""
import json, random, sys, time
import numpy as np
import jax, jax.numpy as jnp

from poker.envs import config, derived, guards
from poker.envs.betting import init
from poker.envs.actions import action_table
from poker.envs.transition import step, step_target
from poker.envs.evaluator import get_rewards, rake_paid
sys.path.insert(0, "tools/audit/envs")
from ref_engine import Ref
from eval_check import rank7

MAXSTACK = 60000
CALLP = 0.55

@jax.jit
def info(s):
    m, t = action_table(s)
    return m, t, derived.min_raise_to(s), derived.bet_level(s)

jstep = jax.jit(step)
jstep_t = jax.jit(step_target)
jrew = jax.jit(lambda s: (get_rewards(s), rake_paid(s)))

def run(hands, seed):
    rng = random.Random(seed)
    stats = {"hands": 0, "decisions": 0, "showdowns": 0, "side_pot_hands": 0, "short_allin_raises": 0,
             "allin_for_less_calls": 0, "grid_actions": 0, "exact_actions": 0, "raked_hands": 0, "by_n": {},
             "mismatches": []}
    for h in range(hands):
        n = rng.choice([2, 3, 3, 4, 5, 6])
        active = [True] * n
        if n >= 4 and rng.random() < 0.3:       # an empty seat (not the button)
            active[rng.randrange(1, n)] = False
            if sum(active) < 2: active = [True] * n
        stacks = []
        for i in range(n):
            r = rng.random()
            stacks.append(rng.randint(101, 1500) if r < 0.35 else rng.randint(1501, 8000) if r < 0.7 else rng.randint(8001, MAXSTACK))
        stacks = [s if a else 0 for s, a in zip(stacks, active)]
        rake_level = rng.choice([0.0, 0.0, 1/3, 2/3, 1.0, rng.random()])
        key = jax.random.PRNGKey(seed * 100003 + h)
        s = init(key, n, jnp.asarray(stacks, jnp.int32), jnp.asarray(active), True, MAXSTACK, rake_level)
        ref = Ref(stacks, active)
        bad = None
        side = False
        while True:
            if bool(s.terminal) != ref.terminal:
                bad = ("terminal", bool(s.terminal), ref.terminal); break
            if ref.terminal:
                break
            stats["decisions"] += 1
            st = [int(x) for x in np.asarray(s.stack)]; bt = [int(x) for x in np.asarray(s.bet_street)]
            if int(s.to_act) != ref.to_act or int(s.street) != ref.street or st != ref.stack or bt != ref.bet:
                bad = ("state", int(s.to_act), ref.to_act, int(s.street), ref.street, st, ref.stack, bt, ref.bet); break
            m, t, mn, lvl = (np.asarray(x) for x in info(s))
            L = ref.legal()
            if bool(m[config.FOLD]) != L["fold"] or bool(m[config.ALLIN]) != L["raise"] or int(t[config.CHECK_CALL]) != L["call_to"]:
                bad = ("legal", m.tolist(), t.tolist(), L); break
            if L["raise"] and (int(mn) != L["min_to"] or int(t[config.ALLIN]) != L["allin_to"]):
                bad = ("raise_bounds", int(mn), L); break
            # grid targets offered must be legal full raises strictly below all-in
            for a in range(2, config.ALLIN):
                if m[a] and not (L["raise"] and L["min_to"] <= t[a] < L["allin_to"]):
                    bad = ("grid_target", a, int(t[a]), L); break
            if bad: break
            # choose an action
            r = rng.random()
            me = ref.to_act
            if L["fold"] and r < 0.15:
                kind, to = "fold", ref.bet[me]
            elif not L["raise"] or r < CALLP:
                kind, to = "call", L["call_to"]
                if L["call_to"] - ref.bet[me] < ref.to_call(): stats["allin_for_less_calls"] += 1
            else:
                kind = "raise"
                lo, hi = L["min_to"], L["allin_to"]
                rr = rng.random()
                grid = [a for a in range(2, config.ALLIN) if m[a]]
                if hi <= lo or rr < 0.25: to = hi
                elif rr < 0.45: to = lo
                elif rr < 0.7 and grid: to = None
                else: to = rng.randint(lo, hi - 1)
            if kind == "raise" and to is None:
                a = rng.choice(grid); to = int(t[a])
                s = jstep(s, jnp.int32(a)); stats["grid_actions"] += 1
            else:
                a = {"fold": config.FOLD, "call": config.CHECK_CALL}.get(kind, config.ALLIN if to == L["allin_to"] else 2)
                guards.target(s, a, to)
                s = jstep_t(s, jnp.int32(a), jnp.int32(to)); stats["exact_actions"] += 1
            if kind == "raise" and to - ref.level < ref.last_full:
                stats["short_allin_raises"] += 1
            ref.act(kind, to)
        stats["hands"] += 1
        stats["by_n"][n] = stats["by_n"].get(n, 0) + 1
        if bad is None:
            hole = np.asarray(s.hole); board = np.asarray(s.board)
            keys = [rank7(list(hole[i]) + list(board)) for i in range(n)]
            bps, cap = int(s.rake_bps), int(s.rake_cap)
            prof_ref, rake_ref = ref.settle(keys, bps, cap)
            rew, rk = jrew(s)
            rew = [int(x) for x in np.asarray(rew)]
            if sum(not ref.folded[j] for j in range(n)) > 1: stats["showdowns"] += 1
            if len(set(ref.total[j] for j in range(n) if not ref.folded[j])) > 1: stats["side_pot_hands"] += 1
            if rake_ref: stats["raked_hands"] += 1
            if rew != prof_ref or int(rk) != rake_ref or sum(rew) != -int(rk):
                bad = ("settle", rew, prof_ref, int(rk), rake_ref, ref.total, ref.folded, n, stacks)
        if bad is not None:
            stats["mismatches"].append({"hand": h, "n": n, "stacks": stacks, "rake_level": rake_level, "detail": repr(bad)[:600]})
            if len(stats["mismatches"]) >= 20: break
    return stats

if __name__ == "__main__":
    hands, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    if len(sys.argv) > 4: CALLP = float(sys.argv[4])
    t0 = time.time()
    st = run(hands, seed)
    st["seconds"] = round(time.time() - t0, 1)
    st["mismatch_count"] = len(st["mismatches"])
    print(json.dumps({k: v for k, v in st.items() if k != "mismatches"}, indent=1))
    for mm in st["mismatches"][:5]: print(mm)
    json.dump(st, open(out, "w"), indent=1)
