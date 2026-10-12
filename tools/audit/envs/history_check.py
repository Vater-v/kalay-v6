"""History capacity vs the longest legal hands: adversarial min-raise wars through step_target (exact chips, the live and
Slumbot path) and through step with the grid's smallest size (1/100 pot, clamped to the min raise).
Policy 'close-raise': call unless the call would close the action, then min-raise (n-1 decisions per BB of level);
'always-raise': min-raise whenever open. Reports decisions vs config.history_bound / the allocated capacity and what
the engine does when the capacity is exceeded (checkify off: the step is a silent no-op). Usage: history_check.py OUT.json"""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp
from poker.envs import config, derived
from poker.envs.betting import init
from poker.envs.actions import action_table
from poker.envs.transition import step_target, step

@jax.jit
def info(s):
    m, t = action_table(s)
    return m, t, derived.min_raise_to(s)
jst = jax.jit(step_target); jstep = jax.jit(step)

def play(n, bb_depth, policy, grid=False, cap_steps=20000):
    stack = bb_depth * config.BB
    s = init(jax.random.PRNGKey(1), n, jnp.full(n, stack, jnp.int32), None, True, max(stack, config.BB + 1))
    capacity = s.history_player.shape[0]
    d = 0; stalled = False
    while not bool(s.terminal) and d < cap_steps:
        m, t, mn = (np.asarray(x) for x in info(s))
        call = (config.CHECK_CALL, int(t[config.CHECK_CALL]))
        raise_ok = bool(m[config.ALLIN])
        choice = call
        if raise_ok and mn < int(t[config.ALLIN]):
            if policy == "always-raise":
                choice = ("grid", 2) if grid else (2, int(mn))
            else:
                after = jst(s, jnp.int32(call[0]), jnp.int32(call[1]))
                closes = bool(after.terminal) or int(after.street) != int(s.street)
                if closes: choice = ("grid", 2) if grid else (2, int(mn))
        prev_len = int(s.history_len)
        if choice[0] == "grid":
            if not m[2]:
                choice = call
                s2 = jst(s, jnp.int32(call[0]), jnp.int32(call[1]))
            else:
                s2 = jstep(s, jnp.int32(2))
        else:
            s2 = jst(s, jnp.int32(choice[0]), jnp.int32(choice[1]))
        if int(s2.history_len) == prev_len:
            stalled = True; break
        s = s2; d += 1
    return {"n": n, "depth_bb": bb_depth, "policy": policy, "grid_1pct": grid, "decisions": d,
            "history_bound": config.history_bound(n, start=max(stack, config.START)), "capacity": capacity,
            "terminal": bool(s.terminal), "stalled_at_capacity": stalled}

out = []
t0 = time.time()
for n, depth, pol, grid in [(2, 200, "always-raise", False), (2, 500, "always-raise", False), (2, 500, "always-raise", True),
                            (6, 500, "close-raise", False), (9, 500, "close-raise", False), (2, 1000, "always-raise", False),
                            (2, 2000, "always-raise", False)]:
    r = play(n, depth, pol, grid); r["seconds"] = round(time.time() - t0, 1); out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
