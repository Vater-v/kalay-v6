"""CPU timings of the jitted engine: step, step_target, action_table, get_rewards, evaluate_hand; single call
(dispatch-bound) and vmap-batched. Single-threaded XLA (set XLA_FLAGS) on a shared 4-core box. Usage: perf.py OUT.json"""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp
from poker.envs import config
from poker.envs.betting import init
from poker.envs.actions import action_table
from poker.envs.transition import step, step_target
from poker.envs.evaluator import get_rewards, evaluate_hand

def timeit(fn, *args, reps=None, budget=2.0):
    out = fn(*args); jax.block_until_ready(out)          # compile
    t0 = time.perf_counter(); k = 0
    while True:
        out = fn(*args); jax.block_until_ready(out); k += 1
        if time.perf_counter() - t0 > budget or (reps and k >= reps): break
    return (time.perf_counter() - t0) / k

res = []
for n, hist in ((2, True), (2, False), (6, True), (6, False)):
    s = init(jax.random.PRNGKey(0), n, None, None, hist)
    a = jnp.int32(config.CHECK_CALL)
    cases = {"step": (jax.jit(step), (s, a)), "step_target": (jax.jit(step_target), (s, a, jnp.int32(100))),
             "action_table": (jax.jit(action_table), (s,))}
    # terminal state for rewards: everybody all-in preflop
    t = s
    for _ in range(n):
        m, tg = action_table(t)
        t = step(t, jnp.int32(config.ALLIN) if bool(m[config.ALLIN]) else jnp.int32(config.CHECK_CALL))
    assert bool(t.terminal)
    cases["get_rewards"] = (jax.jit(get_rewards), (t,))
    for name, (fn, args) in cases.items():
        single = timeit(fn, *args)
        row = {"fn": name, "n": n, "history": hist, "history_capacity": int(s.history_player.shape[0]), "single_us": round(single * 1e6, 1)}
        for B in (1024, 8192):
            bargs = jax.tree.map(lambda x: jnp.broadcast_to(x, (B,) + jnp.shape(x)), args)
            bfn = jax.jit(jax.vmap(fn.__wrapped__ if hasattr(fn, "__wrapped__") else fn))
            tb = timeit(bfn, *bargs)
            row[f"batch{B}_us_per_item"] = round(tb / B * 1e6, 3)
        res.append(row); print(json.dumps(row), flush=True)
cards = jnp.asarray(np.random.default_rng(0).permuted(np.tile(np.arange(52), (8192, 1)), axis=1)[:, :7], jnp.int32)
ev = jax.jit(jax.vmap(evaluate_hand))
tb = timeit(ev, cards)
row = {"fn": "evaluate_hand", "batch8192_us_per_item": round(tb / 8192 * 1e6, 3), "hands_per_s": round(8192 / tb)}
res.append(row); print(json.dumps(row))
json.dump(res, open(sys.argv[1], "w"), indent=1)
