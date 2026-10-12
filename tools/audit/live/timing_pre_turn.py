"""Tree sizes and warm per-iteration solve time of the HU preflop (hand start, 200 BB) and turn (b200c/kk) live
trees at batch 1 on this CPU, T = 100 (extrapolated linearly to the play T: per-iteration cost is constant).
Random critic of the training width and random all-in matrix (timing only).
    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/timing_pre_turn.py
Writes results/audit/live/timing_pre_turn.json."""
import json, time
import jax, jax.numpy as jnp, numpy as np
from poker.envs import config
from poker.envs.betting import init_betting
from poker.solver import critics as CR, subgame as SG
from poker.solver.cards import NUM_HANDS as H
from tools.audit.live.timing import street_root

aux = CR.net_critic(CR.init_critic(jax.random.key(0), (512, 512, 512)), 0.0)
eq = np.random.default_rng(0).random((H, H), np.float32)
out = {}
pre = jax.tree.map(np.asarray, init_betting(2, jnp.full(2, 20000, jnp.int32), history=True, max_stack=20000, rake_level=0.0))
for name, root in (("turn", street_root(2)), ("preflop", pre)):
    t0 = time.perf_counter(); tree = SG.build_tree(root, 1, ()); cold = time.perf_counter() - t0
    r = np.full((2, H), 1 / H)
    row = {"nodes": int(tree.size), "decisions": int(len(tree.internal)), "leaves": int(len(tree.st.leaves)),
           "columns": int(tree.kinds.shape[1]), "tree_build_cold_s": cold}
    for label in ("cold", "warm"):
        t0 = time.perf_counter()
        sol = SG.solve_batch([tree], [r], 100, 50, aux=aux, equity=[eq], ladder=(1,))[0]
        row[f"solve100_{label}_s"] = time.perf_counter() - t0
        row[f"compile_{label}_s"] = sol.compile_seconds
    T = config.PLAY_ITERATIONS[int(root.street)]
    row["play_T"] = T
    row["per_iteration_ms"] = 1000 * row["solve100_warm_s"] / 100
    row["extrapolated_play_T_s"] = row["per_iteration_ms"] * T / 1000
    out[name] = row
    print(name, row, flush=True)
open("results/audit/live/timing_pre_turn.json", "w").write(json.dumps(out, indent=1))
