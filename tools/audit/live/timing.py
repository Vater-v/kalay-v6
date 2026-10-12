"""Wall-clock of the live solve on this CPU: a HU flop and a HU river LiveTree (Slumbot table: 200 BB, a 2.5-BB
open called = 500-chip pot... here b200c: pot 400, 19,800 behind), solve_batch at batch 1 with the play
iterations (config.PLAY_ITERATIONS: flop 400, river 300) and chunk 50, cold (first call: compile) and warm
(second and third calls), the tree build (shape cold, replay warm), spot_batch, peak RSS. The flop's critic is
a RANDOM init_critic of the training default width (512, 512, 512) - the same shapes and FLOPs as a trained
one; the flop all-in matrix is random (timing only).

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/timing.py [flop|river|both] [T_override]
Writes results/audit/live/timing_<street>.json.
"""
import json
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.envs.transition import step_target
from poker.envs.betting import init_betting
from poker.solver import critics as CR
from poker.solver import subgame as SG
from poker.solver.cards import NUM_HANDS as H


def _trivial(board1h, pub, ranges, f, spot=None):
    return f


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def street_root(street):
    s = init_betting(2, jnp.full(2, 20000, jnp.int32), history=True, max_stack=20000, rake_level=0.0)
    s = step_target(s, jnp.int32(2), jnp.int32(200))                        # SB opens to 200
    s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(200))       # BB calls: flop, pot 400
    board = [12, 24, 5, 30, 44]                                             # engine ids (suit*13+rank): A K 7 6 7
    for _ in range(street - 1):                                             # check through to the wanted street
        s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(0))
        s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(0))
    s = s.replace(board=jnp.asarray(board, jnp.int32))
    return jax.tree.map(np.asarray, s)


def run(street, T=None):
    T = T or config.PLAY_ITERATIONS[street]
    out = {"street": street, "T": T, "chunk": 50, "device": jax.default_backend(), "rss_mb_start": rss_mb()}
    root = street_root(street)
    out["root"] = {"street": int(root.street), "pot": int(root.bet_total.sum()), "stack": root.stack.tolist(),
                   "to_act": int(root.to_act), "bucket": SG.bucket_of(root)}
    hero = 1
    t0 = time.perf_counter()
    tree = SG.build_tree(root, hero, ())
    out["tree_build_cold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tree = SG.build_tree(root, hero, ())
    out["tree_build_warm_s"] = time.perf_counter() - t0
    out["nodes"], out["decisions"], out["leaves"] = int(tree.size), int(len(tree.internal)), int(len(tree.st.leaves))
    out["columns"] = int(tree.kinds.shape[1])
    out["state_bytes_per_array"] = int(len(tree.internal) * H * tree.kinds.shape[1] * 4)
    aux = None
    equity = None
    if street in (1, 2):
        params = CR.init_critic(jax.random.key(0), (512, 512, 512))
        aux = CR.net_critic(params, 0.0)
        if "--trivial-critic" in sys.argv:                                     # V = F (check-down): isolates the critic's cost
            aux = jax.tree_util.Partial(_trivial)
            out["critic"] = "trivial (returns F)"
        rng = np.random.default_rng(0)
        equity = rng.random((H, H), np.float32)                                      # timing only
        out["critic_params"] = int(sum(np.prod(l[k].shape) for l in params for k in l))
    ranges = np.full((2, H), 1.0 / H)
    t0 = time.perf_counter()
    SG.spot_batch(tree, ranges, 0.0, equity, None)
    out["spot_batch_s"] = time.perf_counter() - t0
    for label in ("cold", "warm1", "warm2"):
        t0 = time.perf_counter()
        sol = SG.solve_batch([tree], [ranges], T, 50, aux=aux, equity=[equity], ladder=(1,))[0]
        out[f"solve_{label}_s"] = time.perf_counter() - t0
        out[f"solve_{label}_inner_s"] = sol.seconds
        out[f"solve_{label}_compile_s"] = sol.compile_seconds
        out[f"rss_mb_after_{label}"] = rss_mb()
    out["per_iteration_ms_warm"] = 1000 * out["solve_warm2_inner_s"] / T
    tag = ("_trivialcritic" if "--trivial-critic" in sys.argv else "") + ("_mt" if "--mt" in sys.argv else "")
    out["xla_flags"] = __import__("os").environ.get("XLA_FLAGS", "")
    p = Path(f"results/audit/live/timing_{['pre', 'flop', 'turn', 'river'][street]}{tag}.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    T = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    streets = {"flop": [1], "river": [3], "turn": [2], "both": [3, 1]}[which]
    for st in streets:
        run(st, T)
