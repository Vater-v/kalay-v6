"""(4) Where solve time goes on the turn: PCFR+ iterations with street-end leaves valued by
 (a) the check-down oracle V = F (no network), (b) a 64x64 critic, (c) the default 3x512 critic
 (random weights: cost does not depend on the values), projected as the loop calls them.
Also the turn 'prepare' cost (exact turn all-in matrix) WITHOUT the rank7 cache (it is not shipped).
Usage: python turn_cost.py N_SPOTS ITERS
Output: results/audit/training/turn_cost.json
"""
import sys, time, types
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402

import jax, numpy as np  # noqa: E402
loop = common.patch_loop()
from poker.solver import situations as SI, critics as CR, pcfr as SV, rank7  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 50
assert rank7.cached_table() is None, "this measurement is for the no-cache case"
pool = SI.make_training_spots(loop.SEEDS["turn"], 200, 2, rake="mix", street=2)
menus = loop.spot_menus("turn", pool)
spr = pool["stack"] / pool["pot"]
oracle = jax.tree_util.Partial(lambda b, p, r, f, spot=None: f)
crit = {"oracle_V=F": oracle,
        "mlp_64x64": CR.net_critic(CR.init_critic(jax.random.key(0), (64, 64), 2), 0.0),
        "mlp_3x512": CR.net_critic(CR.init_critic(jax.random.key(0), (512, 512, 512), 2), 0.0)}
out = {"iterations_timed": iters, "spots_per_batch": n, "by_menu": {}}
args = types.SimpleNamespace(teacher=loop.FIXED_T, e_target=0.0)
for menu in ("normal", "short"):
    idx = sorted([i for i, m in enumerate(menus) if m == menu], key=lambda i: spr[i])
    pick = [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, n)]
    chunk = {k: v[pick] for k, v in pool.items()}
    topo, st = loop.tree_for(2, 2, menu)
    t0 = time.perf_counter(); batch = SI.prepare(chunk, topo); jax.block_until_ready(batch["equity"])
    prep = time.perf_counter() - t0
    rec = {"tree_nodes": int(topo.size), "street_end_leaves": int(len(st.leaves)), "allin_nodes": int(len(st.allins)),
           "prepare_s_per_spot_no_rank7": prep / n, "spr": spr[pick].tolist()}
    cfg = SV.SolverConfig(cap=iters, e_target=0.0, check_every=iters)
    for name, aux in crit.items():
        SV.solve_fixed(batch, st, cfg, iters, aux=aux)["z"].block_until_ready()          # compile
        t0 = time.perf_counter(); SV.solve_fixed(batch, st, cfg, iters, aux=aux)["z"].block_until_ready()
        dt = time.perf_counter() - t0
        rec[name] = {"ms_per_iteration_per_spot": dt / iters / n * 1e3,
                     "extrapolated_s_per_spot_T1500": dt / iters / n * 1500}
        print(menu, name, rec[name], flush=True)
    base = rec["oracle_V=F"]["ms_per_iteration_per_spot"]
    rec["critic_share_3x512"] = 1 - base / rec["mlp_3x512"]["ms_per_iteration_per_spot"]
    out["by_menu"][menu] = rec
common.dump(f"turn_cost_n{n}_it{iters}.json", out)
