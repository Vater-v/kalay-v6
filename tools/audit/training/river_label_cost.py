"""River label cost on this CPU with the loop's tree, spots and solver (pcfr.solve_fixed, the fixed-T
teacher of loop.batch_solver), plus the label-noise floor: range-weighted |EV(T) - EV(2T)| per spot.
Compile is excluded (a 50-iteration warm-up with the same SolverConfig). The box is shared: the load
average is recorded next to the timings.
Usage: python river_label_cost.py N_PER_MENU [T]
Output: results/audit/training/river_label_cost_T{T}_n{N}.json
"""
import os, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402

import jax, jax.numpy as jnp, numpy as np  # noqa: E402
loop = common.patch_loop()
from poker.envs import config  # noqa: E402
from poker.solver import situations as SI, critics as CR, pcfr as SV  # noqa: E402

n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 4
T = int(sys.argv[2]) if len(sys.argv) > 2 else config.TEACHER_T["river"]
pool = SI.make_training_spots(loop.SEEDS["river"], 200, 2, rake="mix", street=3)
spr = pool["stack"] / pool["pot"]
menus = loop.spot_menus("river", pool)
out = {"T": T, "device": str(jax.devices()[0]), "matmul_precision": str(jax.config.jax_default_matmul_precision),
       "domain_spr_quantiles_200": {str(q): float(np.quantile(spr, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
       "domain_menu_counts_200": {m: int(sum(x == m for x in menus)) for m in set(menus)}, "by_menu": {}}
for menu in sorted(set(menus)):
    idx = sorted([i for i, m in enumerate(menus) if m == menu], key=lambda i: spr[i])
    pick = [idx[int(round(j))] for j in np.linspace(0, len(idx) - 1, n_per)]
    chunk = {k: v[pick] for k, v in pool.items()}
    topo, st = loop.tree_for(3, 2, menu)
    t0 = time.perf_counter(); batch = SI.prepare(chunk, topo); jax.block_until_ready(batch["tables"])
    prep = time.perf_counter() - t0
    cfg = SV.SolverConfig(cap=T, e_target=0.0, check_every=50)
    ex = jax.jit(jax.vmap(lambda z, s: SV.exploitability(z, s, st, cfg)))
    root = jax.jit(jax.vmap(lambda z, s: SV.root_values(z, s, st, cfg)))
    SV.solve_fixed(batch, st, cfg, 50)["z"].block_until_ready()                 # compile
    rec = {"spr": spr[pick].tolist(), "tree_nodes": int(topo.size), "decision_nodes": int(len(st.internal)),
           "terminals": int(len(st.terminals)), "prepare_s_per_spot_incl_compile": prep / n_per}
    evs = {}
    for t in (T, 2 * T):
        load = os.getloadavg()[0]
        t0 = time.perf_counter(); z = SV.solve_fixed(batch, st, cfg, t)["z"]; z.block_until_ready()
        dt = time.perf_counter() - t0
        e = np.asarray(ex(z, batch)["nash_conv"]) / 2
        evs[t] = np.asarray(root(z, batch))
        rec[f"T{t}"] = {"solve_s_per_spot": dt / n_per, "ms_per_iteration_per_spot": dt / n_per / t * 1e3,
                        "load_avg_1min_at_start": load, "E_pot": e.tolist(),
                        "E_bb": (e * chunk["pot"] / config.BB).tolist()}
        print(menu, t, rec[f"T{t}"], flush=True)
    r = jnp.asarray(chunk["ranges"]); valid = jnp.asarray(chunk["valid"], jnp.float32)
    noise = CR.prediction_metrics(jnp.asarray(evs[T]), jnp.asarray(evs[2 * T]), r, valid, jnp.ones((n_per, 2)))
    rec["label_noise_rw_mae_T_vs_2T_pot"] = np.asarray(noise["mae"]).tolist()
    rec["label_noise_worst_substantial_T_vs_2T_pot"] = np.asarray(noise["worst_substantial"]).tolist()
    out["by_menu"][menu] = rec
common.dump(f"river_label_cost_T{T}_n{n_per}.json", out)
