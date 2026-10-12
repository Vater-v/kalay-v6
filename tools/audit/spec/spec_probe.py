"""Spec-area probes: what the engine accepts / refuses for 2, 3, 4+ players, tree sizes for the
multiway distance, the regime/menu facts, the kappa projection under regime 1, the likelihood floor's
re-admitted mass, critic size and forward cost. Read-only on poker/. Output: results/audit/spec/spec_probe.json
    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python3 tools/audit/spec/spec_probe.py
"""
import json, time, traceback, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

from poker.envs import config
from poker.solver import tree as TR, pcfr as SV, critics as CR, showdown as sd
from poker.solver.cards import NUM_HANDS as H
from poker.player import live as LV

OUT = Path("results/audit/spec/spec_probe.json")
res = {}

def attempt(fn):
    t0 = time.perf_counter()
    try:
        v = fn()
        return {"ok": True, "value": v, "s": round(time.perf_counter() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "s": round(time.perf_counter() - t0, 2)}

# 1. regime / menu facts
res["config"] = {
    "regime": config.REGIME, "players": list(config.PLAYERS), "depth_mix": [list(x) for x in config.DEPTH_MIX],
    "rake_mix": [list(x) for x in config.RAKE_MIX], "grid": config.GRID, "LOOP_PLAYERS": config.LOOP_PLAYERS,
    "TREE_MAX_PLAYERS": config.TREE_MAX_PLAYERS, "menus": {m: config.MENU_BOUNDS[m] for m in config.MENUS},
    "PLAY_ITERATIONS": config.PLAY_ITERATIONS, "TEACHER_T": config.TEACHER_T, "TEACHER_E_FLAG": config.TEACHER_E_FLAG,
    "COMMIT_MERGE_POT": config.COMMIT_MERGE_POT, "matmul_precision": str(jax.config.jax_default_matmul_precision),
    "menu_for": {f"street{s}_spr{x}": config.menu_for(s, spr=x) for s in (1, 2, 3) for x in (0.5, 1.99, 2.0, 50.0)}
                | {f"preflop_depth{d}": config.menu_for(0, depth_bb=d) for d in (20, 39.9, 40, 200, 500)},
}

# 2. multiway: topology sizes (teacher trees) and whether the solver accepts them
sizes = {}
for n in (2, 3, 4, 6):
    for street in (3, 2, 1):
        key = f"n{n}_street{street}_normal"
        r = attempt(lambda: TR.rule_topology(n, street, "normal"))
        if r["ok"]:
            topo = r["value"]
            ent = {"nodes": int(topo.size), "decisions": int(len(topo.internal)), "build_s": r["s"],
                   "leaves": int(np.sum(topo.leaf)) if topo.leaf is not None else 0}
            s = attempt(lambda: (SV.Static(topo), "Static ok")[1])
            ent["pcfr_Static"] = s["value"] if s["ok"] else s["error"]
        else:
            ent = {"build_error": r["error"]}
        sizes[key] = ent
        print(key, ent, flush=True)
res["teacher_trees"] = sizes

# 3. the 3+/4+ gates at the function level
w = jnp.asarray(np.random.default_rng(0).random((4, H)), jnp.float32)
res["gates"] = {
    "mass_of_n4": attempt(lambda: (SV.mass_of(w[None], 0, 4), "returned")[1]),
    "mass_of_n3": attempt(lambda: (np.asarray(SV.mass_of(w[None, :3], 0, 3)).shape, "returned")[1]),
    "critics_opponent_mass_4_present_is_nan": attempt(
        lambda: bool(np.isnan(np.asarray(CR.opponent_mass(w, jnp.ones(4), 0))).all())),
}
# projection silently skipped for 4 live slots?
v = jnp.asarray(np.random.default_rng(1).normal(size=(4, H)), jnp.float32)
r4 = w / w.sum(-1, keepdims=True)
proj = CR.project_values(v, r4, jnp.ones(4), jnp.float32(1.0), jnp.float32(1.0), 0.0)
res["gates"]["project_values_4_live_unchanged"] = bool(np.allclose(np.asarray(proj), np.asarray(v)))
proj2 = CR.project_values(v[:2], r4[:2], jnp.ones(2), jnp.float32(1.0), jnp.float32(1.0), 0.0)
wts, live, ok = CR.deal_weights(r4[:2], jnp.ones(2))
res["gates"]["project_values_2_live_total_after"] = float(jnp.sum(wts * proj2))

# 4. kappa under regime 1 (no rake): target total is 1 for any kappa
pub = CR.public_row(jnp.float32(20.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                    jnp.zeros(CR.P_MAX).at[:2].set(1.0), jnp.zeros(CR.P_MAX).at[:2].set(5.0), jnp.zeros(CR.P_MAX).at[:2].set(0.5))
present, net0, lo = CR.payout_terms(pub, 2)
res["kappa_regime1"] = {"net0": float(net0), "lo": float(lo),
                        "target_total_kappa_0.0176": float(CR.target_total(net0, lo, 0.0176)),
                        "KAPPA": CR.KAPPA}

# 5. likelihood floor: mass re-admitted on hands the solve says never take the action
def floor_case(frac_value, like_value, eps=LV.DEFAULT_EPS):
    r = np.full(H, 1.0 / H)
    like = np.zeros(H)
    k = int(frac_value * H)
    like[:k] = like_value                    # hands that take the action
    post = LV._bayes(r, np.maximum(like, eps))
    return float(post[k:].sum())             # posterior mass on hands with L = 0
res["likelihood_floor"] = {
    f"action_taken_by_{int(f*100)}pct_hands_at_L{l}": floor_case(f, l)
    for f in (0.02, 0.1, 0.3) for l in (0.05, 0.3, 1.0)}
# after k consecutive such updates on the same zero-likelihood hands (floor compounds as eps^k relative)
def floor_chain(k, f=0.1, l=0.3, eps=LV.DEFAULT_EPS):
    r = np.full(H, 1.0 / H); like = np.zeros(H); like[:int(f * H)] = l
    for _ in range(k):
        r = LV._bayes(r, np.maximum(like, eps))
    return float(r[int(f * H):].sum())
res["likelihood_floor"]["chain_10pct_L0.3"] = {str(k): floor_chain(k) for k in (1, 2, 3)}

# 6. critic size and forward cost (k = 2 slots, hidden 3x512 as the loop default)
params = CR.init_critic(jax.random.PRNGKey(0), (512, 512, 512))
n_stored = int(sum(np.prod(x.shape) for layer in params for x in layer.values()))
k = 2
used = (52 + CR.PUB + k * H) * 512 + k * H * 512 + 512 * 512 * 2 + 512 * k * H + 512 * 3 + k * H
flop_per_row = 2 * used
fwd = jax.jit(lambda p, b, pub, r, f: CR.apply_net(p, b, pub, r, f))
timing = {}
for L in (1, 13, 19, 64):
    b = jnp.zeros((L, 52)); pb = jnp.zeros((L, CR.PUB)); rr = jnp.full((L, k, H), 1.0 / H); ff = jnp.zeros((L, k, H))
    fwd(params, b, pb, rr, ff).block_until_ready()
    t0 = time.perf_counter(); reps = 20
    for _ in range(reps):
        fwd(params, b, pb, rr, ff).block_until_ready()
    timing[str(L)] = round((time.perf_counter() - t0) / reps * 1e3, 3)
res["critic"] = {"stored_params_P_MAX9": n_stored, "used_params_k2": used, "mflop_per_row_k2": round(flop_per_row / 1e6, 2),
                 "cpu_ms_per_call_by_rows": timing,
                 "note": "rows = street-end leaves evaluated per PCFR iteration (13 flop / 19 turn in the HU normal trees)"}

# 7. vocabulary of the safety mechanism in the code base
import subprocess
def count(pat):
    out = subprocess.run(["grep", "-rIl", "-i", pat, "poker", "--include=*.py"], capture_output=True, text=True).stdout.split()
    return sorted(out)
res["vocabulary_files"] = {p: count(p) for p in ("ceiling", "gadget", "terminate", "restricted nash", "rnr", "coalition",
                                                  "exploitab", "best.response")}
OUT.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k != "teacher_trees"}, indent=1, default=str)[:4000])
