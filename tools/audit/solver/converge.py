"""PCFR+ convergence on HU river spots (and one 3-player river spot): E (pot units, nash_conv / n) every 50
iterations from pcfr.solve_cfr's own exact best-response check; check_exploitability on the final average;
zero-sum identity of the values (no rake): sum_i value_i = 0.
Usage: python3 tools/audit/solver/converge.py out.json --n 2 --cap 2000 [--spots 4]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from poker.solver import pcfr as SV, situations as SI, tree as TR

p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("--n", type=int, default=2); p.add_argument("--cap", type=int, default=2000)
p.add_argument("--spots", type=int, default=4); p.add_argument("--menu", default="normal")
a = p.parse_args()
sprs = {2: [0.5, 2.0, 8.0, 30.0], 3: [1.0, 6.0]}[a.n][:a.spots]
topo = TR.rule_topology(a.n, 3, a.menu)
st = SV.Static(topo)
batch = SI.prepare(common.spots(a.n, 3, sprs, seed=29), topo)
cfg = SV.SolverConfig(cap=a.cap, e_target=0.0, check_every=50)
curve = []
t0 = time.perf_counter()


def log(state):
    curve.append({"iters": int(np.asarray(state["u"])[0]), "E": np.asarray(state["e"]).tolist(),
                  "t": time.perf_counter() - t0})
    if len(curve) % 5 == 0:
        print(curve[-1], flush=True)


res = SV.solve_cfr(batch, st, cfg, log=log)
ex = jax.vmap(lambda z, s: SV.exploitability(z, s, st, cfg))(res["z"], batch)
ex = jax.tree.map(np.asarray, ex)
ok = True
try:
    SV.check_exploitability(ex)
except AssertionError as e:
    ok = str(e)
E = np.asarray([c["E"] for c in curve])                     # [checks, spots]
its = np.asarray([c["iters"] for c in curve])
fits = []
for s in range(E.shape[1]):
    m = (its >= 200) & (E[:, s] > 0)
    slope = float(np.polyfit(np.log(its[m]), np.log(E[m, s]), 1)[0]) if m.sum() > 2 else None
    first = {str(t): (int(its[np.argmax(E[:, s] <= t)]) if (E[:, s] <= t).any() else None) for t in (1e-2, 3e-3, 1e-3, 3e-4)}
    fits.append({"spr": sprs[s], "loglog_slope_200_to_cap": slope, "first_iters_E_below": first,
                 "E_final": float(E[-1, s]), "monotone_violations": int(np.sum(np.diff(E[:, s]) > 0))})
out = {"n": a.n, "menu": a.menu, "decisions": int(len(topo.internal)), "sprs": sprs, "cap": a.cap,
       "check_exploitability": ok, "value_sum_zero_sum_identity": np.sum(ex["value"], axis=-1).tolist(),
       "nash_conv": ex["nash_conv"].tolist(), "fits": fits, "seconds": time.perf_counter() - t0,
       "curve": [{"iters": int(i), "E": e.tolist()} for i, e in zip(its, E)]}
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
