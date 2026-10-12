"""Re-measure B = 1 PCFR+ ms/iteration (river and flop 'normal') with nothing else of this audit running
(perf.py's B = 1 figures were taken while terminals.py ran on another core). Usage: perf_b1.py out.json"""
import json, sys, time
from pathlib import Path
import numpy as np, jax
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from poker.solver import pcfr as SV, situations as SI, tree as TR
common.no_cache_write()
cfg = SV.SolverConfig(cap=100, e_target=0.0, check_every=50)
out = []
for street in (3, 1):
    topo = TR.rule_topology(2, street, "normal"); st = SV.Static(topo)
    batch = SI.prepare(common.spots(2, street, [4.0], seed=11, board=[0, 20, 21, 33, 45] if street == 1 else None), topo)
    aux = jax.tree_util.Partial(lambda board, pub, ranges, f, spot: f) if len(st.leaves) else None
    SV.solve_fixed(batch, st, cfg, 50, aux=aux)["z"].block_until_ready()
    reps = []
    for _ in range(3):
        t0 = time.perf_counter(); SV.solve_fixed(batch, st, cfg, 200, aux=aux)["z"].block_until_ready()
        reps.append((time.perf_counter() - t0) / 200 * 1e3)
    out.append({"street": street, "B": 1, "ms_per_iter_reps": reps, "ms_per_iter_min": min(reps)})
    print(out[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
