"""Tree counts and build times of poker.solver.tree.rule_topology (audit).
Counts per (n, street, menu): nodes, decision nodes, fold terminals, showdowns (river), all-in runouts
(non-leaf boundaries with >= 2 live before the river), street-end leaves, max depth, actions per decision,
first-build seconds (incl. jit of the engine step/action table) and a second build (warm).
Also times tree.replay (the per-batch engine replay that prepare() uses) for a batch of 8 spots.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax
from poker.envs import config
from poker.solver import tree as TR

out = []
cases = [(2, s, m) for s in (3, 2, 1, 0) for m in ("normal", "short")] + [(3, 3, "normal"), (3, 2, "normal"), (3, 1, "normal")]
if len(sys.argv) > 2 and sys.argv[2] == "--with6":
    cases += [(4, 3, "normal"), (6, 3, "normal")]
for n, street, menu in cases:
    t0 = time.perf_counter()
    topo = TR.rule_topology(n, street, menu)
    t1 = time.perf_counter()
    topo2 = TR.rule_topology(n, street, menu)
    t2 = time.perf_counter()
    term = topo.terminal
    leaf = topo.leaf if topo.leaf is not None else np.zeros(topo.size, bool)
    nlive = topo.live.sum(1)
    fold = term & (nlive == 1)
    multi = term & ~leaf & (nlive >= 2)
    rec = {"n": n, "street": street, "menu": menu, "rule": {k: (v if not isinstance(v, tuple) else str(v)) for k, v in config.tree_rule(n, street, menu).items()},
           "nodes": int(topo.size), "decisions": int((~term).sum()), "folds": int(fold.sum()),
           "showdowns" if street == 3 else "allin_runouts": int(multi.sum()), "leaves": int(leaf.sum()),
           "depth": len(topo.levels) - 1,
           "mean_actions_per_decision": float(topo.cap_ok[~term].sum(1).mean()),
           "max_actions": int(topo.cap_ok[~term].sum(1).max()), "A_padded": int(topo.children.shape[1]),
           "build_s_first": t1 - t0, "build_s_warm": t2 - t1}
    out.append(rec)
    print(json.dumps(rec), flush=True)
# replay timing (river normal, 8 spots)
topo = TR.rule_topology(2, 3, "normal")
pot = np.full(8, 1000, np.int32); stack = np.full(8, 3000, np.int32)
t0 = time.perf_counter(); r = TR.replay(topo, pot, stack); jax.block_until_ready(r["legal"]); t1 = time.perf_counter()
r = TR.replay(topo, pot + 2, stack); jax.block_until_ready(r["legal"]); t2 = time.perf_counter()
out.append({"replay_river_normal_B8_first_s": t1 - t0, "replay_warm_s": t2 - t1})
print(out[-1])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
