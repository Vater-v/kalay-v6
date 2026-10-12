"""Reference hashes of the street solver's public trees (brief M2, item 2: HU trees bit for bit).

    POKER_REGIME=1 POKER_GRID=7 PYTHONPATH=. .venv/bin/python -m tools.generate_trees_ref \
        --out checks/data/trees_ref.json

For each argument tuple (players, cap, all-in after the cap, street) of poker.solver.tree.build_topology
it records the node count and the SHA-256 of every Topology array (parent, label, actor, children,
cap_ok, live, terminal, leaf, levels) in a fixed order. Generated once with the builder before the
n <= 6 generalisation (commit in the file); checks/s5_4.py rebuilds the same tuples and compares.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from poker.envs import config
from poker.solver import tree as TR

CONFIGS = [(2, 2, True, s) for s in (0, 1, 2, 3)] + [(2, 3, False, 3), (2, 1, True, 3)] \
    + [(3, 2, True, s) for s in (0, 1, 2, 3)] + [(3, 1, True, 3), (3, 3, False, 3)]


def topology_hash(topo) -> str:
    h = hashlib.sha256()
    h.update(f"n={topo.n} street={topo.street}".encode())
    arrays = [topo.parent, topo.label, topo.actor, topo.children, topo.cap_ok, topo.live, topo.terminal, topo.leaf]
    arrays += list(topo.levels)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if config.GRID != 7:
        raise SystemExit(f"the reference trees are the solver's grid: run with POKER_GRID=7 (this process: {config.GRID})")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "poker/solver/tree.py", "poker/envs"],
                                capture_output=True, text=True).stdout.strip())
    rows = []
    for n, cap, aac, street in CONFIGS:
        topo = TR.build_topology(n, cap, allin_after_cap=aac, street=street)
        rows.append({"n": n, "cap": cap, "allin_after_cap": aac, "street": street, "nodes": int(topo.size),
                     "decisions": int(len(topo.internal)), "sha256": topology_hash(topo)})
        print(rows[-1], flush=True)
    args.out.write_text(json.dumps({"generator": "tools/generate_trees_ref.py", "commit": commit,
                                    "tree_code_dirty": dirty, "grid": config.GRID, "trees": rows}, indent=1))


if __name__ == "__main__":
    main()
