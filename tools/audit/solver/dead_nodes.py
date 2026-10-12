"""Fraction of the fixed topology that is dead (an action on the path illegal at the spot's real chips) by SPR,
HU river/turn/flop 'normal', from tree.replay's `alive` (audit)."""
import json, sys
from pathlib import Path
import numpy as np
from poker.solver import tree as TR
out = []
for street in (3, 2, 1):
    topo = TR.rule_topology(2, street, "normal")
    sprs = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 50.0])
    pot = np.full(len(sprs), 1000, np.int32)
    r = TR.replay(topo, pot, np.round(sprs * 1000).astype(np.int32))
    alive = np.asarray(r["alive"])
    dec = ~topo.terminal
    for s, a in zip(sprs, alive):
        out.append({"street": street, "spr": float(s), "alive_decisions": int(a[dec].sum()), "decisions": int(dec.sum()),
                    "alive_nodes": int(a.sum()), "nodes": int(topo.size)})
        print(out[-1])
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
