"""rank7 indexing without the 268 MB table (not shipped): with table = arange(C(52,7)) hand_ranks_fast returns the
combinadic index itself; compare with combo_index of the sorted 7 cards and with an independent python
combinadic; unrank(combo_index(x)) == x. (The table contents - evaluate_hand classes - are covered by the
evaluator agreement in terminals.py, section A.)"""
import json, sys
from math import comb
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from poker.solver import rank7, cards as C

table = jnp.arange(rank7.N7, dtype=jnp.int32)
rng = np.random.default_rng(3)
bad = 0; checked = 0
for _ in range(20):
    b = rng.choice(52, 5, replace=False)
    got = np.asarray(rank7.hand_ranks_fast(table, jnp.asarray(b, jnp.int32)))
    for h in rng.choice(np.flatnonzero(~np.isin(C.HANDS, b).any(1)), 50, replace=False):
        cards = sorted(list(C.HANDS[h]) + list(b))
        want = sum(comb(c, k + 1) for k, c in enumerate(cards))
        bad += int(got[h] != want); checked += 1
x = np.sort(np.stack([rng.choice(52, 7, replace=False) for _ in range(5000)]), 1).astype(np.int32)
rt = np.asarray(rank7.unrank(rank7.combo_index(jnp.asarray(x))))
res = {"hand_ranks_fast_index_mismatches": bad, "checked": checked, "unrank_roundtrip_ok": bool(np.array_equal(rt, x))}
print(res); Path(sys.argv[1]).write_text(json.dumps(res))
