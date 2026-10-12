"""checkdown.turn_cards on flop [0, 20, 21]: card_average over the 49 turn cards (weights m_c = J without c)
== share from the exact flop all-in matrix (runout.equity_matrix_flop, read from the scratch cache or built
there), and == the spec's (1/45) sum_c cfv_c / J form. Two independent code paths of the engine (1176 river
rank tables vs 1326 runout fori_loop). Also times turn_cards."""
import json, sys, time
from pathlib import Path
import numpy as np
import jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from poker.solver import checkdown as CD, runout as RO, situations as SI, cards as C

flop = np.asarray([0, 20, 21])
W = np.asarray(RO.flop_equity_cached(flop, cache=common.SCRATCH / "flop_equity"))
rng = np.random.default_rng(1)
valid = ~np.isin(C.HANDS, flop).any(1)
w = np.stack([SI.random_range(rng, valid), SI.random_range(rng, valid)]).astype(np.float32)
t0 = time.time()
shares, mass = CD.turn_cards(jnp.asarray(flop, jnp.int32), jnp.asarray(w), None)
shares.block_until_ready(); dt = time.time() - t0
avg = np.asarray(CD.card_average(shares, mass))
direct = np.stack([np.asarray(CD.share_matrix(jnp.asarray(w[1]), jnp.asarray(W))),
                   np.asarray(CD.share_matrix(jnp.asarray(w[0]), jnp.asarray(W)))])
jm = np.stack([np.asarray(CD.j_mass(jnp.asarray(w[1]))), np.asarray(CD.j_mass(jnp.asarray(w[0])))])
spec = np.asarray(jnp.sum(shares * mass, 0)) / 45.0 / np.where(jm > 0, jm, 1)
ok = valid & (jm.min(0) > 0)
res = {"flop": flop.tolist(), "turn_cards_seconds_incl_compile": dt,
       "identity_max_abs": float(np.abs(avg - direct)[:, ok].max()),
       "spec_1_over_45_form_max_abs": float(np.abs(spec - direct)[:, ok].max()),
       "uniform_1_over_49_shrink_factor": 45.0 / 49.0}
print(res); Path(sys.argv[1]).write_text(json.dumps(res))
