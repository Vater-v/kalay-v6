"""3-player turn all-in runout (pcfr.turn_runout_three, the /42 normalisation) vs brute force over the 48 rivers
and all card-disjoint triples (ranges on 120-hand supports, 50 sampled hero hands per player), x64.
Usage: python3 tools/audit/solver/turn3.py out.json"""
import json, sys
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import indep
from poker.solver import pcfr as SV, situations as SI

rng = np.random.default_rng(5)
DIS, _ = indep.disjoint()
rows = []
for b4 in ([4, 17, 30, 44], [0, 12, 25, 26]):
    b4 = np.asarray(b4)
    valid = ~np.isin(indep.HANDS, b4).any(1)
    w = np.zeros((3, indep.F)); sup = []
    for j in range(3):
        idx = np.sort(rng.choice(np.flatnonzero(valid), 120, replace=False))
        w[j, idx] = rng.random(120) + 0.1; w[j] /= w[j].sum(); sup.append(idx)
    cards, tables = SI.turn_river_tables(jnp.asarray(b4, jnp.int32))
    rivers = np.asarray(cards)
    rk = {c: indep.ranks_on_board(np.append(b4, c)) for c in rivers}
    for live in [(True, True, True), (True, True, False), (False, True, True)]:
        contrib = np.array([3.0 if live[j] else 0.5 for j in range(3)]); pot = contrib.sum()
        got = np.asarray(SV.turn_runout_three(jnp.asarray(w[None]), jnp.asarray(contrib[None]), jnp.asarray([pot]),
                                              cards, tables, live, chunk=1))[0]
        heroes = rng.choice(np.flatnonzero(valid), 50, replace=False)
        err = 0.0; scale = 0.0
        for i in range(3):
            bb, cc = [j for j in range(3) if j != i]
            Ob, Oc = sup[bb], sup[cc]
            wbc = w[bb, Ob][:, None] * w[cc, Oc][None, :] * DIS[np.ix_(Ob, Oc)]
            for h in heroes:
                m = wbc * DIS[h, Ob][:, None] * DIS[h, Oc][None, :]
                hold = np.zeros(m.shape)
                cnt = np.zeros(m.shape)
                for c in rivers:
                    if c in indep.HANDS[h]:
                        continue
                    okc = (~np.isin(indep.HANDS[Ob], c).any(1))[:, None] & (~np.isin(indep.HANDS[Oc], c).any(1))[None, :]
                    r = rk[c]; rh = r[h]; rb = r[Ob][:, None]; rc = r[Oc][None, :]
                    best = np.full(m.shape, rh)
                    if live[bb]: best = np.maximum(best, rb)
                    if live[cc]: best = np.maximum(best, rc)
                    n = 1.0 + live[bb] * (rb == best) + live[cc] * (rc == best)
                    share = np.where(rh == best, 1.0 / n, 0.0) if live[i] else 0.0
                    hold += okc * (pot * share - contrib[i]); cnt += okc
                bf = np.sum(m * np.where(cnt > 0, hold / np.maximum(cnt, 1), 0.0))
                assert np.all(cnt[m > 0] == 42), "rivers per compatible triple != 42"
                err = max(err, abs(bf - got[i, h])); scale = max(scale, abs(bf))
        rows.append({"board4": b4.tolist(), "live": live, "max_abs_err": err, "max_rel_err": err / scale})
        print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
