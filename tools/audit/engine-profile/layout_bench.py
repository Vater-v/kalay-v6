"""How much of the per-iteration regret bookkeeping is the padded action axis? (read-only micro-benchmark)

pcfr keeps regret / average / last regret as [internal nodes, 1326, A] with A = config.NUM_ACTIONS = 13
(pcfr.init_cfr_state), while a HU river 'normal' node has 2.8 legal actions on average. This times the
same PCFR+ bookkeeping (regret matching on regret + last, clipped regret update, quadratic-weight
average) on (a) the engine's padded layout [Vi, H, 13] and (b) a compact edge-major layout [E, H]
with E = the number of legal edges (segment sums by node), single core, B spots vmapped.

Usage: POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. taskset -c 3 python3 tools/audit/engine-profile/layout_bench.py
"""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from poker.solver import pcfr as SV
from poker.solver import situations as SI
from poker.solver import tree as TR

H = 1326


def bench(fn, *a, reps=20):
    jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / reps


def main():
    topo = TR.rule_topology(2, 3, "normal")
    st = SV.Static(topo)
    spots = SI.make_training_spots(1101, 64, 2, street=3)
    from poker.envs import config
    first = next(i for i in range(64) if config.menu_for(3, spr=int(spots["stack"][i]) / int(spots["pot"][i])) == "normal")
    batch = SI.prepare({k: v[first:first + 1] for k, v in spots.items()}, topo)
    legal = np.asarray(batch["legal"][0])[topo.internal]                       # [Vi, A]
    vi, a = legal.shape
    node_of_edge = np.repeat(np.arange(vi), legal.sum(1)).astype(np.int32)
    e = len(node_of_edge)
    out = {"internal": int(vi), "A": int(a), "edges": int(e), "mean_legal": float(legal.sum(1).mean()),
           "max_legal": int(legal.sum(1).max())}
    for b in (1, 16):
        key = jax.random.PRNGKey(0)
        reg = jax.random.uniform(key, (b, vi, H, a))
        lg = jnp.broadcast_to(jnp.asarray(legal), (b, vi, a))

        def padded(regret, last, avg, r, own, legal):
            pi = SV.regret_policy(regret + last, legal)
            return jnp.maximum(regret + r, 0.0), avg + 9.0 * own[:, :, None] * pi, r

        f_pad = jax.jit(jax.vmap(padded))
        own = jax.random.uniform(key, (b, vi, H))
        t_pad = bench(f_pad, reg, reg, reg, reg, own, lg)

        seg = jnp.asarray(node_of_edge)

        def compact(regret, last, avg, r, own):
            x = jnp.maximum(regret + last, 0.0)                                   # [E, H]
            tot = jax.ops.segment_sum(x, seg, num_segments=vi)                    # [Vi, H]
            cnt = jax.ops.segment_sum(jnp.ones(e), seg, num_segments=vi)
            t = tot[seg]
            pi = jnp.where(t > 0, x / jnp.maximum(t, 1e-30), 1.0 / cnt[seg][:, None])
            return jnp.maximum(regret + r, 0.0), avg + 9.0 * own[seg] * pi, r

        f_cmp = jax.jit(jax.vmap(compact))
        rc = jax.random.uniform(key, (b, e, H))
        t_cmp = bench(f_cmp, rc, rc, rc, rc, own)
        out[f"B{b}"] = {"padded_ms": 1000 * t_pad, "compact_ms": 1000 * t_cmp, "ratio": t_pad / t_cmp,
                        "padded_state_mb_per_spot": 3 * vi * H * a * 4 / 2**20,
                        "compact_state_mb_per_spot": 3 * e * H * 4 / 2**20}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        open(sys.argv[1], "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
