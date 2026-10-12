"""Solver cost on this CPU (audit).
1) kernels: showdown.apply_all (prefix-sum form) vs a dense [F, F] matvec doing the same job; per_card,
   mass_three, couple_live_folded, the full 3-way showdown of pcfr._three; batched x64 vectors.
2) PCFR+ seconds per iteration: river 'normal' (40 decisions) at B = 1, 8, 32 via pcfr.solve_fixed, and
   flop 'normal' (28 decisions, 13 leaves, 14 all-ins) at B = 1, 8 with a dummy critic that returns F.
   Also the 3-player river 'normal' tree at B = 1.
Usage: python3 tools/audit/solver/perf.py out.json [--skip-solve]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from poker.solver import cards as C, showdown as sd, pcfr as SV, situations as SI, tree as TR

H = C.NUM_HANDS
rec = {"jax": jax.__version__, "device": str(jax.devices()[0])}


def bench(fn, *a, reps=20):
    out = jax.block_until_ready(fn(*a))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = jax.block_until_ready(fn(*a))
    return (time.perf_counter() - t0) / reps


rng = np.random.default_rng(0)
board = jnp.asarray([3, 17, 30, 44, 50], jnp.int32)
tables = C.rank_tables(C.hand_ranks(board))
valid = np.asarray(C.board_valid(np.asarray(board)))
for B in (1, 64):
    w = jnp.asarray(rng.random((B, H)) * valid, jnp.float32)
    w2 = jnp.asarray(rng.random((B, H)) * valid, jnp.float32)
    ranks = np.asarray(tables["ranks"])
    dis = ~(C.MEMBER.T.astype(np.int32) @ C.MEMBER.astype(np.int32)).astype(bool)
    Wd = jnp.asarray(np.where(dis & (ranks[:, None] >= 0) & (ranks[None, :] >= 0),
                              (ranks[:, None] > ranks[None, :]) + 0.5 * (ranks[:, None] == ranks[None, :]), 0.0), jnp.float32)
    k = {}
    k["apply_all"] = bench(jax.jit(jax.vmap(lambda x: sd.apply_all(x, tables))), w)
    k["dense_matvec_same_job"] = bench(jax.jit(lambda x: x @ Wd.T), w)
    k["per_card"] = bench(jax.jit(jax.vmap(lambda x: sd.per_card(x, tables))), w)
    k["mass_three"] = bench(jax.jit(jax.vmap(sd.mass_three)), w, w2)
    k["couple_live_folded"] = bench(jax.jit(jax.vmap(lambda a, b: sd.couple_live_folded(a, b, tables))), w, w2)
    k["three_showdown_all_live"] = bench(jax.jit(jax.vmap(lambda a, b: SV._three(jnp.stack([a, b, a * b]), jnp.ones(3), 3.0, tables, (True, True, True)))), w, w2)
    k["two_showdown"] = bench(jax.jit(jax.vmap(lambda a, b: SV._two(jnp.stack([a, b]), jnp.ones(2), 2.0, tables, (True, True)))), w, w2)
    rec[f"kernels_ms_B{B}"] = {kk: v * 1e3 for kk, v in k.items()}
    print(B, rec[f"kernels_ms_B{B}"], flush=True)

if "--skip-solve" not in sys.argv:
    common.no_cache_write()
    cfg = SV.SolverConfig(cap=100, e_target=0.0, check_every=50)
    runs = [(2, 3, 1), (2, 3, 8), (2, 3, 32), (2, 1, 1), (2, 1, 8), (3, 3, 1)]
    rec["solve"] = []
    for n, street, B in runs:
        topo = TR.rule_topology(n, street, "normal")
        st = SV.Static(topo)
        sprs = np.geomspace(2.0, 20.0, B)
        sp = common.spots(n, street, sprs, seed=11, board=[0, 20, 21, 33, 45] if street == 1 else None)
        t0 = time.perf_counter()
        batch = SI.prepare(sp, topo)
        jax.block_until_ready(batch["legal"])
        t_prep = time.perf_counter() - t0
        aux = jax.tree_util.Partial(lambda board, pub, ranges, f, spot: f) if len(st.leaves) else None
        t0 = time.perf_counter()
        SV.solve_fixed(batch, st, cfg, 50, aux=aux)["z"].block_until_ready()
        t_first = time.perf_counter() - t0
        iters = 100 if B <= 8 else 50
        t0 = time.perf_counter()
        SV.solve_fixed(batch, st, cfg, iters, aux=aux)["z"].block_until_ready()
        dt = time.perf_counter() - t0
        r = {"n": n, "street": street, "B": B, "decisions": int(len(topo.internal)), "prepare_s": t_prep,
             "first_chunk_s_incl_compile": t_first, "iters": iters, "ms_per_iter_batch": dt / iters * 1e3,
             "ms_per_iter_per_spot": dt / iters / B * 1e3}
        rec["solve"].append(r)
        print(r, flush=True)
Path(sys.argv[1]).write_text(json.dumps(rec, indent=1))
