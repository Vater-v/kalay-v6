"""Profile pcfr.solve_fixed and its sub-functions on one HU tree, one batch size, in one process.

Usage (from /home/user/kalay-v6):
  POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  python3 tools/audit/engine-profile/profile_solve.py --street 3 --menu normal --batch 4 --iters 800 \
      [--critic checkdown|net] [--reps 10] [--out results/audit/engine-profile/solve.jsonl]

Prints and appends one JSON line: prepare / compile / steady solve seconds, s/spot, s/iteration,
peak RSS, and the per-iteration breakdown (forward, terminal_values, backward, regret update)
from jitted, vmapped sub-functions timed separately on the same batch. Read-only with respect
to poker/: the flop equity matrix is computed via runout.equity_matrix_flop and cached only in
this script's scratch directory (never results/cache).
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

COMPILE = [0.0]


def _listener(event, duration, *a, **k):
    if event.startswith("/jax/core/compile"):
        COMPILE[0] += float(duration)


jax.monitoring.register_event_duration_secs_listener(_listener)

from poker.envs import config                      # noqa: E402
from poker.solver import critics as CR             # noqa: E402
from poker.solver import pcfr as SV                # noqa: E402
from poker.solver import runout as RO              # noqa: E402
from poker.solver import situations as SI          # noqa: E402
from poker.solver import tree as TR                # noqa: E402
from poker.solver.cards import NUM_HANDS as H      # noqa: E402

SCRATCH = Path(os.environ.get("AUDIT_SCRATCH", "/tmp/engine_profile_cache"))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def block(x):
    return jax.block_until_ready(x)


def timeit(fn, *args, reps=10):
    out = block(fn(*args))                         # warm (compile)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = block(fn(*args))
    return (time.perf_counter() - t0) / reps, out


_FLOP_CACHED = RO.flop_equity_cached               # the original, before the patch below replaces the attribute


def flop_matrix_no_cache_write(flop):
    """runout.flop_equity_cached semantics with the cache in SCRATCH (never results/cache)."""
    return _FLOP_CACHED(flop, cache=SCRATCH / "flop_equity")


def make_spots(street, menu, batch, seed):
    """make_training_spots as the loop draws them (poker/solver/loop.py stage_spots), then the first
    `batch` spots whose size menu (config.menu_for by SPR) is `menu`. Flop: one canonical flop (board_pool=1)."""
    pool = 1 if street == 1 else 0
    spots = SI.make_training_spots(seed, max(8 * batch, 64), 2, rake="mix", street=street, board_pool=pool)
    menus = np.asarray([config.menu_for(street, spr=int(s) / int(p)) for s, p in zip(spots["stack"], spots["pot"])])
    idx = np.flatnonzero(menus == menu)[:batch]
    if len(idx) < batch:
        raise SystemExit(f"only {len(idx)} {menu} spots drawn")
    return {k: v[idx] for k, v in spots.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--street", type=int, default=3)
    p.add_argument("--menu", default="normal")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--critic", default="checkdown", choices=["checkdown", "net"])
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--no-solve", action="store_true", help="breakdown only")
    p.add_argument("--seed", type=int, default=1101)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    rec = {"street": a.street, "menu": a.menu, "batch": a.batch, "iters": a.iters,
           "xla_flags": os.environ.get("XLA_FLAGS", ""), "jax": jax.__version__, "device": str(jax.devices()[0])}
    t0 = time.perf_counter()
    topo = TR.rule_topology(2, a.street, a.menu)
    st = SV.Static(topo)
    rec["tree_build_s"] = time.perf_counter() - t0
    rec["tree"] = {"V": int(topo.size), "internal": int(len(topo.internal)), "terminals": int(len(topo.terminals)),
                   "leaves": int(len(st.leaves)), "allins": int(len(st.allins)), "A": int(topo.children.shape[1])}

    spots = make_spots(a.street, a.menu, a.batch, a.seed)
    if a.street == 1:
        SI.runout.flop_equity_cached = flop_matrix_no_cache_write     # module attribute, this process only
    c0, t0 = COMPILE[0], time.perf_counter()
    batch = SI.prepare(spots, topo)
    block(batch)
    rec["prepare_s"] = time.perf_counter() - t0
    rec["prepare_compile_s"] = COMPILE[0] - c0
    legal = np.asarray(batch["legal"])[:, topo.internal]
    rec["mean_legal_actions"] = float(legal.sum(-1).mean())
    rec["spr"] = [float(s / q) for s, q in zip(spots["stack"], spots["pot"])]

    aux = None
    if len(st.leaves):
        if a.critic == "checkdown":
            aux = jax.tree_util.Partial(lambda board, pub, ranges, f, spot: f)
            rec["critic"] = "checkdown stand-in (returns F)"
        else:
            params = CR.init_critic(jax.random.PRNGKey(0), tuple(a.hidden), k=2)
            aux = CR.net_critic(params, kappa=CR.KAPPA["turn" if a.street == 1 else "river"])
            total = int(sum(np.prod(v.shape) for layer in params for v in layer.values()))
            k = 2
            used = ((52 + CR.PUB + 2 * k * H) * a.hidden[0] + a.hidden[0]
                    + sum(i * o + o for i, o in zip(a.hidden[:-1], a.hidden[1:]))
                    + a.hidden[-1] * k * H + k * H)
            rec["critic"] = f"net_critic random init hidden {a.hidden}"
            rec["critic_params_total"] = total
            rec["critic_params_used_k2"] = int(used)
    cfg = SV.SolverConfig(cap=a.iters, e_target=0.0, check_every=50)

    # ---- whole solve, as the loop calls it (poker/solver/loop.py batch_solver: pcfr.solve_fixed)
    if not a.no_solve:
        c0, t0 = COMPILE[0], time.perf_counter()
        # warm-up: 100 iterations = 2 chunk calls; the second call sees the chunk's own output state, whose avals
        # differ from init_cfr_state's (weak types), and compiles the chunk a second time
        st1 = SV.solve_fixed(batch, st, cfg, 100, aux=aux)
        block(st1["z"])
        rec["first_chunk_wall_s"] = time.perf_counter() - t0
        rec["solve_compile_s"] = COMPILE[0] - c0
        c1, t0 = COMPILE[0], time.perf_counter()
        res = SV.solve_fixed(batch, st, cfg, a.iters, aux=aux)
        block(res["z"])
        dt = time.perf_counter() - t0
        rec["compile_during_timed_solve_s"] = COMPILE[0] - c1
        rec["solve_s"] = dt
        rec["solve_s_per_spot"] = dt / a.batch
        rec["s_per_iter_batch"] = dt / a.iters
        rec["s_per_iter_per_spot"] = dt / a.iters / a.batch
        # loop's per-batch extras: one exploitability evaluation and the root values
        ex = jax.jit(jax.vmap(lambda z, s, x: SV.exploitability(z, s, st, cfg, x), in_axes=(0, 0, None)))
        root = jax.jit(jax.vmap(lambda z, s, x: SV.root_values(z, s, st, cfg, x), in_axes=(0, 0, None)))
        t_ex, r = timeit(ex, res["z"], batch, aux, reps=2)
        t_root, _ = timeit(root, res["z"], batch, aux, reps=2)
        e = np.asarray(r["nash_conv"]) / 2
        rec["exploitability_eval_s"] = t_ex
        rec["root_values_s"] = t_root
        rec["E_pot_after_T"] = {"median": float(np.median(e)), "max": float(np.max(e)), "min": float(np.min(e))}
        rec["rss_after_solve_mb"] = rss_mb()

    # ---- per-iteration breakdown on the same batch (vmapped over spots, jitted separately)
    state = jax.vmap(lambda _: SV.init_cfr_state(st, jnp.float32))(jnp.arange(a.batch))
    state = {**state, "regret": jax.random.uniform(jax.random.PRNGKey(1), state["regret"].shape)}   # non-uniform policy
    lg = lambda s: s["legal"][st.internal]

    def pol(regret, last, spot):
        return SV.regret_policy(regret + last, lg(spot))

    f_policy = jax.jit(jax.vmap(pol))
    f_forward = jax.jit(jax.vmap(lambda pi, spot: SV.forward(pi, spot["ranges"], st)))
    f_term = jax.jit(jax.vmap(lambda reach, spot, x: SV.terminal_values(
        reach, spot["contrib"], SV.payout_pot(spot), spot.get("tables"), st, cfg, spot, x), in_axes=(0, 0, None)))
    f_back = jax.jit(jax.vmap(lambda term, pi, spot: SV.backward(term, pi, lg(spot), st)))

    def regret_from_vals(vals, reach, spot):
        legal = lg(spot)
        kids = np.where(st.topo.children[st.internal] >= 0, st.topo.children[st.internal], 0)
        actors = st.actor[st.internal]
        child = jnp.transpose(vals[kids, actors[:, None], :], (0, 2, 1))
        node = vals[st.internal, actors, :]
        return jnp.where(legal[:, None, :], child - node[:, :, None], 0.0), reach[st.internal, actors, :]

    def update(regret, avg, r, own, pi, t):
        return jnp.maximum(regret + r, 0.0), avg + t * t * own[:, :, None] * pi

    f_rfv = jax.jit(jax.vmap(regret_from_vals))
    f_update = jax.jit(jax.vmap(update, in_axes=(0, 0, 0, 0, 0, None)))
    one_chunk = SV.make_cfr_chunk(st, cfg)

    pi = block(f_policy(state["regret"], state["last"], batch))
    reps = a.reps
    out = {}
    out["policy_rm"], _ = timeit(f_policy, state["regret"], state["last"], batch, reps=reps)
    out["forward"], reach = timeit(f_forward, pi, batch, reps=reps)
    out["terminal_values"], term = timeit(f_term, reach, batch, aux, reps=reps)
    if len(st.leaves):
        f_leaf = jax.jit(jax.vmap(lambda reach, spot, x: SV.leaf_values(reach[st.leaves], st.leaves, spot, st, x),
                                  in_axes=(0, 0, None)))
        out["terminal_values.leaf_values(critic)"], _ = timeit(f_leaf, reach, batch, aux, reps=reps)
        if a.critic == "net":
            f_leaf_cd = jax.jit(jax.vmap(lambda reach, spot, x: SV.leaf_values(reach[st.leaves], st.leaves, spot, st, x),
                                         in_axes=(0, 0, None)))
            cd = jax.tree_util.Partial(lambda board, pub, ranges, f, spot: f)
            out["terminal_values.leaf_values(checkdown, same batch)"], _ = timeit(f_leaf_cd, reach, batch, cd, reps=reps)
    if len(st.allins):
        f_allin = jax.jit(jax.vmap(lambda reach, spot: SV.allin_values(
            reach[st.allins], spot["contrib"][st.allins], SV.payout_pot(spot)[st.allins], spot["equity"], st)))
        out["terminal_values.allin_values"], _ = timeit(f_allin, reach, batch, reps=reps)
    out["backward"], vals = timeit(f_back, term, pi, batch, reps=reps)
    out["regret_from_values"], (r, own) = timeit(f_rfv, vals, reach, batch, reps=reps)
    out["regret_avg_update"], _ = timeit(f_update, state["regret"], state["avg"], r, own, pi, jnp.float32(3.0), reps=reps)
    out["sum_of_parts"] = sum(v for k, v in out.items() if "." not in k)
    # one fused 50-iteration chunk, per iteration (what the solve really runs)
    t_chunk, _ = timeit(one_chunk, state, batch, aux, reps=max(1, reps // 5))
    out["fused_iteration(chunk/50)"] = t_chunk / cfg.check_every
    rec["breakdown_s_per_iter_batch"] = out
    rec["peak_rss_mb"] = rss_mb()
    rec["compile_total_s"] = COMPILE[0]
    line = json.dumps(rec)
    print(line)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as fh:
            fh.write(line + "\n")


if __name__ == "__main__":
    main()
