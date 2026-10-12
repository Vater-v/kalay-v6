"""Live heads-up duplicate matches; both seats play the current critic-backed solver.

Every pair uses the same cards, depth and rake with seats swapped. The required --critics
checkpoint directory and --preflop-table must belong to the current game and the same pre_end.
Table depths and rake follow the regime unless --stack-bb fixes the depth.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config, guards
from poker.envs.betting import init

DEPTH_STREAM = 29
_DEALERS = {}


def table_depths(stack_bb=None):
    """(sample, max_stack) of a live match: sample(rng, count) -> int32 depth in BB of every
    deal (both seats the same) - stack_bb if given, else the regime's depth mix (config.DEPTH_MIX,
    config.sample_depths_bb) - and the deepest stack in chips (static: the history capacity, one
    compiled shape for every batch). Fixed depths only require engine validity; depth_record reports training coverage."""
    if stack_bb is not None:
        depth = int(stack_bb)
        if depth != stack_bb or depth < 2:
            raise ValueError(f"--stack-bb = {stack_bb!r}: a table starts above the big blind, in whole BB (>= 2)")
        max_stack = depth * config.BB
        try:
            config.history_bound(2, start=max_stack)       # host: refuse overflow before the draw or chip cast
        except ValueError as error:
            raise ValueError(f"--stack-bb = {stack_bb!r}: {error}") from None
        return (lambda rng, count: np.full(count, depth, np.int32)), max_stack
    return config.sample_depths_bb, config.depth_range_bb()[1] * config.BB


def depth_record(stack_bb, depths):
    """What a result records of its depths: the fixed one or the regime's mix, with the dealt spread."""
    depths = np.asarray(depths)
    return {"stack_bb": stack_bb if stack_bb is not None else "regime depth mix",
            "depth_mix": None if stack_bb is not None else [list(r) for r in config.DEPTH_MIX],
            "depth_bb_min": int(depths.min()), "depth_bb_median": float(np.median(depths)),
            "depth_bb_max": int(depths.max()), "depth_coverage": config.depth_coverage(depths)}


def deal(keys, stacks, levels, max_stack):
    """Heads-up hands (cards dealt, blinds posted, history on): both seats stacks[i] chips, rake level
    levels[i]; max_stack (static chips) bounds every stack and sizes the history. Jitted once per
    max_stack (an eager vmap dispatched init op by op on every batch)."""
    max_stack = int(max_stack)
    guards.deal_stacks(stacks, max_stack, "match.deal")          # host: the jitted vmap below cannot refuse
    guards.rake_levels(levels, "match.deal")
    if not len(keys) == len(stacks) == len(levels):
        raise ValueError(f"match.deal: {len(keys)} keys, {len(stacks)} stacks, {len(levels)} rake levels")
    if max_stack not in _DEALERS:
        _DEALERS[max_stack] = jax.jit(jax.vmap(lambda k, s, lv: init(k, 2, jnp.full(2, s, jnp.int32), history=True,
                                                                      max_stack=max_stack, rake_level=lv)))
    return _DEALERS[max_stack](keys, jnp.asarray(stacks, jnp.int32), jnp.asarray(levels, jnp.float32))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--critics", type=Path, required=True, help="critics of the current game")
    p.add_argument("--preflop-table", type=Path, required=True, help="canonical table generated from these critics")
    p.add_argument("--flop-cache", type=Path, default=Path("results/cache/flop_equity_v1"))
    p.add_argument("--rank7", type=Path, default=Path("results/cache/rank7_v1.npy"))
    p.add_argument("--preflop-equity", type=Path, default=Path("results/cache/preflop_equity_v1.npy"))
    p.add_argument("--threads", type=int, default=8, help="hands in flight: the solve server batches their solves")
    p.add_argument("--window", type=float, default=0.3, help="the trap collector's window in seconds")
    p.add_argument("--dev-size", type=int, default=0, help="developer toggle: the trap waits for N requests")
    p.add_argument("--budget", type=float, default=4.0, help="wall-clock budget per decision in seconds")
    p.add_argument("--live-t", type=int, help="one T for every street (default config.PLAY_ITERATIONS)")
    p.add_argument("--jax-cache", type=Path, help="the persistent compilation cache base (default config.JAX_CACHE_DIR)")
    p.add_argument("--deals", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stack-bb", type=int, help="every deal at this depth (default: the regime's depth mix; extrapolation is reported)")
    p.add_argument("--no-gadget", action="store_true", help="Bayesian opponent ranges instead of the re-solving gadget (comparisons only)")
    p.add_argument("--gadget-deal", default="mix", choices=("mix", "range", "uniform"), help="the opponent's deal under the gadget")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    try:
        table_depths(args.stack_bb)       # validate the engine input before artifacts and solver warm-up
    except ValueError as error:
        p.error(str(error))
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    from poker.arena.live_match import LiveArena, live_duplicate
    iterations = None if args.live_t is None else {s: args.live_t for s in range(4)}
    arena = LiveArena(critics_dir=args.critics, preflop_table=args.preflop_table, flop_cache=args.flop_cache, rank7=args.rank7,
                      preflop_equity=args.preflop_equity, iterations=iterations, chunk=min(50, args.live_t or 50),
                      window_seconds=args.window, dev_size=args.dev_size, budget_seconds=args.budget, log=log,
                      jax_cache=args.jax_cache, engine_max_stack=table_depths(args.stack_bb)[1], gadget=not args.no_gadget, gadget_deal=args.gadget_deal)
    log(f"live duplicate: {args.deals} deals x 2, {args.threads} threads, T {arena.metadata['iterations']}, window {args.window} s, "
        f"budget {args.budget} s, device {arena.metadata['device']}, turn matrix {arena.metadata['turn_matrix']}")
    log(arena.warm_up())                                                   # the ready gate
    if not arena.ready:
        raise SystemExit("live arena: not ready after the warm-up")
    try:
        res = live_duplicate(arena, args.deals, args.seed, stack_bb=args.stack_bb, threads=args.threads, log=log)
    finally:
        arena.close()
    res.update(a="live", b="live")
    log(f"live duplicate: {res['hands']} hands, failures {res['failed_hands']}, pair sum {res['bb_per_100_pair_sum']} bb/100 "
        f"(+-{res['ci95']}), decisions {res['decision_seconds']}, within budget {res['within_budget']}, server batches "
        f"{res['server']['batches'][:16]}..., errors {res['server']['errors']}")
    print(json.dumps(res))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
