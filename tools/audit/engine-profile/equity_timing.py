"""Equity-matrix and hand-ranking costs on this CPU without the rank7 table (it is not shipped).

Usage (from /home/user/kalay-v6, one core):
  POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 taskset -c 2 \
      python3 tools/audit/engine-profile/equity_timing.py [--flop] [--out results/audit/engine-profile/equity.json]

Times: cards.hand_ranks (one river board, 1326 evaluator calls), situations._tables (rank tables),
runout.equity_matrix_turn(board4, table=None) (the evaluator fallback: 52 x 1326 evaluations),
runout.equity_matrix_flop(board3, table=None) for ONE canonical flop (1326 runouts; --flop),
and rank7._scores on a chunk of combinadic indices -> projected rank7 build time.
Nothing is written under results/cache (flop_equity_cached is not called).
"""

import argparse
import json
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

COMPILE = [0.0]
jax.monitoring.register_event_duration_secs_listener(
    lambda e, d, *a, **k: COMPILE.__setitem__(0, COMPILE[0] + d) if e.startswith("/jax/core/compile") else None)

from poker.solver import cards as C          # noqa: E402
from poker.solver import flops as FL         # noqa: E402
from poker.solver import rank7               # noqa: E402
from poker.solver import runout as RO        # noqa: E402
from poker.solver import situations as SI    # noqa: E402


def timed(fn, *a):
    c0, t0 = COMPILE[0], time.perf_counter()
    out = jax.block_until_ready(fn(*a))
    return out, time.perf_counter() - t0, COMPILE[0] - c0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flop", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    rec = {"rank7_table_present": rank7.TABLE_PATH.exists(), "cached_table_is_none": rank7.cached_table() is None}
    board5 = jnp.asarray([0, 13, 27, 40, 51], jnp.int32)
    hr = jax.jit(C.hand_ranks)
    _, t1, c1 = timed(hr, board5)
    _, t2, _ = timed(hr, jnp.asarray([1, 14, 28, 41, 50], jnp.int32))
    rec["hand_ranks_one_board"] = {"first_s": t1, "compile_s": c1, "steady_s": t2, "evals_per_s": 1326 / t2}
    boards = jnp.asarray(np.stack([np.random.default_rng(i).choice(52, 5, replace=False) for i in range(16)]), jnp.int32)
    _, t1, c1 = timed(SI._tables, boards)
    _, t2, _ = timed(SI._tables, boards + 0)
    rec["rank_tables_16_boards"] = {"first_s": t1, "compile_s": c1, "steady_s": t2, "per_board_s": t2 / 16}

    b4 = jnp.asarray([0, 13, 27, 40], jnp.int32)
    w, t1, c1 = timed(RO.equity_matrix_turn, b4, None)
    _, t2, _ = timed(RO.equity_matrix_turn, jnp.asarray([2, 15, 29, 44], jnp.int32), None)
    rec["equity_matrix_turn_evaluator"] = {"first_s": t1, "compile_s": c1, "steady_s": t2}
    # brute-force spot check of the matrix (runout.check_turn_matrix: evaluator over the 44 rivers)
    chk = RO.check_turn_matrix(np.asarray(b4), samples=5)
    rec["turn_matrix_check_max_abs_err"] = float(max(abs(x - y) for x, y, _ in chk))

    # rank7 build projection: _scores over 2^16 combinadic indices
    idx = jnp.arange(1 << 16, dtype=jnp.int32)
    _, t1, c1 = timed(rank7._scores, idx)
    _, t2, _ = timed(rank7._scores, idx + (1 << 16))
    rec["rank7_scores_65536"] = {"first_s": t1, "compile_s": c1, "steady_s": t2, "evals_per_s": 65536 / t2,
                                 "projected_full_build_s": rank7.N7 / (65536 / t2)}

    if a.flop:
        flop = jnp.asarray(FL.FLOPS[0], jnp.int32)
        w, t1, c1 = timed(RO.equity_matrix_flop, flop, None)
        rec["equity_matrix_flop_evaluator"] = {"flop": FL.FLOPS[0].tolist(), "first_s": t1, "compile_s": c1,
                                               "run_s_excl_compile": t1 - c1}
        chk = RO.check_flop_matrix(np.asarray(flop), samples=2)
        rec["flop_matrix_check_max_abs_err"] = float(max(abs(x - y) for x, y, _ in chk))
    rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(rec, indent=1))
    if a.out:
        a.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
