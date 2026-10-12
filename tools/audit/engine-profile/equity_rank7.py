"""Turn / flop equity matrices WITH the rank7 table (built by run_loop_river.py --rank7-dir into a scratch dir),
on the device path (runout.equity_matrix_turn/_flop with the table) and the host path for the turn
(subgame.turn_equity, the CPU play path's builder). Checks both against the evaluator-path matrices.

Usage: POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. taskset -c 2 python3 tools/audit/engine-profile/equity_rank7.py \
           <dir with rank7_v1.npy> [out.json]
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.solver import flops as FL
from poker.solver import runout as RO
from poker.solver import subgame as SG


def t(fn, *a):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*a))
    return out, time.perf_counter() - t0


def main():
    path = Path(sys.argv[1]) / "rank7_v1.npy"
    host = np.load(path, mmap_mode="r")
    dev = jax.device_put(host)
    rec = {}
    b4 = jnp.asarray([0, 13, 27, 40], jnp.int32)
    w_tab, first = t(RO.equity_matrix_turn, b4, dev)
    _, steady = t(RO.equity_matrix_turn, jnp.asarray([2, 15, 29, 44], jnp.int32), dev)
    w_ev = np.asarray(RO.equity_matrix_turn(b4, None))
    rec["turn_device_rank7"] = {"first_s": first, "steady_s": steady,
                                "max_abs_diff_vs_evaluator": float(np.abs(np.asarray(w_tab) - w_ev).max())}
    w_host, th = t(SG.turn_equity, np.asarray(b4), host)
    _, th2 = t(SG.turn_equity, np.asarray([2, 15, 29, 44]), host)
    rec["turn_host_rank7_subgame.turn_equity"] = {"first_s": th, "steady_s": th2,
                                                   "max_abs_diff_vs_evaluator": float(np.abs(w_host - w_ev).max())}
    flop = jnp.asarray(FL.FLOPS[0], jnp.int32)
    w_f, tf = t(RO.equity_matrix_flop, flop, dev)
    _, tf2 = t(RO.equity_matrix_flop, jnp.asarray(FL.FLOPS[700], jnp.int32), dev)
    rec["flop_device_rank7"] = {"first_s": tf, "steady_s": tf2}
    print(json.dumps(rec, indent=1))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
