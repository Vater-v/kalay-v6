"""Run poker.solver.loop's river stage end-to-end (the user's own entry point), optionally skipping the
rank7 table build that loop.main performs before any stage when results/cache/rank7_v1.npy is absent
(poker/solver/loop.py: `if args.stages and not rank7.TABLE_PATH.exists(): rank7.load_table(log=log)`).

The river stage itself never reads the table (river rank tables come from cards.hand_ranks via
situations._tables); skipping the build only stops loop.main from spending the build time and
writing the 268 MB table under results/cache. Nothing in poker/ is edited: the patch is a module
attribute in this process.

Usage (from /home/user/kalay-v6):
  POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. timeout 600 python3 tools/audit/engine-profile/run_loop_river.py \
      [--skip-rank7-build | --rank7-dir <scratch dir>] -- --out <dir> --quick --stages river_solve
"""

import atexit
import resource
import runpy
import signal
import sys

import functools
import time
from pathlib import Path

argv = sys.argv[1:]
skip = "--skip-rank7-build" in argv
argv = [a for a in argv if a != "--skip-rank7-build"]
rank7_dir = None
if "--rank7-dir" in argv:                       # build the table as loop.main does, but into this directory
    i = argv.index("--rank7-dir")
    rank7_dir = Path(argv[i + 1])
    del argv[i:i + 2]
if argv and argv[0] == "--":
    argv = argv[1:]

if rank7_dir is not None:
    from poker.solver import rank7
    rank7.TABLE_PATH = rank7_dir / "rank7_v1.npy"
    _orig = rank7.load_table

    def _timed_build(*a, **k):
        t0 = time.perf_counter()
        out = _orig(rank7.TABLE_PATH, log=k.get("log", print))
        print(f"[audit] rank7 table built in {time.perf_counter() - t0:.1f} s -> {rank7.TABLE_PATH}", flush=True)
        return out

    rank7.load_table = _timed_build

if skip:
    from poker.solver import rank7

    def _no_build(*a, **k):
        print("[audit] rank7.load_table skipped (table absent; the river stage does not read it)", flush=True)
        return None

    rank7.load_table = _no_build

T0 = time.perf_counter()


def _report(*_):
    print(f"[audit] wall {time.perf_counter() - T0:.1f} s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)


atexit.register(_report)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(124))     # `timeout` -> SystemExit -> atexit report

sys.argv = ["poker.solver.loop"] + argv
runpy.run_module("poker.solver.loop", run_name="__main__", alter_sys=True)
