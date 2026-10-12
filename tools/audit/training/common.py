"""Shared setup for the training-area audit scripts (read-only use of poker/).

- Pins single-threaded CPU execution (shared box).
- Prevents the engine from invoking git: poker.envs.fingerprint._from_git is replaced by a stub
  that raises LookupError, so the fingerprint falls back to its documented "files" mode, and
  poker.solver.loop.code_version (if imported) returns a fixed no-git record.
Nothing under poker/ is modified on disk; the patches live only in the audit process.
"""
import os
os.environ.setdefault("POKER_REGIME", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import json
from pathlib import Path

from poker.envs import fingerprint as FP


def _no_git(root):
    raise LookupError("audit: git disabled")


FP._from_git = _no_git

RESULTS = Path(__file__).resolve().parents[3] / "results" / "audit" / "training"
RESULTS.mkdir(parents=True, exist_ok=True)


def patch_loop():
    from poker.solver import loop
    loop.code_version = lambda: {"commit": None, "dirty": True, "source": "audit-no-git"}
    return loop


def dump(name, obj):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    print("wrote", path)
    return path
