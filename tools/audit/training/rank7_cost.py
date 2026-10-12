"""What the loop blocks on at start (loop.py:1094-1095): rank7.load_table builds all C(52,7) = 133,784,560
seven-card classes when results/cache/rank7_v1.npy is missing. Time one chunk of the builder's own
_scores (1<<20 hands) on this CPU and extrapolate (the builder runs 128 such chunks).
Output: results/audit/training/rank7_cost.json"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402
import jax, jax.numpy as jnp, numpy as np  # noqa: E402
from poker.solver import rank7  # noqa: E402

chunk = 1 << 18
idx = jnp.asarray(np.arange(chunk, dtype=np.int32))
t0 = time.perf_counter(); np.asarray(rank7._scores(idx)); first = time.perf_counter() - t0
idx2 = jnp.asarray(np.arange(chunk, 2 * chunk, dtype=np.int32))
t0 = time.perf_counter(); np.asarray(rank7._scores(idx2)); warm = time.perf_counter() - t0
per = warm / chunk
common.dump("rank7_cost.json", {"chunk": chunk, "first_s_incl_compile": first, "warm_s": warm,
                                "extrapolated_build_minutes_single_thread": per * rank7.N7 / 60,
                                "code_claim": "~10 min on a GTX 1070 (rank7.py:53)",
                                "table_mb_int16": rank7.N7 * 2 / 1e6})
print(per * rank7.N7 / 60, "min")
