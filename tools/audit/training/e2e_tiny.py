"""(6) Tiny end-to-end run of the real loop (poker.solver.loop.main) on this CPU, then the learning curve.

Declared deviations from the production loop (all in this process only, nothing on disk under poker/):
- rank7.TABLE_PATH -> a scratch path (built there once, ~2-3 min; results/cache is left untouched);
- config.TEACHER_T lowered (river 200, turn 100, flop 100 PCFR+ iterations instead of 800/1500/1500)
  so the stages fit the shared box; the layout records the values used;
- git disabled (fingerprint 'files' mode).
Usage: python e2e_tiny.py OUT_DIR STAGE [STAGE ...]   (stages as loop.STAGES; 'curve' = learning_curve on river)
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402
from pathlib import Path  # noqa: E402

loop = common.patch_loop()
from poker.envs import config  # noqa: E402
from poker.solver import rank7  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-kalay-v6/721b9d69-ee04-59bb-a6cc-2f7cda70f7ae/scratchpad/cache")
SCRATCH.mkdir(parents=True, exist_ok=True)
rank7.TABLE_PATH = SCRATCH / "rank7_v1.npy"
loop.rank7.TABLE_PATH = rank7.TABLE_PATH
config.TEACHER_T.update({"river": 200, "turn": 100, "flop": 100})

out = sys.argv[1]
stages = sys.argv[2:]
t0 = time.perf_counter()
if stages == ["curve"]:
    from poker.solver import learning_curve as LC
    LC.main(["--results", out, "--street", "river", "--seeds", "1", "--epochs", "20", "--epoch-steps", "25",
             "--out", str(Path(out) / "learning_curve_river.json")])
else:
    loop.main(["--out", out, "--quick", "--fixed-t", "--stages", *stages])
print(f"AUDIT wall {time.perf_counter() - t0:.1f} s for {stages}", flush=True)
