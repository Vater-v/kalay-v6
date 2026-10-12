# engine-profile (audit experiment)

Where the engine's compute goes on this CPU (4 cores, jax 0.11.2 CPU, no GPU, no caches shipped),
and whether the river data-generation path of `poker.solver.loop` runs. Read-only with respect to
`poker/` and `tools/` (outside this folder); nothing is written under `results/cache`.

Environment for every script (from `/home/user/kalay-v6`):

    export POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1

| Script | What it measures | Output |
|---|---|---|
| `profile_solve.py` | One tree (street, menu), one batch size, one process: `situations.prepare`, compile, steady `pcfr.solve_fixed` (T iterations), the loop's per-batch extras (exploitability, root values), peak RSS, and a per-iteration breakdown from separately jitted, vmapped sub-functions (`regret_policy`, `forward`, `terminal_values` with its `leaf_values`/`allin_values` parts, `backward`, regret-from-values, regret/average update) plus one fused 50-iteration chunk. Flop/turn: `--critic checkdown` (a `jax.tree_util.Partial` returning F) or `--critic net` (`critics.net_critic(critics.init_critic(key, (512,512,512), k=2))`). | one JSON line appended to `--out` |
| `run_river.sh [core]` | River sweep, 'normal' B = 1, 4, 16 (T = 800), 32 (T = 400), 'short' B = 1, 16, pinned to one core | `results/audit/engine-profile/river.jsonl` |
| `run_flop_turn.sh [core]` | Flop 'normal' (B = 1, 4; check-down and net critic) and turn 'normal' (B = 4), T = 500, one T = 1500 check | `results/audit/engine-profile/flop_turn.jsonl` |
| `equity_timing.py [--flop]` | `cards.hand_ranks`, rank tables, `runout.equity_matrix_turn` and `equity_matrix_flop` with `table=None` (the evaluator fallback; the rank7 table is absent), brute-force spot checks of both matrices, and `rank7._scores` throughput -> projected rank7 build | `results/audit/engine-profile/equity.json` |
| `run_loop_river.py` | Runs `python -m poker.solver.loop` unchanged via runpy; `--rank7-dir D` lets `loop.main` build the rank7 table as it always does when absent, but into `D` (timed) instead of `results/cache`; `--skip-rank7-build` skips it | loop's own files in `--out` |
| `summarise.py` | Builds the tables of the report from the JSON files | `results/audit/engine-profile/summary.json` |

Pin with `taskset -c <core>` for per-core numbers. The box is shared (load average 4-6 during the
runs): absolute times carry contention noise of perhaps +-30 %; ratios inside one run are firmer.
