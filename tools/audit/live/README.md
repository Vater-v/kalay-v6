# Audit scripts: live play path and Slumbot runner (area "live")

Run from the repo root with
`POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python3 tools/audit/live/<script>.py`.
Nothing here edits `poker/` or `tools/` outside this folder. Results go to `results/audit/live/`.

| script | what it does | output |
|---|---|---|
| `slumbot_ref.py` | independent re-implementation of Slumbot's `ParseAction` (sample_api.py) and a brute-force 7-card evaluator | (library) |
| `fake_slumbot.py` | simulated Slumbot server (new_hand / act, 'Ac'-style cards, random legal bot incl. off-menu sizes, limps, all-ins, folds, `///` runouts, winnings) | (library) |
| `test_protocol.py` | 19 tests: realistic responses, malformed cards, card mapping vs the engine evaluator, seat mapping, 20k-hand fuzz of `parse_history` vs the reference | `protocol_tests.json` |
| `e2e_stub.py [HANDS] [SEED]` | the real path run_match -> parse_response -> LiveSlumbotBot.choose -> engine replay -> LivePlayer -> build_tree/SLOT -> serialize_action against the fake server; only the solve is a random stub | `e2e_stub.json` |
| `budget_repro.py` | the production runner (parallel.run_parallel) with the REAL river solver behind the 4 s budget on CPU; then a resume attempt | `budget_repro.json`, `budget_run/` |
| `timing.py flop|river [--trivial-critic] [--mt]` | wall clock of solve_batch at the play T, batch 1, cold/warm, tree build, spot_batch, RSS | `timing_*.json` |
| `timing_pre_turn.py` | preflop and turn tree sizes and per-iteration cost (T=100, extrapolated) | `timing_pre_turn.json` |
| `hlo_count.py` | fused kernels per PCFR+ iteration in the compiled loop body (CPU backend), input to a launch-bound GPU estimate | `hlo_count.json` |
| `river_floor.py` | range-only re-solve with a wrong carried range: what the 1e-3 likelihood floor changes (local loss on the river) | `river_floor.json` |
| `slots.py` | SLOT insertion at exact chips, own off-menu size, limp, raises past the cap, 3 players (3-player rows did not complete here) | `slots.json` |

Critics, the preflop table, the flop class cache, rank7 and the preflop matrix do not exist on this box: flop/turn/preflop
timings use a randomly initialised critic of the training width (512, 512, 512) and a random all-in matrix (same shapes
and FLOPs, meaningless values). The river is exact.
