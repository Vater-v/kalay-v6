# Audit scripts: `poker/envs` (game engine)

Read-only checks of the engine; nothing under `poker/` is modified. Run from the repo root with

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python3 tools/audit/envs/<script> ...

Results go to `results/audit/envs/` (small JSON); the report is `results/audit/envs/REPORT.md`.

| script | what it does | wall time here |
|---|---|---|
| `eval_check.py [OUT]` | independent best-of-21 five-card evaluator vs `evaluator.evaluate_hand`: 30k random hands, all 49 205 seven-card rank multisets, 20k flush-heavy, 3k straight-flush, 19 adversarial, 4k board-plays pairs; checks the score is an order isomorphism of the reference key | 16 s |
| `ref_engine.py` | plain-Python NLHE reference (blinds, min-raise, short all-in reopening, side pots, uncalled bets, odd chips, rake) written from the rules, not from the engine | - |
| `betting_check.py HANDS SEED OUT [CALLP]` | differential random play, 2-6 seats, unequal odd stacks, empty seats, random rake: state/legality at every decision, then net profits and rake | 60-90 s per 1000 hands |
| `mutation_check.py HANDS OUT` | re-runs the differential test with deliberately wrong reference rules to show it detects them | 1 min |
| `scenarios.py OUT` | hand-written rule cases (HU blinds, first actor, min-raise after a short all-in, cumulative reopening, all-in for less, rake cases, rake int32 overflow) | 20 s |
| `history_check.py OUT` | adversarial min-raise wars vs `config.history_bound` / capacity; shows the silent stall at capacity | 40 s |
| `illegal_step.py OUT` | what `step` / `step_target` do with an illegal action, with and without checkify / `guards.target` | 10 s |
| `menu_probe.py OUT` | the grid-13 size menus and the commit-merge rule at regime-1 (HU 200 BB) spots; merge sweep over SPR; opponent side | 15 s |
| `menu_bounds_check.py OUT` | `menu_for` vs float32 `menu_index` at the bound edges | 5 s |
| `perf.py OUT` | jitted `step`, `step_target`, `action_table`, `get_rewards`, `evaluate_hand`: single call and vmap batches | 3.5 min |
| `py313_check.py OUT` | compiles every file of `poker/` and `tools/` and imports every `poker.*` module under the local Python 3.13 | 35 s |
