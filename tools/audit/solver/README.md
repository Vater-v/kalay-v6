# solver audit scripts

Read-only with respect to `poker/` and `tools/` (outside this folder). Nothing is written under `results/cache`
(flop matrices go to the scratchpad via `common.no_cache_write`). Results: `results/audit/solver/`.

Environment (from `/home/user/kalay-v6`):

    export POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1

| Script | What it checks / measures | Output |
|---|---|---|
| `indep.py` | Independent references: a numpy 7-card evaluator (best of 21 five-card subsets), the 1326 combos, explicit card-disjointness and dense score matrices. No `poker.solver` math. | library |
| `terminals.py [--x64]` | A: engine ranks vs independent evaluator (30 boards). B: `pcfr._two` (fold both ways, showdown equal/unequal contributions), `pcfr.j_mass`, `showdown.apply_all` vs dense O(F^2) brute force (4 river boards x 3 range families). C: `runout.equity_matrix_turn` and `pcfr.allin_values` vs explicit 44-river enumeration (3 turn boards); check-down identity and the spec's 1/44 form. D: `pcfr._three` for all 7 live patterns, `showdown.mass_three` vs brute force over all card-disjoint triples (200-hand supports, every hero hand); a side-pot case. | `terminals_f64.json`, `terminals_f32.json` |
| `turn3.py` | `pcfr.turn_runout_three` (3-player turn all-in, /42) vs brute force over 48 rivers x triples (x64). | `turn3_runout.json` |
| `rank7_index.py` | `rank7.hand_ranks_fast` combinadic indexing (table = arange) and `unrank` round trip. | `rank7_index.json` |
| `tree_stats.py` | `tree.rule_topology` counts (decisions, folds, showdowns / all-in runouts, leaves, actions) for HU streets x menus and 3-player postflop; build and replay times. | `tree_stats.json` |
| `dead_nodes.py` | Share of the fixed topology alive at real chips by SPR (`tree.replay` alive). | `dead_nodes.json` |
| `perf.py` | Kernel timings (apply_all vs dense matvec, per_card, mass_three, couple_live_folded, 3-way showdown) and PCFR+ ms/iteration via `pcfr.solve_fixed`: river 'normal' B = 1, 8, 32; flop 'normal' B = 1, 8 with a dummy critic returning F; 3-player river B = 1. Pinned with `taskset -c 2`. | `perf.json` |
| `converge.py --n 2/3` | `pcfr.solve_cfr` E (nash_conv / n, pot units) every 50 iterations on 4 HU river spots (SPR 0.5, 2, 8, 30) to 2000 iterations and one 3-player river spot to 500; `check_exploitability`; zero-sum identity. | `converge_hu_river.json`, `converge_3p_river.json` |

The box was shared (load average 5-9 on 4 cores during these runs): absolute times carry contention noise
(B = 1 times here are about 4x the pinned, lighter-load numbers in `results/audit/engine-profile`); ratios
within a run and the B = 32 figure agree with engine-profile.
