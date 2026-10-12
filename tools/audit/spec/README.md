# spec-area audit scripts (engine vs docs/poker-core-mechanism.md, GPU-first, multiway)

Read-only with respect to `poker/` and `tools/` outside this folder. Results: `results/audit/spec/`.
Run from the repo root:

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" python3 tools/audit/spec/<script>

| Script | What it shows | Output | Wall (this box) |
|---|---|---|---|
| `spec_probe.py` | regime / menu facts; teacher-tree sizes for 2, 3, 4, 6 players and whether `pcfr.Static` accepts them; the 3+/4+ gates (`mass_of`, `critics.opponent_mass`, `project_values` on 4 live slots); kappa under regime 1; the likelihood floor's re-admitted posterior mass (one update and chained); critic parameter count and CPU forward cost; which files mention ceilings / gadget / RNR / coalitions | `spec_probe.json` | ~1 min |
| `carried_range_demo.py` | the real `LivePlayer` on a river spot, pot 20 BB, SPR 5: hero holds bluff-catchers, opponent bets pot (off-menu, goes into SLOT). Two runs that differ only in the carried opponent range (air at the likelihood-floor weight vs the true range). Reports the hero's call frequency, the air's gain, the player's state attributes (no ceiling state) | `carried_range_demo.json` | ~40 s |
| `pipeline_costs.py` | rental-pipeline inputs present / missing, bootstrap pins, wall budgets and their dollar value at the spec's $0.5-1.5/GPU-h, the gate's host-RAM rule against 16 GB, critic row bytes (9 slots fp32 vs 2 slots fp16), live (SLOT) vs teacher tree sizes | `pipeline_costs.json` | ~30 s |

Numbers taken from other auditors' runs are cited by file in REPORT.md (`results/audit/engine-profile`,
`results/audit/solver`, `results/audit/training`); they are theirs, not re-measured here.
