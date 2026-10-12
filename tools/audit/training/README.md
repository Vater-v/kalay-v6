# Training-area audit scripts

Read-only checks of `poker/solver/{loop,critics,situations,learning_curve,io,sizing_lab}.py`.
Nothing under `poker/` or `tools/` (other than this directory) is modified. `common.py` sets
single-threaded CPU flags and replaces `poker.envs.fingerprint._from_git` / `loop.code_version` in
the audit process so the engine never runs git. Results go to `results/audit/training/`.

Run from the repo root:

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=/home/user/kalay-v6 OPENBLAS_NUM_THREADS=1 python3 tools/audit/training/<script>

| script | what it measures | output |
|---|---|---|
| `arch_and_projection.py` | critic parameter count / FLOPs (HU slice of the P_MAX=9 net), CPU forward cost, and what `critics.project_values` does to errors (W-orthogonality, per-slot error transfer) on synthetic values | `arch_projection.json` |
| `river_label_cost.py N T` | per-spot river label cost with the loop's trees/spots/solver at fixed T and 2T, E at T, label-noise floor (range-weighted EV(T) vs EV(2T)) | `river_label_cost_T{T}_n{N}.json` |
| `aux_identity.py` | whether turn_end / flop_end labels are exact card averages with the right card-removal weights (independent pure-python evaluator for turn->river; brute-force averaging for flop->turn) | `aux_identity.json` |
| `rank7_cost.py` | cost of building the rank7 cache the loop blocks on at start (extrapolated from one chunk) | `rank7_cost.json` |
| `turn_cost.py N IT` | turn solve: ms/iteration with check-down leaves vs 64x64 vs 3x512 critics (where the time goes), turn prepare without the rank7 cache | `turn_cost_n{N}_it{IT}.json` |
| `e2e_tiny.py OUT STAGES` | the real `loop.main --quick --fixed-t` with T lowered to 200/100/100 and rank7 in a scratch path; `curve` runs `learning_curve.main` on the result | logs + `OUT/*.json`, copied summaries in `e2e_*.json` |

Declared deviations: lowered T in `e2e_tiny.py` (labels noisier than production); timings on a
shared 4-core box with load average 4-8 (recorded where measured), so absolute CPU timings are upper
bounds for an idle core.
