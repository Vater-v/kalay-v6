"""Rental pipeline assumptions, budgets and the storage / memory arithmetic of the critic data, from the code's own
constants (tools/cloud/launch.py, poker/solver/critics.py) and the live topologies' sizes (poker/solver/subgame.py).
No network, no GPU. Output: results/audit/spec/pipeline_costs.json
    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python3 tools/audit/spec/pipeline_costs.py
"""
import json, os, re, shutil, time
from pathlib import Path
import numpy as np

ROOT = Path(".").resolve()
out = {}

# 1. what the pipeline needs that is not in this tree / on this machine
need = {
    "requirements.lock (bootstrap.sh installs it exactly)": ROOT / "requirements.lock",
    "checks/run.py (pack.sh --checks, gate.py start: CHECKS_SUMMARY verdict pass)": ROOT / "checks" / "run.py",
    "~/clore_tools/box.sh (fetch.sh, watch.sh, release.sh)": Path.home() / "clore_tools" / "box.sh",
    "~/clore_tools/clore.py (create / cancel orders)": Path.home() / "clore_tools" / "clore.py",
    "results/pack/JOBS.json (manifest; launch.py manifest)": ROOT / "results" / "pack" / "JOBS.json",
    "tools/cloud/cache_sha256.json": ROOT / "tools" / "cloud" / "cache_sha256.json",
    "tools/cloud/flop_equity_v1.sha256": ROOT / "tools" / "cloud" / "flop_equity_v1.sha256",
    "results/cache/rank7_v1.npy": ROOT / "results" / "cache" / "rank7_v1.npy",
    "results/cache/preflop_equity_v1.npy": ROOT / "results" / "cache" / "preflop_equity_v1.npy",
}
out["pipeline_inputs_present"] = {k: p.exists() for k, p in need.items()}
out["binaries"] = {b: shutil.which(b) is not None for b in ("git", "uv", "nvidia-smi", "tmux", "cargo")}
boot = (ROOT / "tools/cloud/bootstrap.sh").read_text()
out["bootstrap_pins"] = {k: re.search(rf"^{k}=(\S+)", boot, re.M).group(1) for k in ("UV_VERSION", "PYTHON_VERSION")}

# 2. budgets (wall seconds, not money) and what they cost at the spec's GPU-hour calibration
import importlib.util
spec = importlib.util.spec_from_file_location("launch", ROOT / "tools/cloud/launch.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
chain = ["bootstrap", "caches", "preflight", "k0_smoke", "k0"]
hours = sum(L.BUDGET_S[k] for k in chain) / 3600
out["budgets"] = {"BUDGET_S": L.BUDGET_S, "PRODUCTION_spots": L.PRODUCTION, "QUICK_spots": L.QUICK,
                  "worst_case_hours_bootstrap_to_k0": hours,
                  "worst_case_usd_at_0.5_1.5_per_gpu_hour_one_gpu": [round(hours * 0.5, 2), round(hours * 1.5, 2)],
                  "river_shard_hours": L.BUDGET_S["river_shard"] / 3600,
                  "money_in_gate_py": bool(re.search(r"usd|dollar|price|\$/h|cost_per", (ROOT / "tools/cloud/gate.py").read_text(), re.I))}
res = L.RESOURCES
def host_gb(workers, spots_river):
    rows_gb = spots_river * res["critic_kb_per_spot"]["river"] / 1e6
    return workers * res["host_base_gb"] + res["critic_ram_factor"] * rows_gb
out["host_ram_rule"] = {"k0_production_1_worker_gb": host_gb(1, L.PRODUCTION["river"]),
                        "river_45k_1_worker_gb": host_gb(1, 45000),
                        "river_100k_1_worker_gb": host_gb(1, 100000),
                        "home_ram_gb": 16, "gpu_mib_min": res["gpu_mib_min"], "gtx1070_mib": 8192}

# 3. critic row storage
from poker.solver import critics as CR
H, P = CR.NUM_HANDS, CR.P_MAX
row_bytes = 4 * (52 + CR.PUB + 3 * P * H + H + P)
hu_fp16 = 2 * (3 * 2 * H) + 52 // 8 + 4 * 8
out["rows"] = {"bytes_per_row_P_MAX_fp32": row_bytes, "zero_fraction_of_slot_arrays_in_HU": round(1 - 2 / P, 3),
               "bytes_per_row_HU_2slots_fp16": hu_fp16, "ratio": round(row_bytes / hu_fp16, 1),
               "GB_per_1e6_rows_as_stored": round(row_bytes * 1e6 / 1e9, 1),
               "GB_per_1e6_rows_2slot_fp16": round(hu_fp16 * 1e6 / 1e9, 1),
               "np_savez_compressed": "np.savez (uncompressed) in poker/solver/io.py:36-40"}

# 4. labels per dollar from the code's recorded GPU throughput (tools/cloud/launch.py:61, loop.py:77-84)
gpu = {"GTX1070_river_s_per_spot": 0.86, "GTX1070_turn_flop_s_per_spot": 1.7, "RTX4090_grid7_river_spm": 197.1}
out["labels_from_recorded_gpu_speed"] = {
    **gpu,
    "1070_river_labels_per_day": round(86400 / 0.86), "1070_turn_or_flop_labels_per_day": round(86400 / 1.7),
    "k0_production_total_solved_spots": L.PRODUCTION["river"] + L.PRODUCTION["turn"] + L.PRODUCTION["flop"],
    "spec_gate_labels": "1e3-1e4 exact (48 river solves each) + 1e4-1e5 single-card; budget buys 1e4-1e5 exact or ~1e6 single-card per rung (spec 6.3)"}

# 5. live topologies (SLOT columns) vs the teacher's trees, HU, normal menu
from poker.solver import subgame as SG, tree as TR
live = {}
for street in (3, 2, 1, 0):
    for hero in (0, 1):
        t0 = time.perf_counter()
        s = SG.shape_for(street, "normal", 2, hero)
        live[f"street{street}_hero{hero}"] = {"nodes": int(s.size), "decisions": int(len(s.internal)),
                                              "leaves": int(len(s.st.leaves)), "columns": int(s.kinds.shape[1]),
                                              "build_s": round(time.perf_counter() - t0, 2)}
        print(street, hero, live[f"street{street}_hero{hero}"], flush=True)
teacher = {st: int(TR.rule_topology(2, st, "normal").size) for st in (3, 2, 1, 0)}
out["live_vs_teacher_nodes"] = {"live": live, "teacher": teacher}
Path("results/audit/spec/pipeline_costs.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "live_vs_teacher_nodes"}, indent=1, default=str))
