"""Tables and label-cost estimates from the engine-profile JSON files -> results/audit/engine-profile/summary.json.

Usage: python3 tools/audit/engine-profile/summarise.py   (from /home/user/kalay-v6; numpy only, no jax)

Estimates (all labelled as such in the report):
  river label (engine unit)   = one HU river spot at T = 800 (config.TEACHER_T), measured s/spot at the best batch size
                                on one core + steady prepare + the loop's one exploitability + root-values pass
  exact V4 label (spec unit)  = 48 such river solves (docs/poker-core-mechanism.md §6.3 'all 48 unseen river cards')
  turn label (engine unit)    = one HU turn spot at T = 1500 with the 512^3 net leaf (net_critic) + its turn matrix
  flop label (engine unit)    = one HU flop spot at T = 1500 with the net leaf + its flop matrix / 6.67 (300 flops for
                                2000 spots, poker/solver/loop.py FLOP_BOARDS and --flop default)
Scaling: E5-2650 v2 per core = 0.6-0.8 x this box (the brief's assumption), 8 processes in parallel;
GPU references are the code's own recorded numbers (cited in REPORT.md).
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[3] / "results" / "audit" / "engine-profile"


def load(name):
    p = R / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def row(d):
    b = d["breakdown_s_per_iter_batch"]
    parts = {k: v for k, v in b.items() if "." not in k and k not in ("sum_of_parts", "fused_iteration(chunk/50)")}
    tot = sum(parts.values())
    out = {"street": d["street"], "menu": d["menu"], "batch": d["batch"], "T": d["iters"],
           "critic": d.get("critic", "-"), "tree": d["tree"], "mean_legal": d["mean_legal_actions"],
           "s_per_spot_at_T": d["solve_s_per_spot"], "ms_per_iter_per_spot": 1000 * d["s_per_iter_per_spot"],
           "ms_per_iter_batch": 1000 * d["s_per_iter_batch"], "fused_ms_per_iter_batch": 1000 * b["fused_iteration(chunk/50)"],
           "compile_s": d["solve_compile_s"] + d.get("compile_during_timed_solve_s", 0.0),
           "prepare_first_s": d["prepare_s"], "prepare_compile_s": d["prepare_compile_s"],
           "exploitability_eval_s_per_spot": d["exploitability_eval_s"] / d["batch"],
           "root_values_s_per_spot": d["root_values_s"] / d["batch"], "peak_rss_mb": d["peak_rss_mb"],
           "E_pot": d["E_pot_after_T"], "share": {k: v / tot for k, v in parts.items()}}
    for k, v in b.items():
        if "." in k:
            out["share"][k] = v / tot
    return out


def main():
    river = [row(d) for d in load("river.jsonl")]
    ft = [row(d) for d in load("flop_turn.jsonl")]
    eq = json.loads((R / "equity.json").read_text()) if (R / "equity.json").exists() else {}
    summary = {"river": river, "flop_turn": ft, "equity": eq}

    def best(rows, street, menu, critic=None, T=None):
        c = [r for r in rows if r["street"] == street and r["menu"] == menu
             and (critic is None or critic in r["critic"]) and (T is None or r["T"] == T)]
        return min(c, key=lambda r: r["ms_per_iter_per_spot"]) if c else None

    est = {}
    rb = best(river, 3, "normal")
    if rb:
        prep = 0.0108 if not eq else eq["rank_tables_16_boards"]["per_board_s"]
        river_label = rb["ms_per_iter_per_spot"] / 1000 * 800 + prep + rb["exploitability_eval_s_per_spot"] + rb["root_values_s_per_spot"]
        est["river_label_core_s"] = river_label
        est["river_label_T200_core_s"] = rb["ms_per_iter_per_spot"] / 1000 * 200 + prep
        est["exact_V4_label_core_s(48 river solves, T=800)"] = 48 * river_label
        est["exact_V4_label_core_s(48 river solves, T=200)"] = 48 * est["river_label_T200_core_s"]
    for street, name, mat in ((2, "turn", "equity_matrix_turn_evaluator"), (1, "flop", "equity_matrix_flop_evaluator")):
        tb = best(ft, street, "normal", "net")
        if tb:
            m = eq.get(mat, {})
            msec = m.get("steady_s", m.get("run_s_excl_compile", 0.0))
            if street == 1:
                msec = msec * 300 / 2000
            est[f"{name}_label_core_s"] = tb["ms_per_iter_per_spot"] / 1000 * 1500 + msec + tb["exploitability_eval_s_per_spot"] + tb["root_values_s_per_spot"]
            cd = best(ft, street, "normal", "checkdown")
            if cd:
                est[f"{name}_label_checkdown_leaf_core_s"] = cd["ms_per_iter_per_spot"] / 1000 * 1500 + msec
    # scaling
    scaled = {}
    for k, v in est.items():
        e5_core = (v / 0.8, v / 0.6)
        scaled[k] = {"this_box_core_s": v, "e5_core_s": e5_core,
                     "e5_8core_labels_per_hour": (8 * 3600 / e5_core[1], 8 * 3600 / e5_core[0]),
                     "this_box_4core_labels_per_hour_ideal": 4 * 3600 / v}
    summary["estimates"] = scaled
    # GPU references recorded in the code (see REPORT.md for file:line)
    gpu = {"GTX1070_grid13_fixedT_river_s_per_spot": 0.86, "GTX1070_grid13_fixedT_turn_flop_s_per_spot": 1.7,
           "RTX3090_grid7_certificate_45k_river_first_pass_spots_per_min": 45000 / (5 * 60 + 18),
           "RTX4090_grid7_certificate_RIVER_REF_SPM": 197.1, "EXPECT_RIVER_SPM_3090": 150, "EXPECT_RIVER_SPM_4090": 200,
           "k0_size_run_hours_1070_vs_3090_grid13": [2.5, 3.0, 1.0]}
    gpu["GTX1070_river_labels_per_hour"] = 3600 / 0.86
    gpu["GTX1070_turn_or_flop_labels_per_hour"] = 3600 / 1.7
    # 3090 at grid 13 fixed T: the 1070 numbers / (2.5-3) (k0-size run 2.5-3 h vs 1 h, launch.py) - inference
    gpu["RTX3090_grid13_river_s_per_spot_inferred"] = (0.86 / 3.0, 0.86 / 2.5)
    gpu["RTX3090_grid13_turn_flop_s_per_spot_inferred"] = (1.7 / 3.0, 1.7 / 2.5)
    gpu["RTX4090_over_3090_grid7_ratio"] = 197.1 / gpu["RTX3090_grid7_certificate_45k_river_first_pass_spots_per_min"]
    summary["gpu_reference"] = gpu
    (R / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"estimates": scaled, "gpu": gpu}, indent=1))


if __name__ == "__main__":
    main()
