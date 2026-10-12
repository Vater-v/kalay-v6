"""Print a compact view of a profile jsonl (river.jsonl / flop_turn.jsonl)."""
import json
import sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    print(d["street"], d["menu"], "B", d["batch"], "T", d["iters"], d.get("critic", ""), "| prep", round(d["prepare_s"], 1),
          "compile", round(d.get("solve_compile_s", 0), 1), "| s/spot", round(d.get("solve_s_per_spot", 0), 2),
          "ms/it/spot", round(1000 * d.get("s_per_iter_per_spot", 0), 2), "| rss", round(d["peak_rss_mb"]),
          "| E med", round(d.get("E_pot_after_T", {}).get("median", -1), 5), "legal", d["mean_legal_actions"])
    print("   ", {k: round(v * 1000, 2) for k, v in d["breakdown_s_per_iter_batch"].items()},
          "ex", round(d.get("exploitability_eval_s", 0), 3), "root", round(d.get("root_values_s", 0), 3))
