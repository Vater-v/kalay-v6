"""Render every results/*.json into the markdown tables used in README.md."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load, mA


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


e1 = load("e1.json")
g = e1["game"]
print("### Game size\n")
table(["public nodes", "decision", "chance", "fold", "showdown", "fiber (hands)", "info sets", "levels"],
      [[g["public_nodes"], g["decision"], g["chance"], g["fold"], g["showdown"], g["fiber"], g["info_sets"], g["levels"]]])

print("### E1 exact solve (exploitability in mA/hand = milli-antes per hand; % of the 2-ante initial pot)\n")
van = {r["iter"]: r for r in e1["vanilla"]}
rows = []
for r in e1["cfr_plus"]:
    v = van.get(r["iter"])
    rows.append([r["iter"], f(r["seconds"]), f(r["expl_mA"], 2), f(r["expl_pct_pot"], 3),
                 f(v["seconds"]) if v else "-", f(v["expl_mA"], 2) if v else "-", f(v["expl_pct_pot"], 3) if v else "-"])
table(["iter", "CFR+ s", "CFR+ expl mA/h", "CFR+ % pot", "CFR s", "CFR expl mA/h", "CFR % pot"], rows)
print(f"seconds/iteration: CFR+ {e1['sec_per_iter_plus']:.4f}, vanilla {e1['sec_per_iter_vanilla']:.4f}; "
      f"solver arrays {e1['solver_array_MB']:.1f} MB; peak RSS {e1['peak_rss_MB']:.0f} MB; "
      f"game value for P0 {e1['game_value_P0']:.5f} antes/hand.\n")

if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e2a.json")):
    a = load("e2a.json")
    print("### E2a turn-boundary value function (errors in units of x = each player's pot share)\n")
    print(f"turn oracle labels: {a['turn_label']['n']} situations, {a['turn_label']['cpu_ms_per_situation']:.1f} CPU-ms each, "
          f"mean label exploitability {a['turn_label']['mean_label_expl_x_units']:.5f} x; "
          f"exact flop labels: {a['flop_exact_label']['cpu_ms_per_situation']:.0f} CPU-ms each. "
          f"On-policy states: {a['n_onpolicy_turn']}.\n")
    table(["V_turn", "N", "residual", "range-w loss", "train s", "test RMSE", "test RMSE range-w", "on-policy RMSE", "on-policy RMSE range-w"],
          [[r["name"], r["N"], r.get("residual", "-"), r.get("range_weighted", "-"), f(r.get("train_s")),
            f(r["test_rmse"], 3), f(r["test_rmse_rangew"], 3), f(r["onpolicy_rmse"], 3), f(r["onpolicy_rmse_rangew"], 3)]
           for r in a["V_turn"]])
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e2b.json")):
    b = load("e2b.json")
    print("### E2b flop-boundary value function (bootstrapped labels; errors vs EXACT flop labels)\n")
    table(["V_flop", "M_flop", "boot label CPU-ms", "boot-label RMSE vs exact", "range-w", "test RMSE", "test RMSE range-w",
           "on-policy RMSE", "on-policy range-w"],
          [[r["name"], r.get("M_flop", "-"), f(r.get("boot_cpu_ms_per_situation")), f(r.get("boot_label_rmse_vs_exact"), 3),
            f(r.get("boot_label_rmse_rangew_vs_exact"), 3), f(r["exact_test_rmse"], 3), f(r["exact_test_rmse_rangew"], 3),
            f(r["onpolicy_rmse"], 3), f(r["onpolicy_rmse_rangew"], 3)] for r in b["V_flop"]])
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e2c.json")):
    c = load("e2c.json")
    print(f"### E2c continual re-solving agents (K = {c['K']} CFR+ iterations per re-solve, both seats)\n")
    fs, ff = c["full_solve_same_K"], c["full_solve_final"]
    rows = [[f"full solve, {fs['iter']} it", f(fs["expl_mA"]), f(fs["expl_pct_pot"], 2), "-", "-", "-", "-", "-"],
            [f"full solve, {ff['iter']} it", f(ff["expl_mA"], 2), f(ff["expl_pct_pot"], 3), "-", "-", "-", "-", "-"]]
    for k, v in c["agents"].items():
        ps = v["per_street"]
        rows.append([k, f(v["expl_mA"]), f(v["expl_pct_pot"], 2), f(v["seat0_exploitable_by_mA"]), f(v["seat1_exploitable_by_mA"]),
                     f"{f(ps['0']['ms_per_resolve_batched'])} / {f(ps['1']['ms_per_resolve_batched'])} / {f(ps['2']['ms_per_resolve_batched'])}",
                     f(v["wall_s_seat0"] + v["wall_s_seat1"]), sum(ps[s]["n_resolves"] for s in ps)])
    table(["agent", "expl mA/h", "% pot", "seat0 expl", "seat1 expl", "ms per re-solve (batched) pre/flop/turn",
           "extraction s", "# re-solves"], rows)
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e2d.json")):
    d = load("e2d.json")
    print("### E2d exact leaf at one boundary, approximate leaf at the other (safe re-solving)\n")
    table(["agent", "expl mA/h", "% pot", "extraction s (both seats)"],
          [[k, f(v["expl_mA"]), f(v["expl_pct_pot"], 2), f(v["wall_s"])] for k, v in d["agents"].items()])
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e3.json")):
    d = load("e3.json")
    print(f"### E3 off-tree opening bet by P0 at every flop root (we are P1); loss = P0 best-response value - value (mA/hand)\n")
    print(f"Blueprint in the base game: P0 best response gains {d['base_loss_mA']:.2f} mA/h over the game value.\n")
    rows = []
    for s in d["sizes"]:
        st = s["strategies"]
        rows.append([s["size"], f(mA(s["aug_value_P0"] - s["base_value_P0"]), 2), f(mA(s["aug_solve_expl"]), 2)]
                    + [f(st[k]["loss_vs_aug_mA"], 1) for k in ("snap_lo", "snap_hi", "snap_near", "resolve", "aug_nash")]
                    + [f(1000 * s["resolve_s_per_root"], 1), f(s["resolve_single_root_s"], 2),
                       f"{s['unmapped_reached_P1_nodes_lo']}/{s['unmapped_reached_P1_nodes_hi']}/{s['unmapped_reached_P1_nodes_near']} of {s['P1_decision_nodes_offtree']}"])
    table(["size (pot)", "value gain of size for P0", "aug-solve expl", "snap down", "snap up", "snap nearest", "safe re-solve",
           "aug Nash", "re-solve ms/root (batched)", "single re-solve s", "reached P1 nodes w/o translation (down/up/near)"], rows)
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e4.json")):
    e = load("e4.json")
    print(f"### E4 robustness vs exploitation (seat-averaged, mA/hand; RNR with K = {e['K']} CFR+ iterations)\n")
    rows = []
    for r in e["table"]:
        p = r["p"]
        name = "Nash (p=0)" if p == 0.0 else ("exact best response" if p == "BR" else f"RNR p={p}")
        rows.append([r["model"], name, f(r["ev_mA"]), f(r["expl_mA"]), f(1000 * r["seat0_ev"]), f(1000 * r["seat1_ev"])])
    table(["model", "strategy", "EV vs model", "exploitability", "EV as P0", "EV as P1"], rows)
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "e5.json")):
    t = load("e5.json")
    print("### E5 timing\n")
    rows = [["full solve: s/iteration", f(t["full_solve_sec_per_iter"], 4)],
            ["full solve to <1% pot (iter, s)", t["full_solve_to_1pct"]],
            ["full solve to <0.1% pot (iter, s)", t["full_solve_to_0p1pct"]],
            ["turn oracle label, CPU-ms / situation", f(t["turn_oracle_label_ms"])],
            ["exact flop label, CPU-ms / situation", f(t["flop_exact_label_ms"])]]
    for k, v in t["flop_boot_label_ms"].items():
        rows.append([f"bootstrapped flop label ({k}), CPU-ms / situation", f(v)])
    for k, v in t["turn_train_s"].items():
        rows.append([f"train V_turn {k}, s", f(v)])
    for k, v in t["flop_train_s"].items():
        rows.append([f"train V_flop {k}, s", f(v)])
    for r in t["single_resolves"]:
        street = {0: "preflop", 1: "flop", 2: "turn"}[r["key"].count("d")]
        rows.append([f"single re-solve {street} root, {r['leaf']} leaf, K={t['K']} ({r['nodes']} nodes), s", f(r["seconds"], 3)])
    for sz, sec in t["e3_resolve_single_root_s"]:
        rows.append([f"E3 single flop re-solve with off-tree size {sz}, s", f(sec, 3)])
    table(["quantity", "value"], rows)
if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", "label_noise.json")):
    n = load("label_noise.json")
    print(f"### Label noise: per-hand CFVs vs a {n['ref_iters']}-iteration solve (1000 random turn situations, x units)\n")
    table(["CFR+ iters", "label expl (x)", "RMSE all hands", "RMSE hands with p>1e-3", "range-weighted RMSE"],
          [[r["iters"], f(r["label_expl_x"], 5), f(r["rmse_all"], 3), f(r["rmse_in_range"], 3), f(r["rmse_rangew"], 3)]
           for r in n["rows"]])
