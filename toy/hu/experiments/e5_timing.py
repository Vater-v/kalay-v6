"""E5: timing of single (un-batched) re-solves per street, for each leaf type.
Collects the other timing numbers from E1/E2/E3 json files into one table."""
import os, sys, time, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from common import setup, save, load, Log, RES
from cfr import Solver, values
from tree import PublicTree
from valuefn import LeafEvaluator, CheckdownV, ByStreetV

K = int(os.environ.get("E2_K", 300))
cfg, B, T = setup()
bp = [np.load(os.path.join(RES, "blueprint_eff0.npy")), np.load(os.path.join(RES, "blueprint_eff1.npy"))]
S0 = Solver(T, [T.bmask[:1].copy()] * 2)
v0, v1, r0, r1 = values(S0, bp[0], bp[1])
VT = pickle.load(open(os.path.join(RES, "value_nets_turn.pkl"), "rb"))
VF = pickle.load(open(os.path.join(RES, "value_nets_flop.pkl"), "rb"))
CD = CheckdownV(B)
VL = ByStreetV(B, VF["20000"], VT["20000"])


def single(key, leaf, V, reps=3):
    n = T.key_index[key]
    best = np.inf
    for _ in range(reps):
        t0 = time.time()
        F = PublicTree(cfg, B, [T.states[n]], [T.cscale[n]], depth_limited=(leaf != "oracle"))
        lf = LeafEvaluator(F, V) if (leaf != "oracle" and len(F.L)) else None
        gad = None if n == 0 else dict(player=0, w=v0[n:n + 1])
        S = Solver(F, [F.bmask[:1].copy(), r1[n:n + 1]], leaf_fn=lf, gadget=gad)
        S.iterate(K)
        best = min(best, time.time() - t0)
    return dict(key=key, leaf=leaf, nodes=F.M, seconds=best)


if __name__ == "__main__":
    log = Log("e5.log")
    rows = []
    for key in ["", "k/k/d0", "k/k/d0/k/k/d1"]:
        for leaf, V in (("checkdown", ByStreetV(B, CD, CD)), ("learned", VL), ("oracle", None)):
            r = single(key, leaf, V)
            rows.append(r); log(r)
    e1, e2a, e2b, e2c = load("e1.json"), load("e2a.json"), load("e2b.json"), load("e2c.json")
    e3 = load("e3.json")
    summary = dict(
        full_solve_sec_per_iter=e1["sec_per_iter_plus"],
        full_solve_to_1pct=e1["plus_first_check_below_1.0pct"],
        full_solve_to_0p1pct=e1["plus_first_check_below_0.1pct"],
        turn_oracle_label_ms=e2a["turn_label"]["cpu_ms_per_situation"],
        flop_exact_label_ms=e2a["flop_exact_label"]["cpu_ms_per_situation"],
        flop_boot_label_ms={r["name"]: r["boot_cpu_ms_per_situation"] for r in e2b["V_flop"] if r["N"]},
        turn_train_s={r["name"]: r["train_s"] for r in e2a["V_turn"] if r["N"]},
        flop_train_s={r["name"]: r["train_s"] for r in e2b["V_flop"] if r["N"]},
        agent_batched_ms_per_resolve={k: {s: v["per_street"][s]["ms_per_resolve_batched"] for s in v["per_street"]}
                                      for k, v in e2c["agents"].items()},
        e3_resolve_single_root_s=[(s["size"], s["resolve_single_root_s"]) for s in e3["sizes"]],
        single_resolves=rows, K=K)
    save("e5.json", summary)
    log("done", summary)
