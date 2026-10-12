"""Price of the 3-player postflop raise cap.

Turn spots with three players (SPR on a log grid, random ranges, NL5 rake), the turn's betting round
solved to the certificate twice - raise cap 1 and cap 2 (+ all-in after the cap, every grid size) -
and the root values compared. Street ends are checked down exactly (pcfr.turn_runout_three over the
48 rivers, paying the engine's showdown rake of that pot): there is no 3-player river critic yet,
so both trees are solved in the same exact game. Reported per spot and overall, in pots of the turn
start: |dV| per hand of every player (weights = probability of the deal: range x compatible mass of
the opponents; mean and max) and the difference of each player's root EV.

    POKER_REGIME=1 POKER_GRID=7 PYTHONPATH=. .venv/bin/python -m poker.solver.cap_price_n3 --out results/cap_price_n3.json \
        [--spots 24]
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.solver import pcfr as SV
from poker.solver import showdown as sd
from poker.solver import situations as SI
from poker.solver import tree as TR
from poker.solver.io import log, write_text


def spots_on_grid(count, seed=81):
    # NL5 as a fixture (SI.NL5), not the regime's mixture: the measured spots (79a9fb4) stay the same
    # whatever POKER_REGIME (regime 1 deals no 3-player table at all).
    s = SI.make_spots(seed, count, 3, spr_range=(0.5, 50.0), rake=SI.NL5, street=2)
    spr = np.geomspace(0.5, 50.0, 8)[np.arange(count) % 8]
    s["stack"] = np.maximum(np.round(spr * s["pot"]), config.BB).astype(np.int32)
    return s


def solve_tree(spots, cap, args):
    topo = TR.build_topology(3, cap, allin_after_cap=True, street=2)
    flat = dataclasses.replace(topo, leaf=np.zeros_like(topo.leaf))           # street ends -> check-down runouts
    st = SV.Static(flat)
    cfg = SV.SolverConfig(cap=args.cap, e_target=args.e_target, check_every=50,
                          runout_chunk=args.chunk)
    values, es, iters = [], [], []
    t0 = time.perf_counter()
    for lo in range(0, len(spots["pot"]), args.batch):
        part = {k: v[lo:lo + args.batch] for k, v in spots.items()}
        batch = SI.prepare(part, topo)
        leaves = np.flatnonzero(topo.leaf)
        showdown = config.rake_from_terms(batch["rake_bps"][:, None], batch["rake_cap_chips"][:, None],
                                          batch["pot_chips"], 1) / jnp.asarray(part["pot"], jnp.float32)[:, None]
        batch = {**batch, "rake": batch["rake"].at[:, leaves].set(showdown[:, leaves])}
        state = SV.solve(batch, st, cfg)
        ex = jax.vmap(lambda z, s: SV.exploitability(z, s, st, cfg))(state["z"], batch)
        root = jax.vmap(lambda z, s: SV.root_values(z, s, st, cfg))(state["z"], batch)
        values.append(np.asarray(root))
        es.append(np.asarray(ex["nash_conv"]) / 3)
        iters.append(np.asarray(state["u"]))
        log(f"  cap {cap}: spots {lo}-{lo + len(part['pot'])}: E {np.round(es[-1], 5)}, iterations {iters[-1]}, "
            f"{time.perf_counter() - t0:.0f} s")
    info = {"nodes": int(topo.size), "runout_terminals": int(len(st.allins)),
            "seconds": time.perf_counter() - t0}
    return np.concatenate(values), np.concatenate(es), np.concatenate(iters), info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--spots", type=int, default=24)
    p.add_argument("--batch", type=int, default=2, help="spots per solver call (memory)")
    p.add_argument("--chunk", type=int, default=16, help="runout terminals per vectorised call (memory)")
    p.add_argument("--cap", type=int, default=4000, help="iteration cap")
    p.add_argument("--e-target", type=float, default=0.002)
    p.add_argument("--critic-error", type=float, default=0.0657, help="k0 turn critic test MAE (the yardstick)")
    args = p.parse_args(argv)
    spots = spots_on_grid(args.spots)
    log(f"cap price n=3: {args.spots} turn spots, SPR grid {np.round(np.geomspace(0.5, 50.0, 8), 2).tolist()}, "
        f"e_target {args.e_target}, cap {args.cap}, device {jax.devices()[0]}")
    v1, e1, i1, info1 = solve_tree(spots, 1, args)
    v2, e2, i2, info2 = solve_tree(spots, 2, args)
    ranges = np.asarray(spots["ranges"], np.float64)
    rows, all_mean, all_max = [], [], []
    for s in range(args.spots):
        per = []
        for i in range(3):
            b, c = [j for j in range(3) if j != i]
            mass = np.asarray(sd.mass_three(jnp.asarray(ranges[s, b]), jnp.asarray(ranges[s, c])))
            w = ranges[s, i] * mass
            w = w / w.sum()
            d = np.abs(v1[s, i] - v2[s, i])
            ok = w > 0
            per.append({"mean_abs_dv": float(np.sum(w * d)), "max_abs_dv": float(d[ok].max()),
                        "ev_cap1": float(np.sum(w * v1[s, i])), "ev_cap2": float(np.sum(w * v2[s, i]))})
        rows.append({"spr": float(spots["stack"][s] / spots["pot"][s]), "e_cap1": float(e1[s]), "e_cap2": float(e2[s]),
                     "iters_cap1": int(i1[s]), "iters_cap2": int(i2[s]), "players": per})
        all_mean += [x["mean_abs_dv"] for x in per]
        all_max += [x["max_abs_dv"] for x in per]
    ev_diff = [abs(x["ev_cap1"] - x["ev_cap2"]) for r in rows for x in r["players"]]
    result = {"spots": args.spots, "e_target": args.e_target, "iteration_cap": args.cap,
              "cap1": info1, "cap2": info2, "certified_cap1": int(np.sum(e1 <= args.e_target)),
              "certified_cap2": int(np.sum(e2 <= args.e_target)),
              "mean_abs_dv": float(np.mean(all_mean)), "max_abs_dv": float(np.max(all_max)),
              "mean_abs_ev_diff": float(np.mean(ev_diff)), "max_abs_ev_diff": float(np.max(ev_diff)),
              "critic_error": args.critic_error,
              "verdict": "cap 1" if np.mean(all_mean) <= args.critic_error else "cap 2 with two sizes, measure again",
              "rows": rows}
    write_text(args.out, json.dumps(result, indent=1))
    log(f"cap 1 against cap 2: mean |dV| {result['mean_abs_dv']:.4f} pot (max {result['max_abs_dv']:.4f}), root EV |diff| "
        f"mean {result['mean_abs_ev_diff']:.4f} (max {result['max_abs_ev_diff']:.4f}); turn critic error "
        f"{args.critic_error}; certified {result['certified_cap1']}/{args.spots} and {result['certified_cap2']}/{args.spots}; "
        f"-> {result['verdict']}")


if __name__ == "__main__":
    main()
