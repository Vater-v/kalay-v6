"""Critic learning curve: test error of a street critic on one fixed
test set when it is trained on nested 1/8, 1/4, 1/2 and all of the spots of a loop run.

    POKER_REGIME=1 POKER_GRID=7 PYTHONPATH=. .venv/bin/python -m poker.solver.learning_curve --results results/k0b \
        --street river

Splits are by board, as the loop's critics (loop.split_rows: the same test set as the run's
critic): 10% test, 10% validation, the rest is the training pool; a fraction takes the first part
of a seeded permutation of the pool's boards, so smaller sets are nested in larger ones (river
boards are distinct, so this is by spot; the flop pool repeats boards). Rows of certified spots
only (the critic's data); the exact SPR-0 rows of a spot (river) go into training with it, and on
the validation and test boards they are reported apart (loop.critic_split: selection on solved
rows). Rows of an older slot layout are padded (critics.pad_rows). Every size gets the loop's
protocol (critics.PROTOCOL: the same step budget, best epoch on the same validation spots), so the
curve measures data, not optimisation. Errors in pot units of the street start, range-weighted
over hands (critics.row_metrics: MAE, range-EV offset, centred MAE), by SPR bucket, of the critic
as the loop trains and uses it (projected onto chip conservation with the street's kappa,
critics.project_values; the raw network's MAE and total violation alongside).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from poker.envs import config
from poker.solver import critics as CR
from poker.solver.loop import accepted, batch_starts, board_unit, check_stage_file, run_accept, split_cuts
from poker.solver.io import log, write_text

FRACTIONS = (0.125, 0.25, 0.5, 1.0)
SPR_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf)


def load(results: Path, name, count, batch, accept, game_hash=None):
    """Accepted critic rows (E <= accept: the loop's rule 9, poker/solver/loop.py) with their spot
    positions, the exact all-in rows of every spot, and per-spot (spr, accepted, board) from the
    batch files (the loop's batches, loop.batch_starts; every file of the run's game, game_hash)."""
    rows, ids, allin, allin_ids, spr, cert, boards = [], [], [], [], [], [], []
    for s in batch_starts(name, count, batch):
        data = np.load(results / f"{name}_{s:06d}.npz")
        check_stage_file(results / f"{name}_{s:06d}.npz", data, game_hash)
        pos = s + np.arange(len(data["pub"]))
        metrics = json.loads((results / f"{name}_{s:06d}.json").read_text())
        mask = accepted(metrics, accept)
        if len(mask) != len(pos):
            raise ValueError(f"{name}_{s:06d}: {len(mask)} metrics for {len(pos)} rows")
        boards.append(data["board"])
        rows.append(CR.rows_from_npz(data).take(np.flatnonzero(mask)))
        ids.append(pos[mask])
        if "allin_pub" in data.files:
            allin.append(CR.rows_from_npz(data, "allin_"))
            allin_ids.append(pos)
        spr += [m["spr"] for m in metrics]
        cert += list(mask)
    cat = lambda parts: CR.pad_rows(CR.Rows.concat(parts)) if parts else None
    return (cat(rows), np.concatenate(ids), cat(allin), np.concatenate(allin_ids) if allin_ids else None,
            np.asarray(spr), np.asarray(cert, bool), np.concatenate(boards))


def by_spr(spr, errors):
    out = {}
    for lo, hi in zip(SPR_EDGES[:-1], SPR_EDGES[1:]):
        m = (spr >= lo) & (spr < hi)
        out[f"{lo:g}-{hi:g}"] = {"n": int(m.sum()), "mae": float(np.mean(errors[m])) if m.any() else None}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--street", default="river")
    p.add_argument("--seeds", type=int, default=2, help="repeats with other subsets and initialisations")
    p.add_argument("--epochs", type=int, default=CR.PROTOCOL["epochs"], help="default: the loop's protocol")
    p.add_argument("--epoch-steps", type=int, default=CR.PROTOCOL["epoch_steps"])
    p.add_argument("--patience", type=int, default=CR.PROTOCOL["patience"], help="early stop (default: none)")
    p.add_argument("--clip-norm", type=float, default=CR.PROTOCOL["clip_norm"], help="0: no gradient clipping")
    p.add_argument("--hidden", type=int, nargs="+", help="default: the run's critic_hidden")
    p.add_argument("--out", type=Path, help="default: <results>/learning_curve_<street>.json")
    args = p.parse_args(argv)
    layout = json.loads((args.results / "layout.json").read_text())
    config.check_game(layout, str(args.results / "layout.json"))           # the run's game is this process's
    count, batch = layout[args.street], layout["solve_batch"]
    hidden = tuple(args.hidden or layout["critic_hidden"])
    accept = run_accept(layout)[(args.street, 2)]
    # the street critic's chip-conservation projection (critics.project_values): the run's own kappa
    # (layout "critic_projection"), else this code's; trained and measured projected as the loop does
    kappa = layout.get("critic_projection", {}).get("kappa", CR.KAPPA)[args.street]
    rows, ids, allin, allin_ids, spr, cert, boards = load(args.results, args.street, count, batch, accept,
                                                          layout.get("game_hash"))
    u = board_unit(boards)                                  # 0 training pool, 1 validation, 2 test
    a, b = split_cuts(board_unit(rows.board))               # the cuts of the run's critic (loop.critic_stage)
    role = np.where(u < a, 0, np.where(u < b, 1, 2)).astype(np.int8)
    group = np.unique(boards, axis=0, return_inverse=True)[1].reshape(-1)
    pick = lambda r, where: r.take(np.flatnonzero(where))
    test, val = pick(rows, role[ids] == 2), pick(rows, role[ids] == 1)
    test_spr = spr[ids[role[ids] == 2]]
    allin_val = allin_test = None
    if allin is not None:                                   # SPR-0 copies: training, and reported apart
        allin_val, allin_test = pick(allin, role[allin_ids] == 1), pick(allin, role[allin_ids] == 2)
    pool = np.unique(group[role == 0])
    log(f"learning curve {args.street}: {count} spots, {int(cert.sum())} accepted (E <= {accept}); test {len(test)} rows, "
        f"val {len(val)} (solved), pool {int((role == 0).sum())} spots on {len(pool)} boards; hidden {hidden}, "
        f"{args.epochs} x {args.epoch_steps} steps, patience {args.patience}, clip norm {args.clip_norm}")
    runs = []
    for seed in range(args.seeds):
        order = np.random.default_rng(seed).permutation(pool)
        for frac in FRACTIONS:
            chosen = np.isin(group, order[:int(round(frac * len(order)))]) & (role == 0)
            train = pick(rows, chosen[ids])
            n_solved = len(train)
            if allin is not None:
                train = CR.Rows.concat([train, pick(allin, chosen[allin_ids])])
            t0 = time.perf_counter()
            params, info = CR.train(train, val, hidden=hidden, seed=seed, epochs=args.epochs,
                                    patience=args.patience, epoch_steps=args.epoch_steps, clip_norm=args.clip_norm,
                                    kappa=kappa)
            m = CR.row_metrics(params, test, kappa=kappa)
            a_val, a_test = (float(np.mean(CR.row_metrics(params, r, kappa=kappa)["mae"])) if r else None
                             for r in (allin_val, allin_test))
            run = {"seed": seed, "fraction": frac, "spots": int(chosen.sum()), "solved_rows": n_solved,
                   "test_checkdown_mae": float(np.mean(CR.checkdown_errors(test))),
                   "boards": int(len(np.unique(group[chosen]))),
                   "train_rows": len(train), "test_mae": float(np.mean(m["mae"])),
                   "test_raw_mae": float(np.mean(m["raw_mae"])), "test_raw_total_abs": float(np.mean(np.abs(m["raw_total"]))),
                   "test_mae_median": float(np.median(m["mae"])), "test_worst_median": float(np.median(m["worst"])),
                   "test_offset": float(np.mean(m["offset"])), "test_centred": float(np.mean(m["centred"])),
                   "test_worst_substantial_median": float(np.median(m["worst_substantial"])),
                   "test_mae_by_spr": by_spr(test_spr, m["mae"]),
                   "val_mae": info["val_mae"], "val_allin_mae": a_val, "test_allin_mae": a_test,
                   "epochs": info["epochs_run"], "best_epoch": info["best_epoch"], "lr_at_best": info["lr_at_best"],
                   "clip_norm": info["clip_norm"], "early_stop": info["early_stop"],
                   "loss_ratio_max": info["loss_ratio_max"], "seconds": time.perf_counter() - t0}
            runs.append(run)
            log(f"  seed {seed} {frac:>5.3f}: {n_solved} solved rows, test MAE {run['test_mae']:.4f} "
                f"(median {run['test_mae_median']:.4f}), best epoch {run['best_epoch']}/{run['epochs']}, "
                f"{run['seconds']:.0f} s")
    curve = []
    for frac in FRACTIONS:
        r = [x for x in runs if x["fraction"] == frac]
        curve.append({"fraction": frac, "solved_rows": float(np.mean([x["solved_rows"] for x in r])),
                      "test_mae": float(np.mean([x["test_mae"] for x in r])),
                      "test_mae_spread": float(np.ptp([x["test_mae"] for x in r])),
                      "test_mae_by_spr": {k: (float(np.mean([x["test_mae_by_spr"][k]["mae"] for x in r]))
                                              if r[0]["test_mae_by_spr"][k]["mae"] is not None else None)
                                          for k in r[0]["test_mae_by_spr"]}})
    n = np.log([c["solved_rows"] for c in curve])
    e = np.log([c["test_mae"] for c in curve])
    slope = float(np.polyfit(n, e, 1)[0]) if len(curve) > 1 else None
    cert_by_spr = by_spr(spr, cert.astype(float))
    result = {"results": str(args.results), "street": args.street, "spots": count,
              "accept_e": accept, "accepted": int(cert.sum()), "accepted_by_spr": cert_by_spr,
              "test_rows": len(test), "test_spr_counts": {k: v["n"] for k, v in by_spr(test_spr, test_spr).items()},
              "hidden": list(hidden), "epochs": args.epochs, "epoch_steps": args.epoch_steps, "kappa": kappa,
              "patience": args.patience, "clip_norm": args.clip_norm, "seeds": args.seeds, "curve": curve,
              "loglog_slope": slope, "runs": runs}
    out = args.out or args.results / f"learning_curve_{args.street}.json"
    write_text(out, json.dumps(result, indent=1))
    log(f"MAE vs solved rows (mean over {args.seeds} seeds), log-log slope {slope:+.3f}:")
    for c in curve:
        log(f"  {c['fraction']:>5.3f}: {c['solved_rows']:7.0f} rows  MAE {c['test_mae']:.4f} "
            f"(spread {c['test_mae_spread']:.4f})  by SPR {c['test_mae_by_spr']}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
