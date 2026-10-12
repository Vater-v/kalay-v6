"""Training loop, iteration 0, bottom-up and cached stage by stage:

    river_solve -> river_critic -> turn_end -> turn_solve -> turn_critic
                -> flop_end -> flop_solve -> flop_critic -> pre_end

    POKER_REGIME=1 PYTHONPATH=. .venv/bin/python -m poker.solver.loop --out results/loop_r1 [--quick] [--stages ...]
    (the grid is the regime's, 13; explicit grids 5 and 7 remain mathematical test fixtures)

Postflop and street-end spots share the versioned equal-stack HU training domain of
situations.training_domain_record: pot >= 2 BB, positive behind down to one chip, SPR up to
(depth - 1) / 2 (99.5 at 200 BB), jointly constrained by pot / 2 + behind = sampled starting
depth. Low/core/deep coverage weights are initial uncalibrated allocations. Every street-end
stage includes explicit all-in rows; river solves also emit exact SPR-0 check-down copies.
Depth and rake follow config.DEPTH_MIX and config.RAKE_MIX. Every solve's tree is config.tree_rule(PLAYERS, street, menu) (poker/solver/tree.py rule_topology, tree_for below),
the size menu of a training spot is config.menu_for of its SPR at the street start, and a batch holds spots of one tree only (stage_batches). The game record
(config.game_record: grid, regime, tree rules) and its hash are written into the layout, every
critic's metadata and every stage file, and compared at every load (config.check_game).
Teacher stop rule (layout "teacher"): fixed T (--fixed-t, the default under grid 13) runs config.TEACHER_T iterations per solve, computes E once at the end as a diagnostic
(flag e_flag: E > config.TEACHER_E_FLAG x the stage's median) and puts every spot into the data;
the certificate (--certificate, the default under grids 5 and 7: k0 and the checks) stops every
solve on its certificate E_solve <= e_target (exact best response inside the street, given
the next street's critic at street ends); spots without the certificate are counted and kept out of
the critic data. Critics are residual (poker/solver/critics.py): V = network(inputs, F) + CD with the
exact check-down value CD (poker/solver/checkdown.py); every row stores its CD. Every critic call and
every critic's training project the values onto chip conservation (critics.project_values, the
critic's kappa: critics.KAPPA_STREET, recorded in the layout), and every solve stage reports the
share of spots whose root values pay out more than the pot holds (payout_report). Several processes may
share --out (one per GPU): work items are claimed by exclusive file creation, results are written
atomically, and the layout (scheme, sizes, solver, rake mixture, critic form) is published once and
checked by every later process. --shard i/k (separate machines without a shared disk): this
process solves only the first-pass batches with index % k == i and stops; the batch files of all
shards gathered into one directory then resume as one run. Units: pot at the street start.
"""

import argparse
import functools
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.envs import fingerprint as FP
from poker.solver import checkdown as CD                # importing solver sets full float32 matmuls (poker/solver/__init__.py)
from poker.solver import critics as CR
from poker.solver import flops as FL
from poker.solver import pcfr as SV
from poker.solver.critics import load_critic
from poker.solver import rank7
from poker.solver import runout as RO
from poker.solver import situations as SI
from poker.solver import tree as TR
from poker.solver.io import claim, log, publish_once, savez, wait_for, write_text

SEEDS = {"river": 1101, "turn": 1202, "turn_end": 1303, "flop": 1404, "flop_end": 1505,
         "pre_end": 1606, "preflop": 1707}
ROOT = Path(__file__).resolve().parents[2]
PREFLOP_EQUITY = ROOT / "results" / "cache" / "preflop_equity_v1.npy"      # absolute: any working directory
FLOP_BOARDS = 300                        # distinct canonical flops of the flop stage (matrices cached per flop)
SPLIT = (0.8, 0.1, 0.1)                  # train / val / test by board (split_rows)
# Board splits (split_rows / critic_split) group the boards by their suit-
# isomorphism class since split version 2 (efficiency audit 2026-10-10, item 5); 1 = the exact board, every run before
# (k0, the 45k river set: their test and held-out numbers are version-1 numbers, not comparable without a re-baseline).
# Recorded in the layout and every critic's meta; a run resumes only under its own version.
SPLIT_VERSION = 2
SPR_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0, 100.0, float("inf"))     # report buckets
STAGES = ("river_solve", "river_critic", "turn_end", "turn_solve", "turn_critic",
          "flop_end", "flop_solve", "flop_critic", "pre_end")
STREET_OF = {"river": 3, "turn": 2, "flop": 1, "preflop": 0}
# Speed fuse of the HU river: RTX 4090 run of 2026-10-08, 3000 spots, batch 128, first pass.
# Source: results/box/results/k0/river_[0-9]*.json, using pass1 where present.
# Bucket references (upper SPR edge, seconds/spot) sum the solve_seconds_per_spot and
# prepare_seconds_per_spot means, each rounded to four decimal places.
# Whole-stage throughput: 60 * spots / sum(solve + prepare) = 197.1 spots/min.
RIVER_REF = ((1.0, 0.0819), (2.0, 0.1068), (4.0, 0.1903), (8.0, 0.2894), (16.0, 0.3319), (32.0, 0.5582),
             (float("inf"), 0.7193))
RIVER_REF_SPM = 197.1
# Certificate mode accepts only E <= e_target, independent of retired student measurements.
PLAYERS = config.LOOP_PLAYERS                # heads-up training only
# Teacher stop rules (layout "teacher"; a layout without one is a certificate run, as every run before 2026-10-11).
# FIXED_T: exactly config.TEACHER_T[street] PCFR+ iterations per solve in chunks of
# FIXED_T_CHUNK (pcfr.solve_fixed), no E-stop, no second pass, no acceptance threshold (every spot enters the data);
# E is computed once at the end of every solve (E_EVALUATIONS counts the calls: one per batch) and stored per spot,
# flagged (e_flag) when above config.TEACHER_E_FLAG x the median E of its stage. CERTIFICATE: E <= e_target, with a second pass for rejected spots.
FIXED_T, CERTIFICATE = "fixed_t", "certificate"
FIXED_T_CHUNK = 50
FIXED_T_RULE = ("every solve runs exactly T PCFR+ iterations of its street (no E-stop, no second pass); E is computed once "
                "at the end of every solve as a diagnostic, and a spot whose E exceeds e_flag_factor x the median E of its "
                "stage is flagged (e_flag in its metrics, <stage>_eflags.json); every spot enters the critic data")
E_EVALUATIONS = [0]                        # exploitability evaluations of solved batches in this process (one per batch)


def teacher_mode(layout):
    """The stop rule of a run's layout: FIXED_T or CERTIFICATE (no "teacher" record: a run before 2026-10-11)."""
    return (layout.get("teacher") or {}).get("stop", CERTIFICATE)


def accept_thresholds(e_target):
    """Certificate acceptance: the requested teacher E target on every street."""
    return {name: {str(n): e_target for n in range(2, config.TREE_MAX_PLAYERS + 1)} for name in STREET_OF}


def stop_thresholds(e_target):
    """Certificate solving and acceptance use the same E target."""
    return accept_thresholds(e_target)


def accepted(metrics, accept):
    """bool per spot of a batch's metrics: certificate E <= accept."""
    return np.asarray([m["e_solve"] <= accept for m in metrics], bool)


def run_accept(layout):
    """Certificate thresholds from a layout; fixed-T data accepts every finite solved spot (inf)."""
    if teacher_mode(layout) == FIXED_T:
        return {(name, n): float("inf") for name in STREET_OF for n in range(2, config.TREE_MAX_PLAYERS + 1)}
    if "accept_e" in layout:
        return {(name, int(n)): float(v) for name, by_n in layout["accept_e"].items() for n, v in by_n.items()}
    return {(name, n): float(layout["e_target"]) for name in STREET_OF for n in range(2, config.TREE_MAX_PLAYERS + 1)}


class SpeedFuse:
    """With EXPECT_RIVER_SPM = the river spots per minute expected per card on this card (queue.sh:
    150 on a 3090, 200 on a 4090) a river worker stops once its batches after the first (which
    compiles) took more than twice the time the reference predicts for their spots, scaled to the
    expected speed: at half the expected speed the run burns hours for nothing (stop, find out)."""

    def __init__(self, expected_spm):
        self.factor = RIVER_REF_SPM / expected_spm
        self.actual = self.expected = 0.0
        self.batches = 0

    def predict(self, spr):
        return self.factor * sum(next(sec for edge, sec in RIVER_REF if s < edge) for s in spr)

    def add(self, spr, seconds):
        """Returns the expected seconds of this batch; raises SystemExit when the fuse blows."""
        self.batches += 1
        expected = self.predict(spr)
        if self.batches > 1:
            self.actual += seconds
            self.expected += expected
            if self.batches > 3 and self.actual > 2.0 * self.expected:
                raise SystemExit(f"speed fuse: river batches 2..{self.batches} took {self.actual:.0f} s, the reference "
                                 f"predicts {self.expected:.0f} s at {RIVER_REF_SPM / self.factor:.0f} spots/min per "
                                 f"card: below half the expected speed, stopped")
        return expected


_TREES = {}


def tree_key(n, street, menu):
    """config.tree_rule(n, street, menu) as a key: menus with the same rule share one tree (grids 5 and 7: every
    menu is the iteration-0 tree)."""
    r = config.tree_rule(n, street, menu)
    return n, street, r["cap"], r["allin_after_cap"], r["sizes"], r["no_open_limp"]


def tree_for(street, n, menu):
    """(Topology, Static) of the teacher's tree for n players on `street` in size menu `menu`: poker/solver/tree.py
    rule_topology, i.e. config.tree_rule - the one source of the teacher's trees. One object per rule
    and process (the compiled solver and evaluation functions are keyed by it)."""
    key = tree_key(n, street, menu)
    if key not in _TREES:
        topo = TR.rule_topology(n, street, menu)
        _TREES[key] = (topo, SV.Static(topo))
    return _TREES[key]


def spot_menus(name, spots):
    """config.menu_for (the size menu) of every spot of a stage: preflop by the starting stack in BB
    (spots["stack"]: chips per seat at the hand start), postflop by the SPR at the street start (chips behind each /
    pot: the loop's spots have equal stacks)."""
    stack, pot, street = np.asarray(spots["stack"], np.int64), np.asarray(spots["pot"], np.int64), STREET_OF[name]
    if street == STREET_OF["preflop"]:
        return [config.menu_for(street, depth_bb=int(s) / config.BB) for s in stack]
    return [config.menu_for(street, spr=int(s) / int(p)) for s, p in zip(stack, pot)]


@functools.lru_cache(maxsize=None)
def stage_batches(name, count, solve_batch):
    """((start, stop, menu), ...) of a stage's spots in solving order (stage_spots, SPR order): consecutive spots
    of one tree (tree_key) form a run, cut into batches of solve_batch; a batch never mixes two trees, so one
    topology serves one batch. menu: that of the batch's first spot (every spot of the batch has its tree). Under
    one tree (grids 5 and 7; regime 1's preflop) the batches are those of every run before: range(0, count,
    solve_batch). The SPR order puts the spots of the smaller bounds (config.MENU_BOUNDS) first, so a stage has one
    run per tree."""
    spots = stage_spots(name, count)
    menus = spot_menus(name, spots)
    keys = [tree_key(PLAYERS, STREET_OF[name], m) for m in menus]
    out, first = [], 0
    while first < count:
        end = first
        while end < count and keys[end] == keys[first]:
            end += 1
        out += [(s, min(end, s + solve_batch), menus[s]) for s in range(first, end, solve_batch)]
        first = end
    return tuple(out)


def batch_starts(name, count, solve_batch):
    return [s for s, _, _ in stage_batches(name, count, solve_batch)]


def code_version():
    """Commit and dirty flag of the running code: git in a checkout of this tree, else
    results/pack/PACKED.json that tools/cloud/pack.sh writes into the tarball for rented boxes, dirty also when the code
    differs from the pack (poker/envs/fingerprint.py). A tree unpacked inside another repository is not its checkout."""
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                        check=True).stdout.strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() != ROOT.resolve():
            raise OSError("not the repository of this tree")
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        packed = ROOT / "results" / "pack" / "PACKED.json"
        if packed.exists():
            d = json.loads(packed.read_text())
            return {"commit": d["commit"], "dirty": int(d["dirty_paths"]) > 0 or FP.code_fingerprint()["dirty"],
                    "source": "results/pack/PACKED.json"}
        return {"commit": None, "dirty": True, "source": "unknown"}


# ---------------------------------------------------------------- critic parameters on disk

def save_critic(path: Path, params, info):
    """A critic's weights with its meta: the critic config, split version, the game record (config.game_meta) and
    `info`."""
    savez(path, meta=np.asarray(json.dumps({**CR.config_payload(), "split_version": SPLIT_VERSION, **info,
                                            "training_domain": SI.training_domain_record(), **config.game_meta()})),
          **{f"{i}_{k}": np.asarray(v) for i, layer in enumerate(params) for k, v in layer.items()})


# ---------------------------------------------------------------- solves

# Payout bound of a solve (zero-sum audit 2026-10-10, F2): chips are conserved on every line, so the
# sum of the players' root values (pcfr.exploitability "value": whole hand, pot units of the street
# start) is -E[rake] / pot0 <= -R(pot0) / pot0 for ANY strategy profile once the flop is dealt (0
# preflop: no flop, no drop). A sum above the bound is EV the street-end critic created; the k0
# stages had 24.5% (turn), 59.3% (flop) and 21% (preflop) of their spots above it. solve_summary
# reports the fraction above bound + PAYOUT_TOL per SPR bucket and warns above PAYOUT_WARN (not a
# failure yet: the stages of runs before the projection would fail).
PAYOUT_TOL = 1e-4
PAYOUT_WARN = 0.01


def payout_bounds(spots, street):
    """[count] the largest sum of root values the game can pay out: -R(pot0) / pot0, 0 preflop."""
    pot = np.asarray(spots["pot"], np.int64)
    rake = config.rake_from_terms(np.asarray(spots["rake_bps"]), np.asarray(spots["rake_cap"]), pot, street)
    return -np.asarray(rake, np.float64) / pot


def solver_config(args, cap):
    """The SolverConfig of a solve: certificate - stop at E <= e_target or `cap` iterations, E checked every 50;
    fixed T - `cap` = T iterations in chunks of FIXED_T_CHUNK, no E check (pcfr.solve_fixed)."""
    if args.teacher == FIXED_T:
        return SV.SolverConfig(cap=cap, e_target=0.0, check_every=FIXED_T_CHUNK)
    return SV.SolverConfig(cap=cap, e_target=args.e_target, check_every=50)


def batch_solver(street, args, cap, aux=None, equity=None, *, menu):
    """Jitted solve -> run(chunk, positions): critic root EVs and per-spot teacher diagnostics.
    Certificate mode stops at E <= e_target or cap; fixed T runs exactly cap iterations.
    """
    topo, st = tree_for(street, PLAYERS, menu)
    fixed = args.teacher == FIXED_T
    cfg = solver_config(args, cap)
    ex = jax.jit(jax.vmap(lambda z, s, a: SV.exploitability(z, s, st, cfg, a), in_axes=(0, 0, None)))
    root = jax.jit(jax.vmap(lambda z, s, a: SV.root_values(z, s, st, cfg, a), in_axes=(0, 0, None)))
    solver_args = {"aux": aux}                           # every SV.solve call below

    def solve_batch(batch):
        return SV.solve_fixed(batch, st, cfg, cap, **solver_args) if fixed else SV.solve(batch, st, cfg, **solver_args)

    def run(chunk, positions):
        count = len(positions)
        t0 = time.perf_counter()
        batch = SI.prepare(chunk, topo, preflop_equity=equity)
        t1 = time.perf_counter()
        state = solve_batch(batch)
        jax.block_until_ready(state["z"])
        t2 = time.perf_counter()
        res = ex(state["z"], batch, aux)               # the one E evaluation of the batch (fixed T: the diagnostic)
        E_EVALUATIONS[0] += 1
        ev = np.asarray(root(state["z"], batch, aux))
        cd = np.asarray(CR.spot_checkdown(batch, st))
        e = np.asarray(res["nash_conv"]) / st.n
        bad = np.flatnonzero(~np.isfinite(e) | (e < -SV.E_TOL))
        if len(bad):                  # pcfr.solve_cfr's rule (fixed T: the solve's only E check): a bug, not convergence
            raise RuntimeError(f"street {street}: E not finite or below -{SV.E_TOL:g} at positions "
                               f"{np.asarray(positions)[bad].tolist()} (E {e[bad].tolist()}): a solver bug (backward or "
                               f"terminals), not convergence")
        bound = payout_bounds(chunk, street)
        # E / acceptance / broken-solve gates stay in starting-pot units; BB is an additional diagnostic.
        metrics = [{"e_solve": float(e[i]), "e_solve_bb": float(e[i] * chunk["pot"][i] / config.BB),
                    "iters": int(state["u"][i]),
                    "spr": float(chunk["stack"][i] / chunk["pot"][i]), "rake_level": float(chunk["rake_level"][i]),
                    "values": np.asarray(res["value"][i]).tolist(), "payout_bound": float(bound[i]),
                    "solve_seconds_per_spot": (t2 - t1) / count,
                    "prepare_seconds_per_spot": (t1 - t0) / count} for i in range(count)]
        return {"batch": batch, "st": st, "rows": CR.rows_from_solves(chunk, batch, ev, st, cd), "cd": cd, "e": e,
                "certified": None if fixed else e <= args.e_target, "metrics": metrics,
                "iters": np.asarray(state["u"]), "seconds": t2 - t1, "prepare": t1 - t0}
    return run


def solve_stage(name, street, count, args, out, aux=None, equity=None):
    """Solve `count` spots of one street in claimed batches (stage_batches: one tree per batch; critic rows, E and cost per spot), then - certificate - the second pass over the spots left without the certificate,
    or - fixed T - the E flags of the stage (e_flag_pass). Returns the per-spot metrics once every batch exists."""
    spots, order = stage_spots(name, count, with_order=True)
    batches = stage_batches(name, count, args.solve_batch)
    meta = lambda s: out / f"{name}_{s:06d}.json"
    runs = {}                                          # tree_key -> batch_solver (one tree per batch)
    iterations = args.teacher_t[name] if args.teacher == FIXED_T else args.solver_cap
    fuse = SpeedFuse(float(os.environ["EXPECT_RIVER_SPM"])) if street == 3 and "EXPECT_RIVER_SPM" in os.environ else None
    if street == 3:
        log(f"{name}: speed fuse " + (f"on, {os.environ['EXPECT_RIVER_SPM']} spots/min per card expected"
                                      if fuse else "off (EXPECT_RIVER_SPM not set)"))
    for index, (start, stop, menu) in enumerate(batches):
        if args.shard and index % args.shard[1] != args.shard[0]:
            continue                                   # another machine's batch (--shard)
        if meta(start).exists() or not claim(out, f"{name}_{start:06d}"):
            continue
        t_batch = time.perf_counter()
        key = tree_key(PLAYERS, street, menu)
        if key not in runs:
            runs[key] = batch_solver(street, args, iterations, aux, equity, menu=menu)
        chunk = {k: v[start:stop] for k, v in spots.items()}
        r = runs[key](chunk, np.arange(start, stop))
        extra = {}
        if street == 3:              # exact SPR-0 copies (pure showdown), no solve needed
            allin = CR.allin_rows(chunk, r["batch"], r["st"], r["cd"])
            extra = {f"allin_{f}": getattr(allin, f) for f in CR.Rows.__dataclass_fields__}
        if r["certified"] is not None:
            extra["certified"] = r["certified"]
        savez(out / f"{name}_{start:06d}.npz", **game_field(args),
              **{f: getattr(r["rows"], f) for f in CR.Rows.__dataclass_fields__}, **extra)
        metrics = [{"spot": int(order[start + i]), **m, "pass": 1} for i, m in enumerate(r["metrics"])]
        write_text(meta(start), json.dumps(metrics))
        wall = time.perf_counter() - t_batch
        expected = fuse.predict(chunk["stack"] / chunk["pot"]) if fuse else None
        log(f"{name} {start}-{stop}/{count}" + (f" ({menu}, {iterations} iterations)" if args.teacher == FIXED_T else "")
            + f": {r['seconds'] / (stop - start):.3f} s/spot, prepare "
            f"{r['prepare'] / (stop - start):.3f} s/spot, E median {np.median(r['e']):.5f} max {r['e'].max():.5f}, "
            f"iters median {np.median(r['iters']):.0f}"
            + f", wall {wall:.0f} s" + (f" (expected {expected:.0f} s)" if fuse else ""))
        if fuse:
            fuse.add(chunk["stack"] / chunk["pot"], wall)
    if args.shard:
        return None                                    # gathered and resumed elsewhere
    wait_for([meta(s) for s, _, _ in batches], f"{name} batches")
    if args.teacher == FIXED_T:
        return e_flag_pass(name, count, args, out)
    return second_pass(name, street, count, args, out, aux, equity)


def e_flag_pass(name, count, args, out):
    """Fixed T: once every batch of a stage exists, one claimed pass sets "e_flag" (E > config.TEACHER_E_FLAG x the
    stage's median E) in every spot's metrics and writes <name>_eflags.json (T, E median / p90 / max, the threshold,
    the flagged positions). Idempotent on resume. Returns the per-spot metrics."""
    batches = stage_batches(name, count, args.solve_batch)
    meta = lambda s: out / f"{name}_{s:06d}.json"
    marker = out / f"{name}_eflags.json"
    if not marker.exists() and claim(out, f"{name}_eflags"):
        by_batch = {s: json.loads(meta(s).read_text()) for s, _, _ in batches}
        e = np.asarray([m["e_solve"] for s, _, _ in batches for m in by_batch[s]])
        nonfinite = np.flatnonzero(~np.isfinite(e))
        if len(nonfinite):            # the batch check stops them: a stage holding one was made without it
            raise RuntimeError(f"{name}: non-finite E at positions {nonfinite.tolist()}: no E flags on such a stage")
        median = float(np.median(e[np.isfinite(e)]))
        threshold = args.e_flag_factor * median
        flagged = []
        for s, _, _ in batches:
            for i, m in enumerate(by_batch[s]):
                m["e_flag"] = bool(m["e_solve"] > threshold)
                if m["e_flag"]:
                    flagged.append({"position": s + i, "spot": m["spot"], "e": m["e_solve"], "spr": m["spr"]})
            write_text(meta(s), json.dumps(by_batch[s]))
        write_text(marker, json.dumps({
            "iterations": args.teacher_t[name], "spots": len(e), "e_median": median, "e_p90": float(np.percentile(e, 90)),
            "e_max": float(e.max()), "factor": args.e_flag_factor, "threshold": threshold, "nonfinite": len(nonfinite),
            "flagged": flagged}, indent=1))
        log(f"{name}: fixed T {args.teacher_t[name]}, E median {median:.5f} p90 {np.percentile(e, 90):.5f} max "
            f"{e.max():.5f}; {len(flagged)} of {len(e)} spots flagged (E > {args.e_flag_factor:g} x median = {threshold:.5f})")
    wait_for([marker], f"{name} E flags")
    return [m for s, _, _ in batches for m in json.loads(meta(s).read_text())]


def second_pass(name, street, count, args, out, aux=None, equity=None):
    """Brief 7.12: spots not accepted after the first pass (cap --solver-cap; E > accept)
    are solved again from scratch with cap --second-cap, in SPR order and full batches, claimed like
    the first pass. One claimed merge then puts their critic rows, certificates and metrics into
    the first-pass files (first-pass numbers kept under "pass1"). A second-pass chunk holds spots of one tree (stage_batches). Idempotent on resume.
    Returns the per-spot metrics."""
    batches = stage_batches(name, count, args.solve_batch)
    starts = [s for s, _, _ in batches]
    meta = lambda s: out / f"{name}_{s:06d}.json"
    metrics = [m for s in starts for m in json.loads(meta(s).read_text())]
    first_e = np.asarray([m.get("pass1", m)["e_solve"] for m in metrics])
    redo = np.flatnonzero(first_e > args.accept[(name, PLAYERS)])
    marker = out / f"{name}_p2.json"
    if args.second_cap <= args.solver_cap or not len(redo) or marker.exists():
        return metrics
    batch_of = {p: (s, m) for s, e, m in batches for p in range(s, e)}            # position -> (batch start, menu)
    groups = {}                                       # tree -> its positions to redo, in SPR order
    for p in redo:
        groups.setdefault(tree_key(PLAYERS, street, batch_of[int(p)][1]), []).append(int(p))
    chunks = [(np.asarray(ps[i:i + args.solve_batch]), batch_of[ps[0]][1])
              for ps in groups.values() for i in range(0, len(ps), args.solve_batch)]
    tag = lambda k: f"{name}_p2_{k:06d}"
    spots, runs = None, {}
    for k, (sel, menu) in enumerate(chunks):
        if (out / f"{tag(k)}.json").exists() or not claim(out, tag(k)):
            continue
        spots = spots if spots is not None else stage_spots(name, count)
        tree = tree_key(PLAYERS, street, menu)
        if tree not in runs:
            runs[tree] = batch_solver(street, args, args.second_cap, aux, equity, menu=menu)
        r = runs[tree]({key: v[sel] for key, v in spots.items()}, sel)
        savez(out / f"{tag(k)}.npz", **game_field(args), position=sel, certified=r["certified"],
              **{f: getattr(r["rows"], f) for f in CR.Rows.__dataclass_fields__})
        write_text(out / f"{tag(k)}.json", json.dumps([{"position": int(p), **m} for p, m in zip(sel, r["metrics"])]))
        log(f"{name} second pass {k + 1}/{len(chunks)} (cap {args.second_cap}): {r['seconds'] / len(sel):.3f} s/spot, "
            f"certified {int(r['certified'].sum())}/{len(sel)}, E median {np.median(r['e']):.5f} max {r['e'].max():.5f}, "
            f"iters median {np.median(r['iters']):.0f}")
    wait_for([out / f"{tag(k)}.json" for k in range(len(chunks))], f"{name} second-pass batches")
    if claim(out, f"{name}_p2_merge"):
        fields = CR.Rows.__dataclass_fields__
        patches = {}
        for k in range(len(chunks)):
            data = np.load(out / f"{tag(k)}.npz")
            check_stage_file(out / f"{tag(k)}.npz", data, args.game_hash)
            for j, m in enumerate(json.loads((out / f"{tag(k)}.json").read_text())):
                pos = m["position"]
                patches.setdefault(batch_of[pos][0], []).append(
                    (pos, m, {f: data[f][j] for f in fields}, bool(data["certified"][j])))
        for start, items in sorted(patches.items()):
            data = np.load(out / f"{name}_{start:06d}.npz")
            arrays = {f: data[f].copy() for f in data.files}
            mets = json.loads(meta(start).read_text())
            for pos, m, row, ok in items:
                i = pos - start
                for f in fields:
                    arrays[f][i] = row[f]
                arrays["certified"][i] = ok
                first = mets[i].get("pass1") or {key: mets[i][key] for key in
                                                 ("e_solve", "iters", "values", "solve_seconds_per_spot",
                                                  "prepare_seconds_per_spot")}
                mets[i].update({key: v for key, v in m.items() if key != "position"})
                mets[i].update(pass1=first, **{"pass": 2})
            savez(out / f"{name}_{start:06d}.npz", **arrays)
            write_text(meta(start), json.dumps(mets))
        merged = [m for s in starts for m in json.loads(meta(s).read_text())]
        again = [m for m in merged if m.get("pass") == 2]
        write_text(marker, json.dumps({
            "second_cap": args.second_cap, "resolved": len(again), "accept_e": args.accept[(name, PLAYERS)],
            "certified": int(sum(m["e_solve"] <= args.e_target for m in again)),
            "accepted": int(accepted(again, args.accept[(name, PLAYERS)]).sum()),
            "seconds": float(sum(m["solve_seconds_per_spot"] for m in again)),
            "by_spr": spr_table(merged, args, args.accept[(name, PLAYERS)])}, indent=1))
        log(f"{name} second pass merged: {len(again)} spots solved again, "
            f"{sum(m['e_solve'] <= args.e_target for m in again)} now certified, "
            f"{int(accepted(again, args.accept[(name, PLAYERS)]).sum())} accepted (E <= {args.accept[(name, PLAYERS)]:.4f})")
    wait_for([marker], f"{name} second-pass merge")
    return [m for s in starts for m in json.loads(meta(s).read_text())]


def game_field(args):
    """Stage provenance: the current game hash, sampling protocol hash and source fingerprint."""
    fp = FP.code_fingerprint()
    if not args.game_hash:
        raise ValueError("stage data requires the current game hash")
    code = {"code_sha256": np.asarray(fp["sha256"]), "code_commit": np.asarray(str(fp["commit"])),
            "training_domain_hash": np.asarray(SI.training_domain_record()["protocol_hash"])}
    return {"game_hash": np.asarray(args.game_hash), **code}


def check_stage_file(path, data, game_hash):
    """A critic stage file must match the run's current game and sampling protocol."""
    saved = str(data["game_hash"]) if "game_hash" in data.files else None
    if not game_hash or saved != game_hash:
        raise SystemExit(f"{path}: stage file of game {str(saved)[:12]}, the run's layout is game {str(game_hash)[:12]} "
                         f"(config.game_record): a file of another game in this run")
    saved_domain = str(data["training_domain_hash"]) if "training_domain_hash" in data.files else None
    want_domain = SI.training_domain_record()["protocol_hash"]
    if saved_domain != want_domain:
        raise SystemExit(f"{path}: missing or different training domain in stage data; use a new --out")


def load_rows(name, count, args, out):
    """Critic rows of the accepted spots (E <= args.accept[(name, PLAYERS)]: from the batch's metrics;
    fixed T: every spot); river batches also carry exact all-in rows (returned separately so the test split stays
    on solved spots). Every stage file's game is checked (check_stage_file)."""
    parts, allins, kept = [], [], 0
    for s in batch_starts(name, count, args.solve_batch):
        data = np.load(out / f"{name}_{s:06d}.npz")
        check_stage_file(out / f"{name}_{s:06d}.npz", data, args.game_hash)
        mask = accepted(json.loads((out / f"{name}_{s:06d}.json").read_text()), args.accept[(name, PLAYERS)])
        if len(mask) != len(data["pub"]):
            raise ValueError(f"{name}_{s:06d}: {len(mask)} metrics for {len(data['pub'])} rows")
        kept += int(mask.sum())
        parts.append(CR.rows_from_npz(data).take(np.flatnonzero(mask)))
        if "allin_pub" in data.files:
            allins.append(CR.rows_from_npz(data, "allin_"))
    return CR.Rows.concat(parts), (CR.Rows.concat(allins) if allins else None), kept


def mix_unit(keys, seed=0):
    """A uniform number in [0, 1) per uint64 key (splitmix64)."""
    with np.errstate(over="ignore"):
        x = np.asarray(keys, np.uint64) + np.uint64(0x9E3779B97F4A7C15) * np.uint64(seed + 1)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / 2.0 ** 53


def board_class_code(board1h):
    """uint64 [N] id of the suit-isomorphism class of each board (one-hot [N, 52]): the smallest 52-bit
    card-set code over the 24 suit permutations (poker/solver/flops.py CARD_PERM)."""
    bits = (np.asarray(board1h) > 0).astype(np.uint64)                                  # [N, 52]
    images = np.uint64(1) << FL.CARD_PERM.astype(np.uint64)                             # [24, 52]: card -> image bit
    return np.min(np.stack([bits @ w for w in images]), axis=0)


def board_unit(board1h, seed=0):
    """mix_unit of the board's suit-isomorphism class (board_class_code): the same number for every
    board of a class, so rows on one board - and on its suit permutations - always land in the same
    split. The critics train suit-augmented (all 24 permutations), so a test board isomorphic to a
    training board is a training board for them: 24.3% of the 45k river set's test boards had such a
    twin under the exact-board split (efficiency audit 2026-10-10, item 5)."""
    return mix_unit(board_class_code(board1h), seed)


def split_cuts(units):
    """Thresholds (a, b) on split units: below a train, [a, b) validate, from b test. Set on the
    distinct units (boards) so the fractions (SPLIT) are exact in boards and every part has one
    once there are three."""
    uniq = np.unique(units)
    n = len(uniq)
    n_val, n_test = (max(1, round(SPLIT[1] * n)), max(1, round(SPLIT[2] * n))) if n >= 3 else (0, 0)
    a = uniq[n - n_val - n_test] if n_val + n_test else np.inf
    b = uniq[n - n_test] if n_test else np.inf
    return float(a), float(b)


def split_rows(rows: CR.Rows, seed=0, cuts=None, by_board=True):
    """Train / val / test (SPLIT) by board, not by row: the flop pool repeats boards, and a test
    on boards the critic trained on would flatter it. Rows that share a board share a
    part; other row sets on the same boards (river all-in rows) take the same `cuts`. by_board=False
    splits by row (situations without a board: the end of preflop)."""
    u = board_unit(rows.board, seed) if by_board else mix_unit(np.arange(len(rows), dtype=np.uint64), seed)
    a, b = cuts or split_cuts(u)
    return (rows.take(np.flatnonzero(u < a)), rows.take(np.flatnonzero((u >= a) & (u < b))),
            rows.take(np.flatnonzero(u >= b)))


def spr_buckets(rows: CR.Rows, errors):
    """Mean critic error for SPR < 6 and >= 6 (SPR = slot-0 chips behind / pot)."""
    spr = np.expm1(rows.pub[:, 4 + CR.P_MAX])           # log1p(behind / pot) of slot 0
    deep = spr >= 6.0
    return {"spr_lt_6": float(np.mean(errors[~deep])) if np.any(~deep) else None,
            "spr_ge_6": float(np.mean(errors[deep])) if np.any(deep) else None}


def spr_table_rows(rows: CR.Rows, errors, name="mae"):
    """Rows and mean critic error per SPR bucket (SPR_EDGES; SPR = slot-0 chips behind / pot)."""
    spr = np.expm1(rows.pub[:, 4 + CR.P_MAX])
    out = {}
    for lo, hi in zip(SPR_EDGES[:-1], SPR_EDGES[1:]):
        m = (spr >= lo) & (spr < hi)
        out[f"{lo:g}-{hi:g}"] = {"n": int(m.sum()), name: float(np.mean(errors[m])) if m.any() else None}
    return out


def error_report(rows: CR.Rows, metrics):
    """Test numbers beyond the MAE (critics.row_metrics; audit 2026-10-10, F4): mean, median and SPR
    table of the range-EV offset, the centred MAE and the worst substantial hand. Zero-sum audit:
    the raw network's MAE (before the projection), its chip-conservation violation S(V_raw) - T
    (mean, mean |.|, signed by SPR: what the projection removes) and the targets' own S(target) - T
    (mean, |.|, by SPR: the data's extra rake against kappa; where kappa is a placeholder (0),
    -test_target_total estimates it)."""
    out = {}
    for name in ("offset", "centred", "worst_substantial"):
        x = metrics[name]
        out[f"test_{name}"], out[f"test_{name}_median"] = float(np.mean(x)), float(np.median(x))   # as test_mae
        out[f"test_{name}_by_spr"] = spr_table_rows(rows, x, name)
    if "raw_mae" in metrics:
        out["test_raw_mae"] = float(np.mean(metrics["raw_mae"]))
        for name in CR.TOTALS:
            x = metrics[name]
            out[f"test_{name}"], out[f"test_{name}_abs"] = float(np.mean(x)), float(np.mean(np.abs(x)))
            out[f"test_{name}_by_spr"] = spr_table_rows(rows, x, name)
    return out


def critic_protocol(args):
    """The critics' training protocol as a stage report records it. lr: the run's (layout critic_lr,
    read back by main and passed to CR.train by train_seed), CR.PROTOCOL's for callers without one
    (explicit diagnostic callers and checks)."""
    return {"epochs": args.critic_epochs, "epoch_steps": args.critic_epoch_steps, "patience": args.critic_patience,
            "clip_norm": args.critic_clip_norm, "lr": getattr(args, "critic_lr", CR.PROTOCOL["lr"])}


def kappa_of(name, args):
    """kappa of a loop critic's projection (critics.critic_kappa): the run's own values (layout
    "critic_projection", args.kappa) or this code's."""
    return CR.critic_kappa(name, getattr(args, "kappa", None))


def checkpoint_sha256(path):
    """Bind a completed stage report to the exact atomic checkpoint it describes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage_critic(name, args, out, report=None):
    """A completed stage needs its exact valid checkpoint, report and run projection."""
    path = out / f"{name}.npz"
    report = json.loads((out / f"{name}.json").read_text()) if report is None else report
    if report.get("name") != name or report.get("checkpoint_sha256") != checkpoint_sha256(path):
        raise ValueError(f"{name}: stage report does not match its checkpoint; use a new --out or repair the stage")
    params, meta = load_critic(path, name=name)
    kappa = kappa_of(name, args)
    metric = "test_mae_vs_exact_average" if name.endswith("_end") else "test_mae"
    if (meta.get("split_version") != SPLIT_VERSION or meta["kappa"] != kappa or report.get("kappa") != kappa
            or report.get("protocol") != critic_protocol(args)):
        raise ValueError(f"{name}: checkpoint/report training metadata differs from the run")
    fields = (metric, "identity_gap_max", "identity_gap_mean", "identity_gap_p99") if name.endswith("_end") else (metric,)
    if any(not isinstance(report.get(key), (int, float)) or not np.isfinite(report[key]) or report[key] < 0
           for key in fields) or meta.get("test_mae") != report[metric]:
        raise ValueError(f"{name}: invalid or mismatched checkpoint/report metrics")
    return params, meta


def train_seed(train, val, args, seed, kappa):
    """Train this seed once with the run's saved protocol. Numerical failure propagates without
    substitute seeds. Returns (seed, params, info, training seconds)."""
    t0 = time.perf_counter()
    params, info = CR.train(train, val, hidden=args.critic_hidden, seed=seed, epochs=args.critic_epochs,
                            epoch_steps=args.critic_epoch_steps, patience=args.critic_patience,
                            lr=getattr(args, "critic_lr", CR.PROTOCOL["lr"]), clip_norm=args.critic_clip_norm,
                            log=log if args.verbose else None, kappa=kappa)
    return seed, params, info, time.perf_counter() - t0


def train_best(train, val, args, val_allin=None, kappa=0.0):
    """One network of a critic stage: the protocol of CR.PROTOCOL
    (the learning curves' one) with seeds 0 and 1; the network with the better validation MAE is
    kept (validation - solved rows only, critic_split - never the test). Without validation rows
    (toy runs) the first seed's network is kept. Both seeds' training numbers are reported
    (val_allin_mae: the SPR-0 copies of the validation boards, apart from the selection). kappa: the
    critic's projection (kappa_of; trained and measured projected, critics.train). Both fixed seeds
    must finish; a numerical failure stops the stage without trying another seed. Finite loss
    spike warnings remain in each seed's training record."""
    runs = []
    for first in (0, 1):
        seed, params, info, seconds = train_seed(train, val, args, first, kappa)
        t0 = time.perf_counter()
        hist = info.pop("history")
        allin_mae = (float(np.mean(CR.row_metrics(params, val_allin, kappa=kappa)["mae"]))
                     if val_allin is not None and len(val_allin) else None)
        runs.append({"seed": seed, **info, "val_allin_mae": allin_mae,
                     "curve": {"train_loss": [h["train_loss"] for h in hist], "val_mae": [h["val_mae"] for h in hist]},
                     "seconds": seconds + time.perf_counter() - t0, "params": params})
    scored = [r for r in runs if r["val_mae"] is not None]
    best = min(scored, key=lambda r: r["val_mae"]) if scored else runs[0]
    return best["params"], {"chosen_seed": best["seed"], "chosen_val_mae": best["val_mae"],
                            "chosen_val_allin_mae": best["val_allin_mae"],
                            "seeds": [{k: v for k, v in r.items() if k != "params"} for r in runs]}


def critic_split(rows, allin):
    """Board split of a street critic (fit_critic): train = the solved rows of the training boards
    plus the river's exact SPR-0 copies of the spots on them; validation and test = solved rows only
    (audit 2026-10-10, F7: a validation half made of SPR-0 copies, fitted to ~0.001, read half the
    error and halved the selection signal); the SPR-0 copies of the validation and test boards come
    apart (a_val, a_test; None without SPR-0 rows), on the same board cuts."""
    cuts = split_cuts(board_unit(rows.board))
    train, val, test = split_rows(rows, cuts=cuts)
    a_val = a_test = None
    if allin is not None:
        a_train, a_val, a_test = split_rows(allin, cuts=cuts)
        train = CR.Rows.concat([train, a_train])
    return train, val, test, a_val, a_test


def fit_critic(rows, allin, args, kappa=0.0):
    """Board split (critic_split), training (train_best) and test of a street critic: certified
    rows (+ the river's exact SPR-0 rows) -> (params, report). critic_stage and explicit training diagnostics share it. kappa: the critic's projection (kappa_of); every error is of the
    projected values unless named raw."""
    train, val, test, a_val, a_test = critic_split(rows, allin)
    t0 = time.perf_counter()
    params, info = train_best(train, val, args, val_allin=a_val, kappa=kappa)
    m = CR.row_metrics(params, test, kappa=kappa)
    base = CR.checkdown_errors(test)
    result = {"rows": len(rows), "train": len(train), "val": len(val), "test": len(test),
              "val_allin": len(a_val) if a_val is not None else 0, "kappa": kappa,
              "test_allin_mae": (float(np.mean(CR.row_metrics(params, a_test, kappa=kappa)["mae"]))
                                 if a_test is not None else None),
              "test_checkdown_mae": float(np.mean(base)) if len(base) else None,
              "test_checkdown_mae_by_spr": spr_table_rows(test, base) if len(base) else None,
              "test_mae": float(np.mean(m["mae"])), "test_mae_median": float(np.median(m["mae"])),
              "test_worst_median": float(np.median(m["worst"])), "test_mae_by_spr": spr_table_rows(test, m["mae"]),
              **error_report(test, m),
              "val_mae": info["chosen_val_mae"], "val_allin_mae": info["chosen_val_allin_mae"], "training": info,
              "protocol": critic_protocol(args),
              "seconds": time.perf_counter() - t0, "hidden": list(args.critic_hidden)}
    return params, result


def seeds_line(training):
    return ", ".join(f"seed {r['seed']} val {r['val_mae']:.4f} (best epoch {r['best_epoch']}/{r['epochs_run']}, "
                     f"lr {r['lr_at_best']:.1e})"
                     if r["val_mae"] is not None else f"seed {r['seed']} no validation" for r in training["seeds"])


def critic_stage(name, rows_name, count, args, out):
    path = out / f"{name}.npz"
    report = out / f"{name}.json"
    if report.exists() or not claim(out, name):
        wait_for([report], name)
        result = json.loads(report.read_text())
        load_stage_critic(name, args, out, result)
        return result
    rows, allin, kept = load_rows(rows_name, count, args, out)
    params, result = fit_critic(rows, allin, args, kappa=kappa_of(name, args))
    accept = args.accept[(rows_name, PLAYERS)]
    result = {"name": name, "accepted_spots": kept, "accept_e": accept if np.isfinite(accept) else None, **result}
    save_critic(path, params, {"name": name, "test_mae": result["test_mae"], "kappa": result["kappa"]})
    result["checkpoint_sha256"] = checkpoint_sha256(path)
    write_text(report, json.dumps(result, indent=1))
    log(f"{name}: test MAE {result['test_mae']:.4f} pot (median {result['test_mae_median']:.4f}; offset "
        f"{result['test_offset']:.4f}, centred {result['test_centred']:.4f}, worst substantial hand median "
        f"{result['test_worst_substantial_median']:.3f}; check-down alone {result['test_checkdown_mae']}), "
        f"{result['train']} train rows; {seeds_line(result['training'])}; kept seed "
        f"{result['training']['chosen_seed']}; {result['seconds']:.0f} s; {total_line(result)}")
    return result


def total_line(result):
    """Log fragment of a critic report: the projection and what it removes (error_report)."""
    return (f"projected with kappa {result['kappa']:.4f}: raw network MAE {result['test_raw_mae']:.4f}, its total "
            f"S - T {result['test_raw_total']:+.4f} (|.| {result['test_raw_total_abs']:.4f}); targets' S - T "
            f"{result['test_target_total']:+.4f} (|.| {result['test_target_total_abs']:.4f})")


# ---------------------------------------------------------------- end of street (auxiliary critic)

def end_situations(seed, count, street_after):
    """Shared training envelope before the next card, including exact all-in cases."""
    return SI.make_training_spots(seed, count, PLAYERS, rake="mix", street=street_after - 1,
                                  allin_fraction=SI.TRAINING_ALLIN_FRACTION)


def aux_device():
    """Device for the auxiliary critics' targets: the GPU, except Pascal (compute capability 6.x),
    where the card loop around the critic fuses into kernels it cannot load (same limit as the
    engine replay); there the CPU does it."""
    dev = jax.devices()[0]
    if dev.platform != "gpu":
        return dev
    cc = getattr(dev, "compute_capability", None)
    pascal = float(cc) < 7.0 if cc else any(t in dev.device_kind for t in ("GTX 10", "P100", "P40", "P4"))
    return TR.require_cpu("poker.solver.loop.aux_device on a Pascal GPU") if pascal else dev


def situation_rows(ranges_seat, pot, behind_pot, rake_level, rake_bps, rake_cap):
    """(ranges [S, P_MAX, H], present [S, P_MAX], pub [S, PUB]) of two-player situations at a street
    start: slot 0 = seat 1 (acts first postflop), slot 1 = seat 0; both put half the pot in."""
    count = len(pot)
    pot = np.asarray(pot, np.float64)
    ranges = np.zeros((count, CR.P_MAX, CR.NUM_HANDS), np.float32)
    ranges[:, 0], ranges[:, 1] = ranges_seat[:, 1], ranges_seat[:, 0]
    present = np.zeros((count, CR.P_MAX), np.float32)
    present[:, :2] = 1.0
    behind = np.zeros((count, CR.P_MAX), np.float32)
    behind[:, :2] = np.asarray(behind_pot, np.float32)[:, None]
    contrib = np.zeros((count, CR.P_MAX), np.float32)
    contrib[:, :2] = 0.5
    pub = jax.vmap(CR.public_row)(jnp.asarray(pot / config.BB, jnp.float32), jnp.asarray(rake_level, jnp.float32),
                                  jnp.asarray(np.asarray(rake_bps) / 500.0, jnp.float32),
                                  jnp.asarray(np.asarray(rake_cap) / pot, jnp.float32), jnp.asarray(present),
                                  jnp.asarray(behind), jnp.asarray(contrib))
    return ranges, present, np.asarray(pub)


def identity_gap(avg, own, ranges2, valid):
    """Per situation: max |card average of the next street's check-down - own check-down| over the
    valid hands of both slots whose compatible opponent mass is >= 1e-6 (the identity; float32)."""
    opp = ranges2[:, ::-1]
    mass = np.asarray(jax.vmap(jax.vmap(CD.j_mass))(jnp.asarray(opp)))
    ok = (mass >= 1e-6) & (valid[:, None, :] > 0)
    return np.where(ok, np.abs(np.asarray(avg) - np.asarray(own)), 0.0).max(axis=(1, 2))


def end_rows(name, spots, next_critic, table, batch_size):
    """Rows of an auxiliary critic, two players: turn_end = the river critic averaged over the river
    cards, flop_end = the turn critic averaged over the turn cards (card removal). V = network + CD:
    CD and F of a situation come from its board's all-in matrix, the function the solver's street
    ends use; the target is the card average of the next critic's network part plus that CD (brief
    M2, step 0). The card average of the next critic's F is the identity's other side; its gap to
    CD is returned per situation."""
    count = len(spots["pot"])
    pot = np.asarray(spots["pot"])
    ranges, present, pub = situation_rows(spots["ranges"], pot, spots["stack"] / pot, spots["rake_level"],
                                          spots["rake_bps"], spots["rake_cap"])
    net = np.asarray(CD.net_fraction(pot, spots["rake_bps"], spots["rake_cap"]))
    cards = {"turn_end": 4, "flop_end": 3}[name]
    average = {"turn_end": CR.end_of_turn, "flop_end": CR.end_of_flop}[name]
    matrix = {"turn_end": RO.equity_matrix_turn, "flop_end": RO.equity_matrix_flop}[name]
    dev = aux_device()
    critic, table_dev = jax.device_put(next_critic, dev), jax.device_put(table, dev)
    fn = jax.jit(jax.vmap(average, in_axes=(None, 0, 0, 0, 0, None)))
    own_fn = jax.jit(lambda b, r, t: jax.lax.map(
        lambda x: CD.two(lambda w: CD.share_matrix(w, matrix(x[0], t)), x[1]), (b, r)))
    boards = np.asarray(spots["board"])[:, :cards].astype(np.int32)
    res, avg, own = [], [], []
    for i in range(0, count, batch_size):
        sl = slice(i, i + batch_size)
        a, c = fn(critic, *(jax.device_put(x, dev) for x in (boards[sl], pub[sl], ranges[sl, :2], net[sl])), table_dev)
        res.append(np.asarray(a))
        avg.append(np.asarray(c))
        own.append(np.asarray(own_fn(jnp.asarray(boards[sl]), jnp.asarray(ranges[sl, :2]), table)) * net[sl, None, None])
    res, avg, own = (np.concatenate(x) for x in (res, avg, own))
    # With no chips behind, the target is exact showdown; no future decision uses a critic.
    res = np.where(np.asarray(spots["stack"])[:, None, None] > 0, res, 0.0)
    valid = np.asarray(spots["valid"], np.float32)
    gap = identity_gap(avg, own, ranges[:, :2], valid)
    board = np.asarray(jax.vmap(CR.board_onehot)(jnp.asarray(spots["board"])))
    rows = CR.Rows(board, pub, ranges, CR.pad_slots(res + own) * valid[:, None, :], valid, present,
                   CR.pad_slots(own) * valid[:, None, :])
    return rows, gap


def _matvec(matrix, x):
    return x @ matrix.T


def pre_end_rows(seed, count, next_critic, batch_size=1024):
    """Situations at the flop start (no board): pot in whole chips, chips behind, suit-symmetric
    ranges (one weight per 169-class, as a preflop solved in class space leaves them), rake; V =
    network + CD with CD from the exact preflop all-in matrix (the solver's preflop street ends use
    the same); target = the flop critic's network part averaged over all 22,100 flops through the
    1755 classes with card removal (critics.end_of_preflop_classes), F of each flop from
    its exact all-in matrix (cached per flop), plus CD. Returns the rows and the identity gap."""
    rng = np.random.default_rng([seed, 41])
    chips = SI.sample_training_chips(rng, count, allin_fraction=SI.TRAINING_ALLIN_FRACTION)
    pot, spr = chips["pot"], chips["stack"] / chips["pot"]
    seat_ranges = np.zeros((count, 2, CR.NUM_HANDS), np.float32)
    for i in range(count):
        seat_ranges[i, 1] = SI.class_range(rng)                          # slot 0 = seat 1, as before
        seat_ranges[i, 0] = SI.class_range(rng)
    rake = SI.sample_rake(seed, count, PLAYERS, "mix")
    ranges, present, pub = situation_rows(seat_ranges, pot, spr, rake["rake_level"], rake["rake_bps"], rake["rake_cap"])
    net = np.asarray(CD.net_fraction(pot, rake["rake_bps"], rake["rake_cap"]))
    equity = RO.preflop_equity_cached(PREFLOP_EQUITY, log=log)
    own = np.asarray(jax.vmap(lambda r: CD.two(lambda w: CD.share_matrix(w, equity), r))(jnp.asarray(ranges[:, :2])))
    own = own * net[:, None, None]
    matvec_for = lambda flop: jax.tree_util.Partial(_matvec, RO.flop_equity_cached(flop))
    res, avg = [], []
    for i in range(0, count, batch_size):
        sl = slice(i, i + batch_size)
        a, c = CR.end_of_preflop_classes(next_critic, jnp.asarray(pub[sl]), jnp.asarray(ranges[sl, :2]),
                                         jnp.asarray(net[sl]), matvec_for, log=log)
        res.append(np.asarray(a))
        avg.append(np.asarray(c))
    res, avg = np.concatenate(res), np.concatenate(avg)
    res = np.where(chips["stack"][:, None, None] > 0, res, 0.0)  # all-in: exact check-down target
    valid = np.ones((count, CR.NUM_HANDS), np.float32)
    gap = identity_gap(avg, own, ranges[:, :2], valid)
    rows = CR.Rows(np.zeros((count, 52), np.float32), pub, ranges, CR.pad_slots(res + own), valid, present,
                   CR.pad_slots(own))
    return rows, gap


def end_stage(name, next_name, count, args, out):
    report = out / f"{name}.json"
    if report.exists() or not claim(out, name):
        wait_for([report], name)
        result = json.loads(report.read_text())
        load_stage_critic(name, args, out, result)
        return result
    next_params, next_meta = load_stage_critic(next_name, args, out)
    next_critic = CR.net_critic(next_params, next_meta["kappa"])               # the checkpoint's validated projection
    kappa = kappa_of(name, args)
    t0 = time.perf_counter()
    if name == "pre_end":
        rows, gap = pre_end_rows(SEEDS[name], count, next_critic)
    else:
        table = rank7.cached_table()
        if table is None:
            raise RuntimeError(f"{rank7.TABLE_PATH} missing: the loop builds it at the start")
        rows, gap = end_rows(name, end_situations(SEEDS[name], count, {"turn_end": 3, "flop_end": 2}[name]),
                             next_critic, table, batch_size={"turn_end": 64, "flop_end": 8}[name])
    t1 = time.perf_counter()
    train, val, test = split_rows(rows, by_board=name != "pre_end")
    params, info = train_best(train, val, args, kappa=kappa)
    m = CR.row_metrics(params, test, kappa=kappa)
    base = CR.checkdown_errors(test)
    result = {"name": name, "rows": len(rows), "test": len(test), "test_mae_vs_exact_average": float(np.mean(m["mae"])),
              "test_mae_median": float(np.median(m["mae"])), "test_worst_median": float(np.median(m["worst"])),
              "test_mae_by_spr": spr_table_rows(test, m["mae"]), **error_report(test, m),
              "test_checkdown_mae": float(np.mean(base)), "test_checkdown_mae_by_spr": spr_table_rows(test, base),
              "identity_gap_max": float(gap.max()), "identity_gap_mean": float(gap.mean()),
              "identity_gap_p99": float(np.percentile(gap, 99)), "identity_situations": int(len(gap)),
              "val_mae": info["chosen_val_mae"], "training": info, "protocol": critic_protocol(args), "kappa": kappa,
              "target_seconds": t1 - t0, "train_seconds": time.perf_counter() - t1}
    path = out / f"{name}.npz"
    save_critic(path, params, {"name": name, "test_mae": result["test_mae_vs_exact_average"], "kappa": kappa})
    result["checkpoint_sha256"] = checkpoint_sha256(path)
    write_text(report, json.dumps(result, indent=1))
    log(f"{name}: error vs the exact card average {result['test_mae_vs_exact_average']:.4f} pot (offset "
        f"{result['test_offset']:.4f}, centred {result['test_centred']:.4f}; check-down alone "
        f"{result['test_checkdown_mae']:.4f}) on {len(test)} situations; identity gap max "
        f"{result['identity_gap_max']:.2e} over {len(gap)} situations; {seeds_line(info)}; kept seed "
        f"{info['chosen_seed']}; targets {t1 - t0:.0f} s; {total_line(result)}")
    return result


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    p.add_argument("--river", type=int, default=3000)
    p.add_argument("--turn", type=int, default=2000)
    p.add_argument("--turn-end", type=int, default=3000)
    p.add_argument("--flop", type=int, default=2000)
    p.add_argument("--flop-end", type=int, default=3000)
    p.add_argument("--pre-end", type=int, default=3000)
    p.add_argument("--solve-batch", type=int, default=32)
    p.add_argument("--solver-cap", type=int, default=4000)
    p.add_argument("--second-cap", type=int, default=12000,
                   help="cap of the second pass over spots without the certificate (brief 7.12)")
    p.add_argument("--e-target", type=float, default=0.002)
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--fixed-t", dest="teacher", action="store_const", const=FIXED_T,
                      help="fixed T per street (config.TEACHER_T), E once at the end, every spot into the data (the "
                           "default under grid 13; memo section 12, decision #25)")
    stop.add_argument("--certificate", dest="teacher", action="store_const", const=CERTIFICATE,
                      help="stop on E <= --e-target or --solver-cap, second pass, E-target acceptance (the default under "
                           "grids 5 and 7: k0 and the checks)")
    p.add_argument("--critic-hidden", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--critic-epochs", type=int, default=CR.PROTOCOL["epochs"],
                   help="training protocol of every critic (critics.PROTOCOL): epochs of --critic-epoch-steps batches")
    p.add_argument("--critic-epoch-steps", type=int, default=CR.PROTOCOL["epoch_steps"])
    p.add_argument("--critic-patience", type=int, default=CR.PROTOCOL["patience"],
                   help="early stop after this many epochs without a better validation MAE (default: none, the "
                        "protocol runs its whole budget and keeps the best validation epoch)")
    p.add_argument("--critic-clip-norm", type=float, default=CR.PROTOCOL["clip_norm"],
                   help="global-norm gradient clipping of the critics (0: none)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="processes sharing --out (1 = clear stale claims)")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume a run that was started by other code (recorded in code_resumes.json)")
    p.add_argument("--code-change-reason", help="why (required with --allow-code-change; kept in code_resumes.json)")
    p.add_argument("--shard", help="i/k: solve only the river's first-pass batches with index %% k == i and stop "
                                   "(machines without a shared disk; gather the batch files, then resume as one run)")
    args = p.parse_args(argv)
    asked_teacher = args.teacher                     # None: the run's own (a resume) or the grid's default
    if args.shard:
        i, k = (int(x) for x in args.shard.split("/"))
        if not 0 <= i < k or args.stages != ["river_solve"]:
            p.error("--shard i/k needs 0 <= i < k and --stages river_solve")
        args.shard = (i, k)
    if args.quick:
        args.river, args.turn, args.turn_end, args.flop, args.flop_end = 48, 24, 64, 16, 16
        args.pre_end = 32
        args.solve_batch, args.solver_cap, args.second_cap, args.e_target = 16, 200, 400, 0.05
        # the critics' protocol scaled by 1/20 (400 -> 20 epochs of 25 steps; no early stop, as the protocol)
        args.critic_hidden, args.critic_epochs, args.critic_epoch_steps = [64, 64], 20, 25
    args.critic_hidden = tuple(args.critic_hidden)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.workers == 1:            # a lone process owns every unfinished item (resume after a crash)
        shutil.rmtree(args.out / "claims", ignore_errors=True)
    left_out = config.heads_up_part("poker.solver.loop")   # the loop solves two-player spots (PLAYERS)
    new_teacher = asked_teacher or (FIXED_T if config.GRID == config.STREET_SIZES_GRID else CERTIFICATE)
    wanted = {"scheme": config.scheme(), "regime": config.regime_record(), "players_solved": [PLAYERS],
              "training_domain": SI.training_domain_record(),
              "iteration": 0, "river": args.river, "turn": args.turn,
              "turn_end": args.turn_end, "flop": args.flop, "flop_end": args.flop_end,
              "pre_end": args.pre_end, "solve_batch": args.solve_batch}
    if new_teacher == CERTIFICATE:
        wanted.update({"solver_cap": args.solver_cap, "second_cap": args.second_cap, "e_target": args.e_target})
    wanted.update({"critic_hidden": list(args.critic_hidden),
                   "critic_epochs": args.critic_epochs, "critic_epoch_steps": args.critic_epoch_steps,
                   "critic_patience": args.critic_patience, "critic_clip_norm": args.critic_clip_norm,
                   "critic_lr": CR.PROTOCOL["lr"],              # read back on resume (train_seed), as critic_clip_norm
                   "quick": args.quick, "code": code_version(),
                   "matmul_precision": str(jax.config.jax_default_matmul_precision),
                   "rake_mix": [list(r) for r in config.RAKE_MIX], "critic_form": CR.config_payload()["form"]})
    if new_teacher == CERTIFICATE:
        wanted.update({"accept_e": accept_thresholds(args.e_target),
                       "accept_rule": "a spot enters the critic data when E <= e_target; the second pass solves spots above it",
                       "stop_e": stop_thresholds(args.e_target)})
    else:
        wanted["accept_rule"] = "fixed T: every solved spot enters the critic data (no acceptance threshold)"
    wanted.update({"split_version": SPLIT_VERSION,
                   "critic_projection": {"kappa": dict(CR.KAPPA), "street_of": dict(CR.KAPPA_STREET),
                                         "rule": "every critic call and the critics' training loss project the values "
                                                 "onto S(V) = net0 - min(kappa, headroom) (critics.project_values)"},
                   "teacher": ({"stop": FIXED_T, "iterations": dict(config.TEACHER_T), "chunk": FIXED_T_CHUNK,
                                "e_flag_factor": config.TEACHER_E_FLAG, "rule": FIXED_T_RULE} if new_teacher == FIXED_T
                               else {"stop": CERTIFICATE, "rule": "E-stop at e_target or solver_cap, second pass at "
                                                                  "second_cap, E-target acceptance (accept_e)"}),
                   "trees": f"config.tree_rule({PLAYERS}, street, menu) (poker/solver/tree.py rule_topology; players: "
                            f"config.LOOP_PLAYERS); menu = config.menu_for of the spot SPR at the street start; "
                            f"one tree per batch (stage_batches)",
                   **config.game_meta()})
    layout = json.loads(publish_once(args.out / "layout.json", json.dumps(wanted, indent=1)))
    config.check_scheme(layout.get("scheme"), str(args.out / "layout.json"))
    made_under = config.saved_regime(layout.get("scheme"))
    if made_under != wanted["regime"]:              # POKER_REGIME_FORCE let it through check_scheme: not here
        raise SystemExit(f"{args.out}: made under {config.describe_regime(made_under)}; the loop never mixes regimes "
                         f"(POKER_REGIME_FORCE=1 is for evaluations): use a new --out")
    config.check_game(layout, str(args.out / "layout.json"))    # grid, regime and tree rules of the run (SystemExit)
    if "cap_tree" in layout and any(layout["cap_tree"] != config.cap_for(PLAYERS, s, m)
                                    for s in range(4) for m in config.MENUS):
        raise SystemExit(f"{args.out}: a layout from before the game record with cap_tree {layout['cap_tree']}; this "
                         f"process's trees (config.tree_rule) have other caps: use a new --out")
    if layout.get("training_domain") != wanted["training_domain"]:
        raise SystemExit(f"{args.out}: missing or different training domain; use a new --out (old data cannot resume "
                         f"under a changed sampling protocol)")
    args.game_hash = layout.get("game_hash")         # every stage file of the run carries it (check_stage_file)
    args.teacher = teacher_mode(layout)
    if asked_teacher is not None and asked_teacher != args.teacher:
        raise SystemExit(f"{args.out}: a {args.teacher} run (layout teacher); --{asked_teacher.replace('_', '-')} asks "
                         f"for {asked_teacher}: use a new --out")
    check_code(layout, wanted["code"], args)
    if layout.get("split_version", 1) != SPLIT_VERSION:       # a run's critics and held-out spots under one split
        raise SystemExit(f"{args.out}: made with board split version {layout.get('split_version', 1)} (1 = exact board), "
                         f"this code splits with version {SPLIT_VERSION} (suit-isomorphism class); use a new --out")
    for k in ("river", "turn", "turn_end", "flop", "flop_end", "pre_end", "solve_batch",
              "solver_cap", "second_cap", "e_target", "critic_epochs", "critic_epoch_steps",
              "critic_patience"):
        setattr(args, k, layout.get(k, getattr(args, k)))
    if args.teacher == FIXED_T:                      # the run's own T (as every layout value); this code's for reference
        args.teacher_t = dict(layout["teacher"]["iterations"])
        args.e_flag_factor = float(layout["teacher"]["e_flag_factor"])
        args.solver_cap = args.second_cap = args.e_target = None      # certificate settings: not this run's
        if args.teacher_t != dict(config.TEACHER_T):
            log(f"WARNING: this run's T {args.teacher_t} differs from this code's {config.TEACHER_T}; the run's are used")
        if any(t % FIXED_T_CHUNK for t in args.teacher_t.values()):
            raise SystemExit(f"{args.out}: T {args.teacher_t} must be multiples of the chunk {FIXED_T_CHUNK}")
    args.critic_hidden = tuple(layout["critic_hidden"])
    args.critic_clip_norm = layout.get("critic_clip_norm")     # the run's own; a layout without it was not clipped
    if "critic_clip_norm" not in layout:
        log("layout without critic_clip_norm (a run before the critic audit of 2026-10-10): its critics train "
            "without gradient clipping, as that run did")
    args.critic_lr = layout.get("critic_lr", 1e-3)             # the run's own, as critic_clip_norm
    if "critic_lr" not in layout:
        log("layout without critic_lr (a run before the protocol fix): its critics trained at lr 1e-3, the protocol "
            "of the code up to d3f48ee")
    args.accept = run_accept(layout)                 # the run's own thresholds, as every layout value
    if args.teacher == FIXED_T:
        pass                                         # every spot enters the data
    elif "accept_e" not in layout:
        log(f"layout without accept_e: spots accepted at E <= e_target {args.e_target}")
    elif args.accept != run_accept({"accept_e": accept_thresholds(args.e_target)}):   # both {(street, players): accept}
        log(f"WARNING: this run's accept_e {args.accept} differs from this code's "
            f"{run_accept({'accept_e': accept_thresholds(args.e_target)})}; the run's are used")
    projection = layout.get("critic_projection")
    args.kappa = projection["kappa"] if projection else dict(CR.KAPPA)   # the run's own kappa, as accept_e
    if projection is None:
        log(f"WARNING: layout without critic_projection (a run before the zero-sum fix): its critics so far were "
            f"trained and used unprojected; stages run now project with this code's kappa {args.kappa}")
    elif args.kappa != CR.KAPPA:
        log(f"WARNING: this run's kappa {args.kappa} differs from this code's {CR.KAPPA}; the run's are used")
    log(f"loop critics grid {config.GRID} rake scheme {config.RAKE_SCHEME}, {config.describe_regime(made_under)}"
        + (f" - this loop solves {PLAYERS} players only, {list(left_out)} of the regime are not covered" if left_out else "")
        + f": device {jax.devices()[0]}, stages {args.stages}, "
        + (f"teacher fixed T {args.teacher_t} (E once at the end, flag above {args.e_flag_factor:g} x the stage median, "
           f"every spot into the data)" if args.teacher == FIXED_T else f"teacher certificate, accept_e {args.accept}")
        + f", game {str(args.game_hash)[:12]} ({config.describe_game()}), critic projection kappa {args.kappa}"
        + f"; {FP.describe(wanted['code_fingerprint'])}"
        + ("" if (layout.get("code_fingerprint") or {}).get("sha256") == wanted["code_fingerprint"]["sha256"] else
           f" (the run's layout: {FP.describe(layout.get('code_fingerprint'))})"))
    if args.stages and not rank7.TABLE_PATH.exists():   # without it every equity matrix falls back to the slow evaluator
        rank7.load_table(log=log)
    for key in ("rake_mix", "critic_form"):
        if layout.get(key) != wanted[key]:
            raise SystemExit(f"{args.out}: layout {key} {layout.get(key)} != this code's {wanted[key]}; use a new --out")
    summary = {"layout": layout}
    if "river_solve" in args.stages:
        t0 = time.perf_counter()
        m = solve_stage("river", 3, args.river, args, args.out)
        if args.shard:
            mine = [s for i, s in enumerate(batch_starts("river", args.river, args.solve_batch))
                    if i % args.shard[1] == args.shard[0]]
            done = [s for s in mine if (args.out / f"river_{s:06d}.json").exists()]
            write_text(args.out / f"shard_{args.shard[0]}of{args.shard[1]}.json", json.dumps(
                {"shard": list(args.shard), "batches": len(mine), "done": len(done), "seconds": time.perf_counter() - t0,
                 "code": wanted["code"], "device": str(jax.devices()[0])}, indent=1))
            log(f"shard {args.shard[0]}/{args.shard[1]}: {len(done)}/{len(mine)} river batches done")
            return
        summary["river_solve"] = solve_summary(m, args, "river")
    if "river_critic" in args.stages:
        summary["river_critic"] = critic_stage("river_critic", "river", args.river, args, args.out)
    if "turn_end" in args.stages:
        summary["turn_end"] = end_stage("turn_end", "river_critic", args.turn_end, args, args.out)
    if "turn_solve" in args.stages:
        params, meta = load_stage_critic("turn_end", args, args.out)
        m = solve_stage("turn", 2, args.turn, args, args.out, aux=CR.net_critic(params, meta["kappa"]))
        summary["turn_solve"] = solve_summary(m, args, "turn")
    if "turn_critic" in args.stages:
        summary["turn_critic"] = critic_stage("turn_critic", "turn", args.turn, args, args.out)
    if "flop_end" in args.stages:
        summary["flop_end"] = end_stage("flop_end", "turn_critic", args.flop_end, args, args.out)
    if "flop_solve" in args.stages:
        params, meta = load_stage_critic("flop_end", args, args.out)
        m = solve_stage("flop", 1, args.flop, args, args.out, aux=CR.net_critic(params, meta["kappa"]))
        summary["flop_solve"] = solve_summary(m, args, "flop")
    if "flop_critic" in args.stages:
        summary["flop_critic"] = critic_stage("flop_critic", "flop", args.flop, args, args.out)
    if "pre_end" in args.stages:
        summary["pre_end"] = end_stage("pre_end", "flop_critic", args.pre_end, args, args.out)
    summary["cache_sha256"] = cache_hashes()
    log(f"cache hashes {summary['cache_sha256']}")
    write_text(args.out / f"summary_w{os.environ.get('WORKER_ID', '0')}.json", json.dumps(summary, indent=1))
    log("done")


def cache_hashes():
    """SHA-256 of the exact-table caches: artifacts rebuilt from the engine's
    evaluator on every machine, so equal hashes across machines are a check."""
    out = {}
    for path in (rank7.TABLE_PATH, PREFLOP_EQUITY):
        if Path(path).exists():
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 24), b""):
                    h.update(block)
            out[Path(path).name] = h.hexdigest()
    return out


def check_code(layout, code, args):
    """A run continues only on the code it last ran on (the commit, code fingerprint and
    runtime of poker/envs/fingerprint.py: Python and numerical packages), unless --allow-code-change
    with --code-change-reason; every such resume is appended to code_resumes.json next to the layout (commit, dirty,
    code fingerprint, runtime, what changed, the reason: tools/cloud/gate.py prints and checks the chain), and the
    next process compares with that last entry."""
    started = layout.get("code")
    if started is None:
        log("layout has no code version (a run from before brief 7.6): not reproducible")
        return
    if code["dirty"]:
        log(f"WARNING: dirty tree ({code['source']}): numbers of this run do not go into a review")
    fp = FP.code_fingerprint()
    path = args.out / "code_resumes.json"
    seen = json.loads(path.read_text()) if path.exists() else []
    first = layout.get("code_fingerprint") or {}
    last = seen[-1] if seen else {**started, "code_sha256": first.get("sha256"), "runtime": first.get("runtime")}
    changed = ([f"commit {str(last.get('commit'))[:12]} -> {str(code['commit'])[:12]}"]
               if code["commit"] != last.get("commit") else [])
    if last.get("code_sha256") and last["code_sha256"] != fp["sha256"]:
        changed.append(f"code {last['code_sha256'][:12]} -> {fp['sha256'][:12]}")
    runtime = FP.runtime_differences({"runtime": last["runtime"]}, fp) if last.get("runtime") else {}
    if runtime:
        changed.append(f"runtime {runtime}")
    if not changed:
        return
    if not args.allow_code_change:
        raise SystemExit(f"{args.out} was started by other code ({'; '.join(changed)}); use a new --out, or "
                         f"--allow-code-change --code-change-reason TEXT")
    if not args.code_change_reason:
        raise SystemExit(f"{args.out}: --allow-code-change needs --code-change-reason TEXT (kept in code_resumes.json)")
    write_text(path, json.dumps(seen + [{**code, "code_sha256": fp["sha256"],
                                         "runtime": {k: fp["runtime"][k] for k in ("python", "packages")},
                                         "changed": changed, "reason": args.code_change_reason,
                                         "time": time.strftime("%Y-%m-%dT%H:%M:%S")}], indent=1))
    log(f"WARNING: resuming on other code ({'; '.join(changed)}), reason {args.code_change_reason!r}, recorded in {path}")


def stage_spots(name, count, with_order=False):
    """The shared training-domain spots in stable SPR order, with one tree menu per batch.
    Preflop sampling is kept for independent tree/regime fixtures; production preflop export
    is poker.solver.preflop, outside this critic-training pipeline."""
    config.heads_up_part("poker.solver.loop stage spots")
    street = STREET_OF[name]
    if street == 0:
        spots = SI.make_preflop_spots(SEEDS[name], count, PLAYERS, rake="mix")
    else:
        spots = SI.make_training_spots(SEEDS[name], count, PLAYERS, rake="mix", street=street,
                                       board_pool=FLOP_BOARDS if street == 1 else 0)
    order = np.argsort(spots["stack"] / spots["pot"], kind="stable")
    spots = {k: v[order] for k, v in spots.items()}
    return (spots, order) if with_order else spots


def error_bb_summary(values):
    """Absolute local teacher error in BB; diagnostic only, no uncalibrated acceptance threshold."""
    return {"e_bb_median": float(np.median(values)), "e_bb_p95": float(np.percentile(values, 95)),
            "e_bb_max": float(np.max(values))}


def spr_table(metrics, args, accept):
    """Per SPR bucket: spots, certified (E <= e_target) and accepted (E <= accept, certificate rule) fractions after each pass, iterations and seconds per spot (both passes counted for a spot
    solved twice), median E."""
    e = np.asarray([m["e_solve"] for m in metrics])
    e_bb = np.asarray([m["e_solve_bb"] for m in metrics])
    e1 = np.asarray([m.get("pass1", m)["e_solve"] for m in metrics])
    it = np.asarray([m["iters"] for m in metrics])
    sec = np.asarray([m["solve_seconds_per_spot"] + (m["pass1"]["solve_seconds_per_spot"] if "pass1" in m else 0.0)
                      for m in metrics])
    spr = np.asarray([m["spr"] for m in metrics])
    out = {}
    for lo, hi in zip(SPR_EDGES[:-1], SPR_EDGES[1:]):
        m = (spr >= lo) & (spr < hi)
        if m.any():
            out[f"{lo:g}-{hi:g}"] = {"spots": int(m.sum()), "certified_pass1": float(np.mean(e1[m] <= args.e_target)),
                                     "certified": float(np.mean(e[m] <= args.e_target)),
                                     "accepted_pass1": float(np.mean(e1[m] <= accept)),
                                     "accepted": float(np.mean(e[m] <= accept)),
                                     "iters_median": float(np.median(it[m])), "iters_p95": float(np.percentile(it[m], 95)),
                                     "seconds_per_spot": float(np.mean(sec[m])),
                                     "seconds_per_spot_p95": float(np.percentile(sec[m], 95)),
                                     "e_median": float(np.median(e[m])), "e_p95": float(np.percentile(e[m], 95)),
                                     **error_bb_summary(e_bb[m])}
    return out


def payout_report(metrics, bounds=None):
    """Root value sums of the solved spots against their payout bound (payout_bounds, pot units of
    the street start): the fraction above bound + PAYOUT_TOL overall and per SPR bucket, the excess
    (sum - bound) mean / p90 / max. bounds: per spot; default the metrics' "payout_bound"."""
    total = np.asarray([sum(m["values"]) for m in metrics], np.float64)
    bound = np.asarray([m["payout_bound"] for m in metrics] if bounds is None else bounds, np.float64)
    excess = total - bound
    above = excess > PAYOUT_TOL
    spr = np.asarray([m["spr"] for m in metrics])
    by_spr = {}
    for lo, hi in zip(SPR_EDGES[:-1], SPR_EDGES[1:]):
        m = (spr >= lo) & (spr < hi)
        if m.any():
            by_spr[f"{lo:g}-{hi:g}"] = {"spots": int(m.sum()), "above_fraction": float(np.mean(above[m])),
                                       "excess_mean": float(np.mean(excess[m])), "excess_max": float(np.max(excess[m]))}
    return {"tolerance": PAYOUT_TOL, "above_fraction": float(np.mean(above)), "above_spots": int(above.sum()),
            "excess_mean": float(np.mean(excess)), "excess_p90": float(np.percentile(excess, 90)),
            "excess_max": float(np.max(excess)), "by_spr": by_spr}


def depth_report(name, spots):
    """A stage's spots against the regime's depths (config.DEPTH_MIX). Preflop: every spot starts at a
    depth drawn from the mix - the count per fixed depth of the mix and of the rest, min and max.
    Postflop: joint pot/behind draws are bounded by a sampled starting depth; report any
    violation and the sampled SPR-band coverage."""
    deepest = config.depth_range_bb()[1]
    if STREET_OF[name] == 0:
        depth = np.asarray(spots["stack"], np.int64) / config.BB
        fixed = sorted({float(a) for a, b, p in config.DEPTH_MIX if a == b and p > 0})
        counts = {f"{d:g}": int(np.sum(depth == d)) for d in fixed}
        counts["other"] = int(len(depth) - sum(counts.values()))
        return {"spots": int(len(depth)), "depth_mix": [list(r) for r in config.DEPTH_MIX], "depth_bb_counts": counts,
                "depth_bb_min": float(depth.min()), "depth_bb_max": float(depth.max())}
    start = (np.asarray(spots["pot"], np.float64) / 2 + np.asarray(spots["stack"], np.float64)) / config.BB
    deeper = start > deepest
    bands = np.asarray(spots.get("spr_band", np.full(len(start), -2)))
    coverage = {str(int(b)): int(np.sum(bands == b)) for b in np.unique(bands)}
    spr = np.asarray(spots["stack"], np.float64) / np.asarray(spots["pot"], np.float64)
    return {"spots": int(len(start)), "deepest_bb": deepest, "deeper_spots": int(deeper.sum()),
            "spr_band_counts": coverage, "spr_min": float(spr.min()), "spr_max": float(spr.max()),
            "deeper_fraction": float(deeper.mean()),
            "start_depth_bb_quantiles": {str(q): float(np.quantile(start, q)) for q in (0.1, 0.5, 0.9)}}


def fixed_t_report(metrics, args, name):
    """A fixed-T stage's solve summary (metrics in solving order): T, E (the diagnostic computed once at the end of
    every solve) median / p90 / max, the flagged spots (e_flag), iterations, seconds per spot, and the same per
    size menu and per SPR bucket. Every spot entered the data."""
    e = np.asarray([m["e_solve"] for m in metrics])
    e_bb = np.asarray([m["e_solve_bb"] for m in metrics])
    it = np.asarray([m["iters"] for m in metrics])
    spr = np.asarray([m["spr"] for m in metrics])
    sec = np.asarray([m["solve_seconds_per_spot"] for m in metrics])
    prep = np.asarray([m["prepare_seconds_per_spot"] for m in metrics])
    flag = np.asarray([bool(m.get("e_flag", False)) for m in metrics])
    batches = stage_batches(name, len(metrics), args.solve_batch)
    menu = np.asarray([m for s, stop, m in batches for _ in range(s, stop)])

    def part(m):
        return {"spots": int(m.sum()), **error_bb_summary(e_bb[m]),
                "e_median": float(np.median(e[m])), "e_max": float(e[m].max()),
                "flagged": int(flag[m].sum()), "e_p95": float(np.percentile(e[m], 95)),
                "iters_median": float(np.median(it[m])), "iters_p95": float(np.percentile(it[m], 95)),
                "seconds_per_spot": float(np.mean(sec[m])), "seconds_per_spot_p95": float(np.percentile(sec[m], 95)),
                "prepare_seconds_per_spot": float(np.mean(prep[m])),
                "prepare_seconds_per_spot_p95": float(np.percentile(prep[m], 95))}

    by_spr = {f"{lo:g}-{hi:g}": part((spr >= lo) & (spr < hi)) for lo, hi in zip(SPR_EDGES[:-1], SPR_EDGES[1:])
              if np.any((spr >= lo) & (spr < hi))}
    return {"teacher": FIXED_T, "iterations": args.teacher_t[name], "accept_e": None, "accepted_fraction": 1.0,
            "e_units": "starting pot", "e_bb_units": "BB", **error_bb_summary(e_bb),
            "e_median": float(np.median(e)), "e_p90": float(np.percentile(e, 90)), "e_max": float(e.max()),
            "e_flag_factor": args.e_flag_factor, "e_flag_threshold": args.e_flag_factor * float(np.median(e)),
            "e_flagged": int(flag.sum()), "e_nonfinite": int(np.sum(~np.isfinite(e))),
            "e_median_spr_lt_6": float(np.median(e[spr < 6])) if np.any(spr < 6) else None,
            "e_median_spr_ge_6": float(np.median(e[spr >= 6])) if np.any(spr >= 6) else None,
            "iters_median": float(np.median(it)), "iters_min": int(it.min()), "iters_max": int(it.max()),
            "seconds_per_spot": float(np.mean(sec)),
            "prepare_seconds_per_spot": float(np.mean([m["prepare_seconds_per_spot"] for m in metrics])),
            "by_menu": {m: part(menu == m) for m in config.MENUS if np.any(menu == m)}, "by_spr": by_spr}


def solve_summary(metrics, args, name):
    e = np.asarray([m["e_solve"] for m in metrics])
    e_bb = np.asarray([m["e_solve_bb"] for m in metrics])
    e1 = np.asarray([m.get("pass1", m)["e_solve"] for m in metrics])
    it = np.asarray([m["iters"] for m in metrics])
    spr = np.asarray([m["spr"] for m in metrics])
    accept = args.accept[(name, PLAYERS)]
    payout = payout_report(metrics)
    above = payout["above_fraction"]
    log(("WARNING: " if above > PAYOUT_WARN else "")
        + f"{name} solves: {above:.1%} of {len(metrics)} spots pay out more than the pot holds (root value sum > "
          f"-R(pot0)/pot0 + {PAYOUT_TOL:g}; excess p90 {payout['excess_p90']:+.4f}, max {payout['excess_max']:+.4f} pot)"
        + (f", above {PAYOUT_WARN:.0%}: the street-end critic creates EV, by SPR "
           + ", ".join(f"{k} {v['above_fraction']:.0%}" for k, v in payout["by_spr"].items()) if above > PAYOUT_WARN else ""))
    depth = depth_report(name, stage_spots(name, getattr(args, name)))
    log(f"{name} spots, regime {config.REGIME} ({config.REGIME_NAME}): "
        + (f"starting depths in BB {depth['depth_bb_counts']} (min {depth['depth_bb_min']:g}, max "
           f"{depth['depth_bb_max']:g}; depth mix {depth['depth_mix']})" if STREET_OF[name] == 0 else
           f"{depth['deeper_spots']} of {depth['spots']} ({depth['deeper_fraction']:.1%}) start deeper than the table's "
           f"{depth['deepest_bb']} BB (pot / 2 + behind; kept as coverage)"))
    if getattr(args, "teacher", CERTIFICATE) == FIXED_T:
        return {"spots": len(metrics), "payout": payout, "regime_depth": depth, **fixed_t_report(metrics, args, name)}
    return {"spots": len(metrics), "payout": payout, "regime_depth": depth,
            "e_units": "starting pot", "e_bb_units": "BB", **error_bb_summary(e_bb),
            "certified_fraction": float(np.mean(e <= args.e_target)),
            "certified_fraction_pass1": float(np.mean(e1 <= args.e_target)),
            "accept_e": accept, "accepted_fraction": float(np.mean(e <= accept)),
            "accepted_fraction_pass1": float(np.mean(e1 <= accept)),
            "second_pass_spots": int(sum(m.get("pass") == 2 for m in metrics)),
            "by_spr": spr_table(metrics, args, accept),
            "e_median": float(np.median(e)), "e_p90": float(np.percentile(e, 90)),
            "e_median_spr_lt_6": float(np.median(e[spr < 6])) if np.any(spr < 6) else None,
            "e_median_spr_ge_6": float(np.median(e[spr >= 6])) if np.any(spr >= 6) else None,
            "iters_median": float(np.median(it)), "iters_p95": float(np.percentile(it, 95)),
            "seconds_per_spot": float(np.mean([m["solve_seconds_per_spot"] + m.get("pass1", {}).get("solve_seconds_per_spot", 0.0)
                                               for m in metrics])),
            "prepare_seconds_per_spot": float(np.mean([m["prepare_seconds_per_spot"] for m in metrics]))}


if __name__ == "__main__":
    main()
