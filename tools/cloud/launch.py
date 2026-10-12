"""The rental launcher (task #34): the job manifest - the ONE place every rented job takes its game, devices, budgets,
plan and resource needs from - and the pack record. tools/cloud/pack.sh runs the first two, the box the others.

    $PY tools/cloud/launch.py manifest --out results/pack/JOBS.json --regime 1 [--budget k0=43200 ...] [--platforms cpu]
    $PY tools/cloud/launch.py record --out results/pack/PACKED.json --commit SHA --tarball NAME --stamp UTC [--checks DIR|FILE]
        [--allow-dirty]
    python3 tools/cloud/launch.py env results/pack/JOBS.json            (export lines for eval; standard library only)
    python3 tools/cloud/launch.py get results/pack/JOBS.json budgets_s.k0    (one value of any JSON file)
    python3 tools/cloud/launch.py budget results/pack/JOBS.json k0      (a milestone's wall-time budget in seconds)
    python3 tools/cloud/launch.py args results/pack/JOBS.json k0        (the poker.solver.loop arguments of a milestone's plan)
    python3 tools/cloud/launch.py stages results/pack/JOBS.json k0      (all, or the explicit stages, one per line)

The manifest's game is the game poker/envs/config.py defines for the regime named by --regime (pack.sh --regime; the
regime has no default, task #39), read in a process with POKER_REGIME=N and no other POKER_* variable: the grid is
config.GRID, the regime's own grid (13 for regimes 1 and 2), never a legacy POKER_GRID of the caller's shell. It
writes them as the environment of every job (POKER_GRID, POKER_REGIME, POKER_REGIME_FORCE=0, and JAX_PLATFORMS =
RENTAL_PLATFORMS "cuda,cpu": never "cuda" alone - JAX then has no CPU device and the engine replay of poker/solver/tree.py,
which runs on the CPU because the fused scan does not load on Pascal GPUs, fails at the first batch; executor's
repro 2026-10-11 07:2x) with the game they make (config.game_hash and describe_game, computed in a process under
exactly that environment), the teacher the loop runs there (fixed T under the sized grid: config.TEACHER_T,
TEACHER_E_FLAG, TEACHER_E_BOUND), the plan of every milestone (stages and spot counts: the gate refuses a run that
differs), the wall-time budgets, the heartbeat period, the stall limit, the gate's E-flag warning threshold, the
resources a box must have (RESOURCES) and what comes back (EXTRACT). To rent another game, change poker/envs/config.py in a
commit: the pack is clean, so the change is recorded. --platforms cpu writes a CPU rehearsal manifest (a local dry
run of the runner without touching a GPU): the gate refuses it as a rental.

The pack record (results/pack/PACKED.json): commit (read once by pack.sh; the record refuses when HEAD or the tree moved while
packing), tree state (dirty_paths: the number of `git status --porcelain` lines, as poker/solver/loop.py code_version reads
it; dirty_list: the lines), the code fingerprint (poker/envs/fingerprint.py, with the per-file hashes the box checks its
tree against), the manifest's game and its sha256, the packed file list (results/pack/.files0 for tar: tracked files without
results/, results/reports/ and .archive/, plus INPUTS) and the attached checks summary (copied to results/pack/CHECKS_SUMMARY.json,
its sha256 recorded; tools/cloud/gate.py reads its verdict).
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))           # poker.envs.fingerprint (standard library only) when run as a script

MANIFEST_VERSION = 2                    # current critic-only plan and versioned training-domain contract
PACKED_VERSION = 2                      # 1: commit, dirty_paths, packed_utc (pack.sh before task #34)
JOB_ENV = ("POKER_GRID", "POKER_REGIME", "POKER_REGIME_FORCE", "JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR")
RENTAL_PLATFORMS = "cuda,cpu"           # the jobs' devices: the GPU, and the CPU the engine replay runs on
# JAX's persistent compilation cache of every job: a directory of the tree on the box's disk, never /tmp (a tmpfs of a
# few GB that a cold warm-up fills and dies on: the play path's report, 2026-10-11); relative paths are resolved
# against the tree root (resolved_env). The start gate writes and reads one entry there and refuses a tmpfs or a full
# disk (tools/cloud/gate.py compile_cache).
COMPILE_CACHE = "results/jax_cache"
# Wall-time budget per milestone (seconds; every job of the milestone runs under it, `timeout` stops it and the queue
# writes FAILED): a cap on the money a hung or runaway job burns, generous against the k0 numbers (RTX 3090, grid 7:
# loop 2 h 33 min of which the solves 2 h 20 min, reports 13 min; 45k river first pass 5 h 18 min); grid 13 fixed T
# on the home 1070 under load: river 0.86, turn and flop 1.7 s per spot, a k0-size run about 2.5-3 h there and
# about 1 h on a 3090 (memo section 12 (11)). The preflight milestone measures the rented card; then these are set
# from it (pack.sh --budget NAME=SECONDS, recorded here). caches: the one-process build and hash check of the exact tables
# (about 100 s on a 3090: preflop 13 s, the 1755 flop classes 84 s; rental-path sniff M11).
BUDGET_S = {"bootstrap": 3600, "caches": 1800, "selftest": 30, "preflight": 3600, "k0_smoke": 2 * 3600,
            "k0": 6 * 3600, "river_shard": 12 * 3600, "river_finish": 6 * 3600}
HEARTBEAT_S = 30                        # queue.sh rewrites results/HEARTBEAT this often
STALL_S = 1800                          # a job whose worker logs stay silent this long is reported stalled (gate)
E_FLAG_WARN = 0.25                      # gate: WARNING when a fixed-T stage flags more than this share of its spots
# The plan of every milestone: the loop's stages ("all" = poker/solver/loop.py STAGES) and spot counts (its defaults, its
# --quick sizes, the preflight's production-shape sizes, $RIVER for the river milestones). The gate refuses a run whose
# layout counts or stage list differ: a change of the loop's defaults shows up as a refusal, not as another run.
QUICK = {"river": 48, "turn": 24, "turn_end": 64, "flop": 16, "flop_end": 16, "pre_end": 32}
PRODUCTION = {"river": 3000, "turn": 2000, "turn_end": 3000, "flop": 2000, "flop_end": 3000, "pre_end": 3000}
PREFLIGHT = {"river": 256, "turn": 128, "turn_end": 256, "flop": 128, "flop_end": 256, "pre_end": 256}
CRITIC_STAGES = ("river_solve", "river_critic", "turn_end", "turn_solve", "turn_critic",
                 "flop_end", "flop_solve", "flop_critic", "pre_end")
PLAN = {"k0_smoke": {"stages": list(CRITIC_STAGES), "spots": QUICK, "quick": True},
        "preflight": {"stages": list(CRITIC_STAGES), "spots": PREFLIGHT, "quick": False,
                      "args": ["--critic-epochs", "10"]},
        "k0": {"stages": list(CRITIC_STAGES), "spots": PRODUCTION, "quick": False},
        "river_shard": {"stages": ["river_solve"], "spots": {"river": "$RIVER"}, "quick": False},
        "river_finish": {"stages": ["river_solve", "river_critic"], "spots": {"river": "$RIVER"}, "quick": False}}
# What a box needs before a milestone starts (static sniff S6, rental-path sniff M11). Rows on disk per spot, grid 13,
# fixed T (the teacher's grid-13 --quick run of 2026-10-11): critic rows river 298 KB with the SPR-0 copies, turn / flop
# 149 KB for turn/flop. End-row files and reports are covered by the disk margin. The critic stage holds about 2.8x its street's critic row files in host RAM
# (load_rows' parts and concatenation, critic_split's copies: S6 estimates 27-38 GB at 45k river spots), next to every
# worker's own process: a loop process about 5 GB (the --quick fixed-T run peaked at 3.6 GB RSS on the 1070 with
# cuda,cpu; 2.86 GB on the CPU), so host RAM >= workers x 5 GB + 2.8 x the largest street's critic rows. GPU: the k0
# run's high-water mark on the RTX 3090 (grid 7, batch 128, eval-batch 8; dmon fb) is 12.9 GB - a reference until the
# preflight measures grid 13. Disk: the venv (5.6 GB locally), the caches (rank7 0.27 GB + 1755 flop matrices 12.34 GB +
# preflop 7 MB), the rows twice (the job directory and its outbox archive) and a margin. process_gb: the measured
# sizes this table rests on (the play path's live bot: 2-2.5 GB host, GPU process RSS 2.3 GB - not a rental job today).
RESOURCES = {"row_kb_per_spot": {"river": 298, "turn": 149, "flop": 149},
             "critic_kb_per_spot": {"river": 298, "turn": 149, "flop": 149},
             "critic_ram_factor": 2.8, "host_base_gb": 5.0, "gpu_mib_min": 13000,
             "disk_fixed_gb": {"venv": 6.0, "caches": 12.7, "margin": 3.0},
             "process_gb": {"loop_quick_fixed_t_gpu_rss": 3.6, "loop_quick_fixed_t_cpu_rss": 2.86,
                            "critic_stage_45k_river_s6": [27, 38], "live_bot_host": [2.0, 2.5],
                            "live_bot_gpu_process_rss": 2.3}}
EXTRACT = {"summary": "results/outbox/results-<milestone>.tar.gz: results/ without the per-batch row files "
                      "(*_NNNNNN.npz, solve_*.npz), the caches, the claims and the outbox, with results/pack/{PACKED,JOBS,"
                      "CHECKS_SUMMARY}.json",
           "rows": "results/outbox/job-<job>.tar.gz.part-NNN: the job's whole directory (critic rows, critic checkpoints, metrics; "
                   "claims aside) in 512 MB parts, so a dropped transfer repeats one part (the 45k river set is job "
                   "river_big: its own archive, by name)",
           "checksums": "results/outbox/SHA256SUMS (sha256 and bytes of every file above); tools/cloud/fetch.sh "
                        "verifies them at home with sha256sum -c"}
EXCLUDED = ("results/", ".archive/")      # evidence, review records, the archive, packs
# Untracked inputs the box needs, explicitly (paths from the repo root; sha256 into PACKED.json). Empty: the loop's
# exact-table caches are rebuilt on the box in one process before any worker (queue.sh caches step: rank7_v1.npy,
# preflop_equity_v1.npy, the 1755 flop classes; static sniff S3) and checked against tools/cloud/cache_sha256.json
# and tools/cloud/flop_equity_v1.sha256 (equal on this machine and both k0 boxes, 2026-10-09). No job of queue.sh
# reads anything else outside the tracked tree.
INPUTS = ()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _python_json(code, env):
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"launch: {code[:60]}... failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def manifest(args):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("POKER_")}
    clean.update(JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT), POKER_REGIME=str(args.regime))
    grid, regime = _python_json("import json; from poker.envs import config; "      # the regime's own grid: config.GRID
                                "print(json.dumps([config.GRID, config.REGIME]))", clean)
    env = {"POKER_GRID": str(grid), "POKER_REGIME": str(regime), "POKER_REGIME_FORCE": "0"}
    game = _python_json(                                        # the game does not depend on the device: on the CPU
        "import json; from poker.envs import config; from poker.solver.situations import training_domain_record; "
        "print(json.dumps({'grid': config.GRID, 'regime': config.REGIME, "
        "'game_hash': config.game_hash(), 'game': config.describe_game(), 'sized_grid': config.STREET_SIZES_GRID, "
        "'teacher_t': config.TEACHER_T, 'e_flag_factor': config.TEACHER_E_FLAG, 'e_bound': config.TEACHER_E_BOUND, "
        "'training_domain': training_domain_record()}))",
        {**clean, **env})
    if (game["grid"], game["regime"]) != (grid, regime):
        raise SystemExit(f"launch: config under {env} reads grid {game['grid']}, regime {game['regime']}")
    env.update(JAX_PLATFORMS=args.platforms, JAX_COMPILATION_CACHE_DIR=COMPILE_CACHE)
    budgets = dict(BUDGET_S)
    for item in args.budget:
        name, _, value = item.partition("=")
        if name not in budgets or not value.isdigit():
            raise SystemExit(f"launch: --budget NAME=SECONDS with NAME in {sorted(budgets)}, got {item!r}")
        budgets[name] = int(value)
    record = {"version": MANIFEST_VERSION, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "decided_from": f"poker/envs/config.py in a process with POKER_REGIME={args.regime} (pack.sh --regime) and no other "
                              f"POKER_* variable: grid = config.GRID, the regime's own grid",
              "env": env, "rehearsal": args.platforms != RENTAL_PLATFORMS, "game_hash": game["game_hash"],
              "game": game["game"], "training_domain": game["training_domain"],
              "teacher": ({"stop": "fixed_t", "iterations": game["teacher_t"], "e_flag_factor": game["e_flag_factor"],
                           "e_bound": game["e_bound"]} if grid == game["sized_grid"] else {"stop": "certificate"}),
              "workers": "one per GPU of the box (queue.sh GPUS), passed to every loop job as --workers N explicitly: "
                         "--workers 1 clears the claims of a run (static sniff S4)",
              "plan": PLAN, "budgets_s": budgets, "heartbeat_s": HEARTBEAT_S, "stall_s": STALL_S,
              "e_flag_warn": E_FLAG_WARN, "resources": RESOURCES, "extract": EXTRACT}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"manifest {args.out}: {' '.join(f'{k}={v}' for k, v in env.items())}; game {game['game_hash'][:12]} "
          f"({game['game']}); teacher {record['teacher']['stop']}; budgets {budgets}"
          + (f"; REHEARSAL (JAX_PLATFORMS={args.platforms}, not {RENTAL_PLATFORMS}): the gate refuses it as a rental"
             if record["rehearsal"] else ""))


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def record(args):
    from poker.envs import fingerprint as FP
    dist = args.out.parent
    jobs_path = dist / "JOBS.json"
    jobs = json.loads(jobs_path.read_text())
    status = [line for line in _git("status", "--porcelain").splitlines() if line]
    if status and not args.allow_dirty:
        raise SystemExit(f"launch: the tree is dirty ({len(status)} lines) and --allow-dirty was not given")
    tracked = [p for p in _git("ls-files", "-z").split("\0") if p]
    files = [p for p in tracked if not p.startswith(EXCLUDED) and (ROOT / p).exists()]
    inputs = []
    for path in INPUTS:
        if path in tracked or not (ROOT / path).is_file():
            raise SystemExit(f"launch: INPUTS entry {path} is tracked or missing")
        inputs.append({"path": path, "sha256": sha256_file(ROOT / path)})
    (dist / ".files0").write_text("".join(f"{p}\0" for p in files + [i["path"] for i in inputs]))
    checks = None
    target = dist / "CHECKS_SUMMARY.json"
    target.unlink(missing_ok=True)
    if args.checks:
        src = args.checks / "summary.json" if args.checks.is_dir() else args.checks
        shutil.copyfile(src, target)
        summary = json.loads(target.read_text())
        checks = {"source": str(src), "sha256": sha256_file(target), "commit": summary.get("commit"),
                  "suite": summary.get("suite"), "verdict": summary.get("verdict"),
                  "started_utc": summary.get("started_utc")}
    fp = FP.compute(ROOT)
    head = _git("rev-parse", "HEAD").strip()
    after = [line for line in _git("status", "--porcelain").splitlines() if line]
    if not head == fp["commit"] == args.commit or after != status:
        raise SystemExit(f"launch: HEAD or the tree moved while packing (pack.sh read {args.commit[:12]}, now "
                         f"{head[:12]}, fingerprint {str(fp['commit'])[:12]}; status {len(status)} -> {len(after)} lines)")
    packed = {"version": PACKED_VERSION, "commit": args.commit, "dirty_paths": len(status), "dirty_list": status,
              "allow_dirty": bool(args.allow_dirty), "packed_utc": args.stamp, "tarball": args.tarball,
              "files": len(files), "excluded": list(EXCLUDED), "inputs": inputs, "manifest": "results/pack/JOBS.json",
              "manifest_sha256": sha256_file(jobs_path), "env": jobs["env"], "rehearsal": jobs.get("rehearsal"),
              "game_hash": jobs["game_hash"], "game": jobs["game"], "checks": checks, "fingerprint": fp}
    args.out.write_text(json.dumps(packed, indent=1) + "\n")
    print(f"record {args.out}: commit {packed['commit'][:12]}, {len(files)} tracked files + {len(inputs)} inputs, "
          f"{'DIRTY ' + str(len(status)) + ' paths' if status else 'clean'}, {FP.describe(fp)}, game "
          f"{packed['game_hash'][:12]}, checks " + (f"{checks['source']} ({checks['commit']})" if checks else "none"))


def _load(path):
    return json.loads(Path(path).read_text())


def resolved_env(jobs, root) -> dict:
    """The manifest's job environment with a relative compile-cache path resolved against the tree root."""
    out = {k: str(v) for k, v in jobs.get("env", {}).items()}
    cache = out.get("JAX_COMPILATION_CACHE_DIR")
    if cache and not os.path.isabs(cache):
        out["JAX_COMPILATION_CACHE_DIR"] = str((Path(root) / cache).resolve())
    return out


def env(args):
    jobs = _load(args.manifest)
    missing = [k for k in JOB_ENV if k not in jobs.get("env", {})]
    if missing:
        raise SystemExit(f"launch: {args.manifest} has no {missing}")
    values = resolved_env(jobs, args.manifest.resolve().parents[2])       # results/pack/JOBS.json -> the tree root
    print("export " + " ".join(f"{k}={shlex.quote(values[k])}" for k in JOB_ENV))


def get(args):
    value = _load(args.manifest)
    for key in args.key.split("."):
        value = value[key]
    print(value if not isinstance(value, (dict, list)) else json.dumps(value))


def budget(args):
    budgets = _load(args.manifest)["budgets_s"]
    if args.name not in budgets:
        raise SystemExit(f"launch: no budget for {args.name!r} in {args.manifest} ({sorted(budgets)})")
    print(int(budgets[args.name]))


def plan_stages(jobs, name):
    """The manifest owns stage selection; "all" deliberately uses poker.solver.loop's default stages.

    An explicit list reaches --stages unchanged. plan.args cannot override it, including argparse's abbreviations.
    This helper stays standard-library-only so the launcher can run before the job environment is installed.
    """
    plan = jobs["plan"][name]
    extra = plan.get("args", [])
    if not isinstance(extra, list) or any(not isinstance(arg, str) for arg in extra):
        raise SystemExit(f"launch: plan {name} args must be a list of strings")
    for arg in extra:
        option = arg.partition("=")[0]
        if option.startswith("--") and len(option) > 2 and "--stages".startswith(option):
            raise SystemExit(f"launch: plan {name} must set stages in plan.stages, not plan.args ({arg})")
    selected = plan["stages"]
    if selected != "all" and (not isinstance(selected, list) or not selected
                              or any(stage not in CRITIC_STAGES for stage in selected)
                              or len(set(selected)) != len(selected)):
        raise SystemExit(f"launch: plan {name} stages must be 'all' or a nonempty list of unique critic stages")
    return selected


def stages(args):
    selected = plan_stages(_load(args.manifest), args.name)
    print("all" if selected == "all" else "\n".join(selected))


def loop_args(args):
    """The poker.solver.loop arguments of a milestone's plan, one per line: stages ("all" keeps the loop's defaults),
    --quick for a quick plan, else every spot count ($RIVER from the environment), then the plan's extra arguments.
    """
    jobs = _load(args.manifest)
    plan = jobs["plan"][args.name]
    selected = plan_stages(jobs, args.name)
    out = ["--quick"] if plan.get("quick") else []
    if not plan.get("quick"):
        for name, count in plan["spots"].items():
            if isinstance(count, str) and count.startswith("$"):
                count = os.environ.get(count[1:], "")
                if not count.isdigit():
                    raise SystemExit(f"launch: plan {args.name} needs {plan['spots'][name]} in the environment")
            out += [f"--{name.replace('_', '-')}", str(count)]
    if selected != "all":
        out += ["--stages", *selected]
    print("\n".join(out + plan.get("args", [])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--regime", type=int, required=True, help="the table regime of the jobs (poker/envs/config.py REGIMES; "
                                                             "no default, task #39)")
    m.add_argument("--budget", action="append", default=[], help="NAME=SECONDS (milestone wall-time budget)")
    m.add_argument("--platforms", default=RENTAL_PLATFORMS,
                   help=f"JAX_PLATFORMS of the jobs (default {RENTAL_PLATFORMS}; cpu: a rehearsal the gate refuses)")
    m.set_defaults(func=manifest)
    r = sub.add_parser("record")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--commit", required=True, help="HEAD as pack.sh read it once")
    r.add_argument("--tarball", required=True)
    r.add_argument("--stamp", required=True)
    r.add_argument("--checks", type=Path, help="a checks.run output directory (its summary.json) or the file")
    r.add_argument("--allow-dirty", action="store_true")
    r.set_defaults(func=record)
    for name, func in (("env", env), ("get", get), ("budget", budget), ("args", loop_args), ("stages", stages)):
        s = sub.add_parser(name)
        s.add_argument("manifest", type=Path)
        if name == "get":
            s.add_argument("key")
        if name in ("budget", "args", "stages"):
            s.add_argument("name")
        s.set_defaults(func=func)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
