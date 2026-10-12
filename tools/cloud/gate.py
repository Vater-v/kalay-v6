"""Automatic gates of a rented box: technical validity and provenance, never a research verdict.

    python -m tools.cloud.gate start [--host] [--plan k0]  (queue.sh / bootstrap.sh before any job; pack.sh: --host)
    python -m tools.cloud.gate caches                      (queue.sh: build and hash-check the exact tables, one process)
    python -m tools.cloud.gate batches results/k0          (queue.sh, every heartbeat: the solved batches so far)
    python -m tools.cloud.gate loop results/k0 --plan k0   (after a loop job) [--run-checks-only]
    python -m tools.cloud.gate shard results/river_big --plan river_shard
    python -m tools.cloud.gate heartbeat HEARTBEAT [--now EPOCH]   (on the host: a heartbeat fetched from the box)

Exit 0 = go on; exit 1 = stop, with exactly one last line "GATE FAIL: <reason>[; <reason> ...]" (every check of a gate
runs; the reasons are joined; WARNING lines never stop). Research conclusions are read by a human later.

Start gate (task #34), on the tree it runs in (the repo root):
1. results/pack/JOBS.json (the job manifest, tools/cloud/launch.py) and results/pack/PACKED.json (tools/cloud/pack.sh) exist,
   PACKED.json binds the manifest (its sha256) and records a clean tree (no dirty path, not --allow-dirty); the
   manifest is a rental's (JAX_PLATFORMS "cuda,cpu"), not a CPU rehearsal;
2. the game: this process's POKER_GRID / POKER_REGIME / POKER_REGIME_FORCE / JAX_PLATFORMS / JAX_COMPILATION_CACHE_DIR
   are the manifest's (launch.py resolved_env), config follows, and config.game_hash() equals the manifest's and
   PACKED.json's (the game line is printed);
3. the code: poker.envs.fingerprint of this tree equals PACKED.json's fingerprint, and both are clean (printed);
4. the checks: results/pack/CHECKS_SUMMARY.json (pack.sh --checks; its sha256 in PACKED.json) ran on the packed commit on a
   clean tree (commit, empty git_status; code_fingerprint too when the summary carries one), and carries the honest
   gate's top-level verdict "pass" and a complete record (checks/run.py, task #23: fields `verdict` and
   `completeness`; a summary without them is refused - it cannot say "partial" or "skipped"), with every module of
   the packed checks.run.REQUIRED recorded as pass;
5. the box (not with --host, which pack.sh uses on the host): JAX sees the manifest's platforms - a CPU device (the
   engine replay) and a GPU as the default device - the toolkit of the venv against the driver (a newer toolkit than
   the driver's CUDA is a WARNING, recorded), the persistent compilation cache (the manifest's directory: not a tmpfs,
   at least 2 GB free, one entry written and read back - a JAX cache-hit event), the kernel-load smoke (one rule tree
   of the loop and situations.prepare on 2 river spots on the job's device: a kernel that does not load fails here, not
   mid-run) and its negative (poker/solver/tree.py require_cpu must raise, naming JAX_PLATFORMS=cuda,cpu, in a child without
   the CPU device: JAX_PLATFORMS=cuda on a rental box, simulated on a CPU rehearsal), and with --plan the resources of
   that milestone (RESOURCES of the manifest: host RAM for every worker and
   the critic stage, free disk for the rows twice, GPU memory); everything goes into results/BOX.json (jax_platforms,
   devices, GPUs, driver, CUDA, packages, the compile cache).
Loop and shard gates (after a job; --plan names the milestone) add, for the job's output directory: the layout's
grid, regime, game_hash, teacher (the manifest's stop rule, T, flag factor) and spot counts (the plan's); the code
chain - the layout's code record and code fingerprint and every entry of code_resumes.json (poker/solver/loop.py check_code:
a resume across a code change is allowed only with a recorded reason) is clean, and the last of them is the pack's;
the stages of the plan all present; per fixed-T solve stage: finite E, every spot ran T iterations (iters_min ==
iters_max == T), the E median at or below the teacher's bound (config.TEACHER_E_BOUND, in the manifest), the payout
excess share at or below poker/solver/loop.py PAYOUT_WARN outside --quick (a WARNING in --quick: toy critics), and the
E-flag share as a diagnostic (a WARNING above E_FLAG_WARN, default the manifest's e_flag_warn 0.25, never a stop); a
certificate stage (grids 5/7) keeps its rule (outside --quick the river accepted >= 90% after the second pass); every
batch's row and metrics file present (the rows are the product); finite critic MAE; the exact-table caches the job
reported have the reference sha256 (tools/cloud/cache_sha256.json); the GPU rule below. --run-checks-only evaluates
these without the start gate (a diagnostic of a run's artifacts: its OK line says so; queue.sh never uses it).
Batches gate (every heartbeat while a loop job runs; standard library only): every new <street>_NNNNNN.json of the
job has finite E >= -1e-5 for every spot, T iterations (fixed T) and a batch median E at or below the bound - a
broken solve stops the job within minutes, not after the stage.
Caches gate: rank7_v1.npy, preflop_equity_v1.npy and the 1755 flop matrices built where missing in this one process
(static sniff S3: no worker races a first build) and hash-checked against cache_sha256.json and flop_equity_v1.sha256.
Heartbeat gate (on the host): results/HEARTBEAT of queue.sh: exit 1 when it is older than 3 heartbeat periods + 60 s
(the runner died or the box is gone), when the queue reports a failure, when the running job is past its wall-time
budget, or when its worker logs were silent longer than the manifest's stall_s.

GPU utilisation (loop and shard jobs; brief M2, step 1, A8): with GPU_UTIL_LOG (the dmon log of queue.sh) and
GPU_LIST (the GPU of each worker, in order) the utilisation by stage, the spots per minute and the peak framebuffer go
to <out>/gpu_util.json. The peak framebuffer in use above 90% of a card's memory fails the job (an OOM on the next,
larger shape is near). Reviewer's rule for part B (2026-10-09): the river speed fuse (poker/solver/loop.py, half of
EXPECT_RIVER_SPM) is the hard stop; a card's SM mean over the solve stages below 60% fails the job only together
with low speed, i.e. that card's river first-pass spots per minute in this job below EXPECT_RIVER_SPM / 2. Without
river first-pass batches in the job's logs (river_finish: second pass and critic) or without EXPECT_RIVER_SPM the
speed is not judged: low SM is reported, not a stop. Never in --quick runs (toy sizes are compile-bound).
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from tools.cloud.launch import RENTAL_PLATFORMS, plan_stages, resolved_env

MIN_SOLVE_SM = 60.0
FB_MAX_SHARE = 0.9
E_NEGATIVE = -1e-5                       # poker/solver/pcfr.py solve_cfr: E below this is a solver bug, not convergence
CACHE_FREE_GB = 2.0                      # the compile cache's disk must have at least this free
DIST = Path("results/pack")
HERE = Path(__file__).resolve().parent
CACHE_REFERENCE = HERE / "cache_sha256.json"
FLOP_MANIFEST = HERE / "flop_equity_v1.sha256"
BOX_RECORD = Path("results") / "BOX.json"
BATCH_FILE = re.compile(r"^(river|turn|flop|preflop)_(\d{6})\.json$")
PROBLEMS = []


def fail(msg):
    """Stop now: the reasons so far and this one, on one line."""
    print("GATE FAIL: " + "; ".join(PROBLEMS + [msg]))
    sys.exit(1)


def problem(msg):
    """A failed check: reported now, the gate fails at its end (finish)."""
    PROBLEMS.append(msg)
    print(f"  FAIL {msg}", flush=True)


def warning(msg):
    print(f"  WARNING {msg}", flush=True)


def finish(ok_line):
    if PROBLEMS:
        print("GATE FAIL: " + "; ".join(PROBLEMS))
        sys.exit(1)
    print(f"GATE OK: {ok_line}")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path, what: str):
    if not path.exists():
        fail(f"{path} missing ({what})")
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        fail(f"{path} is not JSON ({what}): {error}")


def utilisation(out: Path, quick: bool, name="gpu_util.json"):
    """GPU utilisation gate of a loop/shard job (module doc)."""
    dmon, gpus = os.environ.get("GPU_UTIL_LOG"), os.environ.get("GPU_LIST")
    if not dmon or not gpus:
        print("utilisation gate not applied: GPU_UTIL_LOG / GPU_LIST not set (queue.sh sets them)")
        return
    from tools.cloud import gpu_util
    logs = sorted(out.parent.glob(f"{out.name}.w*.log"), key=lambda p: int(re.search(r"\.w(\d+)\.log$", p.name).group(1)))
    rep = gpu_util.report(dmon, [str(p) for p in logs], [int(g) for g in gpus.split()])
    (out / name).write_text(json.dumps(rep, indent=1))
    for g, v in rep["spots_per_min"].items():
        print(f"  gpu {g}: " + ", ".join(f"{s} {x['spots_per_min']:.1f} spots/min" for s, x in v.items()
                                        if x["spots_per_min"] is not None))
    total = {str(g["index"]): float(g["memory_mib"]) for g in nvidia_smi()[0] if g.get("memory_mib")}
    for g, mb in rep.get("fb_peak_mb", {}).items():
        share = mb / total[g] if total.get(g) else None
        print(f"  gpu {g}: peak framebuffer {mb:.0f} MB" + (f" of {total[g]:.0f} ({share:.0%})" if share else ""))
        if share is not None and share > FB_MAX_SHARE:
            problem(f"gpu {g}: peak framebuffer {mb:.0f} of {total[g]:.0f} MB ({share:.0%} > {FB_MAX_SHARE:.0%}): "
                    f"the next larger shape runs out of memory")
    low = {g: v for g, v in rep["solve_sm_mean"].items() if v < MIN_SOLVE_SM}
    print(f"  solve stages, SM mean by GPU: { {g: round(v, 1) for g, v in rep['solve_sm_mean'].items()} }")
    if not rep["solve_sm_mean"]:
        problem("no utilisation samples in the solve stages")
        return
    if low and not quick:
        expected = os.environ.get("EXPECT_RIVER_SPM")
        speed = {g: rep["spots_per_min"].get(g, {}).get("river_solve", {}).get("spots_per_min") for g in low}
        slow = {g: round(v, 1) for g, v in speed.items() if expected and v is not None and v < float(expected) / 2}
        if slow:
            problem(f"GPU utilisation in the solve stages below {MIN_SOLVE_SM:g}%: {low}, and river speed "
                    f"{slow} spots/min below half of the expected {expected}")
            return
        if not expected:
            notes = ["EXPECT_RIVER_SPM not set, the speed is not judged"]
        else:
            fast = [g for g, v in speed.items() if v is not None]
            unmeasured = [g for g, v in speed.items() if v is None]
            notes = ([f"river speed at least half of the expected {expected} spots/min on GPUs {fast}"] if fast else []) \
                + ([f"river speed not measured in this job on GPUs {unmeasured}"] if unmeasured else [])
        print(f"  SM below {MIN_SOLVE_SM:g}% in the solve stages: {low}; not a stop: " + "; ".join(notes))


def finite(*xs):
    return all(x is not None and math.isfinite(x) for x in xs)


# ---------------------------------------------------------------- start gate: pack, game, code, checks, box

def _verdict(summary):
    """(verdict, complete) of a checks summary of the honest gate (task #23), None where the field is absent. Read
    tolerantly: `verdict` a string or {"status"|"verdict": ...}; `completeness` a bool or {"complete": bool}."""
    verdict, completeness = summary.get("verdict"), summary.get("completeness")
    if isinstance(verdict, dict):
        verdict = verdict.get("status", verdict.get("verdict"))
    if isinstance(completeness, dict):
        completeness = completeness.get("complete")
    return (None if verdict is None else str(verdict).lower()), completeness


def check_checks(packed):
    """Part 4 of the start gate: the attached checks summary binds a PASS to the packed tree."""
    info = packed.get("checks")
    if not info:
        problem("no checks summary in the pack (pack.sh --checks <a full `checks.run --suite required` run of this "
                "commit on a clean tree>)")
        return
    path = DIST / "CHECKS_SUMMARY.json"
    if not path.exists() or sha256_file(path) != info.get("sha256"):
        problem(f"{path} is missing or not the summary PACKED.json recorded")
        return
    s = json.loads(path.read_text())
    if s.get("commit") != packed["commit"]:
        problem(f"the checks ran on commit {str(s.get('commit'))[:12]}, the pack is {packed['commit'][:12]}")
    if s.get("git_status") != "":
        problem(f"the checks ran on a dirty tree (git_status {str(s.get('git_status'))[:120]!r})")
    fp = s.get("code_fingerprint")
    if isinstance(fp, dict) and fp.get("sha256") != packed["fingerprint"]["sha256"]:
        problem(f"the checks ran on code {str(fp.get('sha256'))[:12]}, the pack is {packed['fingerprint']['sha256'][:12]}")
    verdict, complete = _verdict(s)
    if verdict is None or complete is None:
        problem("the checks summary has no top-level verdict / completeness (a checks/run.py before the honest gate, "
                "task #23: it cannot tell a partial or skipped run from a green one)")
    elif verdict != "pass" or complete is not True:
        problem(f"checks verdict {verdict!r}, complete {complete!r}")
    try:
        from checks.run import REQUIRED
    except Exception as error:        # noqa: BLE001 - any import failure of the packed runner is a refusal
        problem(f"checks.run.REQUIRED of the packed code cannot be read: {type(error).__name__}: {error}")
        return
    rows = {str(r.get("module", "")).removeprefix("checks."): r.get("status") for r in s.get("results", [])}
    missing = [m for m in REQUIRED if rows.get(m) != "pass"]
    if missing:
        problem(f"{len(missing)} of {len(REQUIRED)} REQUIRED modules not recorded as pass: {missing[:8]}"
                + (" ..." if len(missing) > 8 else ""))
    print(f"  checks: {info.get('source')} (suite {s.get('suite')}, commit {str(s.get('commit'))[:12]}, verdict "
          f"{verdict}, complete {complete}, {sum(v == 'pass' for v in rows.values())} pass of {len(rows)} recorded)")


def nvidia_smi():
    """([{index, name, driver, memory_mib, compute_cap}], the driver's CUDA version) from nvidia-smi; [] and None
    without one."""
    try:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60,
                              check=True).stdout
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return [], None
    gpus = [dict(zip(("index", "name", "driver", "memory_mib", "compute_cap"), (x.strip() for x in line.split(","))))
            for line in rows.splitlines() if line.strip()]
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    return gpus, (cuda.group(1) if cuda else None)


def _major_minor(version):
    parts = str(version).split(".")
    return (int(parts[0]), int(parts[1])) if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit() else None


def kernel_smoke():
    """One rule tree through the loop's own entry point (poker/solver/loop.py tree_for: config.tree_rule) and
    situations.prepare of 2 river spots of its menu: the engine replay on the CPU, the river tables on the job's
    default device. Returns a line; raises whatever a kernel that does not load raises."""
    import jax
    import numpy as np
    from poker.solver import loop as L
    from poker.solver import situations as SI
    t0 = time.perf_counter()
    spots = SI.make_spots(7, 2, L.PLAYERS, spr_range=(4.0, 20.0), rake="mix", street=3, pot_bb=(3.0, 300.0))
    menus = L.spot_menus("river", spots)
    if len(set(menus)) != 1:
        raise RuntimeError(f"the smoke's two spots fall in different menus {menus}")
    topo, _ = L.tree_for(3, L.PLAYERS, menus[0])
    batch = SI.prepare(spots, topo)
    leaves = jax.block_until_ready(jax.tree.leaves(batch))
    bad = [k for k, v in batch.items() if np.issubdtype(np.asarray(v).dtype, np.floating) and not np.all(np.isfinite(v))]
    if bad:
        raise RuntimeError(f"non-finite prepared arrays {bad}")
    devices = sorted({str(d) for x in leaves if hasattr(x, "devices") for d in x.devices()})
    return (f"river rule tree ({menus[0]}) {topo.size} nodes / {len(topo.internal)} decisions, 2 spots prepared on "
            f"{devices} in {time.perf_counter() - t0:.1f} s")


def _mount(path: Path):
    """(mount point, filesystem type) of the deepest mount holding `path` (/proc/mounts)."""
    best = ("/", "?")
    for line in Path("/proc/mounts").read_text().splitlines():
        parts = line.split()
        if len(parts) > 2:
            point = parts[1].replace("\\040", " ")
            if (str(path) == point or str(path).startswith(point.rstrip("/") + "/")) and len(point) >= len(best[0]):
                best = (point, parts[2])
    return best


def compile_cache():
    """The manifest's persistent compilation cache: a directory on a disk (not tmpfs / ramfs) with CACHE_FREE_GB free,
    where this process writes one entry (a fresh jit, the minimum compile time set to 0) and reads it back after
    jax.clear_caches() - JAX's cache-hit event. Returns a line; raises with the cause."""
    import jax
    import jax.monitoring
    import jax.numpy as jnp
    raw = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not raw:
        raise RuntimeError("JAX_COMPILATION_CACHE_DIR is not set (the manifest sets it)")
    path = Path(raw).resolve()
    path.mkdir(parents=True, exist_ok=True)
    point, fstype = _mount(path)
    free = shutil.disk_usage(path).free / 1e9
    if fstype in ("tmpfs", "ramfs"):
        raise RuntimeError(f"{path} is on a {fstype} ({point}): a cold warm-up fills it and dies; use a disk")
    if free < CACHE_FREE_GB:
        raise RuntimeError(f"{path}: {free:.1f} GB free on {point}, at least {CACHE_FREE_GB:g} GB needed")
    events = []
    jax.monitoring.register_event_listener(lambda name, **kw: events.append(name))
    before = {p.name for p in path.iterdir()}
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    salt = float(time.time_ns() % 1000003)                      # a function no earlier run has cached
    first = float(jax.jit(lambda x: jnp.sin(x) * 3.0 + salt)(1.0))
    new = sorted({p.name for p in path.iterdir()} - before)
    if not new or not all((path / n).stat().st_size > 0 for n in new):
        raise RuntimeError(f"no cache entry written in {path} (JAX events {sorted(set(events))})")
    events.clear()
    jax.clear_caches()
    second = float(jax.jit(lambda x: jnp.sin(x) * 3.0 + salt)(1.0))
    if "/jax/compilation_cache/cache_hits" not in events or first != second:
        raise RuntimeError(f"the entry {new[0]} was not read back (JAX events {sorted(set(events))})")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    return f"{path} on {fstype} ({point}), {free:.0f} GB free: entry written ({new[0][:40]}...) and read back"


def require_cpu_negative(rental: bool):
    """The GPU trap stays closed (task #40, poker/solver/tree.py require_cpu): a process without the CPU device must raise at
    the first engine replay, naming JAX_PLATFORMS=cuda,cpu. On a rental box a child runs with JAX_PLATFORMS=cuda (the
    GPU only); a CPU rehearsal has no GPU to drop the CPU backend for, so its child simulates that state (tree._cpu ->
    None, jax.default_backend -> "gpu") - a test of the guard, not of the box. Returns a line; raises when the child
    does not fail with that message."""
    real = "from poker.solver import tree; tree.require_cpu('the start gate negative test')"
    simulated = ("import jax; from poker.solver import tree; tree._cpu = lambda: None; jax.default_backend = lambda: 'gpu'; "
                 "tree.require_cpu('the start gate negative test (simulated)')")
    env = {**os.environ, "JAX_PLATFORMS": "cuda"} if rental else dict(os.environ)
    child = subprocess.run([sys.executable, "-c", real if rental else simulated], env=env, capture_output=True,
                           text=True, timeout=600)
    last = ((child.stderr or child.stdout).strip().splitlines() or ["<no output>"])[-1]
    if child.returncode == 0:
        raise RuntimeError("require_cpu did not raise in a process without the CPU device: the GPU trap is open")
    if "JAX_PLATFORMS=cuda,cpu" not in child.stderr + child.stdout:
        raise RuntimeError(f"the child failed, but not with require_cpu's message: {last[:200]}")
    return ("JAX_PLATFORMS=cuda child" if rental else "simulated (a CPU rehearsal has no GPU)") + f" raised: {last[:160]}"


def device_checks(jobs, packed, fp):
    """Part 5 of the start gate (module doc): devices, versions, the compile cache, the kernel-load smoke and its
    negative (require_cpu); writes results/BOX.json."""
    from importlib import metadata
    from poker.envs import fingerprint as FP
    gpus, driver_cuda = nvidia_smi()
    packages = {}
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cuda-nvcc-cu12",
                 "nvidia-cuda-runtime-cu12", "nvidia-cudnn-cu12", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
              "jax_platforms": os.environ.get("JAX_PLATFORMS"), "gpus": gpus, "driver_cuda": driver_cuda,
              "packages": packages, "python": FP.runtime()["python"], "commit": packed["commit"],
              "code_sha256": fp["sha256"], "game_hash": jobs["game_hash"]}
    import jax
    try:
        devices = jax.devices()
        record["devices"] = [f"{d.platform}:{d.id} {d.device_kind}" for d in devices]
        record["default_backend"] = jax.default_backend()
    except Exception as error:          # noqa: BLE001 - a backend that does not initialise is the finding
        record["devices_error"] = f"{type(error).__name__}: {error}"
        problem(f"JAX cannot initialise JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}: {str(error)[:200]}")
        devices = []
    try:
        record["cpu_device"] = str(jax.devices("cpu")[0])
    except RuntimeError:
        record["cpu_device"] = None
        problem("JAX has no CPU device: the engine replay of poker/solver/tree.py runs there (JAX_PLATFORMS must name cpu)")
    if "cuda" in jobs["env"].get("JAX_PLATFORMS", "") and devices and record.get("default_backend") != "gpu":
        problem(f"the manifest asks for cuda, the default device is {devices[0]}: the jobs would run on the CPU")
    toolkit, driver = _major_minor(packages.get("nvidia-cuda-nvcc-cu12")), _major_minor(driver_cuda)
    if toolkit and driver and toolkit > driver:
        record["toolkit_vs_driver"] = (f"the venv's CUDA toolkit (nvcc {packages['nvidia-cuda-nvcc-cu12']}) is newer "
                                       f"than the driver's CUDA {driver_cuda}")
        warning(record["toolkit_vs_driver"] + ": PTX the driver cannot load fails at the kernel smoke below; a "
                                              "passing smoke still runs with XLA's slower serial compilation")
    record["jax_compilation_cache_dir"] = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    try:
        record["compile_cache"] = compile_cache()
        print(f"  compile cache: {record['compile_cache']}")
    except Exception as error:          # noqa: BLE001 - the cause is the finding
        record["compile_cache"] = f"FAILED {type(error).__name__}: {error}"
        problem(f"compile cache: {str(error)[:300]}")
    try:
        record["kernel_smoke"] = kernel_smoke()
        print(f"  kernel smoke: {record['kernel_smoke']}")
    except Exception as error:          # noqa: BLE001 - the cause is the finding
        record["kernel_smoke"] = f"FAILED {type(error).__name__}: {error}"
        problem(f"kernel-load smoke on {devices[0] if devices else '?'}: {type(error).__name__}: {str(error)[:300]}"
                + (f" ({record['toolkit_vs_driver']})" if "toolkit_vs_driver" in record else ""))
    try:
        record["require_cpu_negative"] = require_cpu_negative("cuda" in jobs["env"].get("JAX_PLATFORMS", ""))
        print(f"  require_cpu negative: {record['require_cpu_negative']}")
    except Exception as error:          # noqa: BLE001 - the cause is the finding
        record["require_cpu_negative"] = f"FAILED {type(error).__name__}: {error}"
        problem(f"require_cpu negative test: {str(error)[:300]}")
    BOX_RECORD.parent.mkdir(parents=True, exist_ok=True)
    BOX_RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"  box: JAX_PLATFORMS {record['jax_platforms']}, devices {record.get('devices')}, GPUs "
          f"{[g.get('name') for g in gpus]}, driver {gpus[0]['driver'] if gpus else None} (CUDA {driver_cuda}), "
          f"jax {packages['jax']} / jaxlib {packages['jaxlib']}, cuda plugin {packages['jax-cuda12-plugin']}, nvcc "
          f"{packages['nvidia-cuda-nvcc-cu12']}, python {record['python']} -> {BOX_RECORD}")
    return record


def plan_spots(jobs, plan):
    """{stage spots name: count} of a milestone's plan ($RIVER from the environment)."""
    out = {}
    for name, count in jobs["plan"][plan]["spots"].items():
        if isinstance(count, str) and count.startswith("$"):
            value = os.environ.get(count[1:])
            if not (value or "").isdigit():
                fail(f"plan {plan}: {count} is not set in the environment")
            count = int(value)
        out[name] = int(count)
    return out


def resource_checks(jobs, plan, gpus):
    """The box against the milestone's needs (manifest resources; static sniff S6, rental-path sniff M11): host RAM for
    one loop process per worker (GPUS or every GPU) plus the critic stage's copies of the largest street's critic rows,
    free disk for the rows twice and a margin, every GPU's memory."""
    res, spots = jobs["resources"], plan_spots(jobs, plan)
    workers = len(os.environ["GPUS"].split(",")) if os.environ.get("GPUS") else max(1, len(gpus))
    rows_gb = sum(spots.get(s, 0) * kb for s, kb in res["row_kb_per_spot"].items()) / 1e6
    critic_gb = max(spots.get(s, 0) * kb for s, kb in res["critic_kb_per_spot"].items()) / 1e6
    host_gb = workers * res["host_base_gb"] + res["critic_ram_factor"] * critic_gb
    disk_gb = 2 * rows_gb + res["disk_fixed_gb"]["margin"]
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
    available_gb = int(mem["MemAvailable"].split()[0]) / 1e6
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"  resources for {plan} ({spots}; {workers} worker(s)): rows {rows_gb:.1f} GB; host RAM {available_gb:.1f} GB "
          f"available, {host_gb:.1f} GB needed ({workers} x {res['host_base_gb']:g} + {res['critic_ram_factor']:g} x "
          f"{critic_gb:.2f}); disk {free_gb:.1f} GB free, {disk_gb:.1f} GB needed; GPU memory "
          f"{[g.get('memory_mib') for g in gpus]} MiB, {res['gpu_mib_min']} needed")
    if available_gb < host_gb:
        problem(f"host RAM {available_gb:.1f} GB available < {host_gb:.1f} GB {plan} needs ({workers} worker(s) and the "
                f"critic stage)")
    if free_gb < disk_gb:
        problem(f"free disk {free_gb:.1f} GB < {disk_gb:.1f} GB (the rows twice and a margin) for {plan}")
    small = [g for g in gpus if float(g.get("memory_mib") or 0) < res["gpu_mib_min"]]
    if small:
        problem(f"GPUs with less memory than {res['gpu_mib_min']} MiB (the k0 high-water mark): "
                f"{[(g.get('index'), g.get('name'), g.get('memory_mib')) for g in small]}")


def start(host=False, plan=None):
    """The start gate (module doc, parts 1-5); returns (manifest, PACKED.json, this tree's fingerprint)."""
    jobs = load_json(DIST / "JOBS.json", "the job manifest: pack with tools/cloud/pack.sh")
    packed = load_json(DIST / "PACKED.json", "the pack record: pack with tools/cloud/pack.sh")
    if packed.get("version", 1) < 2 or "fingerprint" not in packed or "game_hash" not in packed:
        fail("results/pack/PACKED.json was written by pack.sh before task #34 (no fingerprint, no game hash): re-pack")
    if packed.get("manifest_sha256") != sha256_file(DIST / "JOBS.json"):
        problem("results/pack/JOBS.json is not the manifest PACKED.json was packed with (sha256)")
    if packed.get("allow_dirty") or packed.get("dirty_paths"):
        problem(f"packed from a dirty tree ({packed.get('dirty_paths')} paths: {packed.get('dirty_list', [])[:3]}): "
                f"a rehearsal pack (--allow-dirty) never starts a rental")
    if jobs["env"].get("JAX_PLATFORMS") != RENTAL_PLATFORMS:
        problem(f"a rehearsal manifest (JAX_PLATFORMS={jobs['env'].get('JAX_PLATFORMS')}, a rental runs "
                f"{RENTAL_PLATFORMS}): never a rental")
    if plan is not None and plan not in jobs.get("plan", {}):
        fail(f"no plan {plan!r} in the manifest ({sorted(jobs.get('plan', {}))})")
    want = resolved_env(jobs, Path.cwd())                   # the compile cache resolved against this tree (the cwd)
    keys = [k for k in want if not (host and k == "JAX_PLATFORMS")]             # the host gate runs on its CPU
    wrong = {k: (os.environ.get(k), want[k]) for k in keys if os.environ.get(k) != want[k]}
    if wrong:
        problem(f"this process's environment is not the manifest's (now, manifest): {wrong}")
    from poker.envs import config                 # after the environment check: config reads it at import
    from poker.envs import fingerprint as FP
    if (config.GRID, config.REGIME) != (int(jobs["env"]["POKER_GRID"]), int(jobs["env"]["POKER_REGIME"])):
        problem(f"config reads grid {config.GRID}, regime {config.REGIME}; the manifest says {jobs['env']}")
    game = config.game_hash()
    print(f"  game: {config.describe_game()} (game {game[:12]}); manifest {jobs['game_hash'][:12]}, packed "
          f"{packed['game_hash'][:12]}", flush=True)
    if not game == jobs["game_hash"] == packed["game_hash"]:
        problem(f"this process plays game {game[:12]}, the manifest {jobs['game_hash'][:12]}, the pack "
                f"{packed['game_hash'][:12]}")
    fp = FP.code_fingerprint()
    print(f"  code: {FP.describe(fp)}; packed {FP.describe(packed['fingerprint'])}", flush=True)
    if fp["sha256"] != packed["fingerprint"]["sha256"]:
        diff = FP.differences(packed["fingerprint"], fp)
        problem(f"this tree's code {fp['sha256'][:12]} is not the packed {packed['fingerprint']['sha256'][:12]}: "
                + ", ".join(diff[:5]) + (f" (+{len(diff) - 5})" if len(diff) > 5 else ""))
    elif fp["dirty"] or packed["fingerprint"].get("dirty"):
        problem(f"the code fingerprint is dirty: {(fp['dirty_paths'] or packed['fingerprint'].get('dirty_paths'))[:3]}")
    check_checks(packed)
    if not host:
        record = device_checks(jobs, packed, fp)
        if plan is not None:
            resource_checks(jobs, plan, record["gpus"])
    return jobs, packed, fp


def start_gate(args):
    jobs, packed, fp = start(host=args.host, plan=args.plan)
    finish(f"start{' (host)' if args.host else ''}: game {jobs['game_hash'][:12]} "
           f"({' '.join(f'{k}={v}' for k, v in jobs['env'].items())}), code {fp['sha256'][:12]} (commit "
           f"{packed['commit'][:12]}), checks {(packed.get('checks') or {}).get('source')}")


# ---------------------------------------------------------------- caches: one process builds, everything is hashed

def caches(args):
    """Build the missing exact tables in this one process, then hash every one of them (module doc)."""
    from poker.solver import flops as FL
    from poker.solver import loop as L
    from poker.solver import rank7
    from poker.solver import runout as RO
    t0 = time.perf_counter()
    log = lambda m: print(f"  {m}", flush=True)
    built = []
    if not rank7.TABLE_PATH.exists():
        rank7.load_table(log=log)
        built.append(rank7.TABLE_PATH.name)
    if not L.PREFLOP_EQUITY.exists():
        RO.preflop_equity_cached(L.PREFLOP_EQUITY, log=log)
        built.append(L.PREFLOP_EQUITY.name)
    missing = [f for f in FL.FLOPS if not (RO.FLOP_CACHE / "{}_{}_{}.npy".format(*sorted(int(c) for c in f))).exists()]
    for flop in missing:
        RO.flop_equity_cached(flop)
    if missing:
        built.append(f"{len(missing)} flop matrices")
    t_build = time.perf_counter() - t0
    reference = json.loads(CACHE_REFERENCE.read_text())
    for path in (rank7.TABLE_PATH, L.PREFLOP_EQUITY):
        digest = sha256_file(path)
        if digest != reference.get(path.name):
            problem(f"cache {path.name}: sha256 {digest[:12]}, reference {str(reference.get(path.name))[:12]} "
                    f"({CACHE_REFERENCE.name})")
    wanted = dict(reversed(line.split()) for line in FLOP_MANIFEST.read_text().splitlines() if line.strip())
    present = {p.name for p in RO.FLOP_CACHE.glob("*.npy")}
    if set(wanted) - present:
        problem(f"{len(set(wanted) - present)} flop matrices of {FLOP_MANIFEST.name} missing")
    wrong = [n for n in sorted(set(wanted) & present) if sha256_file(RO.FLOP_CACHE / n) != wanted[n]]
    if wrong:
        problem(f"{len(wrong)} flop matrices differ from {FLOP_MANIFEST.name}: {wrong[:5]}")
    finish(f"caches: built {built or 'nothing'} in {t_build:.0f} s; rank7, preflop and {len(wanted)} flop matrices "
           f"equal to the references (hashed in {time.perf_counter() - t0 - t_build:.0f} s)")


# ---------------------------------------------------------------- job gates

def check_chain(layout, packed, out):
    """The code chain of a run (rental-path sniff M9): the layout's code record and fingerprint, then every entry of
    code_resumes.json (poker/solver/loop.py check_code). Every link clean, every resume with its reason, the last link the
    pack's commit and code."""
    from poker.envs import fingerprint as FP
    first = dict(layout.get("code") or {})
    first["code_sha256"] = (layout.get("code_fingerprint") or {}).get("sha256")
    path = out / "code_resumes.json"
    resumes = json.loads(path.read_text()) if path.exists() else []
    chain = [first] + resumes
    print(f"  {out.name}: code chain " + " -> ".join(
        f"{str(c.get('commit'))[:12]} ({'dirty' if c.get('dirty', True) else 'clean'}, code {str(c.get('code_sha256'))[:12]}"
        + (f", reason {c.get('reason')!r}" if i else ", start") + ")" for i, c in enumerate(chain))
          + f"; {FP.describe(layout.get('code_fingerprint'))}")
    for i, link in enumerate(chain):
        if link.get("dirty", True):
            problem(f"{out.name}: code chain link {i} ({str(link.get('commit'))[:12]}) ran on a dirty or unknown tree")
        if i and not link.get("reason"):
            problem(f"{out.name}: resume {i} on {str(link.get('commit'))[:12]} has no recorded reason")
    last = chain[-1]
    if last.get("commit") != packed["commit"] or last.get("code_sha256") != packed["fingerprint"]["sha256"]:
        problem(f"{out.name}: the run's last code is {str(last.get('commit'))[:12]} / {str(last.get('code_sha256'))[:12]}, "
                f"the pack is {packed['commit'][:12]} / {packed['fingerprint']['sha256'][:12]} (a resume across a code "
                f"change without --allow-code-change, or another pack's run)")


def check_run(layout, jobs, packed, out, plan):
    """The layout of a job's output against the manifest, the plan and the pack."""
    scheme = layout.get("scheme") or {}
    regime = (scheme.get("regime") or {}).get("number")
    if (scheme.get("grid"), regime) != (int(jobs["env"]["POKER_GRID"]), int(jobs["env"]["POKER_REGIME"])):
        problem(f"{out.name}: layout grid {scheme.get('grid')}, regime {regime}; the manifest says {jobs['env']}")
    if layout.get("game_hash") != jobs["game_hash"]:
        problem(f"{out.name}: layout game {str(layout.get('game_hash'))[:12]}, the manifest's {jobs['game_hash'][:12]}")
    if not jobs.get("training_domain") or layout.get("training_domain") != jobs["training_domain"]:
        problem(f"{out.name}: training domain differs from the manifest or is missing")
    teacher, want = layout.get("teacher") or {"stop": "certificate"}, jobs.get("teacher") or {}
    if teacher.get("stop") != want.get("stop") or (want.get("stop") == "fixed_t" and (
            teacher.get("iterations") != want.get("iterations") or teacher.get("e_flag_factor") != want.get("e_flag_factor"))):
        problem(f"{out.name}: teacher {teacher.get('stop')} T {teacher.get('iterations')}; the manifest's "
                f"{want.get('stop')} T {want.get('iterations')}")
    print(f"  {out.name}: layout grid {scheme.get('grid')}, regime {regime}, game {str(layout.get('game_hash'))[:12]}, "
          f"teacher {teacher.get('stop')} {teacher.get('iterations', '')}, quick {layout.get('quick')}")
    if plan is not None:
        spots = plan_spots(jobs, plan)
        differ = {k: (layout.get(k), v) for k, v in spots.items() if layout.get(k) != v}
        if differ:
            problem(f"{out.name}: spot counts (layout, plan {plan}) differ: {differ}")
        if bool(layout.get("quick")) != bool(jobs["plan"][plan].get("quick")):
            problem(f"{out.name}: layout quick {layout.get('quick')}, plan {plan} quick {jobs['plan'][plan].get('quick')}")
    check_chain(layout, packed, out)


def check_solve(name, value, layout, jobs, quick, warn_share):
    """One solve stage's summary: fixed T (iterations, E bound, payout, e-flags) or certificate (the old rule)."""
    street = name.removesuffix("_solve")
    if not finite(*(value.get(key) for key in ("e_bb_median", "e_bb_p95", "e_bb_max"))):
        problem(f"{name}: missing or non-finite BB error diagnostics")
    if value.get("teacher") == "fixed_t":
        if not finite(value.get("e_median"), value.get("e_p90"), value.get("e_max")) or not value.get("spots"):
            problem(f"{name}: non-finite E or no spots ({value.get('e_median')}, {value.get('spots')})")
            return
        t = (layout.get("teacher") or {}).get("iterations", {}).get(street)
        if not value.get("iters_min") == value.get("iters_max") == value.get("iterations") == t:
            problem(f"{name}: iterations min {value.get('iters_min')} max {value.get('iters_max')}, stage T "
                    f"{value.get('iterations')}, layout T {t}: not every spot ran T")
        bound = ((jobs.get("teacher") or {}).get("e_bound") or {}).get(street)
        if bound is not None and value["e_median"] > bound:
            problem(f"{name}: E median {value['e_median']:.5f} above the bound {bound:g} (config.TEACHER_E_BOUND)")
        share = value.get("e_flagged", 0) / value["spots"]
        print(f"  {name}: T {value.get('iterations')}, {value['spots']} spots, E median {value['e_median']:.5f} "
              f"(bound {bound}) p90 {value['e_p90']:.5f} max {value['e_max']:.5f}; flagged {value.get('e_flagged')} "
              f"({share:.1%}, E > {value.get('e_flag_factor')} x median = {value.get('e_flag_threshold', float('nan')):.5f})")
        if share > warn_share:
            warning(f"{name}: {share:.1%} of the spots flagged, above {warn_share:.0%} (a diagnostic, not a stop)")
    else:
        if not finite(value.get("e_median"), value.get("certified_fraction")):
            problem(f"{name}: non-finite certificate numbers")
        if name == "river_solve" and not quick:
            taken = value["accepted_fraction"] if "accepted_fraction" in value else value["certified_fraction"]
            if taken < 0.9:
                problem(f"river accepted {taken:.2%} < 90% after the second pass")
    payout = (value.get("payout") or {}).get("above_fraction")
    if payout is not None:
        from poker.solver.loop import PAYOUT_WARN
        if payout > PAYOUT_WARN and not quick:
            problem(f"{name}: {payout:.1%} of the spots pay out more than the pot holds (> {PAYOUT_WARN:.0%}, "
                    f"poker/solver/loop.py PAYOUT_WARN): a street-end critic creates EV")
        elif payout > PAYOUT_WARN:
            warning(f"{name}: payout excess on {payout:.1%} of the spots (--quick: toy critics, not a stop)")


def inspect_critic_checkpoint(path, name):
    """Use the production loader on the CPU; validate weights without a solve or JIT."""
    import jax
    from poker.solver import critics as CR
    from poker.solver import tree as TR
    with jax.default_device(TR.require_cpu("the loop checkpoint gate")):
        _, meta = CR.load_critic(path, name=name)
    return meta


def check_checkpoints(out, layout, stages, summary):
    """Every completed critic/end stage has its exact valid checkpoint and finite diagnostics."""
    projection = layout.get("critic_projection") or {}
    kappas, street_of = projection.get("kappa") or {}, projection.get("street_of") or {}
    protocol = {"epochs": layout.get("critic_epochs"), "epoch_steps": layout.get("critic_epoch_steps"),
                "patience": layout.get("critic_patience"), "clip_norm": layout.get("critic_clip_norm"),
                "lr": layout.get("critic_lr")}
    for name in stages:
        if not (name.endswith("_critic") or name.endswith("_end")):
            continue
        value = summary[name]
        metric = "test_mae_vs_exact_average" if name.endswith("_end") else "test_mae"
        fields = (metric, "identity_gap_max", "identity_gap_mean", "identity_gap_p99") if name.endswith("_end") else (metric,)
        if any(not finite(value.get(key)) or value[key] < 0 for key in fields):
            problem(f"{name}: non-finite or negative stage metrics ({', '.join(fields)})")
        if value.get("name") != name or value.get("protocol") != protocol:
            problem(f"{name}: stage name/training protocol differs from the run")
        path = out / f"{name}.npz"
        if not path.is_file():
            problem(f"{name}: required checkpoint {path.name} missing")
            continue
        if value.get("checkpoint_sha256") != sha256_file(path):
            problem(f"{name}: checkpoint SHA256 differs from its completed stage report")
            continue
        try:
            meta = inspect_critic_checkpoint(path, name)
        except (OSError, ValueError, KeyError, TypeError, SystemExit) as error:
            problem(f"{name}: invalid checkpoint ({error})")
            continue
        kappa = kappas.get(street_of.get(name))
        if (not finite(kappa) or meta.get("kappa") != kappa or value.get("kappa") != kappa
                or meta.get("split_version") != layout.get("split_version") or meta.get("test_mae") != value.get(metric)):
            problem(f"{name}: checkpoint/stage metadata differs from the run or reported metric")
        print(f"  {name}: checkpoint hash, production loader and stage metadata checked")


def check_rows(out, layout, stages):
    """Every batch of every solve stage left its row and metrics files (the rows are the product)."""
    from poker.solver import loop as L
    for stage in stages:
        if not stage.endswith("_solve"):
            continue
        name = stage.removesuffix("_solve")
        starts = L.batch_starts(name, int(layout[name]), int(layout["solve_batch"]))
        missing = [f"{name}_{s:06d}.{ext}" for s in starts for ext in ("npz", "json")
                   if not (out / f"{name}_{s:06d}.{ext}").exists()]
        print(f"  {name}: {len(starts)} batches, {len(starts) - len(missing) // 2} with critic rows and metrics")
        if missing:
            problem(f"{out.name}: {len(missing)} row/metrics files of {name} missing: {missing[:4]}")


def check_caches(found):
    reference = json.loads(CACHE_REFERENCE.read_text())
    for name, digest in sorted((found or {}).items()):
        if name in reference and digest != reference[name]:
            problem(f"cache {name}: sha256 {digest[:12]} on this box, reference {reference[name][:12]} "
                    f"({CACHE_REFERENCE.name}): another evaluator or a corrupt table")
    print("  caches: " + (", ".join(f"{n} {'= reference' if reference.get(n) == d else d[:12]}" for n, d in
                                    sorted((found or {}).items())) or "none reported"))


def job_context(args):
    """(manifest, PACKED.json) - from the start gate, or read as they are with --run-checks-only."""
    if args.run_checks_only:
        print("  (start gate not evaluated: --run-checks-only)")
        return load_json(DIST / "JOBS.json", "the job manifest"), load_json(DIST / "PACKED.json", "the pack record")
    jobs, packed, _ = start(plan=None)
    return jobs, packed


def loop(args):
    """A loop job (poker/solver/loop.py): the start gate, the run against manifest, plan and pack, every stage."""
    out = args.out
    jobs, packed = job_context(args)
    layout = load_json(out / "layout.json", "the run's layout")
    check_run(layout, jobs, packed, out, args.plan)
    s = {}
    for p in sorted(out.glob("summary_w*.json")):
        s.update(json.loads(p.read_text()))
    if not s:
        fail(f"no summary_w*.json in {out}")
    quick = bool(layout.get("quick"))
    warn_share = float(os.environ.get("E_FLAG_WARN", jobs.get("e_flag_warn", 0.25)))
    stages = [k for k in s if k not in ("layout", "cache_sha256")]
    if args.plan is not None:
        want = plan_stages(jobs, args.plan)
        if want == "all":
            from poker.solver.loop import STAGES
            want = STAGES
        missing = [x for x in want if x not in stages]
        unexpected = [x for x in stages if x not in want]
        if missing or unexpected:
            fail(f"{out.name}: stages differ from plan {args.plan}: missing {missing}; unexpected {unexpected}")
    for name in stages:
        value = s[name]
        if name.endswith("_solve"):
            check_solve(name, value, layout, jobs, quick, warn_share)
    check_checkpoints(out, layout, stages, s)
    check_rows(out, layout, stages)
    check_caches(s.get("cache_sha256"))
    utilisation(out, quick)
    finish(f"loop {out.name}, stages {stages}, game {jobs['game_hash'][:12]}, code "
           f"{packed['fingerprint']['sha256'][:12]} (commit {packed['commit'][:12]})"
           + (" - run checks only, the start gate was not evaluated" if args.run_checks_only else ""))


def shard(args):
    """One machine's share of a river stage (poker.solver.loop --shard i/k): the start gate, the layout, every batch of the
    shard written, finite E (a fixed-T shard has no certificate to count: its flags come with river_finish)."""
    out = args.out
    jobs, packed = job_context(args)
    reports = sorted(out.glob("shard_*.json"))
    if not reports:
        fail(f"no shard_*.json in {out}")
    for path in reports:
        r = json.loads(path.read_text())
        if r["done"] != r["batches"]:
            problem(f"{path.name}: {r['done']}/{r['batches']} batches")
    e = [m["e_solve"] for p in out.glob("river_[0-9]*.json") for m in json.loads(p.read_text())]
    if not e or not finite(*e):
        problem("non-finite or missing E of the river spots")
    layout = load_json(out / "layout.json", "the run's layout")
    check_run(layout, jobs, packed, out, args.plan)
    utilisation(out, bool(layout.get("quick")), "gpu_util_shard.json")
    certified = (f", certified {sum(x <= layout['e_target'] for x in e)}" if "e_target" in layout and e else "")
    finish(f"shard {[p.name for p in reports]}, {len(e)} spots{certified}, game {jobs['game_hash'][:12]}, code "
           f"{packed['fingerprint']['sha256'][:12]}" + (" - run checks only" if args.run_checks_only else ""))


def batches(args):
    """The per-batch gate of a running loop job (module doc); standard library only. Remembers the files it passed in
    <out>/batch_gate.json and reads only the new ones."""
    out = args.out
    layout_path = out / "layout.json"
    if not layout_path.exists():
        finish(f"batches {out.name}: no layout yet")
        return
    layout = json.loads(layout_path.read_text())
    jobs = load_json(DIST / "JOBS.json", "the job manifest")
    teacher = layout.get("teacher") or {"stop": "certificate"}
    bounds = (jobs.get("teacher") or {}).get("e_bound") or {}
    state_path = out / "batch_gate.json"
    seen = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    new = sorted(p.name for p in out.glob("*_[0-9]*.json") if BATCH_FILE.match(p.name) and p.name not in seen)
    for name in new:
        street = BATCH_FILE.match(name).group(1)
        try:
            metrics = json.loads((out / name).read_text())
        except ValueError as error:
            problem(f"{name}: unreadable ({error})")
            continue
        e = [m.get("e_solve") for m in metrics]
        bad = [m.get("spot") for m, x in zip(metrics, e) if not (isinstance(x, (int, float)) and math.isfinite(x)
                                                                 and x >= E_NEGATIVE)]
        if bad:
            problem(f"{name}: E non-finite or below {E_NEGATIVE:g} (a solver bug, not convergence) on spots {bad[:5]}")
        if teacher.get("stop") == "fixed_t":
            t = teacher.get("iterations", {}).get(street)
            off = sorted({m.get("iters") for m in metrics} - {t})
            if off:
                problem(f"{name}: iterations {off}, the run's T is {t}")
        good = sorted(x for x in e if isinstance(x, (int, float)) and math.isfinite(x))
        median = good[len(good) // 2] if good else float("nan")
        if bounds.get(street) is not None and good and median > bounds[street]:
            problem(f"{name}: batch E median {median:.5f} above the bound {bounds[street]:g}")
    if not PROBLEMS:
        state_path.write_text(json.dumps(sorted(seen | set(new))))
    finish(f"batches {out.name}: {len(new)} new, {len(seen) + len(new)} checked")


def heartbeat(args):
    """The host's view of a running queue (module doc): results/HEARTBEAT of queue.sh, key=value lines."""
    path = args.out
    if not path.exists():
        fail(f"{path} missing (fetch it: ~/clore_tools/box.sh NAME get /workspace/poker-huy/results/HEARTBEAT {path})")
    hb = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    now = time.time() if args.now is None else float(args.now)
    period = int(hb.get("period_s") or 30)
    age = now - int(hb.get("epoch") or 0)
    job_age = now - int(hb.get("job_started_epoch") or now)
    log_age = now - int(hb.get("log_mtime_epoch") or hb.get("job_started_epoch") or now)
    budget, stall = int(hb.get("budget_s") or 0), int(hb.get("stall_s") or 0)
    print(f"  queue {hb.get('milestone')} state {hb.get('state')}, job {hb.get('job')} for {job_age:.0f} s "
          f"(budget {budget} s), worker logs {hb.get('log_bytes')} bytes, silent {log_age:.0f} s (stall {stall} s); "
          f"heartbeat {age:.0f} s old; last line: {hb.get('last_line', '')[:160]}")
    if age > 3 * period + 60:
        problem(f"heartbeat {age:.0f} s old (period {period} s): the runner died or the box is unreachable")
    if hb.get("state") == "failed":
        problem(f"the queue reports a failure: {hb.get('last_line', '')[:200]}")
    if hb.get("state") == "running" and budget and job_age > budget:
        problem(f"job {hb.get('job')} running {job_age:.0f} s, past its budget {budget} s")
    if hb.get("state") == "running" and stall and log_age > stall:
        problem(f"job {hb.get('job')}: worker logs silent for {log_age:.0f} s (stall limit {stall} s): hung?")
    finish(f"heartbeat {hb.get('milestone')}: {hb.get('state')}, job {hb.get('job')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="gate", required=True)
    s = sub.add_parser("start")
    s.add_argument("--host", action="store_true", help="pack.sh on the host: no device, resource or JAX_PLATFORMS check")
    s.add_argument("--plan", help="a milestone of the manifest's plan: check the box's resources for it")
    s.set_defaults(func=start_gate)
    sub.add_parser("caches").set_defaults(func=caches)
    for name, func in (("loop", loop), ("shard", shard), ("batches", batches), ("heartbeat", heartbeat)):
        g = sub.add_parser(name)
        g.add_argument("out", type=Path)
        if name in ("loop", "shard"):
            g.add_argument("--plan", help="the milestone of the manifest's plan this job ran")
            g.add_argument("--run-checks-only", action="store_true",
                           help="the run's checks without the start gate (a diagnostic; queue.sh never uses it)")
        if name == "heartbeat":
            g.add_argument("--now", type=float, help="the time to judge the heartbeat at (tests)")
        g.set_defaults(func=func)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
