"""GPU utilisation and speed per stage of a run (brief M2, items 7; step 1, A8: >= 60% in the solve
stages, else the milestone fails).

    python -m tools.cloud.gpu_util results/gpu_util_k0.log results/k0.w0.log results/k0.w1.log \
        --gpus 0,1 [--json out.json]

The dmon log (nvidia-smi dmon -s um -o DT, queue.sh) has one row per GPU per sample; worker i's log
(one line per batch or stage, "HH:MM:SS [wK] <stage> ...") belongs to GPU gpus[i] (one log and no
--gpus: every GPU belongs to it). A sample of a GPU belongs to the first line of its worker's log
written after it (the batch or stage it was working on). Per stage and GPU: samples, mean and 10th
percentile of the SM activity; per solve stage and GPU: spots per minute (batch lines' spots over
the wall time from the worker's previous line to its last batch line; both batch-line formats of
poker/solver/loop.py: "river 0-128/3000: ..." and the fixed-T "river 0-128/3000 (normal, 800 iterations): ...");
per GPU: the peak of the framebuffer memory in use (dmon fb, MB). Queue resumes append both logs
with matching RUN_START IDs and UTC dates: samples stay within their invocation, and solve speed
sums active invocation durations without counting a pause between them. An old unmarked prefix
is one legacy invocation, dated by its dmon samples.
"""

import argparse
import json
import re
from datetime import datetime, timedelta

LINE = re.compile(r"^(\d\d:\d\d:\d\d) \[w(\S+)\] (\S+)")
BATCH = re.compile(r"^(\d\d:\d\d:\d\d) \[w\S+\] (river|turn|flop) (\d+)-(\d+)/\d+(?: \([^)]*\))?:")
STAGE_OF = {"river": "river_solve", "turn": "turn_solve", "flop": "flop_solve"}
SOLVE = ("river_solve", "turn_solve", "flop_solve")
RUN_START = re.compile(r"^# RUN_START (\S+) id=(\S+)(?: gpu=(\d+))?$")


def _run_start(line):
    if not line.startswith("# RUN_START "):
        return None
    m = RUN_START.fullmatch(line.strip())
    if m is None:
        raise ValueError(f"invalid run boundary: {line.strip()}")
    return datetime.strptime(m[1], "%Y-%m-%dT%H:%M:%SZ"), m[2], int(m[3]) if m[3] is not None else None


def read_dmon(path, column="sm"):
    """[(datetime, gpu, value, segment)] from dmon; column="fb" selects framebuffer MB."""
    rows, cols, segment = [], None, None
    for line in open(path):
        marker = _run_start(line)
        if marker is not None:
            segment, cols = marker[1], None
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "#Date":
            cols = [c.lower() for c in parts]
            cols[0] = "date"
            continue
        if parts[0].startswith("#") or cols is None or len(parts) < len(cols):
            continue
        rec = dict(zip(cols, parts))
        if rec.get(column, "-") == "-":
            continue
        when = datetime.strptime(rec["date"] + " " + rec["time"], "%Y%m%d %H:%M:%S")
        rows.append((when, int(rec["gpu"]), float(rec[column]), segment))
    return rows


def _times(path, day, gpu=None):
    """Worker events with dates and invocation IDs; dated boundaries handle arbitrary resume gaps."""
    prev, segment = None, None
    for line in open(path):
        marker = _run_start(line)
        if marker is not None:
            prev, segment, recorded_gpu = marker
            if gpu is not None and recorded_gpu != gpu:
                raise ValueError(f"{path}: recorded GPU {recorded_gpu} differs from GPU {gpu}")
            day = prev.replace(hour=0, minute=0, second=0)
            yield prev, line, None, segment
            continue
        m = LINE.match(line)
        if not m:
            continue
        t = datetime.strptime(day.strftime("%Y%m%d ") + m.group(1), "%Y%m%d %H:%M:%S")
        if prev is not None and t < prev:
            day += timedelta(days=1)
            t += timedelta(days=1)
        prev = t
        yield t, line, m, segment


def read_log(path, day, gpu=None):
    """[(datetime, stage, segment)] of the worker log, including invocation boundaries."""
    return [(t, STAGE_OF.get(m.group(3).rstrip(":"), m.group(3).rstrip(":")) if m else "run start", segment)
            for t, _, m, segment in _times(path, day, gpu)]


def speeds(path, day, gpu=None):
    """Solve spots/minute over active invocation durations, excluding time between resumes."""
    segments = {}
    for t, line, _, segment in _times(path, day, gpu):
        segments.setdefault(segment, []).append((t, line))
    out = {}
    for lines in segments.values():
        for street in ("river", "turn", "flop"):
            idx = [i for i, (_, line) in enumerate(lines) if BATCH.match(line) and BATCH.match(line).group(2) == street]
            if not idx:
                continue
            spots = sum(int(BATCH.match(lines[i][1]).group(4)) - int(BATCH.match(lines[i][1]).group(3)) for i in idx)
            start = lines[idx[0] - 1][0] if idx[0] > 0 else lines[idx[0]][0]
            wall = (lines[idx[-1]][0] - start).total_seconds()
            value = out.setdefault(STAGE_OF[street], {"spots": 0, "seconds": 0.0})
            value["spots"] += spots
            value["seconds"] += wall
    for value in out.values():
        value["spots_per_min"] = 60.0 * value["spots"] / value["seconds"] if value["seconds"] > 0 else None
    return out


def summarise(dmon, logs_by_gpu):
    """Per (stage, GPU) samples; never attribute a sample to a different invocation."""
    per = {}
    for gpu, lines in logs_by_gpu.items():
        segments = {}
        for when, stage, segment in lines:
            segments.setdefault(segment, []).append((when, stage))
        samples = {}
        for when, g, sm, segment in dmon:
            if g == gpu:
                samples.setdefault(segment, []).append((when, sm))
        for segment, rows in samples.items():
            events = sorted(segments.get(segment, []))
            j = 0
            for when, sm in sorted(rows):
                while j < len(events) and events[j][0] < when:
                    j += 1
                stage = events[j][1] if j < len(events) else "after the last line"
                per.setdefault((stage, gpu), []).append(sm)
    table = []
    for (stage, gpu), v in per.items():
        v = sorted(v)
        table.append({"stage": stage, "gpu": gpu, "samples": len(v), "sm_mean": sum(v) / len(v),
                      "sm_p10": v[int(0.1 * (len(v) - 1))]})
    first = {}
    for lines in logs_by_gpu.values():
        for t, stage, _ in lines:
            first[stage] = min(first.get(stage, t), t)
    return sorted(table, key=lambda r: (first.get(r["stage"], datetime.max), r["gpu"]))


def solve_utilisation(table):
    """{gpu: SM mean over the samples of the solve stages}."""
    acc = {}
    for r in table:
        if r["stage"] in SOLVE:
            s, n = acc.get(r["gpu"], (0.0, 0))
            acc[r["gpu"]] = (s + r["sm_mean"] * r["samples"], n + r["samples"])
    return {g: s / n for g, (s, n) in acc.items() if n}


def report(dmon_path, logs, gpus=None):
    dmon = read_dmon(dmon_path)
    if not dmon:
        raise SystemExit(f"{dmon_path}: no samples")
    day = min(d for d, _, _, _ in dmon).replace(hour=0, minute=0, second=0)
    seen = sorted({g for _, g, _, _ in dmon})
    if gpus is None:
        if len(logs) != 1:
            raise SystemExit("several worker logs need --gpus (worker i runs on gpus[i])")
        gpus = seen
        logs = logs * len(seen)
    if len(gpus) != len(logs):
        raise SystemExit(f"{len(logs)} logs for GPUs {gpus}")
    logs_by_gpu = {g: read_log(path, day, g) for g, path in zip(gpus, logs)}
    table = summarise(dmon, logs_by_gpu)
    speed = {str(g): speeds(path, day, g) for g, path in zip(gpus, logs)}
    fb = {}
    for _, g, mb, _ in read_dmon(dmon_path, "fb"):
        fb[str(g)] = max(fb.get(str(g), 0.0), mb)
    return {"stages": table, "solve_sm_mean": {str(g): v for g, v in solve_utilisation(table).items()},
            "spots_per_min": speed, "fb_peak_mb": fb}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dmon")
    p.add_argument("logs", nargs="+")
    p.add_argument("--gpus", help="comma-separated GPU index of each worker log, in order")
    p.add_argument("--json")
    args = p.parse_args(argv)
    gpus = [int(x) for x in args.gpus.split(",")] if args.gpus else None
    rep = report(args.dmon, args.logs, gpus)
    for r in rep["stages"]:
        print(f"{r['stage']:22s} gpu {r['gpu']}: {r['samples']:5d} samples, SM mean {r['sm_mean']:5.1f}%, p10 {r['sm_p10']:5.1f}%")
    print(f"solve stages, SM mean by GPU: { {g: round(v, 1) for g, v in rep['solve_sm_mean'].items()} }")
    for g, st in rep["spots_per_min"].items():
        for stage, v in st.items():
            print(f"gpu {g} {stage:14s}: {v['spots']} spots in {v['seconds']:.0f} s = {v['spots_per_min']:.1f} spots/min")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
