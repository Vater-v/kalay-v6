#!/bin/bash
# River sweep: HU river 'normal' (40 decision nodes) and 'short' trees, pcfr.solve_fixed at T=800
# (config.TEACHER_T['river']), one process per (menu, batch), pinned to one core (per-core numbers).
# Usage: bash tools/audit/engine-profile/run_river.sh [core] ; appends to results/audit/engine-profile/river.jsonl
cd "$(dirname "$0")/../../.."
CORE=${1:-1}
OUT=results/audit/engine-profile/river.jsonl
export POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1
export AUDIT_SCRATCH=${AUDIT_SCRATCH:-/tmp/engine_profile_cache}
run() { echo "== $* (load $(cut -d' ' -f1-3 /proc/loadavg))"; timeout 600 taskset -c $CORE python3 tools/audit/engine-profile/profile_solve.py "$@" --out $OUT 2>&1 | tail -1 | cut -c1-300; }
run --street 3 --menu normal --batch 1  --iters 800 --reps 10
run --street 3 --menu normal --batch 4  --iters 800 --reps 5
run --street 3 --menu normal --batch 16 --iters 800 --reps 3
run --street 3 --menu normal --batch 32 --iters 400 --reps 2
run --street 3 --menu short  --batch 1  --iters 800 --reps 10
run --street 3 --menu short  --batch 16 --iters 800 --reps 3
