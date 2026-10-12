#!/bin/bash
# HU flop 'normal' (28 decision nodes, 13 street-end leaves, 14 all-ins) and turn 'normal' (40 nodes, 19 leaves,
# 20 all-ins): pcfr.solve_fixed with (a) the check-down stand-in critic (returns F) and (b) critics.net_critic on
# random init (512x512x512, k=2 slots). One process per run, pinned to one core. T=500 (extrapolate x3 to
# config.TEACHER_T 1500: fixed chunks, data-independent cost) plus one T=1500 run as a direct check.
# Usage: bash tools/audit/engine-profile/run_flop_turn.sh [core] ; appends to results/audit/engine-profile/flop_turn.jsonl
cd "$(dirname "$0")/../../.."
CORE=${1:-2}
OUT=results/audit/engine-profile/flop_turn.jsonl
export POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1
export AUDIT_SCRATCH=${AUDIT_SCRATCH:-/tmp/engine_profile_cache}
run() { echo "== $* (load $(cut -d' ' -f1-3 /proc/loadavg))"; timeout 600 taskset -c $CORE python3 tools/audit/engine-profile/profile_solve.py "$@" --out $OUT 2>&1 | tail -1 | cut -c1-300; }
run --street 1 --menu normal --batch 1 --iters 500 --critic checkdown --reps 5
run --street 1 --menu normal --batch 4 --iters 500 --critic checkdown --reps 3
run --street 1 --menu normal --batch 4 --iters 500 --critic net --reps 3
run --street 1 --menu normal --batch 4 --iters 1500 --critic net --reps 2
run --street 2 --menu normal --batch 4 --iters 500 --critic checkdown --reps 3
run --street 2 --menu normal --batch 4 --iters 500 --critic net --reps 3
