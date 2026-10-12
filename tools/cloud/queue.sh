#!/usr/bin/env bash
# Run a milestone's GPU jobs back to back on every GPU of the box, with automatic gates.
# Each job runs one worker process per GPU on the same output directory (work items are
# claimed, see poker/solver/io.py), so N GPUs finish a job ~N times faster and none
# idles except at phase barriers. A failed worker stops its job, a failed gate stops the
# queue (no money burnt on a broken step). Logs GPU utilisation every 10 s (per stage:
# tools/cloud/gpu_util.py), packs results after every job, optionally stops the box.
# Task #34: everything a job runs comes from ONE place, results/pack/JOBS.json (the job manifest pack.sh writes with
# tools/cloud/launch.py from poker/envs/config.py): the game and devices (POKER_GRID, POKER_REGIME, POKER_REGIME_FORCE,
# JAX_PLATFORMS=cuda,cpu, JAX_COMPILATION_CACHE_DIR=<tree>/results/jax_cache: exported before anything runs, never the
# repo default or the caller's shell), the plan (the loop's stages, spot counts and arguments, passed explicitly, with
# --workers N), the wall-time budgets, the heartbeat. Before any job: the start gate (tools/cloud/gate.py start
# --plan: game line, code fingerprint, clean pack, checks verdict, devices, the compile cache on a disk written and read,
# the kernel-load smoke, the box's RAM / disk / GPU memory for the plan; refuses on any difference), then
# the caches step (one process builds what is missing and hash-checks all three exact tables: no worker races a first
# build). While a job runs: results/HEARTBEAT (key=value, every heartbeat_s) carries the state, the job, its age and
# budget, its worker logs' size, age and last line (gate.py heartbeat judges it on the host); the batches gate checks
# every solved batch (finite E, T iterations, median E bound) and stops the job within a heartbeat of a broken solve;
# `timeout` stops a job past its budget (exit 124). After a job: its gate (--plan), then results/outbox/job-<job>.tar.gz
# .part-NNN (the job directory with its rows, 512 MB parts). At the end: results/outbox/results-<milestone>.tar.gz
# (results without rows and caches, with results/STATUS, BOX.json and results/pack/{PACKED,JOBS,CHECKS_SUMMARY}.json),
# SHA256SUMS and SIZES of the outbox, and DONE.<milestone> or FAILED (the reason). tools/cloud/box_run.sh chains the
# milestones; tools/cloud/fetch.sh brings the outbox home and runs sha256sum -c; nothing here stops or cancels a box.
#
# Usage (repo root on the box, inside tmux; tools/cloud/box_run.sh runs these):
#        bash tools/cloud/queue.sh check       (the start gate only)
#        bash tools/cloud/queue.sh selftest    (the hang machinery: a self-stopped worker must die at its budget)
#        bash tools/cloud/queue.sh preflight   (production shapes on few spots: speeds, peak GPU memory; ~10-20 min)
#        bash tools/cloud/queue.sh k0_smoke    (the whole loop at --quick sizes, the decided game)
#        bash tools/cloud/queue.sh k0          (iteration 0 at the plan's production sizes)
#        SHARD=i/k RIVER=45000 bash tools/cloud/queue.sh river_shard   (box i of k: its river batches only)
#        RIVER=45000 bash tools/cloud/queue.sh river_finish            (all shards gathered in results/river_big:
#                                                                         E flags, critic, learning curve)
# Env: GPUS=0,1 picks the GPUs, one worker each (default: all). More than one GPU needs a k0_smoke that passed on this
#        box with the same count (results/outbox/DONE.k0_smoke): the multi-GPU path has never run on a rented box.
#      ALLOW_CODE_CHANGE="<reason>" resumes a job started by other code (poker/solver/loop.py --allow-code-change
#        --code-change-reason; recorded in code_resumes.json, printed and checked by the gate).
#      SOLVE_BATCH: spots per solver batch of the loop (default 128 on cards with >= 15 GB, else 32).
#      EXPECT_RIVER_SPM: river spots per minute per card for the speed fuse of poker/solver/loop.py (half of it stops a
#        worker). Its reference (poker/solver/loop.py RIVER_REF) is the grid-7 certificate run: under the fixed-T manifest
#        it is not derived from the card - set it from the preflight's river speed, or the fuse stays off (the
#        budget and the batches gate guard then).
#      AUTO_STOP=1 stops the instance at the end (runpod: runpodctl; vast.ai: vastai + API key; clore: nothing).
#      E_FLAG_WARN: see tools/cloud/gate.py.
set -uo pipefail
cd "$(dirname "$0")/../.."
milestone="${1:?milestone name, e.g. k0 (or check)}"
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false TZ=UTC
mkdir -p results/outbox
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
manifest=results/pack/JOBS.json
status() { echo "$(date -u +%FT%TZ) $*" | tee -a results/STATUS; }
refuse() { status "REFUSED $milestone: $*"; echo "$(date -u +%FT%TZ) refused $milestone: $*" > results/outbox/FAILED; exit 2; }
[ -f "$manifest" ] || refuse "no $manifest (pack with tools/cloud/pack.sh)"
for v in $(env | grep -o '^POKER_[A-Z_]*'); do unset "$v"; done         # the manifest's, nothing else
envline=$("$PY" tools/cloud/launch.py env "$manifest") || refuse "cannot read $manifest"
eval "$envline"
rm -f "results/outbox/DONE.$milestone" results/outbox/FAILED results/.queue_stop
case "$milestone" in
  m1|m1_n3|smoke) refuse "removed RiverLab jobs; choose a supported loop milestone for grid $POKER_GRID" ;;
  check|selftest|preflight|k0|k0_smoke|river_shard|river_finish) ;;
  *) refuse "unknown milestone $milestone" ;;
esac
budget=$("$PY" tools/cloud/launch.py budget "$manifest" "$milestone" 2>/dev/null || echo 0)
period=$("$PY" tools/cloud/launch.py get "$manifest" heartbeat_s)
stall=$("$PY" tools/cloud/launch.py get "$manifest" stall_s)
log_segment="${milestone}-$(date -u +%Y%m%dT%H%M%SZ)-$$"

job_state() { echo "$1 $(date +%s) $2 $3" > results/.queue_job; }    # job_state NAME BUDGET STATE
workers() {  # workers NAME CMD...: one process per GPU, all writing results/NAME, each under the job's budget
  local name="$1"; shift
  rm -rf "results/${name}/claims"                  # nobody runs now: claims are stale
  # Summaries describe this invocation, including reused stages. Keep stage models and metrics for resume.
  rm -f -- "results/${name}"/summary_w*.json || return 1  # only this job, before any worker starts
  local pids=() i pid status=0 rc
  for i in "${!gpus[@]}"; do
    # Keep every invocation of this logical job; the shared ID pairs worker events with dmon samples.
    printf '# RUN_START %s id=%s gpu=%s\n' "$(date -u +%FT%TZ)" "$log_segment" "${gpus[$i]}" \
      >> "results/${name}.w${i}.log" || return 1
    WORKER_ID="$i" CUDA_VISIBLE_DEVICES="${gpus[$i]}" timeout --kill-after=120 "$budget" "$@" --out "results/${name}" \
      --workers "${#gpus[@]}" >> "results/${name}.w${i}.log" 2>&1 &
    pids+=("$!")
  done
  local left=("${pids[@]}")
  while [ "${#left[@]}" -gt 0 ]; do
    local still=()
    for pid in "${left[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then still+=("$pid")
      else
        wait "$pid"; rc=$?
        if [ "$rc" -ne 0 ]; then
          status=1
          [ "$rc" -eq 124 ] && status "job $name: a worker exceeded the wall-time budget of ${budget} s (timeout)"
        fi
      fi
    done
    left=(${still[@]+"${still[@]}"})
    if [ -f results/.queue_stop ] && [ "${#left[@]}" -gt 0 ]; then     # the batches gate stopped the job
      status "job $name stopped by the batches gate: $(cat results/.queue_stop)"; status=1
    fi
    if [ "$status" -ne 0 ] && [ "${#left[@]}" -gt 0 ]; then    # one worker failed: stop the others
      kill "${left[@]}" 2>/dev/null; wait "${left[@]}" 2>/dev/null; left=()
    fi
    [ "${#left[@]}" -gt 0 ] && sleep 5
  done
  return $status
}

if [ "$milestone" = selftest ]; then   # no project code: before the start gate, on any box
  gpus=(0); start=$(date +%s); before=$(cat results/STATUS 2>/dev/null | wc -l)
  status "selftest: a worker that stops itself (kill -STOP) under the selftest budget of ${budget} s"
  job_state selftest "$budget" running
  workers selftest python3 -c "import os, signal; print('selftest worker: stopping myself', flush=True); os.kill(os.getpid(), signal.SIGSTOP)"
  rc=$?; took=$(( $(date +%s) - start ))
  rm -f results/.queue_job
  if [ "$rc" -ne 0 ] && tail -n +"$((before + 1))" results/STATUS | grep -q "job selftest: a worker exceeded the wall-time budget" \
      && [ "$took" -le $(( budget + 30 )) ]; then
    status "selftest OK: the stopped worker was killed by its budget after ${took} s and the job reported failed"; exit 0
  fi
  refuse "selftest: the stopped worker was not stopped by its budget (exit $rc after ${took} s)"
fi

status "queue $milestone: POKER_GRID=$POKER_GRID POKER_REGIME=$POKER_REGIME POKER_REGIME_FORCE=$POKER_REGIME_FORCE JAX_PLATFORMS=$JAX_PLATFORMS JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR ($manifest)"
plan_flag=(); [ "$milestone" = check ] || plan_flag=(--plan "$milestone")
if ! "$PY" -m tools.cloud.gate start "${plan_flag[@]}" 2>&1 | tee results/gate_start.log; then
  refuse "start gate: $(tail -1 results/gate_start.log)"
fi
status "start gate OK: $(grep -E '^  (game|code|box|compile cache|kernel smoke|require_cpu negative):' results/gate_start.log | cut -c3-220 | tr '\n' ' ')"
[ "$milestone" = check ] && exit 0
caches_budget=$("$PY" tools/cloud/launch.py budget "$manifest" caches)
if ! timeout --kill-after=60 "$caches_budget" "$PY" -m tools.cloud.gate caches 2>&1 | tee results/gate_caches.log; then
  refuse "caches: $(tail -1 results/gate_caches.log)"
fi
status "caches OK: $(tail -1 results/gate_caches.log | cut -c1-200)"
IFS=, read -r -a gpus <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
if [ "${#gpus[@]}" -gt 1 ] && [ "$milestone" != k0_smoke ] \
    && ! grep -qs "gpus=${#gpus[@]}" results/outbox/DONE.k0_smoke; then
  refuse "${#gpus[@]} GPUs: the multi-GPU path has not passed k0_smoke on this box (run GPUS=${GPUS:-all} queue.sh k0_smoke first)"
fi
teacher=$("$PY" tools/cloud/launch.py get "$manifest" teacher.stop)
case "$milestone" in
  preflight|k0|k0_smoke|river_shard|river_finish)
    if [ -z "${EXPECT_RIVER_SPM:-}" ] && [ "$teacher" != fixed_t ]; then
      card=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
      case "$card" in
        *Laptop*) ;;
        *"RTX 3090"*) EXPECT_RIVER_SPM=150 ;;
        *"RTX 4090"*) EXPECT_RIVER_SPM=200 ;;
      esac
      [ -n "${EXPECT_RIVER_SPM:-}" ] || refuse "EXPECT_RIVER_SPM unknown for this card (${card:-?}): set it"
    fi
    if [ -n "${EXPECT_RIVER_SPM:-}" ]; then export EXPECT_RIVER_SPM
    else status "speed fuse off: its reference is the grid-7 certificate run; EXPECT_RIVER_SPM from the preflight arms it"; fi ;;
esac
code_change=(); [ -n "${ALLOW_CODE_CHANGE:-}" ] && code_change=(--allow-code-change --code-change-reason "$ALLOW_CODE_CHANGE")
# the gates read the utilisation of the loop's jobs (tools/cloud/gate.py, utilisation)
export GPU_UTIL_LOG="results/gpu_util_${milestone}.log" GPU_LIST="${gpus[*]}"
printf '# RUN_START %s id=%s\n' "$(date -u +%FT%TZ)" "$log_segment" >> "$GPU_UTIL_LOG" || refuse "cannot preserve GPU log"
nvidia-smi dmon -s um -o DT -d 10 >> "$GPU_UTIL_LOG" 2>&1 &
dmon=$!

beat() {   # one heartbeat: results/HEARTBEAT (key=value), written atomically; the batches gate of a running loop job
  local name="" started="" job_budget="" state=starting bytes=0 mtime="" last="" f
  [ -f results/.queue_job ] && read -r name started job_budget state < results/.queue_job
  if [ -n "$name" ]; then
    for f in results/"$name".w*.log; do
      [ -e "$f" ] || continue
      bytes=$((bytes + $(stat -c %s "$f")))
      [ -z "$mtime" ] || [ "$(stat -c %Y "$f")" -gt "$mtime" ] && { mtime=$(stat -c %Y "$f"); last=$(tail -1 "$f"); }
    done
    if [ "$state" = running ] && [ -f "results/$name/layout.json" ] && [ ! -f results/.queue_stop ] \
        && ! python3 -m tools.cloud.gate batches "results/$name" > results/gate_batches.log 2>&1; then
      tail -1 results/gate_batches.log > results/.queue_stop
    fi
  fi
  printf 'utc=%s\nepoch=%s\nmilestone=%s\nstate=%s\njob=%s\njob_started_epoch=%s\nbudget_s=%s\nstall_s=%s\nperiod_s=%s\nlog_bytes=%s\nlog_mtime_epoch=%s\nlast_line=%s\n' \
    "$(date -u +%FT%TZ)" "$(date +%s)" "$milestone" "$state" "$name" "$started" "$job_budget" "$stall" "$period" \
    "$bytes" "$mtime" "$(tr -cd '[:print:]' <<< "$last" | cut -c1-300)" > results/HEARTBEAT.tmp
  mv results/HEARTBEAT.tmp results/HEARTBEAT
}
rm -f results/.queue_job
(while :; do beat; sleep "$period"; done) &
heart=$!
trap 'kill "$heart" "$dmon" 2>/dev/null' EXIT

pack() {  # everything except the bulky per-spot training rows and caches (they go in the job archives), with the record
  local paths=() path
  for path in results/* results/.[!.]* results/..?*; do
    [ -e "$path" ] || [ -L "$path" ] || continue
    [ "$path" = results/pack ] || paths+=("$path")
  done
  local extra=(results/pack/PACKED.json results/pack/JOBS.json)
  [ -f results/pack/CHECKS_SUMMARY.json ] && extra+=(results/pack/CHECKS_SUMMARY.json)
  tar czf "results/outbox/results-${milestone}.tar.gz" --exclude='solve_*.npz' --exclude='*[0-9][0-9][0-9][0-9][0-9][0-9].npz' \
    --exclude='results/cache' --exclude='results/jax_cache' --exclude='claims' --exclude='results/outbox' "${paths[@]}" "${extra[@]}" || return $?
  echo "packed results/outbox/results-${milestone}.tar.gz"
}
JOBS_RUN=()
outbox_job() {  # outbox_job NAME: the job directory (rows included, claims aside) as gzip parts of 512 MB
  rm -f "results/outbox/job-$1.tar.gz.part-"*
  if ! tar cf - --exclude=claims -C results "$1" | gzip -1 | split -b 512M -d -a 3 - "results/outbox/job-$1.tar.gz.part-"; then
    status "outbox: job $1 FAILED to pack (disk?)"; return 1
  fi
  status "outbox: job $1 in $(ls "results/outbox/job-$1.tar.gz.part-"* | wc -l) parts, $(du -ch "results/outbox/job-$1.tar.gz.part-"* | tail -1 | cut -f1)"
}

batch_for_gpu() {  # --solve-batch 128 on cards with >= 15 GB, 32 on smaller ones; the batch changes speed only
  local mib            # (fixed T: every spot runs T iterations whatever the batch)
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "${mib%%.*}" -ge 15000 ]; then echo 128; else echo 32; fi
}


stop_box() {
  [ "${AUTO_STOP:-0}" = "1" ] || return 0
  if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID"
  elif command -v vastai >/dev/null 2>&1 && [ -n "${CONTAINER_ID:-}" ]; then vastai stop instance "$CONTAINER_ID"
  else echo "AUTO_STOP: no runpodctl/vastai found, stop the instance manually"; fi
}

run() {  # run NAME GATE CMD...  (each worker gets --out results/NAME --workers N appended)
  local name="$1" gate="$2"; shift 2
  JOBS_RUN+=("$name")
  status "job $name on GPUs ${gpus[*]} (budget ${budget} s): $*  (logs: results/${name}.w*.log)"
  rm -f "results/$name/batch_gate.json"
  job_state "$name" "$budget" running
  if ! workers "$name" "$@"; then
    job_state "$name" "$budget" failed
    status "job $name FAILED: $(tail -q -n 1 "results/${name}".w*.log | tr '\n' ' ' | cut -c1-300)"; tail -5 "results/${name}".w*.log; return 1
  fi
  job_state "$name" "$budget" gating
  if ! "$PY" -m tools.cloud.gate "$gate" "results/${name}" --plan "$milestone" 2>&1 | tee "results/gate_${name}.log"; then
    job_state "$name" "$budget" failed
    status "gate $gate FAILED for $name: $(tail -1 "results/gate_${name}.log")"; return 1
  fi
  status "gate $gate OK for $name: $(tail -1 "results/gate_${name}.log" | cut -c1-200)"
  job_state "$name" "$budget" idle
  pack
}

planout=$("$PY" tools/cloud/launch.py args "$manifest" "$milestone") || refuse "no plan for $milestone in $manifest"
planargs=(); [ -n "$planout" ] && mapfile -t planargs <<< "$planout"
stageout=$("$PY" tools/cloud/launch.py stages "$manifest" "$milestone") || refuse "invalid stages for $milestone in $manifest"
mapfile -t planstages <<< "$stageout"
LOOP=(.venv/bin/python -m poker.solver.loop "${planargs[@]}" ${code_change[@]+"${code_change[@]}"})
case "$milestone" in
  preflight) # production critic shapes (nets 512x3 and the solve batch of the card) on few spots, minutes
    run preflight loop "${LOOP[@]}" --solve-batch "${SOLVE_BATCH:-$(batch_for_gpu)}"
    ;;
  k0_smoke)  # minutes: the whole iteration-0 loop at toy size (checks the box before k0)
    run k0_smoke loop "${LOOP[@]}"
    ;;
  k0)        # iteration 0 HU from scratch, river included (brief M1b 0.4 and the plan before item 7)
    run k0 loop "${LOOP[@]}" --solve-batch "${SOLVE_BATCH:-$(batch_for_gpu)}"
    ;;
  river_shard)   # one box of k: the river first-pass batches with index % k == i (brief M2 9: data for the curve)
    run river_big shard "${LOOP[@]}" --shard "${SHARD:?SHARD=i/k}" --solve-batch "${SOLVE_BATCH:-128}"
    ;;
  river_finish)  # the gathered shards: E flags (fixed T) or second pass, river critic; saved stage reports only
    for f in results/river_big.w*.log; do [ -e "$f" ] && mv "$f" "${f/river_big./river_big_pass1.}"; done
    run river_big loop "${LOOP[@]}" --solve-batch "${SOLVE_BATCH:-128}"
    ;;
esac
code=$?
kill "$dmon" 2>/dev/null
status "queue $milestone finished, exit $code"
job_state "${milestone}" 0 "$([ "$code" -eq 0 ] && echo done || echo failed)"
beat
for name in ${JOBS_RUN[@]+"${JOBS_RUN[@]}"}; do outbox_job "$name" || code=1; done
pack
(cd results/outbox && shopt -s nullglob && files=(results-*.tar.gz job-*.part-*) && sha256sum "${files[@]}" > SHA256SUMS \
  && stat -c '%s %n' "${files[@]}" > SIZES)
if [ "$code" -eq 0 ]; then echo "$(date -u +%FT%TZ) gpus=${#gpus[@]}" > "results/outbox/DONE.$milestone"
else echo "$(date -u +%FT%TZ) $(grep -E 'FAILED|REFUSED|stopped by' results/STATUS | tail -1)" > results/outbox/FAILED; fi
stop_box
exit $code
