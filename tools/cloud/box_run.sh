#!/usr/bin/env bash
# The box runner (task #34): one command on a rented box, in tmux, from the unpacked tarball of tools/cloud/pack.sh.
#   bash tools/cloud/box_run.sh preflight k0_smoke k0
# 1. the upload: this tree's code against results/pack/PACKED.json (poker.envs.fingerprint, standard library only: before anything
#    is installed, so a broken or edited upload costs seconds, not a bootstrap);
# 2. queue.sh selftest: the hang machinery on this box (a self-stopped worker must die at its budget);
# 3. tools/cloud/bootstrap.sh under the manifest's bootstrap budget: disk, pinned uv and Python, the venv against the
#    lock, the start gate (game line, code fingerprint, clean pack, checks verdict, devices, kernel-load smoke), the GPU
#    probes, the exact-table caches in one process with their hashes, the quick loop of the manifest's game and its gate;
# 4. tools/cloud/queue.sh for every milestone in order (start gate with the milestone's resources, caches, jobs
#    under their budgets with heartbeat and batches gate, job gates, results/outbox/).
# Stops at the first failure. Progress lines: results/STATUS; liveness: results/HEARTBEAT; the pull set:
# results/outbox/ - SHA256SUMS, SIZES, results-<milestone>.tar.gz, job-<job>.tar.gz.part-NNN, DONE after the last
# milestone or FAILED with the reason. Nothing here stops, cancels or releases the box: the user does, after fetching.
#
# The whole rental, in order (host = WSL on the user's machine, NAME = a box of ~/clore_tools/box.sh; rental-path sniff
# M13: these are the only commands, every path and flag exists in this repo or in ~/clore_tools):
#  host 1  clean checkout of the commit:  git -C /home/vater/poker-huy worktree add /tmp/pack-SHA SHA
#  host 2  the checks on it:  cd /tmp/pack-SHA && PYTHONPATH=. /home/vater/poker-huy/.venv/bin/python -m checks.run \
#            --suite required --device cuda --output-dir results/reports/checks/required-SHA      (needs the honest gate, #23)
#  host 3  the pack:  PY=/home/vater/poker-huy/.venv/bin/python tools/cloud/pack.sh --regime 1 \
#            --checks results/reports/checks/required-SHA   (exit 0 = launchable; 3 = packed but refused, the reasons printed)
#  host 4  the user's yes to a sum, then the order: python3 ~/clore_tools/clore.py create SERVER KEYFILE;
#            ~/clore_tools/box.sh NAME set HOST PORT
#  host 5  upload and unpack:  ~/clore_tools/box.sh NAME put results/pack/TARBALL /workspace/
#            ~/clore_tools/box.sh NAME ssh 'mkdir -p /workspace/poker-huy && tar xzf /workspace/TARBALL -C /workspace/poker-huy'
#  host 6  start:  ~/clore_tools/box.sh NAME ssh 'cd /workspace/poker-huy && tmux new -d -s q "bash tools/cloud/box_run.sh preflight k0_smoke k0"'
#  host 7  watch (alerts, never cancels):  tools/cloud/watch.sh NAME
#  host 8  fetch and verify:  tools/cloud/fetch.sh NAME results/rental_DATE_NAME
#  host 9  the user releases the box after step 8 verified:  python3 ~/clore_tools/clore.py cancel ORDER
set -uo pipefail
cd "$(dirname "$0")/../.."
[ $# -gt 0 ] || { echo "usage: bash tools/cloud/box_run.sh MILESTONE [MILESTONE ...]"; exit 2; }
mkdir -p results/outbox
rm -f results/outbox/DONE results/outbox/FAILED
status() { echo "$(date -u +%FT%TZ) $*" | tee -a results/STATUS; }
failed() { status "FAIL $*"; echo "$(date -u +%FT%TZ) $*" > results/outbox/FAILED; exit 1; }
status "box_run $*: package $(python3 tools/cloud/launch.py get results/pack/PACKED.json commit 2>/dev/null | cut -c1-12) packed \
$(python3 tools/cloud/launch.py get results/pack/PACKED.json packed_utc 2>/dev/null), game \
$(python3 tools/cloud/launch.py get results/pack/PACKED.json game_hash 2>/dev/null | cut -c1-12)"
python3 -m poker.envs.fingerprint --check results/pack/PACKED.json > results/fingerprint_upload.log 2>&1 \
  || failed "upload: $(tail -1 results/fingerprint_upload.log)"
status "upload OK: $(head -1 results/fingerprint_upload.log)"
bash tools/cloud/queue.sh selftest > results/queue_selftest.log 2>&1 || failed "selftest: $(tail -1 results/queue_selftest.log)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit --format=csv > results/gpus.csv 2>&1
budget=$(python3 tools/cloud/launch.py budget results/pack/JOBS.json bootstrap) || failed "no bootstrap budget in results/pack/JOBS.json"
timeout --kill-after=120 "$budget" bash tools/cloud/bootstrap.sh > results/bootstrap.log 2>&1 \
  || failed "bootstrap (exit $?; 124 = its budget of $budget s): $(tail -2 results/bootstrap.log | tr '\n' ' ' | cut -c1-400)"
status "bootstrap OK: $(grep -E '^(PROBE|AUX|GATE)' results/bootstrap.log | tr '\n' ' ' | cut -c1-600)"
for m in "$@"; do
  bash tools/cloud/queue.sh "$m" > "results/queue_${m}.log" 2>&1 \
    || failed "queue $m (exit $?): $(tail -3 "results/queue_${m}.log" | tr '\n' ' ' | cut -c1-400)"
  status "queue $m OK"
done
status "DONE $*"
date -u +%FT%TZ > results/outbox/DONE
