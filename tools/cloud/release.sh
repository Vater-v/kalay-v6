#!/usr/bin/env bash
# Release a completed cloud order after fetching and verifying its outputs. Every 10 min: ssh to the box; once
# REMOTE/DONE exists, download every file listed in REMOTE/SHA256SUMS into LOCAL, verify the hashes,
# cancel the order through the clore API (the key from ~/.profile, never sent anywhere else) and
# confirm the order is gone. REMOTE/FAILED or a hash mismatch: logged, nothing cancelled, exit -
# unless --cancel-on-fail MARKER is given and the local file MARKER exists (the main data already
# downloaded and verified): then the outputs listed in SHA256SUMS are fetched and verified the same
# way and the order is cancelled, so a failed late step does not leave the box running (2026-10-10).
#   bash tools/cloud/release.sh BOX ORDER REMOTE LOCAL [--dry] [--once] [--cancel-on-fail MARKER]
#   --dry: everything but the cancel; --once: one poll (tests).
# Log: $LOG (default results/reports/cloud/release.log). Run under systemd-run -p MemoryMax=200M.
set -uo pipefail
usage() { echo "Usage: $0 BOX ORDER REMOTE LOCAL [--dry] [--once] [--cancel-on-fail MARKER]"; }
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then usage; exit 0; fi
[ "$#" -ge 4 ] || { usage >&2; exit 2; }
box=$1; order=$2; remote=$3; local=$4; shift 4
dry=0; once=0; marker=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry) dry=1 ;;
    --once) once=1 ;;
    --cancel-on-fail) marker=$2; shift ;;
  esac
  shift
done
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOG=${LOG:-"$repo_root/results/reports/cloud/release.log"}
mkdir -p -- "$(dirname -- "$LOG")"
B="$HOME/clore_tools/box.sh"
C="python3 $HOME/clore_tools/clore.py"
set +u; source ~/.profile >/dev/null 2>&1; set -u
[ -n "${CLORE_API_KEY:-}" ] || { echo "$(date -u +%FT%TZ) no CLORE_API_KEY" >> "$LOG"; exit 3; }
log() { echo "$(date -u +%FT%TZ) [$box order $order dry=$dry] $*" >> "$LOG"; }

fetch_verify() {   # download every file of REMOTE/SHA256SUMS into LOCAL and check it; 3 attempts
  mkdir -p "$local"
  for attempt in 1 2 3; do
    $B "$box" get "$remote/SHA256SUMS" "$local/SHA256SUMS" || { log "attempt $attempt: no SHA256SUMS"; sleep 60; continue; }
    t0=$(date +%s)
    for f in $(awk '{print $2}' "$local/SHA256SUMS"); do
      $B "$box" get "$remote/$f" "$local/$f" || log "attempt $attempt: download of $f failed"
    done
    if (cd "$local" && sha256sum -c SHA256SUMS >> "$LOG" 2>&1); then
      log "downloaded and verified in $(( $(date +%s) - t0 )) s: $(cd "$local" && du -h $(awk '{print $2}' SHA256SUMS) | tr '\n' ' ')"
      return 0
    fi
    log "attempt $attempt: hash mismatch"; sleep 60
  done
  log "hashes do not match after 3 attempts: order NOT cancelled"
  return 1
}

cancel_order() {
  if [ "$dry" = 1 ]; then log "dry run: would cancel order $order now"; return 0; fi
  log "cancel: $($C cancel "$order" 2>&1 | tr '\n' ' ')"
  sleep 10
  if $C conn 2>&1 | grep -q "\"id\": $order,"; then
    log "order $order still listed after cancel: check by hand"
    return 5
  fi
  log "order $order cancelled (not in my active orders); balance $($C wallets 2>&1 | grep USD | tr '\n' ' ')"
}

log "start: watching $remote, download to $local${marker:+, cancel on fail if $marker exists}"
start=$(date +%s); fails=0
while :; do
  st=$($B "$box" ssh "test -f $remote/DONE && echo MARK_DONE; test -f $remote/FAILED && echo MARK_FAILED; tail -1 /workspace/STATUS" 2>&1)
  if [ $? -ne 0 ]; then
    fails=$((fails + 1)); log "ssh failed ($fails): $(echo "$st" | tail -1 | cut -c1-200)"
  else
    fails=0
    if echo "$st" | grep -q MARK_FAILED; then
      log "night chain FAILED: $($B "$box" ssh "cat $remote/FAILED; tail -2 /workspace/STATUS" 2>&1 | tr '\n' ' ' | cut -c1-400)"
      if [ -n "$marker" ] && [ -f "$marker" ]; then
        log "main data verified earlier ($marker): fetching the failed step's outputs, then cancelling"
        fetch_verify || exit 4
        cancel_order; exit $?
      fi
      log "nothing downloaded or cancelled"
      exit 2
    fi
    if echo "$st" | grep -q MARK_DONE; then
      fetch_verify || exit 4
      cancel_order; exit $?
    fi
    log "waiting: $(echo "$st" | tail -1 | cut -c1-160)"
  fi
  [ "$once" = 1 ] && exit 0
  [ $(( $(date +%s) - start )) -gt $((20 * 3600)) ] && { log "20 h without DONE: stop watching"; exit 6; }
  sleep 600
done
