#!/usr/bin/env bash
# Watch a rented box from the host (task #34; rental-path sniff M6). Every PERIOD seconds: fetch results/HEARTBEAT and
# judge it with tools/cloud/gate.py heartbeat (stale heartbeat = the runner died or the box is gone; a failure; a
# job past its budget; worker logs silent past the stall limit), count ssh failures (3 in a row is an alert) and look
# for the outbox markers. ALERT lines go to the terminal and to LOG. It never stops or cancels the box: the user does,
# after tools/cloud/fetch.sh verified the outbox. Ends when the box shows DONE or FAILED.
#   tools/cloud/watch.sh NAME [PERIOD]        (from the repo root; PERIOD default 300 s; BOX_SH: a test stub)
set -uo pipefail
box=${1:?box name}; period=${2:-300}
B=${BOX_SH:-$HOME/clore_tools/box.sh}
remote=${REMOTE:-/workspace/poker-huy}
LOG=${LOG:-results/watch_${box}.log}
mkdir -p "$(dirname "$LOG")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
fails=0
while :; do
  if "$B" "$box" get "$remote/results/HEARTBEAT" "$tmp/HEARTBEAT" > /dev/null 2>&1; then
    fails=0
    verdict=$(PYTHONPATH=. python3 -m tools.cloud.gate heartbeat "$tmp/HEARTBEAT" 2>&1); rc=$?
    say "$box: $(tail -1 <<< "$verdict")"
    [ "$rc" -eq 0 ] || say "ALERT $box: $(tail -1 <<< "$verdict") | $(head -1 <<< "$verdict" | cut -c1-300)"
  else
    fails=$((fails + 1))
    say "$box: heartbeat not fetched (ssh failure $fails in a row)"
    [ "$fails" -ge 3 ] && say "ALERT $box: $fails ssh failures in a row: the box or its network is gone"
  fi
  markers=$("$B" "$box" ssh "ls $remote/results/outbox 2>/dev/null | grep -E '^(DONE|FAILED)' | tr '\n' ' '" 2>/dev/null)
  if grep -qE '(^| )(DONE|FAILED)( |$)' <<< "$markers"; then
    say "$box: outbox markers [$markers] - fetch: tools/cloud/fetch.sh $box <local dir>"
    exit 0
  fi
  sleep "$period"
done
