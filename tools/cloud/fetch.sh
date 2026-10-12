#!/usr/bin/env bash
# Bring a rented box's outbox home and verify it (task #34; rental-path sniff M5). Never stops or cancels anything.
#   tools/cloud/fetch.sh NAME LOCAL_DIR [REMOTE_OUTBOX]      (REMOTE_OUTBOX: /workspace/poker-huy/results/outbox)
# Fetches SHA256SUMS and SIZES (written by tools/cloud/queue.sh on the box), then every listed file that is not here
# already with its sha256 (~/clore_tools/box.sh NAME get: scp, 3 attempts per file; a dropped transfer repeats one
# 512 MB part, not the run), verifies everything with sha256sum -c, joins the parts of every job archive and lists it
# (tar tzf). Prints the bytes and the rate. Exit 0 only when every file verified; then the box may be released.
# BOX_SH replaces ~/clore_tools/box.sh (a test stub).
set -uo pipefail
box=${1:?box name}; local=${2:?local directory}; remote=${3:-/workspace/poker-huy/results/outbox}
B=${BOX_SH:-$HOME/clore_tools/box.sh}
mkdir -p "$local"
for f in SHA256SUMS SIZES; do
  "$B" "$box" get "$remote/$f" "$local/$f" || { echo "fetch: no $f in $remote on $box"; exit 1; }
done
t0=$(date +%s); fetched=0
while read -r sum name; do
  if [ -f "$local/$name" ] && echo "$sum  $local/$name" | sha256sum -c --quiet - >/dev/null 2>&1; then continue; fi
  ok=0
  for attempt in 1 2 3; do
    if "$B" "$box" get "$remote/$name" "$local/$name" && echo "$sum  $local/$name" | sha256sum -c --quiet - >/dev/null 2>&1; then
      ok=1; fetched=$((fetched + $(stat -c %s "$local/$name"))); break
    fi
    echo "fetch: $name, attempt $attempt failed"; sleep 10
  done
  [ "$ok" = 1 ] || { echo "fetch: $name failed 3 times"; exit 1; }
done < "$local/SHA256SUMS"
took=$(( $(date +%s) - t0 ))
(cd "$local" && sha256sum -c --quiet SHA256SUMS) || { echo "fetch: verification FAILED"; exit 1; }
echo "fetch: $(wc -l < "$local/SHA256SUMS") files verified; $fetched bytes fetched in $took s" \
  "($(( fetched / (took > 0 ? took : 1) / 1000 )) kB/s); total $(awk '{s += $1} END {print s}' "$local/SIZES") bytes"
for job in $(ls "$local" | sed -n 's/^\(job-.*\.tar\.gz\)\.part-[0-9]*$/\1/p' | sort -u); do
  cat "$local/$job".part-* > "$local/$job" && tar tzf "$local/$job" > /dev/null \
    || { echo "fetch: $job does not unpack"; exit 1; }
  echo "fetch: $job joined and listed: $(tar tzf "$local/$job" | wc -l) entries, $(du -h "$local/$job" | cut -f1)"
done
echo "fetch OK: $local"
