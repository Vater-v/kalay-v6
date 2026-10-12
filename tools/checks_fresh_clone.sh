#!/usr/bin/env bash
# What a fresh clone can verify (task #23): clone the committed HEAD with --depth 1 into a temporary directory and
# run the CPU part of the rental gate there: python -m checks.run --suite required --device cpu. Nothing of the
# working tree reaches the clone - no uncommitted change, no untracked file, no results/cache (rank7 table, preflop
# equity), no trained live artifacts: the parts that need them record SKIPPED with the reason, never PASS; CUDA-only modules are
# SKIPPED "requires cuda"; modules of the gate that are not in the tree are SKIPPED "missing from this tree". Each
# module's scope records code_root, the clone's path: the run tested the clone's code. The summary goes to OUT
# (default results/reports/checks/fresh_clone_<UTC stamp> of this checkout, outside the clone); the clone is deleted at the
# end. Exit code = the runner's (0 PASS, 1 FAIL, 3 INCOMPLETE).
#
#   tools/checks_fresh_clone.sh [OUT [checks.run arguments, e.g. --timeout 3600]]
#   PYTHON=/path/to/python tools/checks_fresh_clone.sh      (default: this checkout's .venv, else python3)
set -euo pipefail
repo=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
out=${1:-$repo/results/reports/checks/fresh_clone_$(date -u +%Y%m%dT%H%M%SZ)}
if [ $# -gt 0 ]; then shift; fi
python=${PYTHON:-$repo/.venv/bin/python}
[ -x "$python" ] || python=python3
head=$(git -C "$repo" rev-parse HEAD)
branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD || true)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ -n "$branch" ]; then
    git clone --quiet --depth 1 --branch "$branch" "file://$repo" "$tmp/clone"
else                                    # detached HEAD: fetch the commit itself
    git init --quiet "$tmp/clone"
    git -C "$tmp/clone" fetch --quiet --depth 1 "file://$repo" "$head"
    git -C "$tmp/clone" checkout --quiet FETCH_HEAD
fi
[ "$(git -C "$tmp/clone" rev-parse HEAD)" = "$head" ] || { echo "the clone is not HEAD $head" >&2; exit 2; }
mkdir -p "$out"
{
    echo "fresh clone: $head (branch ${branch:-detached}), git clone --depth 1 of $repo into $tmp/clone"
    echo "not in the clone: $(git -C "$repo" status --short | wc -l) uncommitted or untracked entries of the source"
    echo "results/ in the clone: $(git -C "$tmp/clone" ls-files results | wc -l) tracked files;" \
         "results/cache present: $([ -e "$tmp/clone/results/cache" ] && echo yes || echo no)"
    echo "python: $python; load: $(cat /proc/loadavg)"
} | tee "$out/fresh_clone.txt"
cd "$tmp/clone"
set +e
"$python" -m checks.run --suite required --device cpu --output-dir "$out" "$@"
rc=$?
set -e
echo "fresh-clone gate: exit $rc (0 PASS, 1 FAIL, 3 INCOMPLETE); load $(cat /proc/loadavg); summary $out/summary.json" \
    | tee -a "$out/fresh_clone.txt"
exit $rc
