#!/usr/bin/env bash
# Pack a clean commit for a rented GPU box (task #34). Packs the tracked files without results/ (evidence), results/reports/
# and .archive/, the explicit allow-list INPUTS of tools/cloud/launch.py (empty: the box rebuilds its caches), and
# results/pack/: JOBS.json (the job manifest: every job's game, devices, plan, budgets and resource needs from poker/envs/config.py,
# launch.py manifest), PACKED.json (commit read once, tree state, code fingerprint, game hash, launch.py record) and,
# with --checks, the summary of a full `checks.run --suite required` run of this commit (results/pack/CHECKS_SUMMARY.json).
# A dirty tree (any `git status --porcelain` line) is refused (exit 1); --allow-dirty packs it with a banner, the
# dirty list in PACKED.json and "-dirty" in the tarball's name, and the gate refuses such a pack (a rehearsal, never a
# rental). --platforms cpu writes a CPU rehearsal manifest ("-rehearsal" in the name; the gate refuses it as a rental).
# Ends with the launch gate on the tarball itself: unpacked into a temporary directory, the fingerprint against
# PACKED.json and tools/cloud/gate.py start --host there (everything but the box's devices and resources).
# Exit 0: packed and launchable; 3: packed, the gate refuses (the box would refuse to start); 1: dirty, nothing packed;
# 2: another error, nothing packed.
# Usage (WSL, the root of a clean checkout; PY = a Python with the project's packages, for config; the regime has no
# default, task #39):
#   PY=/home/vater/poker-huy/.venv/bin/python tools/cloud/pack.sh --regime 1 [--checks results/reports/checks/<run>] \
#       [--budget k0=43200 ...] [--platforms cpu] [--allow-dirty]       ->  results/pack/poker-huy-<commit>-<utc>.tar.gz
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { echo "pack: no Python at PY=$PY (one with requirements.lock: PY=/home/vater/poker-huy/.venv/bin/python)"; exit 2; }
PY=$(realpath -s "$PY")                       # absolute, symlinks kept: a venv's python is a symlink
allow_dirty=0; checks=(); manifest_args=(); rehearsal=""; regime=""
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-dirty) allow_dirty=1 ;;
    --checks) checks=(--checks "$(realpath "$2")"); shift ;;
    --regime) regime=$2; manifest_args+=("$1" "$2"); shift ;;
    --budget) manifest_args+=("$1" "$2"); shift ;;
    --platforms) manifest_args+=("$1" "$2"); [ "$2" = cuda,cpu ] || rehearsal="-rehearsal"; shift ;;
    *) echo "pack: unknown argument $1"; exit 2 ;;
  esac
  shift
done
[ -n "$regime" ] || { echo "pack: --regime N is required (poker/envs/config.py REGIMES: the regime has no default, task #39)"; exit 2; }
commit=$(git rev-parse HEAD)                  # read once: launch.py record refuses when HEAD or the tree moved since
dirty=$(git status --porcelain)
suffix=""
if [ -n "$dirty" ]; then
  if [ "$allow_dirty" != 1 ]; then
    echo "pack: REFUSED - the tree is dirty ($(wc -l <<< "$dirty") lines of git status --porcelain):"
    head -20 <<< "$dirty"
    echo "pack: commit the change (or pack another checkout); --allow-dirty makes a rehearsal tarball the gate refuses"
    exit 1
  fi
  suffix="-dirty"
  echo "################################################################################"
  echo "## pack: --allow-dirty: $(wc -l <<< "$dirty") dirty paths go into a REHEARSAL tarball"
  echo "## the gate (tools/cloud/gate.py) refuses this pack: it never starts a rental"
  echo "################################################################################"
  head -20 <<< "$dirty"
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)            # seconds: two packs in one minute never share a name
out="results/pack/poker-huy-${commit:0:7}-${stamp}${suffix}${rehearsal}.tar.gz"
mkdir -p results/pack
rm -f results/pack/PACKED.json results/pack/JOBS.json results/pack/CHECKS_SUMMARY.json results/pack/.files0
"$PY" tools/cloud/launch.py manifest --out results/pack/JOBS.json ${manifest_args[@]+"${manifest_args[@]}"}
dirty_flag=(); [ "$allow_dirty" = 1 ] && dirty_flag=(--allow-dirty)
"$PY" tools/cloud/launch.py record --out results/pack/PACKED.json --commit "$commit" --tarball "$(basename "$out")" \
  --stamp "$stamp" ${checks[@]+"${checks[@]}"} ${dirty_flag[@]+"${dirty_flag[@]}"}
extra=(results/pack/PACKED.json results/pack/JOBS.json); [ -f results/pack/CHECKS_SUMMARY.json ] && extra+=(results/pack/CHECKS_SUMMARY.json)
tar czf "$out" --null -T results/pack/.files0 "${extra[@]}"
echo "$out  $(du -h "$out" | cut -f1), $(tr -cd '\0' < results/pack/.files0 | wc -c) files + ${extra[*]}; sha256 $(sha256sum "$out" | cut -c1-16)"
# the launch gate: what the box gets, checked as the box checks it (without the box's devices and resources)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tar xzf "$out" -C "$tmp"
gate=0
(cd "$tmp" && python3 -m poker.envs.fingerprint --check results/pack/PACKED.json) || gate=1
(cd "$tmp" && envline=$(python3 tools/cloud/launch.py env results/pack/JOBS.json) && eval "$envline" && JAX_PLATFORMS=cpu \
  PYTHONPATH=. "$PY" -m tools.cloud.gate start --host) || gate=1
if [ "$gate" != 0 ]; then
  echo "pack: $out is packed but NOT launchable (the box's start checks refuse it, reasons above)"
  exit 3
fi
echo "launchable. next (tools/cloud/box_run.sh header, host steps 5-6):"
echo "  ~/clore_tools/box.sh NAME put $out /workspace/"
echo "  ~/clore_tools/box.sh NAME ssh 'mkdir -p /workspace/poker-huy && tar xzf /workspace/$(basename "$out") -C /workspace/poker-huy'"
echo "  ~/clore_tools/box.sh NAME ssh 'cd /workspace/poker-huy && tmux new -d -s q \"bash tools/cloud/box_run.sh preflight k0_smoke k0\"'"
