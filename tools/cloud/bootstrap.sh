#!/usr/bin/env bash
# One-time setup on a rented Linux box with NVIDIA GPUs (driver for CUDA 12, any distro).
# Installs Python 3.14 + the exact requirements.lock (JAX CUDA wheels bundle the CUDA libs),
# probes every GPU (JAX sees it, fp32 throughput, the TF32 probe of brief M2 step 0: a 1326^2
# float32 matmul at the pipeline's precision within 1e-5 of float64 - tools/cloud/probe_gpu.py LIMIT -, else
# exit 1), then runs the few-minute iteration-0 loop smoke on GPU 0 with GPU utilisation logged per stage. Idempotent.
# Task #34: free disk checked first; uv and the Python patch pinned (static sniff S11: the checks ran on 3.14.4) and
# the installed packages compared with requirements.lock; everything after the install runs the game and devices of
# the job manifest results/pack/JOBS.json (pack.sh; never the repo default grid); the start gate (tools/cloud/gate.py start:
# game line, code fingerprint, clean pack, checks verdict, devices, kernel-load smoke) runs before any GPU work; the
# exact-table caches are built in one process and hash-checked against the tracked references (gate.py caches, static
# sniff S3) before the probes and the smoke use them; the smoke's output passes the loop gate of the k0_smoke plan.
# Usage (repo root on the box): bash tools/cloud/bootstrap.sh   (tools/cloud/box_run.sh runs it under a budget)
set -euo pipefail
cd "$(dirname "$0")/../.."
UV_VERSION=0.12.24          # the uv both k0 boxes installed (2026-10-09)
PYTHON_VERSION=3.14.4       # the interpreter the checks and every local run used
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
need_gb=$(python3 -c "import json; d = json.load(open('results/pack/JOBS.json'))['resources']['disk_fixed_gb']; print(int(sum(d.values())) + 1)")
free_gb=$(( $(df -Pk . | awk 'NR == 2 {print $4}') / 1048576 ))
[ "$free_gb" -ge "$need_gb" ] || { echo "bootstrap FAILED: ${free_gb} GB free, the venv, the caches and a margin need ${need_gb} GB"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
if [ "$(uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
  python3 -m pip install --user --quiet "uv==$UV_VERSION" 2>/dev/null \
    || curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
[ "$(uv --version | awk '{print $2}')" = "$UV_VERSION" ] || { echo "bootstrap FAILED: uv $(uv --version), not $UV_VERSION"; exit 1; }
uv python install "$PYTHON_VERSION"
[ -x .venv/bin/python ] || uv venv --python "$PYTHON_VERSION" .venv
[ "$(.venv/bin/python -c 'import platform; print(platform.python_version())')" = "$PYTHON_VERSION" ] \
  || { echo "bootstrap FAILED: .venv runs Python $(.venv/bin/python --version), not $PYTHON_VERSION"; exit 1; }
uv pip install --python .venv/bin/python --quiet -r requirements.lock
norm() { sed 's/_/-/g' | tr 'A-Z' 'a-z' | sort; }
if ! diff <(norm < requirements.lock) <(uv pip freeze --python .venv/bin/python | norm); then
  echo "bootstrap FAILED: the venv's packages differ from requirements.lock (above)"; exit 1
fi
envline=$(.venv/bin/python tools/cloud/launch.py env results/pack/JOBS.json) \
  || { echo "bootstrap FAILED: no job manifest results/pack/JOBS.json (pack with tools/cloud/pack.sh)"; exit 1; }
eval "$envline"
echo "bootstrap: the manifest's game and devices: POKER_GRID=$POKER_GRID POKER_REGIME=$POKER_REGIME JAX_PLATFORMS=$JAX_PLATFORMS"
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. .venv/bin/python -m tools.cloud.gate start \
  || { echo "bootstrap FAILED: the start gate refuses this box or tree"; exit 1; }
for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. .venv/bin/python -m tools.cloud.probe_gpu
done
mkdir -p results
# the exact tables in one process (rank7, preflop, the 1755 flop classes), hash-checked against the references
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. .venv/bin/python -m tools.cloud.gate caches \
  || { echo "bootstrap FAILED: the exact-table caches"; exit 1; }
# the end-of-street targets on the GPU against the CPU (brief M2 step 1, A7)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. .venv/bin/python -m tools.cloud.probe_aux
rm -rf results/bootstrap_k0
nvidia-smi dmon -i 0 -s um -o DT -d 2 > results/gpu_util_bootstrap.log 2>&1 &
dmon=$!
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  .venv/bin/python -m poker.solver.loop --quick --workers 1 --out results/bootstrap_k0 > results/bootstrap_k0.log 2>&1 \
  || { kill "$dmon"; tail -30 results/bootstrap_k0.log; echo "bootstrap FAILED: quick loop"; exit 1; }
kill "$dmon"
tail -5 results/bootstrap_k0.log
PYTHONPATH=. .venv/bin/python -m tools.cloud.gate loop results/bootstrap_k0 --plan k0_smoke \
  || { echo "bootstrap FAILED: the gate of the quick loop"; exit 1; }
echo "GPU utilisation of the quick loop by stage (compile-bound at toy size: a smoke; the 60% gate is the loop's):"
.venv/bin/python -m tools.cloud.gpu_util results/gpu_util_bootstrap.log results/bootstrap_k0.log
echo "bootstrap OK"
