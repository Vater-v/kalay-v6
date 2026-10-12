#!/bin/sh
# Runs every experiment in order (E1 must run first: it writes the reference
# equilibrium used by E2-E5).  Total wall-clock ~25-30 min on 4 cores.
set -e
cd "$(dirname "$0")"
python3 experiments/e1_exact.py
python3 experiments/e2a_turn_values.py
python3 experiments/e2b_flop_values.py
python3 experiments/e2c_agents.py
python3 experiments/e2d_boundary.py
python3 experiments/e3_offtree.py
python3 experiments/e4_exploit.py
python3 experiments/e5_timing.py
python3 experiments/label_noise.py
python3 experiments/make_tables.py > results/tables.md
