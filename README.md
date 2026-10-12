# kalay-v6

Foundation material for building a poker agent from zero under hard constraints (2015 PC, $10–15 of rented compute, no blueprint, no card buckets, no snapping of opponent bet sizes, one mechanism for robustness and exploitation).

## Contents

- **`docs/poker-core-mechanism.md`** — the document. The structural principle, the minimal formulation, how hidden information / reactions / future are handled jointly, a worked decision, cost and feasibility (HU and 9-max), failure modes and the exact places where the requirements contradict each other, the evidence from the toy games, what was examined and rejected, and the sprint ladder.
- **`toy/hu/`** — heads-up "mini hold'em" reference implementation (numpy only): public tree × private-hand fiber, CFR⁺, exact best-response exploitability, safe continual re-solving with the gadget, oracle / check-down / learned street-boundary leaves, off-menu bet insertion vs snapping, restricted Nash response. `README.md` there has the game spec and every measured number; `run_all.sh` reproduces them; `tests.py` checks the vectorised EV against brute force.
- **`toy/multi/`** — three-player "Tri-Leduc" (numpy only): exact multiway card removal (brute force, pairwise, Möbius) vs independence, CFR⁺ convergence and per-player regret at 3 players, coalitions and online learning, vectorised CFR⁺ vs MCCFR, cost scaling to 9 seats. `README.md` has the numbers.
- **`toy/TARGETED.md`** — targeted measurements for the claims the adversarial review left open: pre-chance gadget ceilings, single-card labels, what can be certified at 3 players, folded players' dead cards, free exploitation on the equilibrium face.

Every number quoted in the document is either derived there, measured by these scripts (exact best responses, not estimates), or checked by an adversarial review; where a claim did not survive review, the corrected statement is what appears.

## Running the toys

```sh
export OPENBLAS_NUM_THREADS=1
cd toy/hu    && python3 tests.py && ./run_all.sh      # ~30 min on 4 cores
cd toy/multi && python3 tests.py && python3 experiments/m1_card_removal.py   # see README for the rest
```

Python 3.11+ and numpy are the only requirements. Large binary results (`*.npy`, `*.npz`) are not committed; the JSON and log files under `toy/*/results/` are.
