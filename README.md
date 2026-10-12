# kalay-v6

Material for building a poker agent from zero under hard constraints (2015 PC, $10–15 of rented compute for heads-up and a $120 ceiling for everything up to 9-max, no blueprint, no card buckets, no snapping of opponent bet sizes, one mechanism for robustness and exploitation).

Two stages so far:

1. **The structural principle and its toy evidence** (`docs/poker-core-mechanism.md`, `toy/`).
2. **The audit of the existing engine and its fixes** (`docs/engine-audit.md`, `poker/`, `tests/`, `results/audit/`, `tools/audit/`).

## Stage 2: the engine audit

- **`docs/engine-audit.md`** — the report: what the engine (`poker/`, JAX, heads-up no-limit hold'em with learned street-boundary values) does right, what was wrong, what was fixed, measured costs, the keep / fix / reuse / missing map, the budget verdicts for heads-up, 6-max and 9-max, and the open questions only the user's hardware can answer.
- **`poker/`** — the user's engine, with this stage's fixes: the safe re-solving gadget (per-hand opponent ceilings from the post-card critics, Terminate/Follow root, certified gap of the gadget game) in `poker/solver/pcfr.py`, `subgame.py` and `poker/player/live.py`; a Slumbot runner that survives a decision failure and refuses the CPU unless asked; loud illegal actions under `checkify`; the rake-split overflow and history-capacity guards; the fold-terminal shortcut; `poker/solver/validate_v4.py` (exact end-of-turn labels).
- **`tests/`** — the new tests and the gadget measurement script (`tests/README.md` has the commands; all four tests pass on a CPU box).
- **`results/audit/`** — the auditors' raw results per area, the gadget measurement (`gadget/`), the exact end-of-turn labels (`v4/`) and the CPU profile (`engine-profile/`); **`tools/audit/`** — the scripts behind them.

## Stage 1: the principle and the toys

- **`docs/poker-core-mechanism.md`** — the document. The structural principle, the minimal formulation, how hidden information / reactions / future are handled jointly, a worked decision, cost and feasibility (HU and 9-max), failure modes and the exact places where the requirements contradict each other, the evidence from the toy games, what was examined and rejected, and the sprint ladder.
- **`toy/hu/`** — heads-up "mini hold'em" reference implementation (numpy only): public tree × private-hand fiber, CFR⁺, exact best-response exploitability, safe continual re-solving with the gadget, oracle / check-down / learned street-boundary leaves, off-menu bet insertion vs snapping, restricted Nash response. `README.md` there has the game spec and every measured number; `run_all.sh` reproduces them; `tests.py` checks the vectorised EV against brute force.
- **`toy/multi/`** — three-player "Tri-Leduc" (numpy only): exact multiway card removal (brute force, pairwise, Möbius) vs independence, CFR⁺ convergence and per-player regret at 3 players, coalitions and online learning, vectorised CFR⁺ vs MCCFR, cost scaling to 9 seats. `README.md` has the numbers.
- **`toy/TARGETED.md`** — targeted measurements for the claims the adversarial review left open: pre-chance gadget ceilings, single-card labels, what can be certified at 3 players, folded players' dead cards, free exploitation on the equilibrium face.
- **`toy/hu/results/X1_NOISE_CURVE.md`** — stage 2's toy experiment: controlled value error at the street boundary against exact exploitability (the curve the engine audit uses to translate V error into cost per hand).

Every number quoted in the document is either derived there, measured by these scripts (exact best responses, not estimates), or checked by an adversarial review; where a claim did not survive review, the corrected statement is what appears.

## Running the toys

```sh
export OPENBLAS_NUM_THREADS=1
cd toy/hu    && python3 tests.py && ./run_all.sh      # ~30 min on 4 cores
cd toy/multi && python3 tests.py && python3 experiments/m1_card_removal.py   # see README for the rest
```

Python 3.11+ and numpy are the only requirements. Large binary results (`*.npy`, `*.npz`) are not committed; the JSON and log files under `toy/*/results/` are.
