# HU mini hold'em: measuring the "public tree + fiber + CFR+ + re-solving" pattern

A small poker-like game, solved exactly, used to measure whether a set of
design principles for a from-scratch poker AI holds up. Everything is numpy-only
Python with no scipy or torch, and every number below was printed by the scripts
in `experiments/` (raw JSON and logs are in `results/`).

## Summary of the results

* **Exact solving works and is fast.** A full public-tree CFR+ solve takes
  0.067 s per iteration. It reaches **<1% of the pot-ante in 300 iterations
  (20 s)** and 0.020% in 2000 iterations (134 s). That is on one core, with
  25 MB of solver arrays.
* **CFR+ beats vanilla CFR by a wide margin at equal time.** At 134 s CFR+ is at
  0.41 mA/h and vanilla CFR is at 17.4 mA/h, about 42 times worse.
* **Continual re-solving is sound only in the safe version.**
  * With exact leaf values, safe re-solving (the DeepStack gadget) scores
    23.7 mA/h (1.18% pot). A full solve at the same 300 iterations scores 10.5.
  * Naive re-solving, which carries only ranges forward, scores **520 mA/h**
    with the same exact leaves. That is 22 times worse.
* **The learned street-boundary value function did not deliver in this budget.**
  * The learned-leaf agents scored 417–521 mA/h (21–26% pot; up to 550 in a
    replicate run). The checkdown leaf, which has no strategic content, scored
    533 mA/h. Exact leaves scored 24 mA/h.
  * Going from 200 to 20,000 oracle labels moved the agent only within seed
    noise.
  * The residual-over-checkdown trick lowered the regression error (0.230 vs
    0.315 at N = 20,000, range-weighted, pot-share units). It made no
    measurable difference to exploitability.
  * Weighting the loss by the range helped most. It still left the agent about
    18 times worse than exact leaves.
* **Re-solving with the observed size added beats snapping by 20–36×.**
  * Snapping an off-menu size to a menu size lost 51–389 mA/h against an
    opponent who exploits it. The re-solve lost 1.4–18.8 mA/h.
  * Which neighbour to snap to matters a lot, and no rule is safe. For the
    0.75-pot size, snapping down costs 227 and snapping up costs 179. For 1.5
    pot, down costs 51 and up costs 236.
* **Robustness and exploitation come from one solver change.**
  * Restricted Nash response changes 3 lines of the solver.
  * At p = 0.25 it captures 88% (station) and 72% (maniac) of the EV gain from
    Nash to best response, with 154 and 47 mA/h of exploitability. The pure
    best responses have 1570 and 1498 mA/h.
  * Against the nit the frontier is much steeper: p = 0.25 captures only 33%
    of that gain.

## 1. Game specification

| item | value |
|---|---|
| deck | 5 ranks × 2 suits = 10 cards, shared (card removal matters) |
| private cards | 1 per player (the fiber dimension is 10 hands) |
| antes / stacks | each player antes 1. Stacks are 12 chips including the ante, so 11 behind and 5.5 initial pots deep |
| streets | preflop (no board), flop (1 public card), turn (1 public card). Player 0 acts first on every street |
| bet menu | opening bet {½ pot, pot, all-in}. First raise {pot (raise to call + pot after calling), all-in}. Second raise {all-in}. No third raise (raise cap = 2). A size that reaches the stack becomes the all-in. |
| actions | fold `f`, check `k`, call `c`, bet `b0.5`/`b1`, raise `r1`, all-in `A` |
| showdown | strength = rank + 5·(number of board cards of the same rank): trips > pair > high card, higher rank wins, kicker irrelevant, ties split |
| all-in before the turn | the remaining board is averaged exactly inside the showdown matrix `W[board]` |

**Size.** The game has 54,957 public nodes:

* 21,380 decision nodes;
* 339 chance nodes;
* 15,258 fold terminals;
* 17,980 showdown terminals.

The fiber has 10 hands. There are 171,920 information sets (decision node ×
hand compatible with the board) and 13 tree levels. The game value for player 0
is −0.11411 antes/hand: acting first on every street costs about 0.114 antes.

**Units.**
* "mA/h" means milli-antes per hand: the ante plays the role of the big blind
  in mbb/hand.
* "% pot" is the exploitability divided by the 2-ante initial pot.
* **Exploitability** is `(BR0 + BR1)/2`, where `BRi = max_σ u_i(σ, σ_-i)` is
  computed by an exact best-response pass on the full public tree. It equals
  the seat-averaged "best-response value minus game value".
* Seat-wise numbers are `BR_opp − v*_opp`.

## 2. The pattern, as implemented (code map)

| file | what it does (math in the docstrings) |
|---|---|
| `game.py` | Rules, bet menu, public states (`PState`, never containing private cards). `Boards` precomputes for every board `W[b][h,h'] = E_runout sign(str(h) − str(h'))`, which is zero for impossible pairs, antisymmetric, and exact over the remaining cards. **All card removal lives here and in the terminal formulas.** |
| `tree.py` | Builds the **public tree, or a forest of public subtrees**, as flat breadth-first arrays. Each depth is a contiguous block, and the children of each node are contiguous. Node kinds are DEC, CHANCE, FOLD, SHOW, and LEAF (the depth-limit boundary). Strategies live on the child edge. The chance probability that is common to all compatible hand pairs is a scalar `cscale`. The card-removal part of chance is a 0/1 mask on chance edges. |
| `cfr.py` | Vectorised CFR / CFR+ over `[node, hand]` arrays: a forward pass for reach and a backward pass for counterfactual values (CFV), each one numpy op per level, about 13 levels. Regret matching+, linear averaging, alternating updates. Exact best response, expected values, and pure best-response extraction. Two hooks: `gadget` (safe re-solving) and `rnr` (restricted Nash response). |
| `valuefn.py` | Street-boundary value functions in normalised form `u_i = V(board, x, p0, p1)`, with absolute `CFV_i = cscale·x·M_opp·u_i`. Includes `CheckdownV`, oracle labelling by exact sub-solves, a tiny MLP with manual backprop and a zero-sum output projection, and `LeafEvaluator`, which turns V into leaf CFVs. |
| `resolve.py` | The continual re-solving agent and the **extraction of its complete strategy**: all re-solves at the same (street, k-th own decision) are independent and are batched into one forest solve. |
| `opponents.py` | Fixed opponent models (station, maniac, nit) as strategy arrays. |
| `tests.py` | Checks that W is antisymmetric, and that the **vectorised EV of a random profile equals a brute-force walk over all 360 deals with real hidden cards** (agreement to 1e-12, checked on a 6-card variant). |
| `experiments/` | `e1_exact`, `e2a_turn_values`, `e2b_flop_values`, `e2c_agents`, `e2d_boundary`, `e3_offtree`, `e4_exploit`, `e5_timing`, `label_noise`, `make_tables` |

**No hidden state is ever branched on.**
* Reach `π_p[n,h]`, strategy `σ[n.a,h]` and `v_i[n,h]` are arrays of shape
  `[public node, hand]`.
* The only place where two hands meet is a terminal:
  * fold: `v_i(h) = cscale·u_i·(Σ q − q[h])·mask[h]`;
  * showdown: `v_i = cscale·x·(q @ W[board]ᵀ)`.
* `tests.py` confirms this formulation is exact.

**Continual re-solving procedure (`resolve.py`).** At each of its decisions the
agent solves only the current street. The root is the street root for its first
decision on the street, otherwise its own decision node. The inputs are:
* our exact range `π_a(R)`;
* either
  * **naive**: an opponent range carried from the previous re-solve, or
  * **safe**: the opponent's CFVs `w(R,h')` from the previous re-solve,
    enforced with the DeepStack gadget (the opponent may "terminate" with
    `w(h')` per hand);
* leaf values at the next-street boundary from V.

After each re-solve the agent stores the opponent's CFVs and range at its own
next decision nodes and at every next-street root (through V). The **ORACLE
leaf** keeps the remaining game below the leaf and solves it jointly in the same
CFR loop. In the limit this equals calling an exact solver at every leaf on
every iteration, and it is what V tries to approximate.

## 3. How to run

```sh
cd toy/hu
python3 tests.py        # correctness checks (2 s)
./run_all.sh            # all experiments, ~28 min wall-clock on 4 cores; writes results/*.json, results/tables.md
python3 experiments/make_tables.py   # re-render the tables below from results/*.json
```

Run E1 first: it writes the reference equilibrium (`results/blueprint_eff*.npy`)
that E2–E5 use. E2a → E2b → E2c/E2d must run in that order (value nets are
pickled in between). Knobs are environment variables: `E1_ITERS`, `E2_K`,
`E2_LABEL_ITERS`, `E2_NS`, `E2_FLOP_CAP`, `E2_STEPS`, `E3_SIZES`, `E3_K`,
`E4_K`.

Measured wall-clock per experiment:

| E1 | E2a | E2b | E2c | E2d | E3 | E4 | E5 | label noise | total |
|---|---|---|---|---|---|---|---|---|---|
| 273 s | 116 s | 422 s | 222 s | 113 s | 130 s | 268 s | 64 s | 62 s | 1,670 s ≈ 28 min |

E2–E4 use a 4-process pool with one BLAS thread each. E1 and E5 are
single-process. **Caveat:** another user's CPU-bound job ran on the same 4-core
machine during part of these runs, so wall-clock numbers are somewhat inflated.
That job may also have been active on one core during E1.

## 4. Results

### E1 Exact solve: CFR+ vs vanilla CFR, equal wall-clock

| iter | CFR+ s | CFR+ expl mA/h | CFR+ % pot | CFR s | CFR expl mA/h | CFR % pot |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 2329.49 | 116.474 | 0.1 | 2329.49 | 116.474 |
| 2 | 0.2 | 2351.12 | 117.556 | 0.1 | 2286.67 | 114.333 |
| 5 | 0.4 | 1653.58 | 82.679 | 0.3 | 1879.34 | 93.967 |
| 10 | 0.7 | 808.67 | 40.433 | 0.6 | 1168.61 | 58.430 |
| 20 | 1.4 | 400.97 | 20.049 | 1.3 | 683.92 | 34.196 |
| 50 | 3.3 | 135.37 | 6.769 | 3.3 | 371.85 | 18.592 |
| 100 | 6.4 | 56.31 | 2.815 | 6.6 | 222.91 | 11.146 |
| 200 | 13.1 | 20.41 | 1.021 | 13.1 | 127.20 | 6.360 |
| 300 | 19.6 | 10.52 | 0.526 | 19.7 | 90.42 | 4.521 |
| 500 | 32.8 | 4.64 | 0.232 | 33.5 | 58.40 | 2.920 |
| 1000 | 67.1 | 1.42 | 0.071 | 66.9 | 31.62 | 1.581 |
| 1500 | 99.8 | 0.69 | 0.034 | 101.0 | 22.36 | 1.118 |
| 2000 | 134.0 | 0.41 | 0.020 | 135.6 | 17.36 | 0.868 |

* **Speed.** CFR+ takes 0.0670 s/iteration and vanilla CFR 0.0678 s/iteration.
* **Memory.** The solver arrays total 24.7 MB and peak RSS is 198 MB.
* **Time to target.**
  * Below 1% pot: CFR+ needs 300 iterations (19.6 s); vanilla needs 2000
    (135.6 s).
  * Below 0.1% pot: CFR+ needs 1000 iterations (67 s); vanilla never gets there
    in the budget.
* **Convergence rate.** Between 100 and 2000 iterations, CFR+ exploitability
  falls roughly as T^-1.65. Vanilla falls roughly as T^-0.85.

### E2 Continual re-solving with a street-boundary value function

**E2a: the turn-boundary value function V_turn.**
* Oracle labels: 22,000 random situations (random 2-card board, a per-player
  pot share x taken from the reachable set {1, 2, 3, 4, 6, 9}, two random
  ranges), each solved exactly with 300 CFR+ iterations.
* Label cost: 11.2 CPU-ms per situation. Mean label exploitability is
  0.00063 pot-shares.
* Model: MLP 51→128→128→20 (31 inputs for the raw variant) with a zero-sum
  projection, 6000 Adam steps.
* Test sets: 1000 random held-out situations, plus 2211 on-policy turn roots
  labelled with the exact equilibrium (E1) values.
* Errors are in units of x, each player's pot share. "range-w" weights the
  squared error of `u_i(h)` by `p_i(h)`; that is what matters for value.

| V_turn | N | residual | range-w loss | train s | test RMSE | test RMSE range-w | on-policy RMSE | on-policy RMSE range-w |
|---|---|---|---|---|---|---|---|---|
| 200 | 200 | True | False | 10.0 | 1.004 | 0.466 | 0.471 | 0.368 |
| 2000 | 2000 | True | False | 14.3 | 0.824 | 0.348 | 0.367 | 0.291 |
| 20000 | 20000 | True | False | 14.2 | 0.581 | 0.230 | 0.267 | 0.227 |
| 20000_raw | 20000 | False | False | 13.3 | 0.661 | 0.315 | 0.334 | 0.303 |
| 20000_rw | 20000 | True | True | 14.2 | 0.656 | 0.151 | 0.214 | 0.159 |
| checkdown | 0 | - | - | - | 0.948 | 0.274 | 0.381 | 0.313 |

**E2b: the flop-boundary value function V_flop.**
* Labels are bootstrapped DeepStack-style: solve the flop street only, with the
  turn leaves valued by the matching V_turn. The number of situations is
  `M_flop = min(N, 2000)`.
* Errors are measured against 300 **exact** flop labels, each a full flop+turn
  solve costing 255 CPU-ms, and against 80 on-policy flop roots.

| V_flop | M_flop | boot label CPU-ms | boot-label RMSE vs exact | range-w | test RMSE | test RMSE range-w | on-policy RMSE | on-policy range-w |
|---|---|---|---|---|---|---|---|---|
| 200 | 200 | 87.5 | 0.414 | 0.206 | 1.062 | 0.569 | 0.467 | 0.403 |
| 2000 | 2000 | 100.6 | 0.320 | 0.139 | 0.792 | 0.393 | 0.300 | 0.277 |
| 20000 | 2000 | 122.5 | 0.233 | 0.083 | 0.794 | 0.360 | 0.309 | 0.292 |
| 20000_raw | 2000 | 88.1 | 0.282 | 0.145 | 0.891 | 0.441 | 0.379 | 0.356 |
| 20000_rw | 2000 | 115.9 | 0.241 | 0.072 | 0.838 | 0.227 | 0.272 | 0.225 |
| checkdown | - | - | - | - | 1.050 | 0.363 | 0.491 | 0.460 |

**Label noise (side measurement, `experiments/label_noise.py`).** Equilibrium
CFVs are not unique. On the same 1000 turn situations, per-hand CFVs after T
iterations differ from those after 1000 iterations as follows:

| CFR+ iters | label expl (x) | RMSE all hands | RMSE hands with p>1e-3 | range-weighted RMSE |
|---|---|---|---|---|
| 50 | 0.00828 | 0.267 | 0.249 | 0.074 |
| 100 | 0.00293 | 0.197 | 0.174 | 0.043 |
| 200 | 0.00105 | 0.131 | 0.112 | 0.023 |
| 400 | 0.00039 | 0.066 | 0.054 | 0.010 |

At the 300 iterations used for labels, the range-weighted label noise is about
0.015. That is an order of magnitude below the net's error (0.15–0.47), so the
nets are limited by data and capacity, not by label noise. Unweighted errors
are dominated by hands a player does not hold, whose values the equilibrium
does not pin down.

**E2c: continual re-solving agents.**
* Each agent's complete strategy is extracted in both seats: 16,237 re-solves
  per agent, 300 CFR+ iterations each, batched by phase.
* Each strategy is then scored with an exact best response on the full game.
* "ms per re-solve (batched)" is the phase time divided by the number of
  re-solves. Unbatched single re-solve latencies are in E5.

| agent | expl mA/h | % pot | seat0 expl | seat1 expl | ms per re-solve (batched) pre/flop/turn | extraction s | # re-solves |
|---|---|---|---|---|---|---|---|
| full solve, 300 it | 10.5 | 0.53 | - | - | - | - | - |
| full solve, 2000 it | 0.41 | 0.020 | - | - | - | - | - |
| oracle_naive | 520.5 | 26.03 | 757.3 | 283.7 | 3312.3 / 74.8 / 3.2 | 154.8 | 16237 |
| oracle_safe | 23.7 | 1.18 | 24.7 | 22.7 | 3252.5 / 76.8 / 3.4 | 158.3 | 16237 |
| checkdown_safe | 532.5 | 26.63 | 393.8 | 671.2 | 52.5 / 9.0 / 3.4 | 59.5 | 16237 |
| learned200_safe | 521.2 | 26.06 | 549.9 | 492.5 | 87.2 / 33.0 / 3.3 | 74.9 | 16237 |
| learned2000_safe | 511.8 | 25.59 | 441.7 | 582.0 | 71.9 / 24.9 / 2.8 | 60.9 | 16237 |
| learned20000_safe | 499.1 | 24.95 | 469.5 | 528.7 | 71.0 / 23.9 / 2.8 | 60.3 | 16237 |
| learned20000_raw_safe | 506.0 | 25.30 | 488.6 | 523.4 | 63.6 / 18.1 / 2.7 | 54.3 | 16237 |
| learned20000_rw_safe | 417.4 | 20.87 | 412.6 | 422.2 | 75.1 / 23.6 / 2.6 | 57.7 | 16237 |
| learned20000_naive | 824.4 | 41.22 | 1057.8 | 590.9 | 72.1 / 23.3 / 3.1 | 64.7 | 16237 |

**Seed replicate.** An earlier run of the same pipeline differed only in the
random flop-situation sample (results kept in
`results/e2_run1_combined.json`). It gave:

| agent | replicate |
|---|---|
| learned200 | 549.9 |
| learned2000 | 502.2 |
| learned20000 | 518.3 |
| learned20000_raw | 544.4 |
| learned20000_naive | 1052.0 |
| oracle_safe, oracle_naive, checkdown | identical |

The differences among the non-range-weighted learned variants (499–550) are
within seed noise.

**E2d: which boundary's value function hurts?** Each agent below uses the exact
leaf at one boundary and an approximate leaf at the other.

| agent | expl mA/h | % pot | extraction s (both seats) |
|---|---|---|---|
| checkdown@preflop + oracle@flop | 416.5 | 20.82 | 101.2 |
| oracle@preflop + checkdown@flop | 379.9 | 18.99 | 106.3 |
| learned20000_rw@preflop + oracle@flop | 215.6 | 10.78 | 106.1 |
| oracle@preflop + learned20000_rw@flop | 258.7 | 12.94 | 102.5 |

Both boundaries contribute comparably. Making either one exact removes only
20–50% of the exploitability: checkdown drops from 532 to 417 or 380, and the
range-weighted net drops from 417 to 216 or 259. No single "bad street"
explains the gap.

**How to read E2.** The re-solving machinery is correct: with exact leaves the
agent stays within 2.3 times the full-solve exploitability at the same
iteration count. **The gadget is not optional**: the same exact leaves without
it give 22 times more exploitability. The weak link is V. A range-weighted
error of about 0.15–0.23 pot-shares at one boundary costs about 0.2–0.26
antes/hand (E2d), comparable to having no value function at all (checkdown).

### E3 Off-tree bet sizes

At every flop root, player 0 may open with a size f that is not in the menu.
We are player 1. "Loss" is player 0's best-response value against our strategy
in the augmented game minus the augmented game's value, in mA/h over all hands.

* **Snap down / up / nearest** play the blueprint at the lower, upper or
  nearest menu size (nearest by pot fraction). Actions are translated by label.
  * Where the translated line has already ended in the base tree, the
    augmented-game equilibrium is used instead. This is generous to snapping.
  * Example: a small bet snapped to the only menu bet, an all-in, while the
    real hand continues.
  * The last column counts the player-1 nodes that this rescue reached.
* **Safe re-solve** re-solves the flop+turn subgame from the flop root with f
  added. The gadget uses player 0's blueprint CFVs, and player 1's blueprint
  range is the root range, at 300 iterations. The re-solved strategy is used
  only below the off-tree action.
* **Aug Nash** is player 1's strategy from a 1000-iteration solve of the
  augmented game, as a reference.
* The blueprint in the base game is exploitable by 0.53 mA/h.

| size (pot) | value gain of size for P0 | aug-solve expl | snap down | snap up | snap nearest | safe re-solve | aug Nash | re-solve ms/root (batched) | single re-solve s | reached P1 nodes w/o translation (down/up/near) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 7.12 | 1.56 | 389.1 | 389.1 | 389.1 | 18.8 | 1.8 | 318.0 | 0.66 | 360/360/360 of 3700 |
| 0.75 | -0.07 | 1.60 | 227.2 | 179.0 | 227.2 | 7.2 | 2.2 | 277.7 | 0.67 | 0/0/0 of 1860 |
| 1.5 | 0.01 | 1.50 | 50.8 | 235.5 | 50.8 | 1.4 | 1.9 | 256.4 | 0.60 | 0/720/0 of 750 |

For 0.25 there is no lower menu size, so all three snaps map to the smallest
menu bet. The off-menu sizes are worth only 0–7 mA/h to player 0 in
equilibrium (column 2). Snapping hands the exploiter 51–389 mA/h. The
translation-free safe re-solve costs 1.4–18.8 mA/h, which is 20–36 times less
than snapping to the nearest size, and needs one re-solve of 0.6–0.7 s at 300
iterations.

### E4 One mechanism for robustness and exploitation (restricted Nash response)

The three fixed opponents (`opponents.py`):
* **station**: check or call, never folds or raises;
* **maniac**: all-in whenever it may bet or raise, otherwise calls;
* **nit**: continues only with a top-2-rank card preflop or a pair or better
  postflop. It bets pot or calls with those and check-folds everything else.

RNR(p) is CFR+ in which **our** traversal sees the opponent reach
`p·reach(model) + (1−p)·reach(free CFR+ player)`. It runs 600 iterations per
solve. All numbers are seat-averaged mA/h.

| model | strategy | EV vs model | exploitability | EV as P0 | EV as P1 |
|---|---|---|---|---|---|
| station | Nash (p=0) | 811.9 | 0.4 | 694.9 | 929.0 |
| station | RNR p=0.25 | 2861.8 | 154.4 | 2922.9 | 2800.6 |
| station | RNR p=0.5 | 3085.9 | 278.3 | 3078.2 | 3093.5 |
| station | RNR p=0.75 | 3122.8 | 341.9 | 3120.3 | 3125.4 |
| station | RNR p=1.0 | 3142.8 | 2526.3 | 3142.8 | 3142.8 |
| station | exact best response | 3142.9 | 1570.2 | 3142.9 | 3142.9 |
| maniac | Nash (p=0) | 540.9 | 0.4 | 164.9 | 916.9 |
| maniac | RNR p=0.25 | 1178.0 | 47.1 | 1127.3 | 1228.6 |
| maniac | RNR p=0.5 | 1298.0 | 137.9 | 1306.2 | 1289.7 |
| maniac | RNR p=0.75 | 1428.6 | 334.4 | 1428.6 | 1428.6 |
| maniac | RNR p=1.0 | 1428.6 | 2302.2 | 1428.6 | 1428.6 |
| maniac | exact best response | 1428.6 | 1498.4 | 1428.6 | 1428.6 |
| nit | Nash (p=0) | 421.6 | 0.4 | 316.6 | 526.5 |
| nit | RNR p=0.25 | 1393.2 | 108.1 | 1016.2 | 1770.3 |
| nit | RNR p=0.5 | 2900.5 | 1114.5 | 2971.2 | 2829.7 |
| nit | RNR p=0.75 | 3296.2 | 1591.4 | 3234.7 | 3357.7 |
| nit | RNR p=1.0 | 3344.4 | 4920.2 | 3284.1 | 3404.7 |
| nit | exact best response | 3344.4 | 9039.3 | 3284.1 | 3404.8 |

* **Nash vs bad players.** Nash already wins 422–812 mA/h against these
  players.
* **RNR is a better point than the best response.** For the maniac, RNR at
  p = 0.75 earns the **full best-response EV (1428.6) at 334 mA/h of
  exploitability**, against 1498 for the best response.
* **The best response's exploitability is arbitrary.**
  * It comes mostly from off-path tie-breaking: actions at nodes the model
    never reaches.
  * That is why the pure best response against the nit is exploitable by
    9.0 antes/hand while earning the same EV as RNR(1.0).
  * RNR settles those choices through the free player.

**The diff from Nash to exploitation.** Everything is in `cfr.py`:

```python
# cfr.py, Solver.__init__ (lines 59-60): reach of the fixed model, computed once
if rnr is not None:
    self.fix_reach = self.reach(rnr["player"], eff=rnr["eff"])

# cfr.py, Solver.opp_reach (lines 90-92): the opponent reach seen by the exploiter
if self.rnr is not None and self.rnr["player"] == o:      # <- RNR: the whole diff
    p = self.rnr["p"]
    q = p * self.fix_reach + (1.0 - p) * q
```

With `p = 0` this is exactly the Nash solver. Nothing else changes: not the
regret update, not the averaging, not the tree. The safe re-solving gadget is
the same kind of hook. It changes the opponent's root reach to
`root_reach * g_follow` (lines 87–88), and adds a two-action regret-matching+
update per hand at the root (lines 157–168).

### E5 Timing summary

| quantity | value |
|---|---|
| full solve: s/iteration (55k nodes × 10 hands, 1 core) | 0.0670 |
| full solve to <1% pot | 300 it, 19.6 s |
| full solve to <0.1% pot | 1000 it, 67.1 s |
| turn oracle label (exact 1-street solve, 300 it), CPU-ms / situation | 11.2 |
| exact flop label (flop+turn solve, 300 it), CPU-ms / situation | 255.0 |
| bootstrapped flop label (flop street + V_turn leaves), CPU-ms / situation | 87.5–122.5 |
| train V_turn (6000 Adam steps), s | 10.0–14.3 |
| train V_flop, s | 10.6–12.9 |
| single re-solve, preflop root, checkdown / learned / oracle leaf | 0.162 / 0.376 / 16.6 s |
| single re-solve, flop root, checkdown / learned / oracle leaf | 0.154 / 0.325 / 0.593 s |
| single re-solve, turn root (no leaves), any | 0.10 s |
| single off-tree flop re-solve (E3), s | 0.60–0.67 |
| batched re-solve (all re-solves of one phase in one forest), ms/re-solve | preflop 52–87 (depth-limited), flop 9–33, turn 2.6–3.4 |

All re-solves use 57 public nodes plus leaves when depth-limited. The oracle
preflop re-solve is the whole game (54,957 nodes), and the oracle flop re-solve
has 2,082 nodes.

## 5. What the data says

### Supported

1. **The public tree with a hand fiber plus vectorised CFR+ is exact and cheap
   at this scale.**
   * The vectorised EV equals the brute-force over-deals EV to 1e-12.
   * 172k information sets are solved to 0.07% pot in 67 s on one core with
     numpy.
   * The entire per-iteration work is about 13 levels × a handful of array
     operations.
2. **CFR+ (RM+, linear averaging, alternating updates) beats vanilla CFR.**
   It is 8.6 times lower in exploitability at 300 iterations and 42 times lower
   at 2000, for the same time per iteration.
3. **Safe continual re-solving reconstructs a near-equilibrium from street-local
   solves when the leaf values are exact.** It scores 23.7 mA/h, which is
   1.18% pot and 2.3 times the full solve at equal iterations.
4. **Naive range-only re-solving is unsound in practice, not only in theory.**
   It scores 520 mA/h with exact leaves and 824–1052 mA/h with learned leaves.
5. **Translation-free re-solving of off-menu sizes beats snapping** by 20–36
   times. Snapping's loss depends on which neighbour is chosen (179 vs 227 for
   0.75 pot; 51 vs 236 for 1.5 pot), so no fixed snapping rule is safe.
6. **Restricted Nash response is a 3-line change to the Nash solver.** It gives
   a usable EV–exploitability frontier. For example, against the station it
   reaches 91% of the best-response EV (88% of the gain over Nash) at 10% of
   the best response's exploitability.

### Contradicted, or not achieved in this budget

1. **"A learned street-boundary value function makes depth-limited re-solving
   near-Nash" was not achieved.**
   * With 200, 2,000 or 20,000 oracle labels (4 CPU-min of labelling for 22k),
     agents score 417–550 mA/h across two runs.
   * The checkdown leaf scores 533 mA/h. Exact leaves score 24 mA/h.
   * More data reduced the regression error (range-weighted turn error
     0.47 → 0.23) but did not move agent exploitability beyond seed noise.
2. **The residual-over-checkdown baseline improves regression error but not
   play.**
   * Regression error: turn 0.230 vs 0.315, flop 0.360 vs 0.441.
   * Exploitability: 499 vs 506 mA/h, and in the replicate 518 vs 544.
3. **Agent quality is very sensitive to the leaf.** About 0.2 pot-shares of
   range-weighted error at a single boundary costs about 0.2–0.26 antes/hand
   (E2d). The learned value function, not the solver, is where this
   whole approach lives or dies. This toy shows no evidence that it is cheap.

## 6. Limitations

* **The toy is small.** It has 10 cards, 1 private card, 3 streets and 55k
  public nodes. HUNL has 1,326-hand fibers and far larger public trees, so the
  value-function result could change in either direction at scale.
* **Value-net design was only lightly tuned.**
  * One MLP architecture, one random-range generator, and 6000 Adam steps.
  * The flop net was trained on at most 2,000 bootstrapped labels.
  * Range-weighted loss, the one variant that helped, was tried only at
    N = 20,000.
  * Bigger N (≥ 200k labels, about 40 CPU-min here), on-policy sampling of
    situations, or larger nets were not tested.
* **Re-solve iterations.** Every re-solve uses K = 300 CFR+ iterations, and
  only that K was measured.
* **The ORACLE leaf is the joint full-depth solve.** It is the limit-equivalent
  of a per-iteration exact oracle; that oracle is not called literally on
  every iteration.
* **E3 scope.**
  * It tests only off-menu opening sizes by player 0 at the flop root.
  * The snapping baseline is deterministic. Randomised pseudo-harmonic
    translation was not implemented.
  * The safe re-solve uses the full remaining game (exact leaves). With a
    learned leaf it would inherit E2's leaf error.
* **E4 scope.**
  * The opponent models are known exactly. Estimating them from finite data
    (the HUD problem) is not tested.
  * The best response's exploitability depends on arbitrary tie-breaking at
    unreached nodes.
* **Two players only.** Nothing here measures multiway pots, where these
  zero-sum guarantees do not apply.
* **Timings were taken on a shared machine.** Another CPU-heavy job was running.
