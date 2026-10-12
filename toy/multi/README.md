# toy/multi — multiway hidden-state structure in a 3-player toy poker

Reference code ("the fishing rod") plus measurements for five questions about solving
**multiplayer** poker with a *public tree + private-hand fiber* representation and vectorised
CFR+. Everything is pure numpy (Python 3.13, numpy 2.x, no scipy/torch/numba), deterministic
(fixed seeds), and runs in about 30 minutes on 4 cores.

Units: 1 ante = the betting unit. Exploitability is reported in **mA/hand = milli-antes per
hand** (the analogue of mbb/hand: the ante plays the role of the big blind).
NashConv = sum over players of delta_i, delta_i = BR_i(sigma_-i) - EV_i(sigma).

---------------------------------------------------------------------------------------------

## 1. The game ("Tri-Leduc")

| item | value (main game; every parameter is a field of `GameConfig`) |
|---|---|
| players | N = 3 (2- and 4..9-player variants are the same code with seats added/removed) |
| deck | D = 12 cards = 6 ranks x 2 suits; card c has rank c // 2 |
| private info | one private card per player; joint hidden state = ordered triple of distinct cards (12*11*10 = 1320 deals) |
| antes | 1 per player (pot starts at N) |
| stacks | 15 antes each, ante included; equal stacks => no side pots ever |
| streets | 2: betting, then ONE public board card, then betting |
| bet menu | not facing a bet: check / bet pot / all-in; facing a bet: fold / call / raise pot (call, then raise by the pot after calling) / all-in |
| raise cap | 2 aggressive actions (bet or raise) per street; a pot-size action that would reach the stack is not offered (all-in covers it) |
| order | lowest seat still able to act opens both streets (seat 0 is always "out of position") |
| showdown | pairing the board beats any unpaired card; else higher rank; equal strength splits the pot |
| size | 6,295 public nodes, 2,574 public decision nodes, 30,888 information sets, 3,669 terminals (2,532 showdowns) |

Public-tree sizes of the same game with other seat counts (counted exactly by `m5_scaling.py`):
N=2: 1,119 nodes; N=4: 15,053; N=5: 64,253; N=6: 257,437; N=7: 988,749; N=8: 3,683,277;
N=9: 13,405,261 (71.9 M infosets).

## 2. Architecture (the pattern under test)

```
game.py       rules only (legal actions, transitions, showdown share)                    ~150 lines
tree.py       PUBLIC tree, BFS-flattened into arrays; children contiguous; depth levels    ~140
terminal.py   terminal counterfactual values with explicit card removal:                   ~250
              EXACT (brute force over disjoint tuples), INDEPENDENT, PAIRWISE, MOBIUS (exact)
cfr.py        vectorised tree passes (reach forward, values backward), CFR+, exact BR      ~220
scalar.py     one-deal-at-a-time view of the same tree (brute-force reference, sampling)
mccfr.py      external-sampling MCCFR; model-free outcome-sampling learner vs a fixed env
coalition.py  colluding joint strategies for seats 1+2
tests.py      correctness checks (brute-force EV and BR, method agreement, 2p convergence)
experiments/  m1..m5 scripts; results/ holds the logs (*.log) and raw numbers (*.json)
```

* **Never branch on hidden state.** A node is a betting sequence plus the board card. Every
  private-information quantity is an array with a trailing fiber axis of length D:
  `reach[p, node, card]`, `sigma[node, card]` (probability of the edge INTO `node`, so per-action
  arrays are plain `[M, D]` arrays), `val_i[node, card]`.
* **Tree passes are ~12 numpy steps** (one per depth level): forward `reach[:, lo:hi] =
  reach[:, parent] * sigma` and backward `np.add.reduceat(w * val, sibling_starts)`; the best
  response uses `np.maximum.reduceat` at the BR player's nodes.
* **Chance is uniform and factored out**: chance edges weigh 1/(D-N) (board), the deal weight
  1/n_deals multiplies the root: `EV_i = sum_c val_i[root, c] / n_deals`.
* **Card removal lives only in the terminal evaluator.** The reach of every player at the board
  child is zeroed at the board card; disjointness between private cards is enforced when the
  per-player reach vectors are combined at the terminals (section M1).
* **CFR+**: regret matching+, alternating updates (player p's update sees players < p already
  updated), linear averaging (weight t) of the strategy actually played.
* **Exact exploitability**: one forward pass + one max-backward pass per player using EXACT
  terminal utilities, no sampling.

Verified (`python3 tests.py`, 4 s): vectorised EV equals the brute-force EV obtained by walking
the tree for every one of the 360 deals x boards of a 6-card 3-player game (to 1e-12); the BR
value from the max-pass equals the brute-force EV of the explicit argmax policy (to 1e-12, all
three players); MOBIUS equals EXACT for N = 2, 3, 4 (max abs diff 7e-13); PAIRWISE equals EXACT
for N = 3 and differs for N = 4; CFR+ on the 2-player variant reaches NashConv 0.79 mA after
1000 iterations.

## 3. How to run

```
cd toy/multi
python3 tests.py                                  #   4 s
cd experiments
python3 m2_convergence.py   # M2 (writes the strategy used by M3)   ~5.5 min
python3 m1_card_removal.py  # M1 part A (~5 min) + part B (~3 min); 'a' or 'b' to run one part
python3 m3_coalition.py     # M3                                    ~3.5 min
python3 m4_mccfr_vs_cfr.py  # M4 (run alone: it measures wall clock) ~8.5 min
python3 m5_scaling.py       # M5                                    ~1 min
```
Wall-clock numbers below are single-process, measured with nothing else running (M1-B and M3
overlapped once during development; their timings are not used for any claim).

---------------------------------------------------------------------------------------------

## M1. Card removal in multiway terminal values

### The math

Hero i holds c, board b, opponents j in O (m = N-1 of them) with own-reach vectors r_j:

    cfv_i(c) = sum_{(c_j) all distinct, != c, != b}  prod_j r_j(c_j) * u_i(c, (c_j), b)

* **(a) EXACT** – materialise the joint opponent reach `P = r_j (x) r_k` ([T, D^m]) and multiply by
  precomputed kernels `valid(c, c_-i, b)` and `valid*share_i` ([D^m, D]). Cost O(T D^N).
* **(b) INDEPENDENT** – drop the constraint between opponents (keep c_j != c, != b). The sum then
  factorises: each opponent contributes masses L_j(c) (weaker than c), E_j(c) (tying), T_j(c)
  (any card != c); the hero's expected pot share is read off the generating polynomial
  prod_j (L_j + E_j z) as sum_k coef_k/(k+1). Cost O(m D^2 + m^2 D) per terminal (the D^2 is a
  comparison matmul; sorting makes it D log D).
* **Exact correction = Moebius inversion on the partition lattice of the opponents:**

      sum_{distinct} f = sum_{partitions pi of O} mu(pi) * sum_{c constant on each block of pi} f,
      mu(pi) = prod_{blocks B} (-1)^{|B|-1} (|B|-1)!

  For a fixed partition the sum still factorises: block B acts as ONE opponent with range
  rho_B(d) = prod_{j in B} r_j(d) whose |B ∩ alive| members all hold d (factor L_B + E_B z^{a_B}).
  **(d) MOBIUS** = all Bell(m) partitions -> exact.
  **(c) PAIRWISE** = finest partition (+1) and the C(m,2) single-pair partitions (-1), i.e.
  "subtract the mass where two opponents hold the same card".
  **For N = 3 (m = 2) the only partitions are {j}{k} and {jk}, so PAIRWISE == MOBIUS == EXACT
  identically** — measured: max |difference| 3e-16 relative. The 4-player run is what tests (c).

**Error scaling** (relative error of the probability mass; first order in the collision
probability; p_j = normalised range of opponent j):

    INDEPENDENT omits  sum_{j<k} P(c_j = c_k)                         ~ C(m,2) / D     (uniform)
    PAIRWISE    omits  2 sum_{j<k<l} P(c_j=c_k=c_l) + sum_{2 disjoint pairs} P(.)P(.)
                                                                ~ (2 C(m,3) + 3 C(m,4)) / D^2

So independent-range error grows **quadratically in the number of opponents and decays only
as 1/D**; pairwise error is one order higher in 1/D but grows like m^3..m^4. With correlated
ranges (both opponents hold strong cards after a raise) the collision probability is the
overlap sum_d p_j(d) p_k(d), which can be much larger than 1/D; in real hold'em a "collision"
is two hands sharing ANY card (~8% for two random hands from 50 cards, ~2.2 expected colliding
pairs among 8 opponents), so (b) is unusable at 9-max and the partition expansion over
*hands* does not apply directly (it handles equality collisions; card-sharing needs a card-level
inclusion-exclusion instead — not implemented here).

### Part A — CFR+ with each evaluator, scored exactly (same iteration budget)

3 players, D = 12, 2000 CFR+ iterations each:

| method | ms/iter | total s | d0 mA | d1 mA | d2 mA | NashConv mA | terminal rel. L1 err (at exact eq.) | max root cond.-EV err mA (at exact eq.) | self-assessed EV err mA (own sol.) |
|---|---|---|---|---|---|---|---|---|---|
| exact | 26.4 | 52.9 | 0.24 | 0.21 | 0.66 | 1.12 | 0 | 0.0 | 0.0 |
| independent | 34.5 | 68.9 | 19.40 | 29.91 | 22.98 | **72.30** | 7.41e-02 | 304.79 | 61.0 |
| pairwise (= exact at N=3) | 37.5 | 75.0 | 0.25 | 0.21 | 0.67 | 1.13 | 3.0e-16 | 0.0 | 0.0 |

4 players, D = 8 (4 ranks x 2 suits), stack 15, raise cap 1, 6,693 public nodes, 400 iterations each:

| method | ms/iter | total s | d0 mA | d1 mA | d2 mA | d3 mA | NashConv mA | terminal rel. L1 err (at exact eq.) | max root cond.-EV err mA | self-assessed EV err mA |
|---|---|---|---|---|---|---|---|---|---|---|
| exact (brute force D^4) | 218.0 | 87.2 | 8.69 | 15.27 | 7.89 | 15.35 | 47.21 | 0 | 0.0 | 0.0 |
| independent | 38.3 | 15.3 | 488.10 | 235.45 | 171.76 | 232.01 | **1127.33** | 7.08e-01 | 3199.92 | 1406.27 |
| pairwise | 66.1 | 26.5 | 174.04 | 98.97 | 103.58 | 61.09 | **437.68** | 7.53e-02 | 559.35 | 78.52 |
| mobius (exact) | 87.1 | 34.8 | 10.00 | 15.86 | 9.96 | 11.51 | 47.34 | 4.8e-16 | 0.0 | 0.0 |

("root cond.-EV err" = max over players and private cards of the error in E[u_i | c_i = c] under
the exact solution's profile; "self-assessed" = how wrong each approximate solver is about its
own solution's EV.) With 4 players and an 8-card deck the exact solution is still far from
converged at 400 iterations (47 mA) but the approximations are 9x and 24x worse and the gap
does not close with iterations because it is a bias, not noise.

Wall-clock at N=3, D=12 is a wash: the brute-force exact evaluator is the *fastest* (26 ms/iter
vs 34-38) because D^3 = 1728 is tiny and it is one BLAS matmul per (board, alive set), while the
factorised methods pay numpy overhead for small-array polynomial algebra. At N=4 the exact
brute force is already 2.5x slower than MOBIUS.

### Part B — isolated showdown, error and cost scaling

Random opponent ranges at one all-alive showdown (40 random boards / range draws per cell; hero =
seat 0). "uniform" = every card weight 1; "concentrated" = independent Gamma(0.5) weights per
opponent. Reference = brute force where D^N <= 4e5 (checked equal to MOBIUS to 1e-9), MOBIUS
otherwise. "pred" columns are the first-order formulas above evaluated on the actual ranges.
Equity error = absolute error in the hero's expected pot share given its card.
Full 56-row table: `results/m1b.log`.

| ranges | N | D | indep mass rel err | pred | indep equity err (mean / max) | pairwise mass rel err | pred | pairwise equity err mean | pairwise mass<=0 frac | us/terminal indep / pairwise / mobius / brute |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 3 | 8 | 2.0e-01 | 1.4e-01 | 2.4e-02 / 4.3e-02 | 0 (exact) | 0 | 0 | 0.00 | 7 / 9 / 6 / 78 |
| uniform | 3 | 16 | 7.7e-02 | 6.7e-02 | 1.1e-02 / 1.8e-02 | 0 (exact) | 0 | 0 | 0.00 | 9 / 12 / 9 / 227 |
| uniform | 3 | 32 | 3.4e-02 | 3.2e-02 | 5.4e-03 / 8.4e-03 | 0 (exact) | 0 | 0 | 0.00 | 15 / 16 / 14 / 1505 |
| uniform | 3 | 64 | 1.6e-02 | 1.6e-02 | 2.6e-03 / 4.1e-03 | 0 (exact) | 0 | 0 | 0.00 | 18 / 22 / 23 / 15026 |
| uniform | 4 | 8 | 8.0e-01 | 4.3e-01 | 3.7e-02 / 7.7e-02 | 1.0e-01 | 4.1e-02 | 2.4e-02 | 0.00 | 7 / 14 / 14 / 198 |
| uniform | 4 | 16 | 2.6e-01 | 2.0e-01 | 1.7e-02 / 3.2e-02 | 1.3e-02 | 8.9e-03 | 3.0e-03 | 0.00 | 9 / 16 / 20 / 3368 |
| uniform | 4 | 32 | 1.1e-01 | 9.7e-02 | 8.1e-03 / 1.5e-02 | 2.5e-03 | 2.1e-03 | 6.0e-04 | 0.00 | 15 / 27 / 32 / - |
| uniform | 4 | 64 | 5.0e-02 | 4.8e-02 | 4.0e-03 / 7.2e-03 | 5.5e-04 | 5.0e-04 | 1.3e-04 | 0.00 | 25 / 54 / 57 / - |
| uniform | 5 | 8 | 2.6e+00 | 8.6e-01 | 4.5e-02 / 1.2e-01 | 1.0e+00 | 2.2e-01 | n/a | 1.00 | 8 / 27 / 47 / 1799 |
| uniform | 5 | 16 | 6.0e-01 | 4.0e-01 | 2.0e-02 / 4.8e-02 | 8.6e-02 | 4.9e-02 | 1.2e-02 | 0.00 | 12 / 35 / 59 / - |
| uniform | 5 | 32 | 2.3e-01 | 1.9e-01 | 9.7e-03 / 2.2e-02 | 1.5e-02 | 1.1e-02 | 2.0e-03 | 0.00 | 15 / 59 / 100 / - |
| uniform | 5 | 64 | 1.0e-01 | 9.5e-02 | 4.8e-03 / 1.0e-02 | 3.1e-03 | 2.8e-03 | 4.2e-04 | 0.00 | 33 / 115 / 190 / - |
| uniform | 6 | 8 | 9.8e+00 | 1.4e+00 | 5.1e-02 / 1.6e-01 | 8.2e+00 | 7.1e-01 | n/a | 1.00 | 12 / 52 / 157 / 26523 |
| uniform | 6 | 16 | 1.2e+00 | 6.7e-01 | 2.3e-02 / 6.1e-02 | 3.6e-01 | 1.6e-01 | 4.9e-02 | 0.00 | 12 / 77 / 236 / - |
| uniform | 6 | 32 | 4.2e-01 | 3.2e-01 | 1.1e-02 / 2.8e-02 | 5.3e-02 | 3.6e-02 | 4.7e-03 | 0.00 | 18 / 103 / 441 / - |
| uniform | 6 | 64 | 1.8e-01 | 1.6e-01 | 5.3e-03 / 1.3e-02 | 1.0e-02 | 8.8e-03 | 8.9e-04 | 0.00 | 39 / 197 / 764 / - |
| concentrated | 3 | 8 | 2.3e-01 | 1.4e-01 | 2.3e-02 / 2.7e-01 | 0 (exact) | 0 | 0 | 0.00 | 5 / 6 / 9 / 62 |
| concentrated | 3 | 32 | 3.7e-02 | 3.4e-02 | 6.4e-03 / 3.8e-02 | 0 (exact) | 0 | 0 | 0.00 | 11 / 11 / 11 / 1335 |
| concentrated | 4 | 8 | 8.4e-01 | 4.1e-01 | 4.0e-02 / 3.6e-01 | 1.1e-01 | 3.7e-02 | 4.8e-02 | 0.00 | 9 / 14 / 17 / 206 |
| concentrated | 4 | 32 | 1.1e-01 | 9.6e-02 | 7.5e-03 / 4.3e-02 | 2.3e-03 | 2.0e-03 | 5.4e-04 | 0.00 | 16 / 30 / 31 / - |
| concentrated | 5 | 8 | 2.9e+00 | 8.3e-01 | 4.0e-02 / 3.1e-01 | 1.1e+00 | 1.9e-01 | 7.4e-01 | 0.34 | 8 / 25 / 48 / 1737 |
| concentrated | 5 | 32 | 2.3e-01 | 1.9e-01 | 1.1e-02 / 4.0e-02 | 1.5e-02 | 1.1e-02 | 2.2e-03 | 0.00 | 17 / 70 / 95 / - |
| concentrated | 6 | 16 | 1.3e+00 | 6.6e-01 | 2.3e-02 / 1.2e-01 | 3.9e-01 | 1.6e-01 | 7.7e-02 | 0.03 | 14 / 85 / 238 / - |
| concentrated | 6 | 64 | 1.8e-01 | 1.6e-01 | 5.3e-03 / 2.2e-02 | 1.1e-02 | 9.0e-03 | 8.9e-04 | 0.00 | 43 / 384 / 1055 / - |

Read-off:
* INDEPENDENT mass error halves when D doubles (N=3: 3.4e-2 -> 1.6e-2 from D=32 to 64; N=6:
  4.2e-1 -> 1.8e-1) and multiplies by C(m,2) across N (D=64: 1.6e-2, 5.0e-2, 1.0e-1, 1.8e-1 for
  m = 2, 3, 4, 5 vs. C(m,2) = 1, 3, 6, 10). PAIRWISE error quarters when D doubles (N=4: 2.5e-3 ->
  5.5e-4). The first-order predictions match within 10% once the error is below ~0.1, and
  under-predict badly when it is not (small D, many players): the expansion is asymptotic.
* For small decks PAIRWISE is *not* a safe approximation: it over-subtracts and produces
  non-positive masses (100% of cases at N=5, D=8 and N=6, D=8..12; 34% at N=5, D=8 concentrated).
* Conditional equity errors (1e-2..5e-2 for INDEPENDENT at D<=16) look small, but Part A shows
  what they do inside a solver: 7.4% terminal error -> 72 mA NashConv at N=3 (65x the exact
  solver's 1.12 mA), and 1127 mA at N=4.
* Cost: brute force grows as D^N (15 ms per terminal at N=3, D=64; 27 ms at N=6, D=8); MOBIUS
  stays in the 6-1100 us range up to N=6, D=64 (Bell(5) = 52 partitions).

## M2. CFR+ self-play convergence with 3 players

3 players (and the 2-player reduction as a control), exact evaluator, 10,000 iterations,
average strategy scored exactly at each checkpoint. R_i/T = average external regret of player
i against the realised self-play sequence (the opponents' strategies it actually faced in its
alternating update), computed exactly by one best-response pass on the accumulated terminal
values.

3 players (273 s for 10,000 iterations, 27 ms/iter):

| iter | sec | d0(avg) mA | d1(avg) mA | d2(avg) mA | NashConv(avg) mA | loglog slope | NashConv(cur) mA | R0/T mA | R1/T mA | R2/T mA | max R/T*sqrt(T) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 4951.66 | 4805.77 | 4491.37 | 14248.8 |  | 8852.86 | 4951.66 | 5600.79 | 4596.05 | 5600.8 |
| 10 | 0.3 | 967.73 | 851.43 | 600.94 | 2420.09 | -1.2 | 2121.34 | 1946.34 | 1843.87 | 1529.88 | 6154.9 |
| 100 | 2.7 | 23.95 | 31.11 | 27.8 | 82.87 | -1.47 | 177.62 | 214.16 | 213.24 | 173.68 | 2141.6 |
| 200 | 5.4 | 7.71 | 8.18 | 10.82 | 26.71 | -1.63 | 55.79 | 107.18 | 105.23 | 86.34 | 1515.8 |
| 500 | 13.0 | 2.07 | 2.44 | 2.56 | 7.07 | -1.45 | 14.09 | 42.99 | 41.81 | 34.36 | 961.4 |
| 1000 | 26.1 | 0.70 | 0.68 | 1.08 | 2.46 | -1.52 | 14.02 | 21.39 | 20.91 | 17.13 | 676.4 |
| 2000 | 52.2 | 0.24 | 0.21 | 0.66 | 1.12 | -1.14 | 10.25 | 10.69 | 10.47 | 8.56 | 478.0 |
| 3000 | 79.3 | 0.16 | 0.13 | 0.58 | 0.87 | -0.62 | 9.36 | 7.13 | 6.98 | 5.71 | 390.3 |
| 5000 | 134.1 | 0.11 | 0.08 | 0.59 | 0.78 | -0.21 | 8.06 | 4.28 | 4.19 | 3.43 | 302.5 |
| 7000 | 189.5 | 0.16 | 0.07 | 0.53 | 0.76 | -0.09 | 8.66 | 3.06 | 2.99 | 2.46 | 255.7 |
| 10000 | 273.2 | 0.22 | 0.07 | 0.67 | 0.96 | +0.67 | 10.06 | 2.14 | 2.09 | 1.71 | 214.1 |

2 players, same deck/stacks/menu (32 s for 10,000 iterations):

| iter | d0(avg) mA | d1(avg) mA | NashConv(avg) mA | loglog slope | NashConv(cur) mA | R0/T mA | R1/T mA |
|---|---|---|---|---|---|---|---|
| 100 | 34.7 | 21.18 | 55.88 | -1.52 | 95.35 | 201.08 | 144.88 |
| 1000 | 0.47 | 0.42 | 0.89 | -1.77 | 20.18 | 20.27 | 14.73 |
| 3000 | 0.07 | 0.07 | 0.15 | -1.67 | 9.31 | 6.77 | 4.94 |
| 10000 | 0.01 | 0.01 | 0.02 | -1.42 | 5.79 | 2.03 | 1.49 |

(full tables with every checkpoint: `results/m2.log`.)

Findings:
* **3 players: fast convergence, then a plateau with drift.** NashConv of the average falls like
  t^-1.5 (the usual CFR+ rate) down to ~1 mA by 2000 iterations, then flattens at 0.76-0.96 mA
  and even rises between 7,000 and 10,000 iterations (slope +0.67). The plateau is carried by
  player 2 (delta_2 = 0.53-0.67 mA from iteration 3000 on) and player 0 (0.11 -> 0.22 mA
  after 5000). Over the same budget the 2-player control keeps converging (0.02 mA, slope -1.4).
  So in this game CFR+ self-play reaches an epsilon-Nash with epsilon ~ 1 mA/hand (0.1% of an
  ante, ~0.07% of a 15-ante stack) but **does not converge to exact Nash**.
* **The current iterate does not converge in either case**: it oscillates at 8-10 mA for 3p (and
  is still 5.8 mA for 2p); only averages are usable.
* **No-regret holds regardless of N.** Average external regret against the realised sequence
  decays as ~1/T (R_i^T itself stays ~17-21 antes from T = 1000 to 10,000), i.e. faster than the
  O(1/sqrt T) guarantee, identical in behaviour for 2 and 3 players. In 3 players this
  guarantee (coarse-correlated-equilibrium-type) is exactly what does NOT pin down a Nash
  equilibrium: regret -> 0 while NashConv of the average plateaus.
* Positional EVs of the final 3p average strategy (antes/hand): seat 0: -0.1018, seat 1: -0.0232,
  seat 2: +0.1250 (last to act wins). 2p: seat 0: -0.1147, seat 1: +0.1147.

## M3. "Opponents are weather" against a colluding coalition

Seats 1 and 2 collude against seat 0. Both coalitions are built from the M2 self-play strategy
sigma* (the colluders still play their own card sensibly) and override it where collusion matters
(`coalition.py`; no card sharing):

* **softplay**: if seat 0 has folded, colluders only check/call; they never bet/raise when the
  partner is the last aggressor on the street (fold/call proportions of sigma* kept, raise mass
  moved to call).
* **whipsaw** = softplay + whenever seat 0 has voluntarily put chips in on the current street
  (bet, raised or called) and a raise is still allowed, the colluder raises (pot if available,
  else all-in) **with any card**.

Seat 0's exact EV (antes/hand; positive = seat 0 wins):

| coalition | seat0 EV in self-play | (i) frozen sigma*_0 | (ii) best response | coalition EV (vs frozen) | (iii) CFR+ fresh, avg @2000 | (iii) CFR+ warm, avg @2000 | fresh: iters to 90% / 99% of gap | warm: iters to 90% / 99% |
|---|---|---|---|---|---|---|---|---|
| softplay | -0.1018 | -0.1059 | -0.0283 | +0.1059 | -0.0283 | -0.0285 | 5 / 20 | 100 / 500 |
| whipsaw | -0.1018 | -0.0311 | +5.6797 | +0.0311 | +5.6797 | +5.6796 | 10 / 50 | 20 / 50 |

(iii) = CFR+ for seat 0 only, coalition fixed, exact counterfactual feedback each iteration
(11 ms/iteration; 2000 iterations in 22 s). "fresh" = zero regrets (first iterate = sigma*_0),
"warm" = M2's self-play regrets. Current iterate reaches 90% of the gap after 1 (softplay) / 2
(whipsaw) iterations fresh, 50 / 5 warm. Selected checkpoints of the average strategy's EV:

| coalition | start | t=1 | t=5 | t=20 | t=100 | t=500 | t=2000 | BR |
|---|---|---|---|---|---|---|---|---|
| softplay | fresh | -0.1059 | -0.0352 | -0.0289 | -0.0284 | -0.0283 | -0.0283 | -0.0283 |
| softplay | warm | -0.1060 | -0.0932 | -0.0751 | -0.0346 | -0.0288 | -0.0285 | -0.0283 |
| whipsaw | fresh | -0.0311 | 4.7646 | 5.6137 | 5.6769 | 5.6795 | 5.6797 | 5.6797 |
| whipsaw | warm | -0.0340 | 3.2296 | 5.4441 | 5.6660 | 5.6791 | 5.6796 | 5.6797 |

(iv) **Model-free online learning**: outcome-sampling regret matching+, one real hand per
iteration, using only the observed payoff and its own sampling probabilities (no model of the
coalition). eps = exploration of the sampling policy; "warm" = M2 regrets divided by n_deals
(their per-hand scale). EV of the average strategy (exact) and realised mean payoff per hand
played so far:

| coalition | variant | 1k hands avg EV | 10k | 100k | 1M | realised/hand over 1M | wall s (1M) |
|---|---|---|---|---|---|---|---|
| softplay | fresh eps=0.6 | -1.703 | -1.392 | -0.884 | -0.481 | -1.285 | 14.1 |
| softplay | fresh eps=0.2 | -1.630 | -1.411 | -0.925 | -0.516 | -0.797 | 13.7 |
| softplay | warm eps=0.2 | -1.391 | -1.211 | -0.855 | -0.515 | -0.790 | 13.6 |
| softplay | warm eps=0.05 | -1.526 | -1.223 | -0.926 | -0.575 | -0.637 | 12.8 |
| whipsaw | fresh eps=0.6 | -0.658 | 2.102 | 4.290 | 5.155 | -0.225 | 16.2 |
| whipsaw | fresh eps=0.2 | -0.636 | 2.184 | 4.388 | 5.184 | 2.966 | 16.2 |
| whipsaw | warm eps=0.2 | -1.178 | 2.240 | 4.370 | 5.178 | 2.954 | 16.2 |
| whipsaw | warm eps=0.05 | -1.099 | 1.795 | 4.117 | 5.072 | 4.461 | 17.0 |

Findings:
* **The frozen self-play strategy is robust to these coalitions but does not exploit them.**
  Against softplay it loses 4.1 mA/hand more than in self-play (-0.1059 vs -0.1018); against
  whipsaw it actually does better than in self-play (-0.0311), because raising any card is a
  losing coalition strategy. In 3-player poker there is no guarantee this holds for every
  coalition (no minimax theorem for 1 vs 2); for these two it did.
* **The exploitation available is large**: +74 mA/hand vs softplay (from -0.106 to -0.028) and
  +5.7 antes/hand vs whipsaw.
* **With counterfactual feedback (i.e. knowing the coalition's strategy) regret minimisation
  gets there in 5-50 iterations** (fresh) — essentially a best-response computation. Warm
  starting from self-play regrets is *slower* (100/500 iterations to 90/99% vs softplay): stale
  regrets must be unlearned.
* **Without a model, it is a different story.** The model-free learner starts from scratch
  (its first-hand policy is uniform, EV -1.7/hand) and, after one million hands, is still at
  -0.48..-0.58 vs softplay — **far worse than simply keeping sigma*_0 (-0.106)** — while its
  realised losses during learning average 0.64-1.29 antes/hand. Against the grossly exploitable
  whipsaw it reaches 91% of the BR value after 1M hands. The warm start in "per-hand units" is
  wiped out within the first 1000 hands (the prior's scale is tiny next to the importance-
  weighted sampled regrets), so it does not protect the learner either; making it protect
  would need a hand-picked prior weight. "Opponents are weather" + pure outcome-based
  no-regret learning is sound but needs 10^5-10^6+ hands here; the fast path requires an
  explicit (statistical) model of the opponents' strategy to compute counterfactual values.

## M4. External-sampling MCCFR vs vectorised CFR+ at equal wall-clock

Both on the same public tree, scored by the same exact NashConv; clocks exclude evaluation.
ES-MCCFR: scalar Python, regret matching, linear-weighted average at opponents' nodes (in a pilot, linear
averaging was equal or better than plain averaging: 1525 vs 1538 mA after 60k iterations). Main game, mean of 2 ES seeds:

| budget s | CFR+ iters | CFR+ NashConv mA | ES iters | ES node visits | ES NashConv mA | winner | ES/CFR+ ratio |
|---|---|---|---|---|---|---|---|
| 5 | 180 | 31.38 | 28,200 | 2.23e6 | 2477.55 | CFR+ | 79 |
| 15 | 534 | 6.29 | 80,100 | 6.58e6 | 1306.50 | CFR+ | 208 |
| 45 | 1677 | 1.35 | 231,200 | 1.93e7 | 597.58 | CFR+ | 442 |
| 90 | 3329 | 0.87 | 462,000 | 3.86e7 | 460.08 | CFR+ | 530 |

Sweep over player count and deck size (budgets 3/10/30 s, 1 ES seed; full tables in
`results/m4.log`), values at 30 s:

| game | public nodes | infosets | CFR+ NashConv mA | ES NashConv mA | ES/CFR+ | ES speed-up needed to tie (~ratio^2) |
|---|---|---|---|---|---|---|
| N=2, D=12, cap 2 | 1,119 | 5,016 | 0.02 | 240.88 | 10,877 | ~1e8 |
| N=3, D=8, cap 2 | 4,239 | 13,872 | 0.50 | 431.40 | 855 | ~7e5 |
| N=3, D=12, cap 2 (45 s) | 6,295 | 30,888 | 1.35 | 597.58 | 442 | ~2e5 |
| N=3, D=16, cap 2 | 8,351 | 54,624 | 6.22 | 1364.28 | 219 | ~5e4 |
| N=4, D=8, cap 1 | 6,693 | 24,800 | 39.74 | 1398.32 | 35 | ~1.2e3 |

Findings:
* **Vectorised CFR+ wins at 3 players by 2-3 orders of magnitude** in NashConv at every budget,
  and the gap widens with time (ES converges ~t^-0.5, CFR+ ~t^-1.5 until its plateau).
* Python penalises the scalar ES walker (430k node visits/s; a C implementation would do
  ~50-100x more). Because ES error falls like work^-1/2, closing a ratio r needs ~r^2 more ES
  work: ~2e5x at the main game. Even granting 100x, CFR+ wins here.
* **The crossover moves toward sampling as D grows and as N grows** — measured, both
  directions: the ES/CFR+ ratio falls 855 -> 442 -> 219 for D = 8 -> 12 -> 16 at N = 3, and
  10,877 -> 855 -> 35 for N = 2 -> 3 -> 4 (D = 8..12). Reason: a CFR+ iteration costs
  O(N passes x M_public(N) x D) + terminals x W(N, D), i.e. linear in the fiber size D and
  linear in the public tree, which grows ~4x per added player; an ES traversal costs one path
  through opponents' nodes (sampled) times the traverser's branching, independent of D, and
  grows far more slowly with N (only the traverser's own decisions are enumerated). ES needs
  more traversals when there are more infosets (variance), but per-unit-of-work it gains
  relative to full-width passes as D and N grow. At 4 players with a 30 s budget, a ~1000x
  faster ES implementation would already tie.

## M5. Cost scaling with the number of players

Same game (D = 12, stack 15, pot + all-in, cap 2), seats added or removed. Measured:

| N | evaluator | public nodes M | showdown terminals | infosets | measured ms/iter | model ms/iter | fiber memory MB |
|---|---|---|---|---|---|---|---|
| 2 | exact | 1,119 | 396 | 5,016 | 3.9 | 2.4 | 0.6 |
| 2 | mobius | 1,119 | 396 | 5,016 | 3.8 | 3.4 | 0.6 |
| 2 | independent | 1,119 | 396 | 5,016 | 2.9 | 3.3 | 0.6 |
| 3 | exact | 6,295 | 2,532 | 30,888 | 27.4 | 34.1 | 4.0 |
| 3 | mobius | 6,295 | 2,532 | 30,888 | 49.0 | 53.0 | 4.0 |
| 3 | independent | 6,295 | 2,532 | 30,888 | 33.5 | 42.1 | 4.0 |
| 4 | exact | 15,053 | 7,260 | 74,784 | 366.3 | 343.1 | 11.0 |
| 4 | mobius | 15,053 | 7,260 | 74,784 | 310.5 | 342.6 | 11.0 |
| 4 | independent | 15,053 | 7,260 | 74,784 | 127.0 | 179.1 | 11.0 |
| 5 | mobius | 64,253 | 32,352 | 327,000 | 4995.9 | 4451.3 | 52.9 |
| 5 | independent | 64,253 | 32,352 | 327,000 | 792.6 | 1194.5 | 52.9 |

(3 -> 2 players: 5.6x fewer public nodes, 7x faster iterations. 3 -> 4: 2.4x nodes, 13x
slower with the brute-force exact evaluator. Fiber memory = (N + 4) arrays of [M, D] float64.)

**Cost model** (fitted by relative least squares on the measured points; a, b per evaluator):

    t_iter(N) = N * [ a * M(N) * D * N  +  b * T_sd(N) * W(N, D) ]
      N alternating passes; each pass recomputes N reach arrays over M public nodes x D cards
      W_exact  = D^N,  W_mobius = (2^m - 1) D^2 + Bell(m) m^2 D,  W_indep = m D^2 + m^2 D,  m = N-1
      fitted: exact a=4.3e-8 s, b=3.6e-10 s;  mobius a=5.3e-8, b=4.2e-9;  independent a=6.2e-8, b~0

Assumptions: same rules and bet menu at every N; M(N) and T_sd(N) are exact counts from the
public-tree counter (not extrapolated); constants from numpy on this machine; no
public-chance sampling, no pruning, no abstraction; float64.

| N | public nodes M | showdown terminals | infosets | fiber memory GB | s/iter exact (model) | s/iter mobius (model) | s/iter independent (model) | hours for 1000 iters (mobius) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1,119 | 396 | 5,016 | 0.0006 | 0.0024 | 0.0034 | 0.0033 | 0.0009 |
| 3 | 6,295 | 2,532 | 30,888 | 0.004 | 0.034 | 0.053 | 0.042 | 0.015 |
| 4 | 15,053 | 7,260 | 74,784 | 0.011 | 0.34 | 0.34 | 0.18 | 0.095 |
| 5 | 64,253 | 32,352 | 327,000 | 0.052 | 15.4 | 4.45 | 1.19 | 1.24 |
| **6** | 257,437 | 132,996 | 1,333,872 | 0.23 | 870 | 73.2 | 6.89 | 20.3 |
| 7 | 988,749 | 518,712 | 5,195,400 | 0.97 | 4.7e4 | 1.5e3 | 36 | 419 |
| 8 | 3,683,277 | 1,949,484 | 19,574,592 | 3.95 | 2.4e6 | 3.5e4 | 175 | 9.8e3 |
| **9** | 13,405,261 | 7,128,336 | 71,915,256 | 15.6 | 1.2e8 | 8.7e5 | 807 | 2.4e5 |

Read-off:
* The public tree of this tiny 2-street game grows ~4x per added seat (x5.6, x2.4, x4.3, x4.0,
  x3.8, x3.7, x3.6 from N=2..9). The fiber (D per player) is the cheap dimension; the
  multiway *betting* tree is the expensive one.
* Brute-force exact card removal (D^N) is hopeless beyond 5 players. The Moebius-exact
  evaluator as implemented is dominated by Bell(N-1) partitions (4140 at 9-max) and is also
  impractical at 9 players (8.7e5 s/iter). The independent approximation is "only" 807 s/iter
  at 9 players — but M1 shows its bias is unacceptable.
* At 6 players the full-width approach is still feasible for this toy (independent: 7 s/iter,
  Moebius: 73 s/iter -> 20 h per 1000 iterations); at 9 players it is not (2.4e5 h).
* Two implementation-level savings not taken here: recomputing only the updated player's
  reach per pass (removes one factor N from the tree term: ~9x at N=9) and an exact
  subset-DP card-removal evaluator (cards in strength order, state = set of opponents already
  placed: O(D * m * 2^m) per hero card, 2^8 = 256 at 9-max instead of Bell(8) = 4140). Neither
  changes the ~4^N public-tree growth.

---------------------------------------------------------------------------------------------

## Verdict on the candidate principles (blunt)

Supported by the data:
1. Public tree + per-player private-card fibers, with card removal handled only at terminals,
   is correct (brute-force verified) and cheap in memory: O(N * M * D) instead of O(M * D^N).
   The joint belief r_1 x r_2 x r_3 restricted to disjoint tuples never needs to be stored.
2. Card removal among opponents must be exact. Ignoring it (INDEPENDENT) costs 65x the exploit-
   ability at N=3, D=12 (72.3 vs 1.12 mA) and 24x at N=4, D=8; the error is a bias that more
   iterations do not remove. At 3 players the pairwise correction is exactly exact.
3. Exact multiway card removal can be done without enumerating D^N tuples: Moebius inversion over
   partitions of the opponents, verified equal to brute force to 1e-12.
4. Vectorised CFR+ with linear averaging reaches ~1 mA/hand NashConv in 3-player play within
   ~2000 iterations (52 s) and beats external-sampling MCCFR by 2-3 orders of magnitude at equal
   wall-clock in every 2-4 player configuration tested.
5. Regret minimisation is no-regret against whatever the other seats do (average external regret
   ~1/T at 2 and 3 players), and, given counterfactual feedback against a fixed coalition, reaches
   the best-response value within 5-50 iterations.

Contradicted or qualified by the data:
1. "CFR+ self-play converges in multiplayer": it does not converge to Nash here; NashConv
   plateaus at ~0.8-1.0 mA and drifts upward after 7000 iterations, while 2p goes to 0.02 mA.
   The plateau is small for practical purposes, but it is a plateau, not convergence.
2. "Pairwise (inclusion-exclusion) correction is good enough": exact for N=3 only; at N=4 it is
   still 9x worse than exact (437.7 vs 47.2 mA) and it produces negative masses for small decks.
3. "Opponents are weather + online no-regret learning extracts the profit": only with
   counterfactual feedback (a model of the opponents). Model-free outcome-based regret matching
   is still far worse than the frozen equilibrium strategy after 1M hands vs the softplay
   coalition, losing 0.6-1.3 antes/hand during learning.
4. "The approach scales to 9 players": full-width vectorised CFR+ on even this 2-street,
   1-card toy needs 13.4M public nodes and an estimated 807 s/iter (independent) to 8.7e5 s/iter
   (exact Moebius) at 9-max. The public betting tree (~4x per seat), not the hidden-state fiber, is
   the binding constraint; MCCFR's relative position improves with N and D.

## Limitations

* One toy game family; 1-card hands make collisions "equal cards". Hold'em hands collide when
  they share any card, which the partition formula does not cover.
* The plateau in M2 is measured for one game, one seat order (no rotation) and linear averaging;
  other weighting schemes (e.g. DCFR) were not tried.
* The coalitions do not share cards; a card-sharing coalition needs a joint fiber [node, c_1, c_2]
  for seats 1+2 (the exact evaluator accepts a joint opponent tensor in place of the outer
  product, so the architecture supports it, but it was not run).
* ES-MCCFR is scalar Python: wall-clock comparisons penalise it by an unknown constant (est.
  50-100x). The "speed-up needed to tie" column is the honest translation.
* Cost extrapolations to 6 and 9 players use exact node counts but constants fitted on N <= 5
  numpy runs; they are order-of-magnitude estimates.
* No neural approximation, abstraction, pruning or subgame solving was tested.
