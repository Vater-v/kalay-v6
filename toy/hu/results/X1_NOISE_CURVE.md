# X1: what street-boundary value error costs in exploitability

Script: `experiments/x1_noise_curve.py`. Raw results are in `results/x1_<setting>.json`, the aggregate is in `results/x1.json`, the generated tables are in `results/x1_tables.md`, and the run logs are in `results/x1_*.log`. Everything is numpy-only with `OPENBLAS_NUM_THREADS=1` and a 4-process pool. No existing file was modified.

## 1. Definitions (read these first: three different "pots" appear below)

* **x** is the number of chips *each* player has put into the pot at a street boundary (the flop root or the turn root). The **leaf pot** is therefore 2x. In this game x is one of {1, 2, 3, 4, 6, 9} antes at the turn boundary and one of {1, 2, 3, 6, 9} at the flop boundary.
* **Leaf value u_i(h)** is player i's counterfactual value with hand h at the boundary, divided by x and normalised by the opponent's range mass. This is the normalisation `valuefn.py` uses. **"Pot-share units" (x units)** are units of x. An error of 0.15 x units is 0.15·x chips, which is **7.5 % of the leaf pot** (2x), not 15 %. The toy README and the audit brief both call 0.15 "15 % of the pot". That is half right: it is 15 % of one player's share, which is 7.5 % of the whole leaf pot. In this file every table gives the x-unit number, and the "% leaf pot" column is x-units / 2 × 100.
* **Range-weighted RMSE** (the same formula as `e2_common.rmse_w`):
  `sqrt( mean over situations S of [ Σ_h p0(h)·err0(h)² + Σ_h p1(h)·err1(h)² ] / 2 )`.
  * `err_i(h)` is the perturbed u minus the exact u, in x units.
  * `p_i` is player i's normalised range at S.
  * This is roughly "the typical value error, in x units, of a hand a player actually holds".
* **In-situ RMSE** applies that formula to the boundary nodes the agent's own re-solves actually reach.
  * It is measured at the end of each re-solve, against the exact value of the jointly solved subtree under the final average profile.
  * A node counts when both normalised reaches exceed 1e-3 of the re-solve root's mass. That gives about 6,000 flop→turn nodes and 140–260 preflop→flop nodes per agent, both seats combined.
  * The "all" column is node-count weighted, so the flop→turn boundary dominates it. Both boundaries are also shown separately.
* **Test RMSE** applies the same error model to E2's exact held-out labels: 1,000 random turn situations and 300 random exact flop situations. This is the column the E2 nets were judged on.
* **Exploitability of the agent** is `(BR0 + BR1)/2`.
  * `BRi` is the exact best-response value of a player who knows the agent's complete strategy, computed on the full public tree (`cfr.exploitability`). The agent's complete strategy is its behaviour at every public node and for every hand, in both seats.
  * The quantity is the average over the two seats of what a perfect exploiter wins per hand, over the whole game from the deal.
  * Units: **mA/h** = milli-antes per hand. In this toy the ante plays the role of the big blind, so mA/h is the analogue of mbb/h.
  * **% init pot** = exploitability / (initial pot of 2 antes) × 100. For example, 23.7 mA/h = 0.0237 antes/hand = 1.18 % of the initial pot.
* **"18x"** (README §4 E2) compared two agents' *exploitabilities*: the best learned-leaf safe re-solver (`learned20000_rw_safe`, 417.4 mA/h) and the exact-leaf safe re-solver (`oracle_safe`, 23.7 mA/h), both measured by the same whole-game measure. 417.4 / 23.7 = 17.6. It is *not* a ratio of regression errors. The learned net's error was 0.15 (turn) / 0.23 (flop) in x units on random test situations. The 23.7 floor is **not** caused by leaf error: it is what safe re-solving with K = 300 CFR+ iterations per re-solve costs when the leaves are exact.

## 2. Method

* **Agent.** The agent is E2c's safe (gadget) continual re-solver with the ORACLE leaf. Each re-solve keeps the rest of the game below the street boundary and solves it jointly, so its boundary values are exact. Everything else is unchanged: K = 300, the same phases and batching, extraction of the complete strategy in both seats, and an exact best response on the full game. The harness copies `resolve.extract_agent` and subclasses `cfr.Solver`.
* **Where the error is injected.** For each boundary node S (the next-street root right after the card is dealt, where a depth-limited agent calls V), the backward pass passes `v_i(S,h) + δ_i(S,h)` up to S's parent chance node.
  * `v[S]` itself stays exact, so the regrets inside the subtree are untouched. The subtree only supplies "the exact V".
  * The opponent CFVs stored at S for the next street's gadget are the *perturbed* ones, exactly as a depth-limited agent stores V's output.
  * Only the current street's boundary is perturbed. Turn re-solves have no boundary.
* **Error models**, with δ written in x units and the absolute δ = cscale·x·M_opp·δ_u:
  * **(a) iid**: `δ_u = 2·eps·z`, with z ~ N(0,1) fixed per (public node S, hand, player, seed). The noise std is **eps × leaf pot** in chips, and the RMSE is about 2·eps in x units. The noise is a fixed function of the public state; it does not depend on the ranges. Seeds 0 and 1.
  * **(b) zs**: the same z, then the MLP's zero-sum projection per situation (`e = p0·n0 + p1·n1; n_i −= e/2`). Chip conservation `p0·u0 + p1·u1` is then unchanged. The noise draws are the same as in (a), so (a) and (b) are paired.
  * **(c) bias**: `u ← u − λ (u − u_checkdown)`, i.e. a fraction λ of the residual over check-down is removed. λ = 1 is exactly the check-down leaf.
  * **learned**: `u ← V_learned(board, x, p0, p1)` with E2's 20000_rw nets.
* **Validation of the harness.** It reproduces E2c's depth-limited agents:

  | case | X1 harness | E2c |
  |---|---|---|
  | no perturbation | 23.7 mA/h | oracle_safe 23.7 |
  | λ = 1 | 532.5 mA/h | checkdown_safe 532.5 |
  | learned 20000_rw | 417.7 mA/h | learned20000_rw_safe 417.4 |

  So "perturbed oracle" and "depth-limited agent with that V" are the same agent.
* **Wall-clock.** Each setting took 90–260 s for both seats. The box was shared with other agents' jobs, at load averages of 5–9 on 4 cores. Every invocation stayed under 10 minutes. The total was about 37 minutes of runs that produced results, plus one 10-minute group run that hit its timeout and was discarded. The script now saves each setting as soon as both of its seats finish, and skips settings that already have results.

## 3. Results

### 3.1 Per setting (all seeds)

### Noise curve: iid (a) and zero-sum-projected (b) Gaussian noise

eps = noise std as a fraction of the LEAF pot (2x).  RMSE columns in x units (1.0 = one player's pot contribution at the boundary = half the leaf pot); '% leaf pot' = RMSE_x / 2 * 100.  Exploitability = (BR0+BR1)/2 of the agent over the whole game, mA/h = milli-antes per hand; '% init pot' = mA/h / 2000 * 100.

| model | eps | seed | in-situ RMSE_rw (x) | as % leaf pot | in-situ pre->flop | in-situ flop->turn | test RMSE_rw turn/flop (x) | expl mA/h | % init pot | excess over exact mA/h |
|---|---|---|---|---|---|---|---|---|---|---|
| iid | 0.001 | 0 | 0.0020 | 0.10 | 0.0019 | 0.0020 | 0.0020 / 0.0020 | 24.5 | 1.23 | +0.8 |
| iid | 0.001 | 1 | 0.0020 | 0.10 | 0.0020 | 0.0020 | 0.0020 / 0.0020 | 24.4 | 1.22 | +0.8 |
| iid | 0.003 | 0 | 0.0060 | 0.30 | 0.0058 | 0.0060 | 0.0059 / 0.0060 | 26.3 | 1.31 | +2.6 |
| iid | 0.003 | 1 | 0.0060 | 0.30 | 0.0060 | 0.0060 | 0.0059 / 0.0060 | 26.9 | 1.35 | +3.3 |
| iid | 0.01 | 0 | 0.0200 | 1.00 | 0.0191 | 0.0200 | 0.0198 / 0.0200 | 39.7 | 1.98 | +16.0 |
| iid | 0.01 | 1 | 0.0200 | 1.00 | 0.0198 | 0.0200 | 0.0198 / 0.0200 | 40.9 | 2.05 | +17.2 |
| iid | 0.03 | 0 | 0.0599 | 3.00 | 0.0574 | 0.0600 | 0.0594 / 0.0601 | 78.6 | 3.93 | +55.0 |
| iid | 0.03 | 1 | 0.0598 | 2.99 | 0.0589 | 0.0599 | 0.0594 / 0.0601 | 81.9 | 4.10 | +58.2 |
| iid | 0.1 | 0 | 0.2005 | 10.03 | 0.2008 | 0.2005 | 0.1981 / 0.2003 | 200.7 | 10.04 | +177.0 |
| iid | 0.1 | 1 | 0.1994 | 9.97 | 0.1975 | 0.1995 | 0.1981 / 0.2003 | 234.6 | 11.73 | +210.9 |
| iid | 0.3 | 0 | 0.6064 | 30.32 | 0.5965 | 0.6068 | 0.5944 / 0.6009 | 549.8 | 27.49 | +526.1 |
| iid | 0.3 | 1 | 0.6012 | 30.06 | 0.6072 | 0.6009 | 0.5944 / 0.6009 | 620.2 | 31.01 | +596.5 |
| zs | 0.001 | 0 | 0.0019 | 0.09 | 0.0019 | 0.0019 | 0.0017 / 0.0018 | 24.3 | 1.21 | +0.6 |
| zs | 0.001 | 1 | 0.0019 | 0.09 | 0.0019 | 0.0019 | 0.0017 / 0.0018 | 23.4 | 1.17 | -0.2 |
| zs | 0.003 | 0 | 0.0056 | 0.28 | 0.0056 | 0.0056 | 0.0052 / 0.0053 | 26.0 | 1.30 | +2.3 |
| zs | 0.003 | 1 | 0.0056 | 0.28 | 0.0057 | 0.0056 | 0.0052 / 0.0053 | 26.6 | 1.33 | +2.9 |
| zs | 0.01 | 0 | 0.0187 | 0.93 | 0.0183 | 0.0187 | 0.0172 / 0.0176 | 38.2 | 1.91 | +14.5 |
| zs | 0.01 | 1 | 0.0187 | 0.93 | 0.0188 | 0.0187 | 0.0172 / 0.0176 | 37.8 | 1.89 | +14.1 |
| zs | 0.03 | 0 | 0.0556 | 2.78 | 0.0552 | 0.0556 | 0.0516 / 0.0528 | 75.3 | 3.77 | +51.7 |
| zs | 0.03 | 1 | 0.0553 | 2.77 | 0.0557 | 0.0553 | 0.0516 / 0.0528 | 75.3 | 3.76 | +51.6 |
| zs | 0.1 | 0 | 0.1821 | 9.11 | 0.1851 | 0.1820 | 0.1721 / 0.1761 | 185.3 | 9.27 | +161.6 |
| zs | 0.1 | 1 | 0.1817 | 9.08 | 0.1865 | 0.1815 | 0.1721 / 0.1761 | 198.6 | 9.93 | +174.9 |
| zs | 0.3 | 0 | 0.5305 | 26.53 | 0.5562 | 0.5294 | 0.5163 / 0.5284 | 389.5 | 19.47 | +365.8 |
| zs | 0.3 | 1 | 0.5314 | 26.57 | 0.5583 | 0.5302 | 0.5163 / 0.5284 | 398.7 | 19.94 | +375.0 |

### 3.2 The curve (seed means). Read "x % of leaf pot error → y exploitability" off this table

| noise std eps (fraction of leaf pot) | iid RMSE_rw, x units | = % of leaf pot | iid expl mA/h (seeds 0, 1) | iid mean mA/h | mean % init pot | excess over exact-leaf agent, mA/h | zs RMSE_rw (x) | zs expl mA/h (seeds 0, 1) | zs mean |
|---|---|---|---|---|---|---|---|---|---|
| 0 (exact) | 0 | 0 | 23.7 | 23.7 | 1.18 | 0 | 0 | 23.7 | 23.7 |
| 0.001 | 0.0020 | 0.10 | 24.5, 24.4 | 24.5 | 1.22 | +0.8 | 0.0019 | 24.3, 23.4 | 23.9 |
| 0.003 | 0.0060 | 0.30 | 26.3, 26.9 | 26.6 | 1.33 | +2.9 | 0.0056 | 26.0, 26.6 | 26.3 |
| 0.01 | 0.0200 | 1.0 | 39.7, 40.9 | 40.3 | 2.02 | +16.6 | 0.0187 | 38.2, 37.8 | 38.0 |
| 0.03 | 0.0599 | 3.0 | 78.6, 81.9 | 80.3 | 4.01 | +56.6 | 0.0555 | 75.3, 75.3 | 75.3 |
| 0.1 | 0.2000 | 10.0 | 200.7, 234.6 | 217.7 | 10.9 | +194.0 | 0.1819 | 185.3, 198.6 | 192.0 |
| 0.3 | 0.6038 | 30.2 | 549.8, 620.2 | 585.0 | 29.2 | +561.3 | 0.5310 | 389.5, 398.7 | 394.1 |

* **Shape.** From an RMSE of 0.02 to 0.6 x units, the excess exploitability is close to linear in the error.
  * The excess is about 0.93 mA/h per 0.001 x units of range-weighted RMSE (measured ratios 0.83–0.97).
  * Equivalently: **excess exploitability as a % of the initial pot ≈ 0.93 × (leaf error as a % of the leaf pot)**.
* **Below 1 % of the leaf pot the cost is sub-linear and small.**
  * At 0.1 % and 0.3 % of the leaf pot, the excess is 0.8 and 2.9 mA/h. That is about half the linear rate.
  * There the agent's exploitability is dominated by the 23.7 mA/h floor of K = 300 re-solving.
* **Seed spread.** About ±1 mA/h at small eps, rising to ±8 % at eps ≥ 0.1.
* **Projection (zs vs iid).**
  * The projection removes about 6–12 % of the RMSE.
  * At equal RMSE it costs about the same up to eps = 0.1: 0.77–0.93 mA/h per 0.001 RMSE for zs, against 0.83–0.97 for iid, at eps 0.01–0.1.
  * It matters clearly only for gross errors. At eps = 0.3 the projected noise costs 394 mA/h against 585, which is 0.70 vs 0.93 mA/h per 0.001 RMSE.
  * So chip conservation is cheap insurance, but it is not what separates a good V from a bad one in the range that matters.

### 3.3 Systematic bias (shrink toward check-down) and the harness validation rows

### Systematic bias (c): shrink toward check-down by lam; validation rows

| setting | in-situ RMSE_rw (x) | in-situ pre->flop | in-situ flop->turn | test RMSE_rw turn / flop (x) | on-policy RMSE_rw turn / flop (x) | expl mA/h | % init pot | E2c depth-limited agent mA/h |
|---|---|---|---|---|---|---|---|---|
| base | 0.0000 | 0.0000 | 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 23.7 | 1.18 | 23.7 |
| bias_l0.1 | 0.0318 | 0.0479 | 0.0312 | 0.0274 / 0.0363 | 0.0313 / 0.0460 | 50.9 | 2.55 | - |
| bias_l0.3 | 0.1011 | 0.1465 | 0.0989 | 0.0821 / 0.1088 | 0.0940 / 0.1380 | 129.0 | 6.45 | - |
| bias_l0.5 | 0.1731 | 0.2542 | 0.1691 | 0.1369 / 0.1813 | 0.1566 / 0.2300 | 218.7 | 10.94 | - |
| bias_l1.0 | 0.3877 | 0.5111 | 0.3809 | 0.2737 / 0.3626 | 0.3132 / 0.4600 | 532.5 | 26.62 | 532.5 |
| learned_20000_rw | 0.1789 | 0.2873 | 0.1732 | 0.1508 / 0.2271 | 0.1586 / 0.2254 | 417.7 | 20.89 | 417.4 |

### 3.4 Do the E2 learned nets and the check-down leaf lie on the iid-noise curve?

The method is to read the iid curve (seed means, log-log interpolation, 23.7 added back) at each leaf's own RMSE, and compare that with the leaf's measured exploitability. The prediction is given as a range because the leaves' errors differ by boundary:
* first table: in situ, at the "all", preflop→flop and flop→turn RMSEs;
* second table: on E2's turn test, flop test, turn on-policy and flop on-policy sets.

Leaves measured in situ in this harness:

| leaf | in-situ RMSE_x all / pre→flop / flop→turn | iid curve predicts, mA/h | measured mA/h | measured ÷ iid prediction |
|---|---|---|---|---|
| bias λ=0.1 | 0.032 / 0.048 / 0.031 | 51–68 | 50.9 | 0.75–1.00 |
| bias λ=0.3 | 0.101 / 0.146 / 0.099 | 118–165 | 129.0 | 0.78–1.09 |
| bias λ=0.5 | 0.173 / 0.254 / 0.169 | 187–268 | 218.7 | 0.82–1.17 |
| check-down (λ=1) | 0.388 / 0.511 / 0.381 | 384–502 | 532.5 | 1.06–1.39 |
| learned 20000_rw | 0.179 / 0.287 / 0.173 | 191–299 | 417.7 | **1.40–2.18** |

E2 nets, placed by the RMSEs E2 reported (not re-run here; E2c exploitability):

| leaf (E2) | RMSE_x turn test / flop test / turn on-pol / flop on-pol | iid curve predicts, mA/h | E2c measured mA/h | measured ÷ iid prediction |
|---|---|---|---|---|
| 200 | 0.466 / 0.569 / 0.368 / 0.403 | 372–554 | 521.2 | 0.94–1.40 |
| 2000 | 0.348 / 0.393 / 0.291 / 0.277 | 289–395 | 511.8 | 1.30–1.77 |
| 20000 | 0.230 / 0.360 / 0.227 / 0.292 | 242–365 | 499.1 | 1.37–2.06 |
| 20000_raw | 0.315 / 0.441 / 0.303 / 0.356 | 313–439 | 506.0 | 1.15–1.62 |
| 20000_rw | 0.151 / 0.227 / 0.159 / 0.225 | 169–243 | 417.4 | **1.72–2.47** |
| check-down | 0.274 / 0.363 / 0.313 / 0.460 | 286–456 | 532.5 | 1.17–1.86 |

* **Uniform shrinkage toward check-down lies on the iid curve**: it falls inside the prediction bracket for λ ≤ 0.5. Plain under-fitting of the residual costs what its RMSE says.
* **The check-down leaf is at or somewhat above the curve** (1.06–1.39× in situ).
* **The learned nets lie clearly above it.**
  * The better the net, the further above: 20000_rw is 1.4–2.5× what iid noise of the same RMSE costs.
  * 20000_rw has the same in-situ flop→turn RMSE as λ=0.5 shrinkage (0.173 vs 0.169). It costs 418 mA/h against 219.
* **Interpretation, inferred and not separately tested.** The nets' errors are structured. They are correlated across hands and across nearby ranges, and they vary with the input ranges. A re-solver can be steered by such errors consistently, which fixed, independent per-hand noise cannot do. So **RMSE alone under-predicts the damage of a learned V by up to about 2.5× in this toy.**
* **Second reading.** To get to iid-noise behaviour, a net needs both a low RMSE and errors that do not line up with what the re-solver exploits. A validation of V should therefore include agent exploitability, or at least a re-solve-in-the-loop check, not only RMSE.

## 4. Answer: does a 0.3 %-of-pot leaf error cost 30 % of the pot, 0.3 % of the pot, or something else?

**Something else, and much less than 30 %.** These numbers are for this toy only.

* **The measured case.** A leaf error with a std of 0.3 % of the *leaf* pot means:
  * eps = 0.003;
  * a range-weighted RMSE of 0.006 x units;
  * for scale, 0.3 bb on a 100 bb pot.

  The safe re-solving agent with that leaf is exploitable for 26.6 mA/h, which is **1.33 % of the initial (2-ante) pot per hand**. The agent with exact leaves is exploitable for 23.7 mA/h (1.18 %).
* **What the error itself costs.** The *additional* cost of the error is 2.9 mA/h ≈ **0.15 % of the initial pot per hand**. In big-blind terms, with the ante as the bb, that is about 3 mbb/h. This is half of "0.3 %" and two hundred times smaller than "30 %".
* **The general rule in the toy.**
  * From about 1 % to 30 % leaf-pot error, x % of leaf-pot error costs about 0.93·x % of the initial pot per hand in exploitability.
  * Below 1 % it costs less than that.
  * A 30 % leaf-pot error (eps = 0.3) is where "30 % of the pot" appears: 585 mA/h, 29 % of the initial pot.
* **The units mix-up behind the question.**
  * "0.3 % of a 100 bb pot" is 0.3 bb, not 30 bb. "30 bb out of 100" is a 30 % error, two orders of magnitude larger.
  * The leaf error is measured relative to the *leaf* pot, which in this toy is 2–18 antes.
  * The exploitability is measured per hand relative to the *initial* pot of 2 antes, averaged over every hand including those that end early.
  * The two percentages are therefore different quantities. Their near-1:1 ratio here is an empirical fact about this toy, not an identity.
* **Two caveats.**
  1. The rule holds for *iid* error. The learned nets' structured error cost 1.4–2.5× more than iid noise of the same RMSE (§3.4). A real net with 0.3 % RMSE could cost proportionally more than 0.15 %, but nothing here suggests it would cost anything near 30 %.
  2. The toy has 10 cards, one private card, 3 streets and K = 300. HUNL has 1,326-hand ranges and deeper trees, so the slope of 0.93 may not carry over. What carries over is the method: measure the curve, and measure the learned V against it.

## 5. Limitations

* **The noise is a fixed function of the public node.** It does not depend on the ranges, and the same draw is reused by every re-solve that reaches S. A noise model that varies with the ranges, or that is redrawn per call, was not tested.
* **The in-situ RMSE is unweighted across situations.** All counted boundary nodes have equal weight. A reach-weighted variant is stored in the JSON (`rmse_rangew_reachweighted`); it differs from the unweighted value by at most 18 % (check-down, preflop→flop) and does not change the conclusions.
* **Only K = 300 was measured.** The 23.7 mA/h floor depends on K, so the sub-linear region below 1 % may shift with more iterations.
* **The learned nets were not all re-run.** Only 20000_rw was re-run inside the harness. The other E2 nets are placed by the RMSEs E2 reported and by their E2c exploitability. The harness reproduces the depth-limited agents exactly (§2), so this is a units issue (test-set vs in-situ RMSE), not a different agent.
* **Two seeds per noise setting.** The seed spread reaches ±8 % at eps ≥ 0.1.
