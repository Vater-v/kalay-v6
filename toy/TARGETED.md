# Targeted toy measurements for the contested claims

These are new numpy-only experiments that extend the two existing toys,
`toy/hu` (HU mini hold'em, 10 cards, 3 streets, exact best response) and
`toy/multi` (Tri-Leduc, N = 3–4, exact NashConv). They target load-bearing
claims that the refuters left open, or where a toy measurement decides between
the claim and the refuters' correction. No existing file was changed. All new
code is in `toy/hu/experiments/t1_*.py, t2_*.py, t5_*.py`, `toy/multi/team.py`
and `toy/multi/experiments/t3_*.py, t3b_*.py, t4_*.py`. Raw numbers are in
`toy/*/results/t*.json` and `*.log`.

Units and conventions:
* **mA/h** is milli-antes per hand, the analogue of mbb/hand.
* **HU exploitability** is `(BR0 + BR1)/2` on the full game.
* **Multiplayer** results are reported as NashConv, `sum_i (BR_i − EV_i)`, computed exactly.
* **"x units"** are value-net units: a player's pot share, as in `toy/hu/README.md` §E2.

## 0. Which claims needed a new measurement

| claim | test_method | already answered by existing toys? | new test here |
|---|---|---|---|
| 1. Belief factorization / fiber locality | toy | Partly. Both `tests.py` show vectorised EV = brute force over deals (1e-12), so the product-of-reaches posterior is exact at terminals. Refuters checked D = J − MMᵀ + I. **Untested:** whether folded players can be dropped from the joint. | **T4** |
| 2. Regularized (KL) target | toy | No. Refuters solved the clairvoyance river exactly; that settles the analytic parts (kink, O(eps) band, reach-mass dependence). The learnability part needs a regularized solver plus nets at several tau. It is beyond the remaining budget and does not decide anything the refuters' analysis leaves open. | not run |
| 3. Pre-chance boundaries: labels and gadget ceilings | toy | No. E2 uses post-card leaves only. | **T1** (safety), **T2** (labels) |
| 4. Exploitation frontier / declared risk | math + toy | Partly. E4 measures the RNR frontier at p ≥ 0.25. **Untested:** whether the frontier starts with free gains at zero exploitability (refuters' Kuhn counterexample). | **T5** |
| 5. HU real-time feasibility, menu, "off-menu costs nothing" | toy | The timings need Hold'em-scale code (refuters measured a C emulation). **E3 already refutes "costs nothing":** safe re-solve with the off-menu size costs 1.4–18.8 mA/h against 1.5–2.2 for the augmented-game Nash. | not run |
| 6. Label sufficiency | toy | Toy analogue already in E2: 200 → 20,000 labels left agent exploitability flat at 417–550 mA/h, against 24 with exact leaves. The Hold'em learning curve is outside the budget. | not run |
| 7. Multiway card removal | toy | Toy deck: M1 (independence costs 65× NashConv at N = 3). Hold'em equities: refuters measured them. | not run |
| 8. Multiplayer guarantees | toy | Partly. M2 shows the self-play plateau. **Untested:** run-to-run multiplicity, mixing runs, coalition maxmin, security levels. | **T3**, **T3b** |
| 9. Live-count principle | toy | No. | **T4** (cost of dropping the folded players' dead cards when the pot goes to an L-live map) |

## T1. Are chance-averaged opponent CFVs valid gadget ceilings? (claim 3b)

**Setup.** `toy/hu/experiments/t1_prechance_ceilings.py`.
* The agent is E2c's continual re-solving agent with **oracle leaves**: the exact remaining game below each re-solve root. Every variant below therefore uses exact values; only the gadget ceilings differ.
* Each variant's complete strategy is extracted in both seats and scored with an exact best response on the full game.
* K = 300 CFR+ iterations per re-solve, with a K = 1000 rerun for two variants.

The variants differ only in the opponent ceilings w(S_c, h′) used at each next-street root S_c (the street root after card c is dealt):

| variant | ceilings at S_c | K | exploitability mA/h | % pot | seat 0 / seat 1 exploitable by |
|---|---|---|---|---|---|
| (i) `percard`: E2c `oracle_safe` | exact per-card opponent CFVs from the previous solve | 300 | **23.7** | 1.18 | 24.7 / 22.7 |
| (i) `percard` | same | 1000 | **18.6** | 0.93 | 22.1 / 15.1 |
| (ii) `avg` | chance-averaged CFV v_C translated to card c in pot-normalised units, which is exactly what a pre-chance value net returns. Σ_c w_c = v_C holds exactly. | 300 | **608.2** | 30.41 | 559.0 / 657.5 |
| (iii) `start_cfv` | start-of-street unsafe solve with the opponent range carried from the previous gadget solve; ceilings = its opponent CFVs | 300 | **68.0** | 3.40 | 67.8 / 68.1 |
| (iii) `start_cfv` | same | 1000 | **62.4** | 3.12 | 65.9 / 58.9 |
| (iii) `start_br` | as `start_cfv`, but ceilings = opponent best-response CFVs | 300 | 90.6 | 4.53 | 72.5 / 108.7 |
| (iii) `start_br_mix` | as `start_br`, opponent range mixed with 1% uniform (full support) | 300 | 72.0 | 3.60 | 58.3 / 85.7 |
| (iii) `start_min` | min(`start_cfv`, `avg`): per-card solve capped by the previous leaf's commitment | 300 | 93.2 | 4.66 | 108.6 / 77.9 |
| (iv) `chance_root` | **one gadget re-solve rooted at the pre-chance node C.** All next cards are in one forest, the opponent's follow/terminate choice is per hand at C, ceiling = v_C | 300 | **25.2** | 1.26 | 23.8 / 26.5 |

References:
* Full solve at the same 300 iterations: 10.5 mA/h.
* Naive range-only re-solving (E2c): 520.5.
* Checkdown leaf (E2c): 532.5.

Wall time per seat: 55–75 s for `percard`, `avg` and `chance_root`; 100–120 s for the `start_*` variants, which do two solves. All at K = 300.

**Verdict on claim 3b.** Half of the claim holds, and the proposed fix fails.
* **Averaged ceilings break safety.** Using the chance-averaged opponent CFVs as per-card ceilings is 26× more exploitable than per-card ceilings (608 vs 23.7 mA/h). That is worse than having no strategic leaf at all (checkdown, 533).
* **The start-of-street fix does not "match (i) within solver tolerance".** Every variant is 2.9–3.9× worse than per-card ceilings (62–93 mA/h).
  * The gap does not close with more iterations: 62.4 vs 18.6 at K = 1000.
  * Full support does not close it (72.0), and neither does capping with the leaf's commitment (93.2).
  * So it is structural, not solver noise. The start-of-street solve picks a different equilibrium selection of opponent values than the one the previous solve committed to, as both refuters predicted.
* **The chance-rooted gadget does work.** This is the refuters' sound alternative: carry the averaged constraint across chance by rooting the gadget at the pre-chance node. It matches per-card ceilings: 25.2 vs 23.7 mA/h at equal K.
* **What the document should say.** A pre-chance leaf requires a chance-spanning gadget re-solve at the first decision of each street. In Hold'em this costs about 49× the net calls for that one re-solve. The cheap start-of-street variant should not be offered as safe.

## T2. Pre-chance labels: normalisation and the single-card "lever" (claim 3a)

**Setup.** `toy/hu/experiments/t2_prechance_labels.py`.
* Situations sit at the end of flop betting, before the turn card: a random flop card, x from the reachable set, and two random ranges (E2's generator).
* The exact label is u_pre(h) = 1/(n_left − 2) · Σ_{c≠h} (1 − p_opp(c)) · u^c(h), where n_left = 9 legal cards and u^c is an exact turn-subgame solve (300 CFR+ iterations; label exploitability 0.0007 x).
* Held-out test set: 400 situations with all 9 cards solved.

**(1) Identity and bias of naive averages.**

| check | result |
|---|---|
| formula vs exact runout-averaged checkdown W[b]·p (no solver, 300 situations) | max abs error **0.0** (identity exact) |
| claim as written: "uniform average over legal next cards with 1[c ∉ h]", i.e. Σ_{c≠h} u^c / 9 | shrinks labels to **0.84** of the truth; mean abs error 0.106 x on held hands; range-weighted RMSE 0.118 x |
| uniform over cards ∉ h, normalised per-card values, i.e. Σ u^c / 8 | 0.946 of the truth; mean abs error 0.059 x; range-weighted RMSE 0.075 x |
| for scale: checkdown value vs exact pre-chance label | range-weighted RMSE 0.221 x |

**(2) Noise of the unbiased single-card label** L_c(h) = 9/7 · 1[c≠h] · (1 − p_opp(c)) · u^c(h):
* range-weighted RMS noise across cards: **0.704 x**;
* range-weighted RMS spread of the true labels across situations: 0.623 x.

The noise is larger than the signal.

**(3) Equal-compute regression.** Every row spends the same number of turn-subgame solves. The model is the same in every row: MLP 128-128, residual over the exact flop checkdown, zero-sum projection, range-weighted loss, 6000 Adam steps. The score is held-out range-weighted RMSE against exact labels.

| solves spent | m = 9 (all cards, exact labels) | m = 3 sampled cards, 3× situations | m = 1 sampled card, 9× situations |
|---|---|---|---|
| 4,500 | **0.1895** (500 situations) | 0.2223 (1,500) | 0.2209 (4,500) |
| 18,000 | **0.1527** (2,000) | 0.1778 (6,000) | 0.1796 (18,000) |

**Verdict on claim 3a.**
* **The identity is right, but only with the correct weights.** Implemented literally as "uniform average over legal cards", the label is biased (×0.84 here). The bias is of the same order as the net's entire error.
* **The economic lever is not supported.** At equal compute, exact all-card labels beat single-card labels at both budgets. Single-card labels needed about 4× the solves to roughly match exact labels (0.180 at 18,000 solves vs 0.190 at 4,500). The effective gain was therefore about 0.25×, not the nominal 9×.
* **Caveat on transfer.** This toy is a worst case for label noise: 1-card hands make the next card decide "pair or no pair", so the noise exceeds the signal. Hold'em turn-to-river swings are smaller relative to the signal (the refuter's proxy gives noise 0.2–0.33 pot against signal about 0.5). A per-card control variate (subtracting the card's checkdown value) was not tested.
* **What the document should say.** The 48× lever is a per-label cost ratio, not an accuracy-at-equal-cost ratio. It must be measured, and in this toy it is below 1×.

## T3. What can be certified with 3 players? (claim 8)

**Setup.** `toy/multi/experiments/t3_guarantees.py` and `t3b_paranoia_price.py`, on Tri-Leduc (main game, N = 3, D = 12).
* `team.py` turns "hero vs the other two seats colluding **with shared cards**" into a 2-player zero-sum game on the same public tree. The coalition's fiber is the 144 ordered card pairs.
* Checked: embedding the 3p self-play strategy as a non-colluding coalition reproduces the 3p EVs to 1e-16.
* The security level of a fixed strategy is one coalition best-response pass. The coalition maxmin comes from 1500 CFR+ iterations and is bracketed between the security of the average hero strategy and the hero's best response to the average coalition.

**(a) Self-play multiplicity.** 8 CFR+ self-play runs, 2000 iterations each, with exact card removal. Run 0 is M2's setup; runs 1–7 use random initial regrets and a random update order.

| quantity | value |
|---|---|
| NashConv of each run's average profile | 0.87 – 1.58 mA (median 1.12) |
| per-seat EV range across runs (antes/hand) | seat 0 −0.1029…−0.1016, seat 1 −0.0234…−0.0229, seat 2 0.1250…0.1258 |
| strategy difference between runs (reach-weighted mean L1 per decision, per seat; max 2) | seat 0 0.10, seat 1 0.11, seat 2 0.17 |
| NashConv when **one seat's average strategy comes from another run** (56 pairs × 3 seats) | min 4.4, median 15–34 by seat, **max 109 mA** |
| NashConv when **all three seats come from three different runs** (5 triples) | 53 – 77 mA |

**(b) Coalition maxmin and security levels** (antes/hand). Check/folding every hand guarantees −1, the ante.

| seat | self-play EV | security of self-play strategy, 8 runs (min … max) | coalition maxmin v_i (bracket) | maxmin strategy's EV vs self-play others | EV change from playing maxmin |
|---|---|---|---|---|---|
| 0 (first to act) | −0.102 | −1.151 … −1.019 | −0.2114 … −0.2110 | −0.1186 | −16.7 mA |
| 1 | −0.023 | −1.137 … −0.939 | −0.1641 … −0.1629 | −0.0511 | −27.9 mA |
| 2 (last to act) | +0.125 | −0.891 … −0.546 | **+0.0061 … +0.0073** | +0.1028 | −22.2 mA |
| sum over seats | 0 | | **−0.369** | | |

**Verdict on claim 8.**
* **Supported.** "The product of average marginals carries no security value" and "V_k is defined only operationally":
  * The 8 runs agree on values to about 1 mA, yet their strategies differ by 5–9% per decision.
  * Recombining marginals from different runs raises NashConv 3–100× (4.4 to 109 mA) above the single runs (0.87–1.58 mA).
* **The self-play strategy has essentially no worst-case guarantee.** Against a card-sharing coalition, its security is −0.55 to −1.15 antes/hand. For seats 0 and 1 that is usually *worse than check/folding everything* (−1).
* **Two statements in the claim are wrong.**
  * *"The coalition maxmin is at most the cost of folding every hand."* Folding (−1) is a **lower** bound, and the maxmin is far above it (−0.21, −0.16, +0.006).
  * *"The coalition maxmin is typically negative for a player in the blinds."* It is not negative for every seat: the last-to-act seat has a slightly **positive** maxmin even against full collusion. The math refuter's "v_i ≤ 0" is also false per seat. Only the **sum over seats** (an orbit) is ≤ 0, here −0.369.
* **"Never an exploitability certificate" is too strong.** A computable worst-case certificate exists: one team best-response pass. A strategy can raise it from about −1 to the maxmin at a modest price: 17–28 mA/hand against the self-play population.
* **What the document should promise for N ≥ 3:**
  * measured NashConv;
  * per-player regret;
  * a computable but conservative coalition-security number, with the maxmin as an optional, priced dial;
  * never "unexploitable".

## T4. Dropping the folded players' card removal (claims 1 and 9)

**Setup.** `toy/multi/experiments/t4_folded_removal.py`.
* The approximation, "drop ν": at terminals, the reach vector of every **folded** opponent is replaced by its mean over non-board cards. This treats the dead cards as uniform, which is what handing the pot to an L-live HU or 3-way solver does. Live opponents keep exact card removal.
* `drop_nu_post` applies this only after the board, where the live-count principle hands off to V_L. The street-1 trunk, including all-fold terminals, stays exact.
* Same CFR+ budgets as M1 part A. Every result is scored by exact NashConv.

| game | evaluator | iterations | ms/iter | NashConv mA | × exact | terminal rel. L1 err (at exact eq.) | max conditional-EV err mA (at exact eq.) |
|---|---|---|---|---|---|---|---|
| N = 3, D = 12 | exact | 2000 | 27.6 | **1.12** | 1 | 0 | 0 |
| N = 3, D = 12 | drop ν (all terminals) | 2000 | 29.0 | **199.5** | 178 | 8.2e-2 | 387 |
| N = 3, D = 12 | drop ν post-board only | 2000 | 28.3 | **143.9** | 128 | 8.1e-2 | 156 |
| *M1 reference:* N = 3 | independent (live opp–opp removal ignored) | 2000 | 34.5 | 72.3 | 65 | 7.4e-2 | 305 |
| N = 4, D = 8, cap 1 | exact (Möbius) | 400 | 86.1 | **47.3** | 1 | 0 | 0 |
| N = 4, D = 8, cap 1 | drop ν (all terminals) | 400 | 87.5 | **1141** | 24 | 2.5e-1 | 339 |
| N = 4, D = 8, cap 1 | drop ν post-board only | 400 | 84.2 | **1406** | 30 | 2.5e-1 | 371 |
| *M1 reference:* N = 4 | independent | 400 | 38.3 | 1127 | 24 | 7.1e-1 | 3200 |

**Verdict on claims 1 and 9.**
* **The belief state must include the folded players' reach vectors.** That is the refuters' correction to claim 1: n counts every dealt player. Ignoring the folders' dead cards is a bias that iterations do not remove. Here it costs more than ignoring card removal between live opponents (144–200 mA vs 72 mA at N = 3; 1141–1406 vs 1127 at N = 4).
* **The live-count principle is therefore not exact.** Even keeping the street-1 trunk exact and approximating only after the board costs 128× (N = 3) and 30× (N = 4) the exact solver's NashConv. Claim 9's "L-player game on a slightly non-uniform deck" needs the non-uniformity carried into V_L, or a declared, measured approximation. "N enters only through the preflop trunk" is false as an exact statement.
* **Caveat on transfer.** In this toy a folder holds 1 of 8–12 cards. In Hold'em a folder holds 2 of 52 and folding ranges are wide, so the per-folder effect is smaller. But 9-max has up to 7 folders, and the refuters estimate 5–12% relative shifts in Hold'em hand probabilities. The toy establishes the mechanism and its sign, not the Hold'em magnitude.

## T5. Is there free exploitation on the equilibrium face? (claim 4)

**Setup.** `toy/hu/experiments/t5_free_exploitation.py`.
* Same as E4: restricted Nash response RNR(p), 600 CFR+ iterations, both seats, seat-averaged, against fixed station, maniac and nit models.
* The p = 0 rows are the E1 blueprint (2000 iterations). The p = 1e-12 rows are plain CFR+ at the same 600 iterations; this is the noise floor for exploitability.

| model | p | EV vs model mA/h | exploitability mA/h |
|---|---|---|---|
| station | blueprint / 1e-12 | 811.9 / 809.2 | 0.41 / 4.15 |
| station | 0.001 / 0.003 / **0.01** | 829.6 / 850.5 / **988.0** | 4.34 / 4.41 / **4.38** |
| station | 0.03 | 1417.6 | 12.86 |
| maniac | blueprint / 1e-12 | 540.9 / 549.2 | 0.41 / 3.58 |
| maniac | **0.001** / 0.003 / 0.01 | **728.5** / 739.6 / 759.9 | **3.41** / 3.56 / 3.55 |
| maniac | 0.03 | 817.3 | 4.68 |
| nit | blueprint / 1e-12 | 421.6 / 428.1 | 0.41 / 3.90 |
| nit | 0.001 / 0.003 / **0.01** | 492.9 / 513.2 / **565.6** | 3.85 / 4.06 / **4.27** |
| nit | 0.03 | 707.8 | 7.19 |

For comparison, E4 at p = 0.25 gives station 2861.8 / 154.4, maniac 1178.0 / 47.1, nit 1393.2 / 108.1 (EV / exploitability, mA/h).

**Verdict on claim 4.** The refuters' correction holds in a real poker toy, and the effect is large. Selecting among near-equilibria by the forecast gains EV against the model with no increase in exploitability beyond the solver's noise floor:
* +179 mA/h against the maniac at p = 0.001 (exploitability 3.41 vs 3.58);
* +179 against the station at p = 0.01 (4.38 vs 4.15);
* +138 against the nit at p = 0.01 (4.27 vs 3.90).

That is 5–20% of the whole Nash-to-best-response EV gain, obtained for free: 7.7% for the station (812 → 3143 in E4), 20% for the maniac (541 → 1429) and 4.7% for the nit (422 → 3344). Only beyond that point, roughly p ≥ 0.03 here, does exploitation start costing exploitability.

The document should state the frontier as two parts:
1. A free part: the best member of the equilibrium face against the forecast, the RNR limit as p → 0+.
2. A priced part: beyond the face, a declared epsilon, confidence level or gift budget sets the price.

"Caution from epsilon, never from data" applies only to the priced part.

## How to run

All runs are numpy-only and use 4 processes.

```sh
cd toy/hu        # needs results/blueprint_eff*.npy and e1.json from E1
python3 experiments/t1_prechance_ceilings.py                     # ~4 min (5 variants x 2 seats)
T1_MODES=chance_root,start_min T1_LOG=t1b.log T1_OUT=t1b.json python3 experiments/t1_prechance_ceilings.py   # ~2 min
T1_K=1000 T1_MODES=percard,start_cfv T1_LOG=t1c.log T1_OUT=t1c.json python3 experiments/t1_prechance_ceilings.py  # ~7 min alone
python3 experiments/t2_prechance_labels.py                       # ~3.5 min
python3 experiments/t5_free_exploitation.py                      # ~5 min alone
cd ../multi/experiments   # needs results/m2_sigma_3p.npy from M2
python3 t3_guarantees.py          # ~4 min alone (7 min measured under contention)
python3 t3b_paranoia_price.py     # ~2-3 min
python3 t4_folded_removal.py      # ~2 min
```

Measured wall-clock: 247 s, 117 s, 677 s, 204 s, 538 s, about 420 s, (not timed), about 120 s, in the order above. About 40 minutes in total. Several of these runs overlapped on the 4-core machine, so T1c, T3 and T5 are inflated by contention: T1c ran over 10 minutes only because of it.

## Limitations

* **T1.** Uses oracle (exact) leaves to isolate the ceiling question. With learned leaves, E2's leaf error would come on top. The `avg` translation is one natural way to turn a pre-chance output into per-card ceilings. Any translation that preserves only the chance average has the same defect, but other translations were not tested.
* **T2.** The 9-card chance node and 1-card hands make per-card swings extreme. Results come from one net size and one range generator. No control variate was tested.
* **T3.** The coalition shares cards, which is the strongest adversary. Security against non-communicating colluders is at least as high and was not computed. 8 runs, 1 game.
* **T4.** Toy decks make each folder's dead card weigh 8–12% of the deck. Hold'em magnitudes will be smaller per folder.
* **T5.** The exploitability noise floor at 600 iterations is about 3.5–4.2 mA/h. "Free" means no measurable increase above that floor, not exactly zero.
