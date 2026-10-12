# Audit of the existing engine and its optimisation within $10-120

Stage 2 deliverable. The engine is the user's `poker/` package (JAX, heads-up no-limit hold'em, re-solving with learned street-boundary values), audited against `docs/poker-core-mechanism.md` with the toy results as reference. Everything measured here ran on a shared 4-core CPU box without a GPU, the user's caches, test package or trained critics; where a number is an inference from the code's own recorded GPU figures or the auditors' estimates it says so. Audit scripts and raw results are under `tools/audit/` and `results/audit/`; new tests under `tests/`.

Contents: 0 verdict; 1 what was done; 2 the re-solving gadget (measurement on the exact river game); 3 the Slumbot card and history path; 4 engine correctness by area (4.1 rules, 4.2 solver, 4.3 training and critics, 4.4 live path against the specification, 4.5 live path and Slumbot runner); 5 the trained-V failure; 6 economics and units; 7 GPU-first on the home hardware; 8 keep / fix / reuse / missing; 9 budget verdicts; 10 the non-negotiables; 11 contradictions and open items; appendices A-E (reproduction, the 22-item plan with status, disagreements, open questions, sources).

## 0. The verdict in one page

**What the engine is.** A correct, exact, heads-up solver and rules engine with a DeepStack-shaped value ladder and a live re-solver. Five independent audits (rules engine, solver core, training loop and critics, live path and Slumbot runner, specification compliance) found the two-player mathematics exact: terminals to 1e-14, the showdown kernel in the spec's O(F) form, the all-in matrices, the card-removal identities, the accounting chain, the certificate, the rules on 110 000 hands and 18 000 decisions, the protocol parser on 105 684 fuzzed histories. Nothing in it was rewritten for uniformity.

**What was wrong, and what was done.**

1. **The live path carried the opponent's range** (a Bayesian posterior with a 1e-3 floor) where the specification requires per-hand counterfactual-value ceilings; its own documentation listed the safety gadget as out of scope. Measured on the real river tree with exact leaves: range-only re-solving loses +16 to +942 mbb/hand against the full solve, and with the kind of wrong range the floor produces +1.1 to +7.7 bb/hand. The re-solving gadget is now in the solver and the live player (ceilings from the post-card critics at every street start, a Terminate/Follow root, the certified gap of the gadget game); with exact ceilings it reproduces the full solve within +0 to +6 mbb/hand in eight of nine cases (+37 in one; range deal) and stays below +40 in eight of nine wrong-range cases (the mixed deal that is now the default). Its requirement on the critics is quantified: about 0.01-0.02 pot of **per-hand** ceiling error; the engine's best recorded critic is at 0.043 and the check-down baseline at 0.07-0.11 (first exact end-of-turn labels, new validator).
2. **The engine's quality record was circular**: every end-of-street critic was trained and scored against the card average of another learned critic. `validate_v4.py` now computes exact end-of-turn labels (48 river solves) and scores any critic against them.
3. **The Slumbot evaluation could not finish**: one decision failure stopped the match and blocked resume, and the runner forced the CPU, where every postflop decision exceeds the 4 s budget (measured 5-13 s). Both fixed. The card-parsing crash itself was **not reproduced**: the parser, the card mapping and the replay are verified correct offline (18 histories, 105 684 fuzzed prefixes, 1 500 simulated hands); slumbot.com is unreachable from this box; the failed run's own hand records will name the payload.
4. Three latent engine defects fixed (illegal actions applied silently under jit, an int32 overflow in the rake split of large side pots, a history-capacity no-op), one solver throughput item taken (fold terminals: 1.13x per river iteration), cache paths resolved at call time.

**Point 3, the trained-V failure.** "18x" was the ratio of two exploitabilities of the whole toy agent (417.7 vs 23.7 mA/h), produced by a value map with 7.5-11 % of the leaf pot of regression error. The measured curve (toy, exact everything else): an iid value error of x % of the leaf pot costs about 0.93 x % of the initial pot per hand between 1 % and 30 %; 0.3 % costs +0.15 %; learned nets cost 1.4-2.5x what iid noise of the same size costs. On the real river tree, 1 % per-hand ceiling error costs +1 to +64 mbb/hand (mostly below +11), 5 % up to +207, and a 5 % bias up to +324.

**Point 4, units and money.** 0.3 % of a 100 bb pot is 0.3 bb per hand on that decision's values, not 30 bb; 30 bb/100 appears only at a 30 % error. The reference is the exact best response (computable exactly in the toy and inside a street or the whole river here; LBR is the proxy for the whole game); regression error and exploitability are linked only by the measured curves above. Measured costs: a river label at T 800 is 11.6 core-seconds on this CPU, 0.86 s on the GTX 1070, about 0.3 s on a 3090; $10-15 buys 90 000-375 000 river labels on a rented 3090, the home 1070 adds 100 000 per day for free; DeepStack's 10^7 turn labels would cost $900-2 600. The spec's CPU unit prices were 15-60x too optimistic for this engine's JAX path; its GPU path is the one to use.

**Budgets.** HU $10-15: the compute is enough for everything except the unknown, which is whether a critic reaches about 0.01-0.02 pot per hand within 10^5-10^6 labels (a power law from the recorded 0.043 at 45 k says no; model and data fixes may say yes). The gate that decides it costs a day of the home 1070 and is specified in section 9.1. 6-max: no cost can be given; the solver stops at three players and has no 3-player critics or side pots. 9-max within min(fact, $120): not impossible in principle, infeasible with today's engine within $120 unless the heads-up gate passes, the home GPU does the labelling and 3-way pots use the coarsest menu with declared approximations for 4+ players; the multiway code does not exist yet.

**Where time went** (CPU, per PCFR+ iteration): regret bookkeeping on padded arrays 36-45 %, backward 27-29 %, terminals 21-27 %; a compact edge layout is 10.7-28x faster on the bookkeeping part (the first throughput item after this stage); dead subtrees at low SPR solve 2-10x more decisions than are alive; labels run 4-7.5x the spec's iteration count.

## 1. What was done in this stage

Code (all under `poker/`, the user's engine; nothing was rewritten for uniformity):

| Change | Files | Kind | Validation |
|---|---|---|---|
| Re-solving gadget (CFR-D / DeepStack) in the live path: per-hand opponent ceilings at every street start from the post-card street critics, Terminate/Follow at the root of every live solve | `poker/solver/pcfr.py` (gadget_follow, gadget_ceiling, gadget_scale, the step, exploitability of the gadget game), `poker/solver/subgame.py` (gadget inputs, Follow output), `poker/solver/critics.py` (load_post_critics), `poker/player/live.py` (ceilings, leaf reach, warm-up), `poker/arena/*` (flags) | missing component of the spec (section 2.4), now present | `tests/test_gadget.py`, `tests/test_warm_up_gadget.py`, `tests/test_live_gadget_hand.py`, `tests/gadget_river_resolve.py` (the measurement, section 2) |
| Loud illegal actions in the engine under checkify; the Slumbot replay guards every event on the host before the jitted step | `poker/envs/transition.py`, `poker/arena/slumbot/live_bot.py` | correctness (envs audit F3) | checkify reports a masked grid action, an over-stack total and a sub-minimum raise; legal actions pass |
| Rake split int32 overflow: the table guard refuses raked tables whose chips x the largest cap exceed 2^31; docstring corrected | `poker/envs/evaluator.py`, `poker/envs/guards.py`, `poker/envs/betting.py`, `poker/envs/config.py` | correctness (envs audit F2) | 9 x 300000 chips with rake refused, without rake accepted (the solver's deep roots), 6 x 500 BB with rake accepted |
| History capacity: the live path refuses a step the engine would drop silently; the bound's docstring now says what it proves (grid raises) and what it does not (exact min-raise wars beyond ~850 BB heads-up) | `poker/player/live.py`, `poker/envs/config.py` | correctness (envs audit F1) | - |
| Slumbot card and history path, offline conformance test | `tests/test_slumbot_replay.py` | test | 18 histories across all streets and both seats replay onto the engine, 6 malformed responses are refused, 52 card strings round-trip |

Experiments (results under `results/audit/`):

- `results/audit/gadget/river_resolve_T800_trunk{13,34,32}.json` and `crushed_trunk{13,34,32}.json`: safe re-solving on the real HU river tree with the right and a wrong carried range, section 2.
- `results/audit/v4/validate_v4.json`: the first exact end-of-turn labels of the engine, section 5.5.
- `toy/hu/results/X1_NOISE_CURVE.md`: value error to exploitability curve in the toy, section 5.
- `results/audit/engine-profile/summary.json`: where the solver's time goes on this CPU, section 6.
- `results/audit/{envs,solver,training,live,spec}/`: the auditors' checks (scripts under `tools/audit/`).

Not done, and why: no run against slumbot.com (the box's network policy refuses the host; section 3), no training run (no GPU here, the user's caches and tests were not in the upload), no 6-max or 9-max code (the solver's terminals stop at 3 players; the audit says where).

## 2. The live path: the re-solving gadget

### 2.1 What the engine did

`poker/player/live.py` solved every street from its start with OUR exact range and the opponent's **Bayesian range**: his range at the previous street's end times the likelihood of every action he took under the solve's average strategy, floored at 1e-3 per hand (`_update_opponent`). The module documentation listed the safety gadget as "out of scope". The specification (section 2.4) and the toy (E2: 520 vs 23.7 mA/h, T1: 25.2 vs 608) say the re-solve must carry the opponent's **per-hand counterfactual-value ceilings** instead: at the root he chooses per hand between Terminate (take the ceiling) and Follow (play the subgame), so our strategy is safe against any opponent whose values at the street start are at most the ceilings (Burch, Johanson, Bowling 2014; DeepStack).

### 2.2 What the fix does

- `pcfr.py`: a live spot carries `gadget_ev` [n, 1326] (whole-hand EV per hand of every seat, street-start pot units; the solver multiplies by our range's compatible mass to get CFV units) and `gadget_mask` [n]. Each PCFR+ iteration scales the masked seats' deal ranges by their current Follow probability (RM+ over Follow/Terminate, predictive like the nodes), computes the regrets of the two gadget actions from the root counterfactual values (which do not depend on the seat's own reach), and accumulates the quadratic-weighted average Follow. `exploitability()` with `follow` is the exact best-response gap of the gadget game. The teacher's spots have no gadget keys and compile to the same program as before (test 1 of `tests/test_gadget.py`: bitwise equal policies with the mask off).
- `live.py`: at every street end the reach of both seats at the leaf is recorded (`_leaf_reach_from_solution` / `_leaf_reach_from_table`: ours exact, his = deal x Follow x his average action probabilities along the path). At the first solve of the next street `_gadget()` evaluates the new street's post-card critic (`critics.load_post_critics`: the loop's `flop_critic`, `turn_critic`, `river_critic`) at that leaf for the card that was dealt, with the exact check-down input F from the street's all-in matrix or rank tables; the opponent's value per hand, converted from relative to whole-hand accounting, is the ceiling. One critic call per street start (27-64 ms on this CPU, compiled at the warm-up: `warm_gadget`). The Bayesian update stays as the diagnostic belief and, with `gadget_deal="range"`, as his deal range.
- The compiled set stays closed: a gadget solve after the warm-up compiles nothing (`tests/test_warm_up_gadget.py`; the one 8 ms compile found on the way was a dtype conversion on the device, removed).

### 2.3 Measurement on the real river tree

`tests/gadget_river_resolve.py`: the full HU river game (`config.tree_rule(2, 3, "normal")`, 117 nodes, exact showdown terminals) is solved on three training spots (T = 800 PCFR+ iterations; E_full = 1.4e-3, 1.1e-3, 2.8e-3 pot; pots 74, 72 and 11 BB at SPR 2.2, 2.3 and 17.6). Trunk: seat 1 checks, seat 0 bets (1/3, 3/4 or 3/2 of the pot: three trunks); the subgame (18 nodes) starts where seat 1, the hero, faces the bet. The hero's subgame strategy is re-solved (T = 800) and spliced into the full strategy; the exact exploitability of the spliced profile in the FULL game minus E_full is the price of the re-solving method. Everything is exact; nothing is sampled or learned. Results in `results/audit/gadget/`.

Gap over the full solve in mbb/hand (pot units x 1000 x pot in BB), the three spots per trunk and the range over all nine cases:

| Re-solve at the subgame root | trunk bet 1/3 pot | trunk bet 3/4 pot | trunk bet 3/2 pot | all nine |
|---|---|---|---|---|
| **range only** (what the engine did: his equilibrium reach, no ceilings) | +481 / +347 / +288 | +189 / +53 / +16 | +942 / +40 / +254 | +16 .. +942 |
| gadget, uniform deal, exact ceilings | +3 / +4 / +2 | +89 / +60 / +23 | +21 / +14 / +31 | +2 .. +89 |
| gadget, uniform deal, ceilings + iid noise 0.01 pot per hand | +22 / +6 / +2 | +148 / +160 / +30 | +11 / +21 / +32 | +2 .. +160 |
| same, noise 0.02 | +44 / +21 / +3 | +157 / +287 / +43 | +4 / +47 / +33 | +3 .. +287 |
| same, noise 0.05 | +67 / +35 / +12 | +230 / +104 / +151 | +25 / +92 / +117 | +12 .. +230 |
| gadget, uniform deal, ceilings biased -0.05 pot | +69 / +73 / +41 | +202 / +493 / +108 | +99 / +190 / +32 | +32 .. +493 |
| same, biased +0.05 | +55 / +217 / +54 | +415 / +407 / +102 | +172 / +213 / +63 | +54 .. +415 |
| gadget, ceilings = his check-down EV (no strategic content) | +333 / +595 / +117 | +2046 / +1942 / +516 | +3002 / +2842 / +802 | +117 .. +3002 |
| **gadget, range deal, exact ceilings** | +1 / +1 / +0 | +6 / +3 / +6 | -0 / +37 / +0 | -0 .. +37 |
| gadget, range deal, noise 0.01 | +64 / +8 / +2 | +11 / +4 / +3 | +1 / +5 / +0 | +0 .. +64 |
| same, noise 0.02 | +55 / +7 / +3 | +77 / +265 / +9 | +6 / +10 / +2 | +2 .. +265 |
| same, noise 0.05 | +179 / +7 / +11 | +116 / +40 / +58 | +207 / +34 / +6 | +6 .. +207 |
| gadget, range deal, biased -0.05 | +274 / +96 / +32 | +114 / +36 / +14 | +505 / +49 / +4 | +4 .. +505 |
| same, biased +0.05 | +13 / +198 / +27 | +230 / +324 / +321 | +18 / +109 / +20 | +13 .. +324 |

Reading:

1. **Range-only re-solving, the engine's path before this stage, is exploitable by +16 to +942 mbb/hand on one river subgame** (median about +190); its worst cases are the unusual trunk sizes, where "his range at this node" is least informative. That is the hold'em-scale counterpart of the toy's 520 vs 23.7 mA/h.
2. **The gadget with the range deal and exact ceilings reproduces the full solve** (+0 to +6 in eight of nine cases, +37 in one), and so does the uniform-deal gadget within the solves' convergence error in most cases (+2 to +89). The range deal (his carried range is the deal, Terminate can only remove hands; Burch, Johanson, Bowling 2014's original form) is the best when the carried range is right; see below for when it is not.
3. **The gadget needs per-hand ceiling accuracy of about 0.01 pot.** iid noise of 0.01 pot per hand costs +1 to +11 mbb/hand in eight of nine cases (+64 in one); 0.02 costs up to +77 (one outlier +265); 0.05 up to +207. A **bias** is the dangerous direction: ceilings 0.05 pot too high make him Terminate with everything and the hero plays against an empty range (+13 to +324); ceilings 0.05 too low make him Follow with everything, which is range-only re-solving again (+4 to +506).
4. A crude value as ceiling (his check-down EV, 0.38-0.48 pot RMS error) costs 0.1 to 3.0 bb/hand: the toy's 533 mA/h check-down leaf has its hold'em counterpart.

**When the carried range is wrong.** The spec auditor showed the live player's Bayesian update with its 1e-3 likelihood floor leaving the opponent's bluffs at 0.25 % of his range and the hero folding every bluff-catcher (air gains +1 pot per hand; `results/audit/spec/carried_range_demo.json`). The same experiment with that wrong range (his 30 % lowest-equity betting mass crushed to 1e-3 of its weight; `results/audit/gadget/crushed_trunk*.json`):

| Re-solve at the subgame root | trunk bet 1/3 pot | trunk bet 3/4 pot | trunk bet 3/2 pot | all nine |
|---|---|---|---|---|
| **range only, wrong range** (his bluffs at 1e-3 of their weight: the likelihood floor's effect) | +5391 / +6024 / +1332 | +7580 / +7736 / +1803 | +4820 / +7378 / +1119 | +1119 .. +7736 |
| gadget, range deal, wrong range, exact ceilings | +4238 / +2474 / +641 | +708 / +1600 / +429 | +1 / +26 / +25 | +1 .. +4238 |
| same, ceilings + noise 0.01 | +4143 / +3274 / +645 | +863 / +1834 / +345 | +17 / +95 / +19 | +17 .. +4143 |
| gadget, uniform deal (no range), exact ceilings | +3 / +4 / +2 | +89 / +60 / +23 | +21 / +14 / +137 | +2 .. +137 |
| same, ceilings + noise 0.01 | +22 / +6 / +1 | +128 / +232 / +28 | +11 / +21 / +35 | +1 .. +232 |
| **gadget, half wrong range + half uniform, exact ceilings** | +0 / +2 / +1 | +37 / +161 / +42 | +1 / +7 / +4 | +0 .. +161 |
| same, ceilings + noise 0.01 | +4 / +18 / +1 | +105 / +147 / +24 | +1 / +21 / +4 | +1 .. +147 |

The range deal inherits the wrong range (Terminate cannot add hands back); the uniform deal never sees it but pays for ceiling noise; **the half-and-half deal keeps the range's information when it is right and full support when it is wrong**: below +40 mbb/hand in eight of nine cases with exact ceilings and in seven of nine with 0.01 pot of noise. It is the live default (`gadget_deal="mix"`, `gadget_mix=0.5`); "range" and "uniform" stay as options for A/B matches.

Consequences for the engine: `gadget_deal="mix"` is the default; `--no-gadget` keeps the old path for A/B matches; the critics' **per-hand** error on the hands in range (the loop reports `worst_substantial`, `centred`, `offset`) is the number the gadget needs at about 0.01-0.02 pot, and the recorded river critic errors (test MAE 0.043 on the 45k set, 0.063 on k0b) are 2-6x above it. Until a critic reaches that, the gadget's safety claim is approximate and the duplicate self-play match (`poker/arena/match.py`, gadget on vs `--no-gadget`) is the arbiter; this is a measurement the GPU box must run (section 6.5).

What the measurement does not cover: the turn and flop (their leaves use critics, so an exact splice test needs the exact turn+river solve the spec's gate describes, 10-40 s wall each on 4 cores; not run here), and the compounding over streets that the toy measured (a re-solve at every street with ceilings from the previous one).

## 3. The Slumbot card and history path

The bot could not be run against slumbot.com from this box (the egress proxy refuses the host). What could be checked offline, `tests/test_slumbot_replay.py`: response payloads of the shape documented by slumbot.com/sample_api.py (card strings like "Ac", "Th"; `client_pos` 0 = BB, 1 = SB; the action string with and without the street separator) go through `protocol.parse_response`, `LiveSlumbotBot._hand_start` / `_replay` (the engine stepped through every event at its exact chips) and back through `serialize_action`:

- 18 histories (preflop opens, 3-bets, 4-bets, flop/turn/river with on- and off-menu bets, a minimum bet, an open shove, a called shove with and without separators, limps) replay onto the engine with the right seat to act, street, stacks, street bets and visible board; every candidate reply of the engine (fold, call, minimum raise, a middle raise, all-in) serialises and parses back as a continuation.
- 6 malformed responses are refused: not the client's turn, a missing board, a duplicate card, a hole card on the board, an invalid card string, an under-minimum raise that is not all-in.
- All 52 card strings round-trip through `card_id` and `our_card`.

So the crash is not reproducible from the code as uploaded with payloads of the documented shape. Two leads the user can check in their run directory: (1) the runner writes every request and response (redacted) into `<out>/hands/*.json` with the `Protocol failure: ...` reason; that record names the exact payload; (2) `poker/envs/transition.py` referenced `poker/arena/slumbot/bot.py`, a module not in the upload: if the crash happened with that older adapter, it is not the path audited here. One hardening went in regardless: the replay now runs `guards.target` before every jitted step (a payload the parser accepts but the engine's rules reject is refused with a ProtocolError instead of being applied silently; envs audit F3).

## 4. Engine correctness: what the audits found and what was fixed

### 4.1 Rules engine (`poker/envs`)

Verified (auditor's independent references, scripts under `tools/audit/envs/`, results under `results/audit/envs/`): the 7-card evaluator is exact on 110 224 hands including all 49 205 rank multisets; settlement (side pots, uncalled bets, odd chips, rake) matches an independent reference on 2 000 random 2-6 seat hands and 1 239 raked hands; legality, minimum raise and all-in targets match on 18 000 decisions; blinds and first actor are right for 2-6 seats; exact opponent totals are applied unchanged by `step_target` (an opponent bet inside the commit-merge zone is applied exactly: no snapping of observed bets); stacks 20-500 BB run on one path at 2, 6 and 9 seats; every module imports under Python 3.13. The engine is not a bottleneck anywhere (a batched `step_target` costs 0.25 us per table).

Fixed in this stage (all three were latent in regime 1; none changes a legal hand):

| # | Finding | Fix | Where |
|---|---|---|---|
| F3 | An illegal action index or an exact total the rules forbid was applied silently by the jitted engine (a masked raise created a chip; a 5000-chip raise with a 300-chip stack left stack -4700); `checkify` did not report it; the Slumbot replay stepped the engine without the host guard | `checkify.debug_check` of the offered mask in `apply_action` and of the exact-total predicate (the same as `guards.target`) in `apply_target`: reported under checkify, legal actions untouched; the replay calls `guards.target` before every jitted step and refuses with a ProtocolError; stale docstring (a non-existent `bot.py`) corrected | `poker/envs/transition.py`, `poker/arena/slumbot/live_bot.py` |
| F2 | `rake_split` forms cap x base_k in int32; above about 2.1 M chips in a side pot (with the 10 BB cap) the rake is misattributed between main and side pots (717 chips on a 6 M-chip example); the docstring claimed int32 safety | `guards.table` refuses a **raked** table whose chips in play x the largest cap reach 2^31 (`config.RAKE_CAP_MAX_CHIPS`; 2 147 BB at 9 seats); unraked tables (the solver's 10^8-chip deep roots, regime 1) are unaffected; docstring states the bound | `poker/envs/guards.py`, `poker/envs/betting.py`, `poker/envs/config.py`, `poker/envs/evaluator.py` |
| F1 | `config.history_bound` is a proof for grid raises only: exact min-raise wars need up to n x stack / BB decisions; measured: 1000 BB heads-up stalls at 832 of 832 history slots and the step becomes a silent no-op; 500 BB holds at 2, 6, 9 seats with 7-10 % margin | the live path checks the history capacity on the host before every step and raises instead of letting the engine drop the step (`check_legal`); the docstring now says what is proven and the measured limit; the array sizes are unchanged (a chip-based bound would multiply the history arrays, which every step copies, by 3-9x) | `poker/player/live.py`, `poker/envs/config.py` |

Not fixed, by decision (heuristics that cost EV, not correctness; they need a measurement, not a patch):

- `COMMIT_MERGE_POT = 0.25` (a size that leaves at most a quarter of the pot behind becomes all-in) diverges from the spec's example rule (67 % of the stack) and prunes the opponent's modelled near-all-in sizes as well as ours; the auditor's sweep puts the effective threshold at 60-80 % of the stack depending on SPR. `STREET_SIZES`, `TREE_CAPS`, `MENU_BOUNDS` are declared heuristics ("a poker heuristic, not the result of a measurement", config.py) with no recorded cost. The tool to price them exists (`poker/solver/sizing_lab.py`); the measurement (own-menu loss in decision units against a richer menu on exact river/turn solves) is a GPU-box job and is listed in section 9.
- The 52-card deck is hard-coded in the evaluator (variants would need a plug-in); `menu_index` uses float32 and disagrees with `menu_for` above 2^23 chips (stacks are at most 50 000); history arrays make every batched step O(capacity) (the solver already strips them).

Multiway: rules verified for 2-6 seats; the multiway caps are placeholders and `LOOP_PLAYERS = 2` (the training loop deals two seats). The engine is one foundation for the rules; the solver and the loop are not yet (section 8).

### 4.2 Solver core (`poker/solver`: pcfr, showdown, checkdown, runout, tree, situations)

Verified exact by the auditor's independent brute force (`tools/audit/solver/`, `results/audit/solver/`): fold and showdown terminals (1e-14 relative in float64, 6e-7 in float32), the uncalled excess, the turn all-in matrix against explicit 44-river enumeration (5e-8), the 3-player terminals for every live pattern at equal contributions (1e-15) including folded players' dead cards, the exact 3-player turn runout (/42), the chance-averaging identity (the engine's `card_average` with mass weights equals the spec's 1/44 and 1/45 forms to 3e-7), the accounting chain label -> critic -> leaf (whole-hand vs relative), the best-response certificate and its invariants, the zero-sum identity (8e-8), rank7's combinadic index. The spec's O(F) prefix-sum showdown is implemented and is 8x a dense matvec at batch 1. The two-player solver is correct; nothing in it was changed for correctness.

Findings and what was done:

| Severity | Finding | Decision in this stage |
|---|---|---|
| major (multiway only) | `_three` pays the whole pot by 3-way share: **no side pots**; unreachable today because every street root gives equal stacks (`tree.street_root`); 101 % error measured on a constructed (3, 3, 1) contribution case | not fixed: multiway with unequal stacks is not yet a code path; listed as the first multiway item (per-seat stacks in `street_root`/`replay`, a layered side-pot payoff in `_three`, the auditor's D-side case as the regression test) |
| major (throughput) | regret/average/last arrays padded to 13 action slots (114 real edges of 520 slots on the river: 78 % padding); backward gathers [L, 13, n, H]; about 70 % of a river iteration is this traffic | not changed (a memory-layout change of the whole solver); the measured ceiling is 1.6-1.8x per iteration (section 6.3); first throughput item after this stage |
| major (throughput) | dead subtrees (illegal at the real chips) solved at full cost: at SPR 2 only 12 of the 40 river decisions are alive, at SPR 4 24, at SPR 50 40; 60 % of training spots are at SPR 0.25-16 | not changed; prune per SPR band or group batches by alive signature |
| major (cost) | T = 800 / 1500 iterations per label with simultaneous PCFR+ updates; E on HU river spots: 0.3 % of pot at SPR 2 after 300 iterations, 0.05 % after 800, 0.01 % after 2000 (log-log slope -1.6 to -1.7; SPR 30: 1.1 % / 0.33 % / 0.1 %) | not changed; the measured curve is now the basis for T per SPR band; alternating updates and a DCFR option are the auditor's recommendation, to be chosen by E against wall time on the GPU |
| major (scope) | no solver for 4+ players; 3-player street-end leaves and check-down raise NotImplementedError; the tree builder goes to 6 players but nothing solves those trees | not fixable in this stage; see the 9-max verdict (section 9) |
| minor | 3-player flop all-in runouts are a seeded ratio estimate over 48 of 1176 turn-river pairs (bias O(1/k), SE function exists but is not reported) | declared approximation; report the SE per solve when 3-way play starts |
| minor | fold terminals ran the full showdown kernel for the compatible mass (38 of 77 river boundaries) | **fixed**: `_two` uses the board-free `j_mass` at folds (equal to 1e-14 per the audit); before/after timing in section 6.3 |
| minor | one 7 MB float32 all-in matrix per spot even when spots share a flop; the flop cache is 12.3 GB (1755 x 7 MB) against the spec's 4.8 GB fp16; the training cache is keyed by the raw sorted flop, the live path by the canonical class | not changed; deduplicate per batch, key by class everywhere, store 1176^2 fp16 after a precision check |
| minor | the HU turn tree is 40 decisions with 19 leaves (three opening sizes), not the 28 / 13 of the spec's cost model: 1.46x the critic calls | carried into the cost figures of section 6 |
| note | float32 floor of about 1e-6 pot on E; production E is a diagnostic (computed once at the end), not a stop rule; labels for hands we never reach are on-policy values under uniform default play; certificate mode keeps computing finished spots | documented; the zero-reach label convention is a V-quality decision for the next stage |

Comparison with the specification for this area: 11 DONE, 3 PARTIAL (flop cache format, 3-player exactness, certificate as a stop rule), 2 MISSING (side pots, 4+ players), 5 DIVERGES (labels at 800-1500 iterations instead of about 200; the turn tree's size; a 1326-fiber preflop tree with a learned pre-flop-cards critic instead of the spec's 169-class tree or sampled flops; the padded layout; the V interface per compatible mass instead of per total mass, internally consistent).

### 4.3 Training loop and critics (`poker/solver/loop.py`, `critics.py`, `situations.py`, `learning_curve.py`)

Verified: the residual critic (network + exact check-down F), the range-weighted Huber loss, the 24-fold suit augmentation, the board-class train/test split, the chip-conservation projection (an exact W-orthogonal projection in the solver's own metric; W-weighted MSE never increased in 192 synthetic cases; exact in regime 1 where T = 1), the card-removal identity behind the end-of-street critics (2.8e-5 against an independent brute force), the training protocol's collapse record (10 of 10 runs collapse-free at lr 5e-4 / clip 0.05), the stage machinery (claims, provenance, game hash: it caught the auditor's own fingerprint change).

The one blocking finding, and what was done about it:

- **Nothing in the loop measures a critic against exact deeper values.** `turn_end` is trained and scored against the card average of the *learned* river critic (`loop.end_stage`, `test_mae_vs_exact_average`), `flop_end` against the learned turn critic, and so on up to `pre_end`: two learned maps per street cut, each inheriting the previous one's error (spec section 7.1, item 3). The auditor's tiny end-to-end run makes the circularity visible: the river critic learned nothing (test MAE 0.0818 against 0.0825 for check-down alone, best epoch 0 for both seeds), yet `turn_end` reported 0.0055 "against the exact average" (check-down 0.0104). **No reported critic MAE in the engine's records says anything about play.**
- Added in this stage: `poker/solver/validate_v4.py` computes **exact end-of-turn labels** (all 48 river cards solved on the rule tree, card-removal average with the engine's own weights) for the loop's turn situations and scores against them the exact check-down value (what the residual network has to learn), a `turn_end` checkpoint when given, and the labels' own noise (T against 2T). On one CPU core a situation costs 48 x T x 21 ms (about 7 minutes at T = 400); on the 1070 about 48 x 0.4 s. First numbers are in section 5.5. This is the cheap half of the spec's gate; the decision-unit half (turn solve with learned against exact leaves, best-response gain in the exact turn+river game) is not built: it needs the hero's river play defined at every river card of the test and about 10-40 s wall per situation on 4 cores, and it belongs on the GPU box.

Other findings:

| Severity | Finding | Decision |
|---|---|---|
| major | ranges are synthetic, board-unaware, independent per player (Chen-percentile mixtures with log-normal noise); no recursive board-strength splits, no ranges harvested from the loop's own solves, no perturbed real ranges; the live path's ranges fall outside this distribution | not changed; listed for the next training run (tag every row with its source, report error per source) |
| major | fixed T keeps noisy deep-SPR labels: on the loop's own river trees at T = 800, E = 0.0009 / 0.0021 / 0.0038 / 0.019 pot at SPR 2.3 / 5.4 / 27 / 90, and the labels drift by 0.0015 / 0.0043 / 0.0056 / 0.017 pot (range-weighted) between T and 2T, up to 0.30 pot on a substantial hand; at SPR >= 27 that noise is 15-40 % of the best recorded river critic MAE (0.043) | not changed; T per SPR band or the certificate above SPR 16; the validator reports the noise floor |
| major | on Pascal the end-of-street target computation (`loop.aux_device`) runs on the CPU: measured 0.27 s per turn_end situation and 5.6 s per flop_end situation here, about 4.7 h for the default 3 000 flop_end situations; likely the bottleneck of a home-GPU run | not fixable here (no GPU); the per-card loop needs a kernel that loads on Pascal or one batched critic call over the 52 cards without the fused scan |
| minor | the projection moves error onto the correct player (slot 1 exact, slot 0 off by 0.1: slot 1's MAE becomes 0.05); `KAPPA` for turn/flop/preflop is a placeholder 0 (wrong under rake, exact without rake) | keep; report per-slot offsets; estimate kappa before any raked regime |
| minor | one loss in street-start pot units over SPR 0-99.5: deep rows dominate (test MAE 0.02 at SPR < 1, 0.12-0.13 at SPR 4-16 in the tiny run) | report and weight per SPR band |
| minor | no suit canonicalisation at inference (equivariance from augmentation only) | canonicalise board, ranges and F before the call |
| minor | `learning_curve.py` measures pot MAE only (no decision units, street critics only, PROTOCOL lr rather than the run's) | extend; the validator above supplies exact labels for the end critics |
| minor | cache paths bound at definition time (`rank7.load_table`, `runout.flop_equity_cached`): the auditor's redirect did not take and a 268 MB table landed in results/cache | **fixed**: defaults resolved at call time |
| minor | `sizing_lab.py` values turn/flop/preflop street ends by check-down only, so its menu costs for those streets ignore future betting; `REAL_RIVER` is a relative path to a file not shipped | report only |
| note | default label counts (3 000 river, 2 000 turn, 2 000 flop) are 1-3 orders below what the budget buys; half the river rows are SPR-0 copies with zero residual; `io.wait_for` has no timeout | size the counts from the learning curve; subsample the copies |

Comparison with the specification for this area: 10 DONE (normalised ranges, public numbers, per-mass values, projection, residual on exact F, range-weighted loss, pre-chance cut with one call per leaf, 1/(N_left-4) weights, augmentation, board split), 2 PARTIAL (suit canonicalisation, SPR stratification), 3 DIVERGES (the network is 4.6 M used parameters / 9.3 MFLOP per heads-up call against the spec's 3.3 M / 6.5 MFLOP because F doubles the first layer's input; end labels are averages of learned critics instead of exact or unbiased single-card labels; 800-1500 PCFR+ iterations per label instead of about 200), 5 MISSING (harvested and perturbed ranges, decision-unit validation, the learning-curve gate in decision units, a measured label-noise floor, the exact-vs-single-card exchange rate). The validator added here closes "measured label-noise floor" and gives the exact labels the other three missing items need.

### 4.4 The live path against the specification (spec audit)

The spec auditor's verdict on the engine as uploaded: a heads-up, DeepStack-shaped engine whose solver, terminals, value-ladder structure and off-menu handling match the specification, and whose re-solving layer does not: it carried the opponent's **range** (a Bayesian posterior with a 1e-3 likelihood floor) where the spec requires per-hand **ceilings**. Measured with the real `LivePlayer` at T = 300 on a river spot (pot 20 BB, SPR 5, hero holding only bluff-catchers, opponent betting pot off-menu into the SLOT): when the carried range has the opponent's air at the floor-survivor weight (0.25 % of the range), the hero calls with frequency 1.4e-7 and every air hand gains +1.0 pot; with the true 50/50 range the hero calls 58 % and air loses 0.16 pot. The floor itself distorts beliefs both ways (2 % of hands taking an action at likelihood 0.05 end up with 50 % of the posterior on hands the solve says never take it) and does not protect against a multi-action deviation (three updates on the same zero-likelihood hands leave 3.4e-7 of the mass). This is the finding section 2 fixes; the section 2.3 "crushed range" rows measure exactly this failure on the exact river game and decide the deal.

Other findings of this audit and their status:

| Severity | Finding | Status |
|---|---|---|
| major | re-solves start after the card with carried state; the spec's chance-rooted gadget (turn/river) and precomputed flop-start ceilings do not exist | partly addressed: the ceilings now come from the post-card critic at the leaf for the dealt card (the DeepStack construction), which is the per-card form of the chance-rooted gadget; a chance-rooted joint solve over the 48 next cards costs 48 solves per decision and does not fit 4 s on the 1070 (section 6.5) |
| major | no certified local gap at play (`certificate=False` everywhere); fixed-T labels accepted without a gate | not changed: the certificate costs one exploitability evaluation per solve (15-22 ms per spot on CPU, measured); it should be on in matches, logged per decision; the gadget game's certificate is implemented (`pcfr.exploitability` with `follow`) |
| major | latent: 3-player side pots wrong (1-3 pot); `critics.project_values` silently skips 4+ live slots (`opponent_mass` returns NaN) | not changed (not reachable heads-up); listed for multiway |
| major | a decision failure (budget, unreachable table row, non-finite, missing artifact) voids the hand and stops a Slumbot match; CPU river decision 4.6-5.0 s at T = 300 | the match no longer stops (section 4.5: the hand fails, is recorded with its reason and scored at the worst case); no default action is played, by decision: it would hide the failure the user said they want to see; the budget must be met on the GPU (section 6.5), and a counted fallback remains the user's call |
| major | critic rows stored as 9 slots fp32 (149 KB per row, 78 % zeros heads-up): 45 k river rows need 42.5 GB by the gate's own rule, the home host has 16 GB | not changed; store the live slots in fp16 (`pad_slots` exists for loading): 9.3x smaller |
| major | V labels and validation diverge (averages of learned critics; 2 000-3 000 labels per rung; synthetic ranges; no decision units) | section 4.3; the exact-label validator is added |
| minor | stacks 20-500 BB: engine yes, trained no (regime 1 deals 200 BB only; the preflop table fits 200 BB only; other depths run on an extrapolated pre_end critic, counted) | not changed; a regime-2 training run or a depth grid for pre_end |
| minor | the rental pipeline cannot be launched from this tree (no `requirements.lock`, no `checks/`, no clore tools) and has no money accounting (wall-time budgets only; `gpu_mib_min` 13 000 rejects 8 and 12 GB cards) | not fixable here; the user has the missing files |
| minor | the live topology has a SLOT at every opponent node: live trees are 2x the teacher's (231 vs 117 nodes river/turn, 165 vs 81 flop), dead SLOT subtrees solved every iteration | not changed; prune off-path slots with compile caching, or accept after measuring GPU latency |

Spec comparison for the whole engine (this auditor): DONE: no bucketing, no snapping, own sizes by street and depth, pre-chance cut with one call per leaf, residual on exact check-down, range-weighted loss. PARTIAL: stacks 20-500 BB (engine yes, training no), exact terminals (2p yes, 3p without side pots, 4+ refused), off-menu re-solve (inserted, but rooted at the street start), certified local gap (teacher yes, play no), learning-curve gate (regression only), 3-player multiway (terminals and trees only), folded players' dead cards (kept at terminals, dropped at critic leaves), inference <= 4 s (CPU no; GPU unmeasured at batch 1). DIVERGES: the ladder's label source and counts; chance-rooted starts. MISSING before this stage: ceilings / gadget (now present), decision-unit validation (exact labels now present, decision units not), RNR exploitation (the `fixed_mask` hook is the place), HUD model, coalitions, 4+ players, variant plug-ins, the test package.

### 4.5 Live path and Slumbot runner (live audit)

Verified by the live auditor (`tools/audit/live/`, `results/audit/live/`): the protocol parser agrees with an independent re-implementation of Slumbot's `ParseAction` on 105 684 fuzzed action prefixes and 19 hand-written cases; all 52 card strings map to the engine's suit x 13 + rank; the engine's evaluator agrees with a brute-force evaluator on 300 showdowns; the engine's first actor (seat 0 small blind preflop, seat 1 postflop) matches the API's `client_pos` 0 = big blind; 1 500 simulated hands replay through `LiveSlumbotBot` with 0 failures; off-menu sizes (137, 1 999, a preflop 250) land in the SLOT at exact chips and the `translated` counter stays 0. **The card and seat mapping is not the cause of the crash** (section 3).

Findings and what was done:

| Severity | Finding | Decision |
|---|---|---|
| blocking | one decision failure (budget, non-finite, unreachable row, missing artifact) set the global stop of the whole match and blocked its resume; the reason kept only the exception's type | **fixed**: a decision failure now fails that one hand (status `failed`, the reason with its message in the record), the match continues, resume passes failed hands, and the summary reports them with a worst-case rate that scores every failed hand as a lost stack; an ambiguous transport outcome (the server may have applied an action) still stops everything, as before |
| blocking | `run.py` forced `JAX_PLATFORMS=cpu`; on the CPU every postflop decision exceeds the 4 s budget (measured at batch 1: river 4.9 s at T 300, flop 10.5 s at T 400, turn about 13 s, off-table preflop 7.9 s), so every decision failed | **fixed**: no device default; a finite budget on the CPU backend is refused unless `--allow-cpu` |
| major | range-only re-solving, no ceilings, no certificate in play | **fixed** (section 2); the certificate is on by default in play (`--no-certificate` to turn it off), compiled at the warm-up, logged per decision in the player's `certificates` |
| major | the 1e-3 likelihood floor is a numerical guard presented as robustness (it caps the evidence per action at 1000:1 and discards the ordering of all hands below it; min ratio measured 0.0041) | no longer the solve's input (the deal is half tracked range, half uniform; the ceilings carry the safety); kept as the diagnostic belief |
| major | the solve server runs topology groups serially and the runner defaults to 16 workers: at 16 flop spots a trap is about 8 GFLOP per iteration | not changed; the worker count must be set from a measured p99 on the 1070 (no GPU latency is recorded anywhere in the code) |
| major | the only duplicate match is self-play (EV 0 by construction) and it drops failed pairs; no LBR; the Slumbot estimate has no AIVAT or baseline control variate (`baseline_winnings` is recorded but unused) | not changed; evaluation work for the GPU box: duplicate against a fixed reference agent, LBR against the live player, AIVAT over >= 4 x 10^4 hands |
| minor | a closing solve could run after the hand was over (likelihoods of a terminal state) | **fixed**: no solve after the hand ends |
| minor | `TableUnreachable` fails the hand when our own reach of the node is below 1e-7 (a rare sampled action of ours) | not changed: the user's design ("never played, counted"); it now fails one hand, not the match |
| minor | `slumbot/cards.py` `engine_cards` is dead code; the equity prefetch overlaps nothing in the Slumbot flow | left as is |

Live decision costs (this CPU, batch 1, measured by the auditor): river live tree 231 nodes 16.5 ms per iteration; flop 165 nodes 26.9 ms (20.7 with the critic replaced by F: the per-leaf check-down matvecs, 1 326 x 1 326 for both slots at every leaf and iteration, are most of the rest); turn 231 nodes 32.4 ms; off-table preflop 166 nodes 26.4 ms. GPU: the code records no solve latency at batch 1; the auditor's estimate from kernel counts (142-194 fused kernels per iteration) is 1-1.5 ms per iteration, i.e. 0.25-0.8 s per solve, plus the 0.3 s trap window: a single table plausibly fits 4 s with a 3-5x margin, **unverified**. Risks: 16 workers with serial groups; a chance-rooted first-turn solve (48 subgames) would not fit.

## 5. The trained-V failure, precisely (point 3)

### 5.1 What "18x worse" was

- The agent: toy/hu E2c, a depth-limited re-solving agent on the 3-street HU toy with the safety gadget and exact leaves below the boundary. Exploitability (exact best response over the whole game, both seats averaged): **23.7 mA/h** (1.18 % of the initial 2-ante pot per hand).
- The same agent with the learned turn-boundary value map (MLP, 20 000 exactly solved turn situations, range-weighted loss): **417.7 mA/h** (20.9 %). With the equity-only leaf (check-down, no strategic content): 532.5. With 200 labels: 521.2; 2 000: 511.8; 20 000 unweighted: 499.1.
- "18x" is 417.7 / 23.7: a ratio of **exploitabilities of the whole agent**, not a regression metric.
- The regression error that produced it: range-weighted RMSE 0.15 (turn) / 0.23 (flop) in x units (x = one player's contribution at the boundary, the leaf pot is 2x), i.e. **7.5-11 % of the leaf pot**; the nets trained on 200 labels: 0.47 / 0.57 x. The exact labels' own noise floor was 0.015 x (0.75 % of the leaf pot).

### 5.2 The curve: value error to exploitability (toy, X1)

`toy/hu/experiments/x1_noise_curve.py` injects controlled errors into the exact boundary values of the same agent and measures exact exploitability (two seeds). Excess over the exact-leaf agent (23.7 mA/h):

| error model | RMSE (x) | as % of leaf pot | exploitability mA/h | excess mA/h | excess, % of initial pot |
|---|---|---|---|---|---|
| iid noise, std 0.1 % of leaf pot | 0.0020 | 0.10 | 24.5 | +0.8 | +0.04 |
| iid 0.3 % | 0.0060 | 0.30 | 26.6 | +2.9 | +0.15 |
| iid 1 % | 0.0200 | 1.00 | 40.3 | +16.6 | +0.83 |
| iid 3 % | 0.0599 | 3.00 | 80.3 | +56.6 | +2.8 |
| iid 10 % | 0.2000 | 10.0 | 217.7 | +194 | +9.7 |
| iid 30 % | 0.6038 | 30.2 | 585.0 | +561 | +28 |
| zero-sum projected noise 1 % / 3 % / 10 % | 0.019 / 0.056 / 0.18 | 0.9 / 2.8 / 9.1 | 38.0 / 75.3 / 192.0 | +14 / +52 / +168 | |
| shrink toward check-down by 10 % / 30 % / 50 % / 100 % | 0.032 / 0.101 / 0.173 / 0.388 | 1.6 / 5.1 / 8.7 / 19 | 50.9 / 129.0 / 218.7 / 532.5 | +27 / +105 / +195 / +509 | |
| the learned net (20000, range-weighted) | 0.179 | 8.9 | 417.7 | +394 | +19.7 |

Between 1 % and 30 % of the leaf pot the excess is linear: **x % of leaf-pot error costs about 0.93 x % of the initial pot per hand** (about 0.93 mA/h per 0.001 x of RMSE). Below 1 % it is sub-linear and the re-solver's own 23.7 mA/h (K = 300 iterations) dominates. The learned nets lie **1.4-2.5x above** the iid curve at the same RMSE: their errors are correlated and depend on the ranges the re-solver can push; RMSE alone under-predicts what a learned V costs. Zero-sum projection (the engine's chip-conservation projection has the same purpose) buys 5-30 % at the same noise.

### 5.3 The same question on the real engine

The gadget experiment (section 2.3) is the hold'em-scale analogue for one street: per-hand ceiling noise of 0.01 pot costs +3 to +11 mbb/hand, 0.02 up to +265, 0.05 up to +116, and the check-down value (the toy's 533 mA/h leaf) +0.5 to +2 bb/hand. A critic that is accurate on average but biased per hand by +0.05 pot is the worst case measured (+230 to +324 mbb/hand).

### 5.4 Scalability killers, as measured

- **Label count.** The toy moved from 521 to 499 mA/h between 200 and 20 000 labels (raw loss) and to 418 with the range-weighted loss: in that regime the error is not data-limited in the way a power law would predict; the toy's labels were single-situation exact solves of a 10-card game, so this is evidence about the toy's architecture, not about hold'em.
- **The engine's recorded critic errors** (`poker/solver/critics.py` comments): river residual critic test MAE 0.043 pot on the 45k-row HU river set (10 of 10 seeds within 0.0422-0.0458 under the current protocol), 0.063 on k0b (0.103 for the plain critic on the same rows); the end-of-street critics are far easier (pre_end validation MAE 0.0039): they predict card averages. The gadget needs 0.01-0.02 per hand on the river. No learning curve of the engine's critics against label count is in the upload (`poker/solver/learning_curve.py` exists and implements the protocol); it is the gate.
- **Units of the critic.** Values in pot units of the street start; a 0.3 % target is 0.3 % of that pot. The toy's "0.3 % of pot" (0.006 x) cost +2.9 mA/h = +0.15 % of the initial pot per hand.
- **Training instability** is documented and handled (lr 5e-4, clip 0.05: 10 of 10 runs collapse-free; lr 1e-3 collapsed in 1-5 of 5-7 runs).

### 5.5 First exact end-of-turn labels of the engine (new validator, section 4.3)

`poker/solver/validate_v4.py` on three of the loop's own turn situations (one CPU core, 48 exact river solves each at T = 400; results in `results/audit/v4/validate_v4.json`):

| situation | pot | SPR | river E after T = 400 (median / max) | check-down vs exact V(4): range-weighted MAE, seat 0 / 1 | worst in-range hand | label noise T = 400 vs 800 |
|---|---|---|---|---|---|---|
| 1 | 8.9 BB | 21.9 | 0.0086 / 0.0163 | 0.089 / 0.110 pot | 0.88 / 1.96 pot | MAE 0.0042, worst 0.10-0.12 pot |
| 2 | 22.7 BB | 8.3 | 0.0046 / 0.0061 | 0.066 / 0.102 | 0.55 / 0.73 | - |
| 3 | 76.0 BB | 2.1 | 0.0024 / 0.0033 | 0.105 / 0.094 | 0.80 / 0.56 | - |

Reading: the exact check-down value, which is the residual critics' baseline and input, is 0.07-0.11 pot away from the true end-of-turn value on the hands in range, and 0.5-2 pot away on the worst in-range hands. The gadget needs per-hand ceilings within about 0.01-0.02 pot (section 2.3), so the network has to remove 85-90 % of that residual; the engine's best recorded river critic removed about half (0.043 from a check-down baseline of about 0.08 on the 45k set). The labels' own noise at T = 400 is 0.004 pot range-weighted (0.1 pot on the worst hand) at SPR 22; deep spots need T >= 800 or the certificate, as the training audit says. Each situation cost 200-370 s on one CPU core (about 20 s on the 1070 by the recorded speed), so 10^4 exact labels is a day of the 1070, not a rental.

## 6. Economics and the units question (point 4)

### 6.1 Units: "0.3 % of pot" is 0.3 bb per 100 bb pot, not 30 bb

A value error of 0.3 % of the pot at a decision with a 100 bb pot is 0.3 bb on that decision's values. If an adversary extracted the whole error at every decision of every hand, the loss would be at most about 2 x error x (pot at the boundary) x P(reaching it) per hand, so roughly 0.3-0.6 bb per hand = 30-60 bb/100 is the **worst-case bound**, not the measurement. The measurements: in the toy, 0.3 % leaf-pot error costs +0.15 % of the initial pot per hand (section 5.2); on the real river tree a 1 % per-hand ceiling error costs +3 to +11 mbb/hand with the range deal (section 2.3). "30 BB/100" (0.3 bb/hand, 300 mbb/hand) appears in the toy only at a 30 % error.

### 6.2 What "exploitability" refers to

The reference is the **exact best response**: the most a perfect adversary who knows our whole strategy wins against it, minus the game value (NashConv / 2 heads-up). It is computed exactly where the game is small enough: in the toy over the whole game; in the engine inside one street given the leaf values (`pcfr.exploitability`, the E certificate), and in the experiment above for the whole river. For the whole hold'em game nobody computes it exactly; the published proxy is LBR (local best response), a lower bound. Regression error (MAE / RMSE of V against solved labels, in pot units) and exploitability (bb or mbb per hand) are different quantities linked only by measured curves like sections 2.3 and 5.2; the engine's loop reports the first, the self-play duplicate match and these experiments the second.

Would an agent exploitable by 300 mbb/hand be a fish? Against the perfect adversary it loses 0.3 bb/hand. Slumbot 2016 was exploited by LBR for 1 000-4 000 mbb/g depending on LBR's action set (Lisý and Bowling 2017); DeepStack was not exploitable by LBR. A strategy at 300 mbb/hand of exploitability would be far less exploitable than Slumbot. Exploitability is not win rate: how much Slumbot (which does not best-respond) actually wins against it is the match's question, and a 95 % confidence interval of +-25 mbb/hand needs about 4 x 10^5 hands without variance reduction.

### 6.3 Measured costs

CPU, this box (one Xeon core pinned, shared with other jobs; absolute times +-30 %), `results/audit/engine-profile/summary.json`:

| Item | measured | per label |
|---|---|---|
| River "normal" tree (40 decisions, 77 terminals), one PCFR+ iteration, batch 1 | 14.5 ms | T = 800: 11.6 core-s; T = 200: 2.9 core-s |
| same, batch 4 / 16 / 32 (per spot) | 18.7 / 20.4 / 21.3 ms | batching does not help on CPU |
| River "short" tree | 6.0 ms | 4.8 core-s at T = 800 |
| Flop tree (28 decisions, 13 leaves), net critic 512x512x512, batch 4 | 21.7 ms | T = 1500: 35 core-s |
| Turn tree (40 decisions, 19 leaves), net critic, batch 4 | 33.1 ms | T = 1500: 50 core-s |
| Where a river iteration goes | terminals 21-27 %, backward 27-29 %, forward 5-13 %, regret-matching bookkeeping 36-45 % | |
| the bookkeeping on [nodes, 1326, 13] padded arrays (2.85 legal actions on average, max 5) vs a compact edge layout | 10.7-28x slower than compact | the one measured throughput lever: about 1.6x per iteration |
| fold terminals with the board-free compatible mass instead of the showdown kernel (done in this stage, `pcfr._two`) | river 13.5 -> 12.0 ms/iteration (1.13x), flop 22.3 -> 20.1 (1.11x), same core back to back; E of the solved spot equal to 3e-7 | a free 11-13 % |
| Equity matrices, one core | turn 0.78 s (evaluator) / 0.033 s (rank7 table); flop 17.4 s / 0.79 s | the rank7 table (184 s to build, 268 MB) is required |
| E after T, river | 0.00086 pot (T 800, batch 1), 0.0019 (T 400) | |

GPU, the code's own recorded numbers (`tools/cloud/launch.py`, `poker/solver/loop.py`): GTX 1070 grid 13 fixed T: river 0.86 s/spot (T 800), turn/flop 1.7 s/spot (T 1500): **4 190 river and 2 120 turn-or-flop labels per hour**; a 3090 is about 3x the 1070 (a k0-size run: 2.5-3 h on the 1070, 1 h on a 3090); an RTX 4090 did 197 river spots/min under grid 7 with the certificate. Against one Xeon core the 1070 is 13x faster per river spot; against the whole E5-2650 v2 (8 cores, 1.25-1.67x slower per core than this box) it is 2-6x.

The engine's label costs against the spec's unit prices (spec section 6.3): a 200-iteration river solve costs 2.9 core-s here, the spec assumed 0.05-0.2 compiled (0.64 numpy); the engine also runs 4x (river) to 7.5x (turn, flop) the spec's 200 iterations per label (`config.TEACHER_T`). The spec's CPU bill (320-790 core-hours) does not describe this engine; the engine's own path is the GPU.

### 6.4 What the budget buys, in labels

At the spec's calibration ($0.5-1.5 per GPU-hour; clore.ai boxes are often cheaper, the user's invoices are the truth) and the 3090's inferred speed (12 500 river or 6 400 turn/flop labels per hour):

| Budget | GPU-hours (3090) | river labels at T 800 | or turn/flop labels at T 1500 |
|---|---|---|---|
| $10-15 (HU) | 7-30 | 90 000-375 000 | 45 000-190 000 |
| $120 (everything, 9-max ceiling) | 80-240 | 1.0-3.0 M | 0.5-1.5 M |

The loop's default run (3 000 river, 2 000 turn, 3 000 turn-end, 2 000 flop, 3 000 flop-end, 3 000 pre-end spots, two seeds per critic) is about 1 GPU-hour on a 3090: trivially inside the budget and, at those counts, data-starved by the loop's own record (45k river rows gave MAE 0.043). DeepStack used 10^7 turn and 10^6 flop situations; at the 3090's speed that is 1 600 + 160 GPU-hours, $900-2 600: not reachable. What is reachable is 10^5-10^6 labels per rung, and whether that gets a per-hand river error to 0.01-0.02 pot is **unknown**: it is the one number the gate must produce before paid runs (section 9).

### 6.5 Inference on the home PC

Live solves measured here (one Xeon core, T = 100): flop 165-node live tree 2.3-2.9 s, turn 231 nodes 2.9-3.1 s, river 1.4-1.5 s; the configured play iterations are 300-400 (`config.PLAY_ITERATIONS`), so a CPU-only decision costs 4-12 s per solve single-threaded: **over the 4 s budget**. On the GTX 1070 (13x per spot) a river solve at T 300 is about 0.3 s and a turn solve about 0.6 s plus the critic; the ceilings add 30-60 ms; tree replay 20-100 ms. The 4 s budget holds on the GPU with the batch ladder; it does not hold on the E5 alone unless T is cut to about 100 (the river's E at T 100-200 is 0.002-0.005 pot).

## 7. GPU-first with JAX/XLA for a GTX 1070 and an E5-2650 v2 (point 2)

- **Solving (labels and live play) belongs on the GPU.** Per river spot the 1070 is 13x one Xeon core and 2-6x the whole E5; the CPU does not gain from batching (18.7-21.3 ms per spot at batch 4-32 against 14.5 at batch 1) while the GPU's recorded numbers are at batch 128. The 1070's 8 GB hold a river batch of 128 (the state is about 10 MB per spot in float32).
- **What stays on the CPU, and already does:** tree replay (`subgame.replay`, `tree.replay`: the fused engine scan does not load on Pascal, `tree.require_cpu`), the turn all-in matrix from the memory-mapped rank7 table (0.033 s; the device builder is 0.2 s on the 1070 per the code), the flop matrices from the 1755-class cache, the preflop matrix once. Rank7 table build: 184 s once on 2 cores.
- **Pascal specifics:** no TF32 (irrelevant: the solver runs fp32 with `jax_default_matmul_precision = "highest"`), CUDA_ERROR_NOT_SUPPORTED on the fused scan (handled), 8 GB. JAX/XLA runs on the card today (the code's measurements are from it); recent CUDA/jaxlib releases have been dropping older compute capabilities, so pin the jaxlib version that works and record it in `requirements.lock` (not in the upload).
- **Not GPU-first:** the Python-side bookkeeping of the live player (ranges, trees, the solve server) is host work by nature and is not the bottleneck (replay 20-100 ms, ceilings 30-60 ms).
- **The one throughput lever measured:** the regret-matching bookkeeping on padded [nodes, 1326, 13] arrays is 36-45 % of a river iteration and 10.7-28x slower than a compact edge layout on CPU; the same padding wastes GPU bandwidth. Converting `pcfr.py`'s regret/average/policy arrays to [edges, 1326] is the first optimisation to make (section 8 lists it as "fix: throughput"); it was not done in this stage because it changes the solver's memory layout everywhere and the correctness fixes came first.

## 8. Keep / fix / reuse / missing: the map

Consolidated from the five audits and this stage's work. "fixed here" marks what changed in this stage. Exact / approximate / engineering constraint is stated per component where it matters.

### Keep (correct, verified, unchanged)

| Component | Why | Nature |
|---|---|---|
| `poker/envs` rules engine: evaluator, settlement, blinds, legality, exact totals (`step_target`), guards, provenance (game record, fingerprint) | exact against independent references (110 k hands, 2 000 settlements, 18 k decisions); 2-12 seats in the rules | exact |
| `poker/solver/showdown.py`, `cards.py`, `checkdown.py`, `runout.py`, `rank7.py`, `flops.py` | the spec's O(F) prefix-sum showdown, exact to 1e-14; all-in matrices exact to 5e-8; card-removal identities verified | exact |
| `poker/solver/pcfr.py` two-player core: forward, terminals, backward, PCFR+, best-response certificate, root values, pre-chance leaves with exact check-down F | exact terminals; E monotone and certified; accounting chain verified | exact given the leaf values |
| `poker/solver/tree.py`, `subgame.py` topology, SLOT insertion and chip-exact replay | the spec's off-menu mechanism, verified at exact chips | exact |
| `poker/solver/critics.py` architecture: residual on exact F, range-weighted loss, chip-conservation projection, augmentation, training protocol | as the spec asks; the projection is exact in regime 1 | approximation (the network) with exact baseline |
| `poker/arena/slumbot` protocol, client, exact replay | 105 684 fuzzed prefixes agree with an independent parser; 1 500 simulated hands replay | exact |
| `poker/player/live.py` bookkeeping: exact own reach, frozen own nodes, slot likelihood plumbing, solve server | clean and exact; the base the gadget was built on | exact |
| `poker/solver/loop.py` stage machinery (claims, shards, provenance, resumability) | robust; the auditors' own fingerprint changes were caught | engineering |

### Fixed here

| Component | What changed | Evidence |
|---|---|---|
| live re-solving | the opponent's per-hand ceilings (gadget) from the post-card critics at every street start; deal = half tracked range + half uniform; the certificate on in play | section 2.3: range only +16..+942 mbb/hand, wrong range +1 100..+7 700; the fix +0..+161 (exact ceilings), 8 of 9 cases below +40 |
| Slumbot runner | a decision failure fails one hand (recorded with its reason, scored at the worst case), not the match; resume passes it; no CPU default; `guards.target` before every replayed step; failure messages kept | section 4.5; `tests/test_slumbot_replay.py` |
| engine legality | illegal action index / exact total reported under checkify; the live path refuses a step the engine would drop (history full) | section 4.1 |
| rake split | int32 product bounded at the table guard for raked tables | section 4.1 |
| fold terminals | board-free compatible mass | 1.13x river, 1.11x flop per iteration, E unchanged to 3e-7 |
| cache paths | resolved at call time | section 4.3 |
| no closing solve after a hand ends | | section 4.5 |
| exact end-of-turn labels | `poker/solver/validate_v4.py` | section 5.5 |

### Fix (correct in intent, wrong or incomplete in a measured way; not changed here)

| Component | Problem | Nature |
|---|---|---|
| `pcfr._three`, `tree.street_root` | no side pots, equal stacks only (1-3 pot error on unequal contributions) | correctness, multiway only |
| `pcfr.py` state layout | padded [nodes, 1326, 13], dead subtrees solved: 1.6-3x per iteration | throughput |
| `config.TEACHER_T`, PCFR+ simultaneous updates | labels at 800-1 500 iterations; deep-SPR labels noisy (E 0.019 at SPR 90); no per-SPR T; no DCFR / alternating option measured | throughput, label quality |
| `loop.end_stage`, end-critic targets | averages of learned critics; no exact validation (the validator now exists); label counts 1-3 orders below the budget | V quality (the blocking item of the training audit) |
| `situations.random_range` | synthetic, board-unaware, independent ranges | V quality |
| `critics.py` storage, `KAPPA` placeholders, suit canonicalisation, loss over SPR bands, zero-reach label convention | rows 9.3x larger than needed (45 k rows need 42.5 GB by the gate's rule); kappa 0 under rake; inference not canonicalised; deep rows dominate the loss | V quality, memory |
| `loop.aux_device` on Pascal | end-of-street targets on the CPU: 5.6 s per flop_end situation, about 4.7 h for the default 3 000 | throughput at home |
| `COMMIT_MERGE_POT`, `STREET_SIZES`, `TREE_CAPS`, `MENU_BOUNDS` | declared heuristics without a measured cost (the tool exists: `sizing_lab.py`, but it values turn/flop street ends by check-down) | EV, not safety |
| `config.history_bound` | proof holds for grid raises only; the live path now refuses the silent no-op | correctness at depths beyond 850 BB heads-up |
| live solve server, worker count, anytime solve | serial topology groups; the 4 s budget unverified on the 1070; no deadline inside a solve | decision speed |
| evaluation | self-play duplicate only; no fixed reference opponent; no LBR; no AIVAT | measurement |
| `tools/cloud` | not launchable from this tree (lock file, checks, clore tools missing); no money accounting; `gpu_mib_min` rejects 8-12 GB cards | engineering |

### Reuse (right tool for a later rung)

`sizing_lab.py` (exploitability of restricted and translated policies: the core of a decision-unit validator), `cap_price_n3.py` and `tools/measure_runout_sampling.py` (3-player measurements), `preflop.py` (publication gate for the preflop table), `learning_curve.py` (protocol-faithful curves; needs decision units), `table.TableSampler` (9-seat fixtures), `guards.py` (host checks for every exact-amount path).

### Missing

| Component | Needed for | Status |
|---|---|---|
| decision-unit gate (turn solve with learned vs exact leaves, best-response gain in the exact turn+river game) | any claim about V quality in play | not built; exact labels exist (section 5.5) |
| harvested and perturbed ranges, SPR-stratified sampling | V quality | not built |
| 3-player street-end leaves, check-down and critics; 4+ player terminals (sampled or declared); side pots; 7-9 seat trees; preflop router | 6-max, 9-max | not started beyond 3-player terminals and trees to 6 seats |
| RNR exploitation and the HUD opponent model | later exploitation | the `fixed_mask` hook is the place; the tracked belief is the input |
| coalitions | 9-max security tests | nothing |
| variant plug-ins (deck, evaluator, betting grammar) | variants | 52-card hold'em hard-coded |
| the user's `checks/` package, `requirements.lock` | the rental gate | not in the upload |
| chance-rooted turn/river forest, flop-start ceilings | the spec's own form of the gadget | replaced by per-card ceilings from the post-card critics (DeepStack's construction) for cost reasons; exact only when the critic is |

## 9. Budget verdicts

All dollar figures use the spec's calibration ($0.5-1.5 per GPU-hour; clore.ai boxes are often cheaper and the user's invoices replace these numbers), the GPU speeds the code itself recorded (GTX 1070: 4 190 river or 2 120 turn/flop labels per hour; a 3090 about 3x that), and the measurements of this stage. Nothing was spent; no run is proposed before the gate below.

### 9.1 Heads-up against Slumbot: $10-15

What the engine can already do within the budget: a full iteration-0 run (3 000 river, 2 000 turn, 2 000 flop labels, the end critics, two seeds each) is about 1 GPU-hour on a 3090 or 2.5-3 hours on the home 1070, i.e. $0.5-1.5 rented or free at home. The budget buys 90 000-375 000 river labels or 45 000-190 000 turn/flop labels on a 3090; the home 1070 adds 100 000 river or 50 000 turn/flop labels per day for nothing but electricity. The live path now re-solves with ceilings (section 2); on the 1070 a river decision costs about 0.3 s and a turn decision about 0.6 s at the configured iterations, inside the 4 s budget; on the CPU alone it does not fit (section 6.5).

What is not known, and what the money depends on: whether a critic trained on 10^5-10^6 labels reaches about 0.01-0.02 pot of **per-hand** error at the street starts (section 2.3's requirement) from a check-down baseline that is 0.07-0.11 pot off (section 5.5). The engine's best recorded river critic (45 k rows) is at 0.043. If the error falls like a power law in the label count with the usual exponents (1/4 to 1/3), 0.01 needs 80-250x more labels than 45 k, i.e. 3.6-11 M river labels: 290-900 GPU-hours on a 3090, $150-1 300, outside both budgets. If the error is dominated by the model and the ranges (the training audit's findings: synthetic independent ranges, no suit canonicalisation, one loss over all SPRs, half the rows SPR-0 copies), fixing those is cheap and the count may be far smaller. **This is the one number the gate must produce** before paid runs: train the river critic on nested subsets of the labels the home 1070 generates in a day (10^4, 3 x 10^4, 10^5), score each on the exact end-of-turn labels of `validate_v4.py` and in the gadget's own units (section 2.3's splice test with the critic's ceilings), fit the curve, extrapolate to the count the budget buys. Cost of the gate: a day of the 1070 and an hour of a CPU.

Verdict: **$10-15 is enough for the compute of a heads-up agent with this engine only if the gate shows the curve reaching about 0.01-0.02 pot per hand within 10^5-10^6 labels; the measured evidence so far (0.043 at 45 k; the check-down baseline at 0.07-0.11; the toy's nets 1.4-2.5x above the iid curve) says the prior is against it.** What $10-15 buys for certain: the exact river and turn solver as a trainer and analyser (certified per solve), the ceilings-based live path, and a self-play duplicate match that measures gadget against range-only play. Beating Slumbot is a match result, not a cost estimate; the match itself (4 x 10^5 hands for +-25 mbb/hand without variance reduction) is days of the home PC on top.

### 9.2 Six-max

The budget is "substantially less than 9-max" and nothing in the engine solves a 4-, 5- or 6-handed pot (section 4.2: terminals stop at three players; three-player street-end leaves and check-down values are not implemented; side pots are wrong for unequal stacks; the training loop deals two seats). A 6-max agent needs the three-player ladder (turn and flop leaves, critics, labels at 2.6 ms per 3-way river terminal and 48 sampled runouts per flop all-in) and a declared approximation for 4+ live players; none of it is costed by a measurement yet. Verdict: **no 6-max cost can be given from this engine today; the next measurable step is the 3-player river and turn ladder, whose per-label cost the solver audit measured (0.85-0.93 s per iteration per spot at batch 1 on one CPU core for the 3-player river tree of 102 decisions: about 70x the heads-up river).**

### 9.3 Nine-max within min(fact, $120)

The hard ceiling is $120 for all future paid computation. The spec's cost model (section 6.5 of `docs/poker-core-mechanism.md`) put 9-max at about 9-10x a heads-up unit at the coarsest 3-way menu and at $66-268 depending on the menu, before evaluation, with the heads-up unit itself unverified. This stage adds: the heads-up unit's one unknown (the critic's label curve) is still unknown; the engine has no 4+-player terminal, no 3-player critic, no side pots, no 7-9 seat trees, no preflop router; the padded layout and dead subtrees make every label 1.6-3x more expensive than it need be; the fold-terminal fix bought 11-13 %. **Fundamental impossibility is not established; infeasibility within $120 at today's engine is**: with the heads-up ladder alone possibly exceeding $120 (section 9.1), and 9-max needing at least the coarse 3-way ladder on top (10x), the $120 ceiling is met only if (a) the gate shows the heads-up critic curve reaching its target within about 10^5 labels, (b) the label generator runs on the home 1070 for weeks rather than on rented boxes, and (c) 3-way pots use the one-size-plus-all-in menu with check-down leaves for 4+ players as a declared approximation. Each of (a)-(c) is a measurement or a decision, not a fact today. The honest statement for the user: **the $120 buys the heads-up gate and, if it passes, the heads-up agent; 9-max within the same $120 requires the home GPU to do the labelling and the multiway code that does not exist yet.**

## 10. The non-negotiables, checked

| Requirement | Status after this stage |
|---|---|
| No hand bucketing | holds: 1 326 combos everywhere (envs, solver, critics, live); the 169 classes appear only in exact suit-symmetric averaging |
| No rounding of opponents' bet sizes | holds: exact totals through `step_target`; an off-menu size goes into the tree's SLOT and is re-solved; verified on the engine (envs audit) and the Slumbot replay (section 3) |
| No unjustified predictors / no hacks as principles | the likelihood floor eps = 1e-3 was one (an undeclared heuristic that fabricated beliefs; measured: +1 pot per air hand) and is no longer the solve's input; `COMMIT_MERGE_POT`, `STREET_SIZES`, `TREE_CAPS`, `MENU_BOUNDS` are declared heuristics without a measured cost (section 4.1); the chip-conservation projection is exact in regime 1 and a placeholder (kappa 0) under rake |
| Exact vs approximate vs engineering constraint, per component | section 8's map; the gadget's ceilings are an approximation whose error the critics' per-hand error sets, measured in section 2.3 |
| Regression error is not exploitability | kept apart throughout: sections 2.3 and 5 measure the map between them; the engine's reported critic MAE is a regression number against labels that are themselves averages of learned critics (section 4.3) |
| Robustness (no lucky runs, no endless tuning) | the training protocol's collapse record is documented and collapse-free in 10 runs; the solver's E is monotone and certified; the gadget's results hold over 3 trunks x 3 spots; nothing here was tuned to a seed |
| One foundation HU / 6-max / 9-max / variants | holds for the rules engine (2-12 seats verified for 2-6) and the tree builder (to 6 seats); does not hold for the solver (3 players at most, no side pots), the critics (2 players), the loop (2 seats), the evaluator (52-card hold'em hard-coded) |
| Stacks 20-500 BB, arbitrary opponent sizes, own sizes by street and depth | engine yes; trained for 200 BB only (regime 1), the preflop table fits 200 BB only; opponent sizes exact; own sizes configurable |
| HUD exploitation later | the `fixed_mask` / `fixed_pi` hook in the solver is the place for the spec's p x model + (1 - p) x regret mixture; the Bayesian belief is still tracked (now as a diagnostic) and is the natural input; not built |
| Coalitions later | nothing in the code |
| No circular reasoning | the end-critic validation was circular (learned critic scored against the average of a learned critic); the exact-label validator breaks it |

## 11. Contradictions and open items

1. **The gadget wants a per-hand accuracy the critics do not have.** Section 2.3: about 0.01-0.02 pot per hand; section 5.5: the baseline is 0.07-0.11 off and the best recorded critic 0.043. Until a critic reaches it, the ceilings are approximate and the mixed deal is the measured compromise; the duplicate match (gadget vs `--no-gadget`) on the GPU box decides whether the approximate ceilings already help.
2. **Chance-rooted re-solving is unaffordable on the 1070 at 4 s per action**, so the ceilings come from the post-card critic (DeepStack's construction) rather than a joint solve over the 48 next cards; the two agree exactly only when the critic is exact.
3. **The spec's unit prices were 15-60x too optimistic for this engine's CPU path** and about 25x for its GPU path (section 6.3); the spec's CPU bill does not apply, the GPU bill (section 9) does. The padded layout and dead subtrees (1.6-3x) and the iteration budget (4-7.5x the spec's 200 per label) are where the factor comes from.
4. **The engine's own quality record was circular** (section 4.3). Every reported MAE of a turn_end / flop_end / pre_end critic is against another learned critic; the first exact labels are in section 5.5.
5. **A decision failure (budget, unreachable table row, non-finite) now fails one hand, scored at the worst case, instead of the match.** The user said failures must be visible; whether the bot should instead play a counted fallback action (the anytime solve of plan item 2) is a product decision the user must take; without the GPU the CPU path cannot meet 4 s at the configured iterations.
6. **The Slumbot crash is not reproduced** (section 3); the recorded request/response files of the failed run are what will find it.
7. **Multiway is not started** beyond 3-player terminals and trees; 9-max within $120 depends on the heads-up gate and on the home GPU doing the labelling.

## Appendix A. Reproducing the tests and measurements

All scripts run from the repository root against the user's engine (`poker/`) with

```sh
export POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
```

| Script | What it checks or measures | Result on this box |
|---|---|---|
| `python3 tests/test_slumbot_replay.py` | Slumbot payloads through parser, replay, candidate replies and serialisation; malformed responses refused; 52 card strings round-trip | exit 0 |
| `python3 tests/test_gadget.py` | gadget in `pcfr`: mask off == no gadget (bitwise), ceilings far below the root values give the plain solve, ceilings far above give Follow 0, finite certificate | exit 0 |
| `python3 tests/test_warm_up_gadget.py` | a certified gadget solve and the ceiling computation after `subgame.warm_up` and `warm_gadget` compile nothing | exit 0 |
| `python3 tests/test_live_gadget_hand.py` | a hand from the flop through `LivePlayer` with random critics: ceilings at the turn and river, Follow statistics, legal actions; `gadget=False` still runs | exit 0 |
| `python3 tests/gadget_river_resolve.py --spots 3 --iters 800 --sub-iters 800 --trunk-frac 3 4 --out results/audit/gadget/river_resolve_T800_trunk34.json` | section 2.3 (repeat with `--trunk-frac 1 3` and `3 2`; `--noise`, `--bias` add ceiling errors) | 5 min per trunk |
| `python3 -m poker.solver.validate_v4 --count 3 --iters 400 --double --out results/audit/v4/validate_v4.json` | section 5.5: exact end-of-turn labels and the check-down baseline, label noise T vs 2T; `--critic <dir>` scores a `turn_end` checkpoint | 200-370 s per situation on one core |
| `tools/audit/{envs,solver,training,live,spec,engine-profile}/` | the auditors' checks; each directory has its own README with the command lines; raw outputs under `results/audit/<area>/` | |
| `toy/hu/experiments/x1_noise_curve.py` | section 5.2 (the toy's value-error curve); results in `toy/hu/results/X1_NOISE_CURVE.md` | |

The user's own `checks/` package (not in the upload) is not replaced by these scripts. Nothing here needs a GPU; everything that needs the GTX 1070 or slumbot.com is listed in Appendix D.

## Appendix B. The consolidated fix plan (22 items) and its status after this stage

The synthesis of the five audits produced this ordered plan before the fixes of this stage were made. Line numbers refer to the engine as uploaded (commit `2932d39`). "Status" is what this stage did about each item; everything marked "open" is costed in hours of engineering by the auditors and in machine time where it applies.

| # | Action | Files | Why | Verification | Effort | Status after this stage |
|---|---|---|---|---|---|---|
| 1 | Make the Slumbot evaluation survive a bad decision: explicit device (refuse CPU without `--allow-cpu`); a failure keeps the hand in the match as a counted failure; error message in the hand record; a worst-case bound that scores failed hands as losses | `arena/slumbot/run.py`, `parallel.py`, `client.py`, `metrics.py` | one decision failure ended the match and blocked resume; the record kept only the exception type | 60/60 hands complete under `budget_repro.py`; resume on a partial directory; 1 500 stub hands with 0 lost | 6-10 h | **done** except the fallback action: a failed hand is recorded with its message and scored at the worst case (lost stack); playing a default action instead is the user's decision (section 4.5) |
| 2 | Anytime `solve_batch` with a deadline: return the average strategy at the last completed chunk with the T reached and the certificate, instead of raising `BudgetExceeded` | `solver/subgame.py`, `player/live.py` | CPU solves (river 4.9 s, flop 10.5 s, turn ~13 s) and any GPU hiccup become hard failures | 0 `BudgetExceeded` at a 4 s budget on CPU; `T_reached` logged per decision | 8-12 h | open (a product decision: a lower-T decision is a declared approximation only if its certified gap is logged; the certificate is now on in play, so the plumbing exists) |
| 3 | Find the card-parse crash: the crashed run's `hands/`, `summary.json`, stderr; 100-200 real Slumbot hands on the 1070; `guards.target` in the replay | `arena/slumbot/live_bot.py`, `cards.py`, `client.py` | not reproduced by any auditor (105 684 fuzzed prefixes, 1 500 e2e hands clean) | a 200-hand real run with 0 ProtocolErrors; every ProtocolError arrives with its message | 3-5 h + network + 1070 | `guards.target` **done**, offline conformance test **done** (section 3); the real run needs the user's network and GPU |
| 4 | Replace the carried range and likelihood floor with the two-player safe re-solving gadget: per-hand opponent ceilings at street close, Terminate/Follow at the `pcfr` root, ceilings in `spot_batch`, `certificate=True` in play, a gadget audit (realised opponent value vs ceiling) | `player/live.py`, `solver/pcfr.py`, `solver/subgame.py` | unsafe by construction: air gains +1.0 pot per hand in the demo; toy 520 vs 23.7 mA/h | air gain <= 0 within the certified gap; local loss <= E; realised value <= ceiling + E on 1 000 decisions | 30-50 h | **done** (section 2): ceilings from the post-card critics at every street start, Follow/Terminate root, certified gadget game, certificate on in play; measured on the exact river game (section 2.3). Not done: the per-decision audit "realised opponent value vs ceiling" (needs a match) |
| 5 | Decide the first-of-street construction: chance-rooted exact river forest vs ceilings carried from the previous street | `player/live.py`, `solver/subgame.py`, `tree.py`, `pcfr.py` | the gadget needs ceilings at the first decision of a street | wall time per river start on the 1070; local exploitability on 50 river starts | 20-30 h | **decided and done in the per-card form**: ceilings come from the post-card critic evaluated at the previous street's leaf for the dealt card (DeepStack's construction); a chance-rooted forest (48 solves per decision) does not fit 4 s on the 1070 (section 11, item 2) |
| 6 | Make every bad engine step loud: `checkify` of the offered mask and of the exact-total predicate; history overflow raises; a chip-based or host capacity check; stale docstring | `envs/transition.py`, `envs/config.py`, `envs/guards.py` | an illegal action silently yields a negative stack; 1 000 BB stalls silently | both illegal cases raise; no stall up to the deepest accepted stack; 2 000 hands still 0 mismatches | 6-9 h | **done**: `checkify.debug_check` in `apply_action` and `apply_target`; host capacity check in the live path; docstrings. The array bound was not changed (section 4.1, F1) |
| 7 | `rake_split` int32-safe | `envs/evaluator.py` | latent overflow above 2^31 / cap | the overflow scenario passes; raked hands 0 mismatches | 1-2 h | **done** via the table guard (`RAKE_CAP_MAX_CHIPS`; section 4.1, F2) |
| 8 | Latent multiway bugs: layered side pots in `_three`, per-seat stacks in `street_root`/`replay`; `opponent_mass`/`project_values` raise on 4+ live slots; state the scope | `solver/pcfr.py`, `showdown.py`, `tree.py`, `critics.py` | 3-player side pots off by 1-3 pot; critics return NaN for 4+ | the auditor's side-pot case < 1e-6 in float64 | 10-16 h | open (not reachable heads-up; first multiway item, section 8) |
| 9 | Exact V(4) validator (48 river solves per end-of-turn situation) and the decision-unit scorer (turn solve with learned vs exact leaves, best-response gain in the exact turn+river game) | new `solver/validate_v4.py`, `sizing_lab.py`, `pcfr.py`, `loop.py` | end critics were scored against the learned critic they average | exact leaves reproduce a direct turn+river solve within E; the scorer gives ~0 for exact leaves | 16-24 h | **half done**: the exact-label validator exists and ran (section 5.5); the decision-unit scorer is open (GPU-box job) |
| 10 | Calibrate T per street and SPR band from the E(T) curves; certificate-stopped solving above SPR 16; end-of-solve E per label; the same for `PLAY_ITERATIONS` | `envs/config.py`, `solver/loop.py`, `situations.py` | up to 4x on low-SPR labels; SPR-90 labels noisy at T 800 | range-weighted drift T vs 2T below 0.002 pot in every band | 6-10 h + 3 h CPU | open; the curves and the noise measurement exist (sections 4.2, 4.3, 5.5) |
| 11 | Shape generation for the hardware: B = 1-4 per pinned CPU process; rank7 only when needed; device rank7 path for the turn matrix; fixed batch shapes and the persistent compile cache; cache paths at call time; a timeout in `io.wait_for` | `solver/loop.py`, `subgame.py`, `rank7.py`, `io.py` | batching hurts on CPU; rank7 build 184 s of a 600 s run; turn matrix 0.78 s instead of 0.033 s | `run_loop_river.py --quick` 48/48 spots in 10 min | 6-10 h | cache paths **done**; the rest open |
| 12 | Compact per-edge PCFR+ state [E, 1326], dead-subtree pruning, one check-down GEMM per spot per iteration, alternating updates and a DCFR option compared by E against wall time | `solver/pcfr.py`, `tree.py`, `envs/config.py` | the main lever on label price and latency: bookkeeping 36-45 %, 10.7-28x faster compact; 108 of 117 nodes dead at SPR <= 0.5 | identical policies (max diff < 1e-4) on 50 spots per street; >= 3x per river iteration at B = 1 | 24-40 h | open; the fold-terminal shortcut from the same analysis is **done** (1.13x river, 1.11x flop; section 6.3) |
| 13 | Critic rows as k live slots in fp16, compressed; update the RAM rule in `gate.py` | `solver/io.py`, `critics.py`, `tools/cloud/gate.py` | rows 9.3x smaller; 10^5 labels fit 16 GB | round-trip error < 1e-3 pot; identical loss curve | 4-6 h | open |
| 14 | Measure label throughput on the GTX 1070 at the calibrated T, end targets on the GPU and aux on the E5 in parallel; decide home vs rental from the measured price | `solver/loop.py`, `tools/cloud/launch.py` | the 1070 figures in the code are not reproduced anywhere; every budget line depends on them | spots per minute per street logged to `results/` | 3-4 h + 4 h of 1070 | needs the user's hardware |
| 15 | Run the spec's gate: train on exact V4 labels, single-card labels and critic-averaged `turn_end` at 10^3 / 3 x 10^3 / 10^4 and 10^4 / 3 x 10^4 / 10^5 rows, two seeds each; score in decision units on 100 held-out turn situations with the extrapolation rule fixed in advance | `solver/loop.py`, `critics.py`, the item-9 validator, `learning_curve.py` | decides whether $10-15 heads-up is reachable; RMSE under-predicts damage by 1.4-2.5x | 12 networks with best-response gain in % of the turn pot and an extrapolated label count | 8-12 h + 1-2 days of the home PC | needs the user's hardware (section 9.1 gives the shortened form: nested subsets of one day's labels, scored on exact V4 labels and in the gadget's units) |
| 16 | V-quality inputs: zero-reach label convention; board-aware, recursive, harvested and perturbed ranges tagged by source; per-SPR-band loss; suit canonicalisation at inference; per-slot projection offsets; `learning_curve` in decision units; the real turn tree in the cost model | `solver/situations.py`, `critics.py`, `learning_curve.py`, `loop.py` | synthetic ranges do not cover live play; SPR-90 rows dominate the loss | gate score by range source and SPR band, no band worse than 2x the median | 14-22 h | open |
| 17 | Measure the own-menu restrictions (`STREET_SIZES`, `COMMIT_MERGE_POT` 25 % of pot vs the spec's 67 % of stack, `MENU_BOUNDS`, `TREE_CAPS`) in decision units against a richer menu on exact river and turn solves; record the numbers next to the constants | `envs/config.py`, `solver/sizing_lab.py` | declared heuristics without a measured cost | best-response gain differences per menu change on 50 spots per street | 8-12 h | open (GPU-box job) |
| 18 | Evaluation that resolves differences: AIVAT or the win-minus-baseline control variate on Slumbot over >= 4 x 10^4 hands; LBR against the live player; a duplicate against a fixed reference agent | `arena/slumbot/metrics.py`, `arena/live_match.py`, new LBR module | the 54-hand CI was [-1 319, 4 860] bb/100; raw hands need 4 x 10^5 for +-25 mbb/hand | AIVAT unbiased on 10^4 simulated hands with variance cut >= 5x; LBR ~0 against an exact river equilibrium | 16-30 h | open |
| 19 | Preflop and depth scope: a `pre_end` depth grid or V2 for 20-500 BB; log every snap or extrapolation; estimate kappa before any rake regime; flop-start ceilings per on-menu preflop line | `solver/preflop.py`, `loop.py`, `player/live.py` | exactly 200 BB, no rake today | decision-unit gate at 3 depths | 20-40 h | open (only when depths other than 200 BB are played) |
| 20 | Multiway scope decision: implement 3-player check-down leaves plus a declared or sampled n >= 4 terminal and fix the 3-way live tree memory (6 GB), or state "heads-up plus 3-way river" as the scope | `solver/pcfr.py`, `tree.py`, `subgame.py` | 6-max and 9-max cannot be solved by this code today | a 3-way live river decision built and solved within 8 GB with E certified | 2 h decision / 40-80 h implementation | the user's decision (section 9.2, 9.3) |
| 21 | Batch-1 decision latency per street on the GTX 1070 (cold and warm, p50 / p99); workers and per-street T so that p99 is about 2 s with E reported; history off in the arena dealer | `envs/config.py`, `player/live.py`, `arena/match.py` | no GPU solve time is recorded anywhere; 142-194 fused kernels per iteration suggest a launch-bound GPU | p99 under 4 s over 1 000 decisions with the real critic | 3-5 h + 1070 time | needs the user's hardware |
| 22 | Prune dead SLOT subtrees in live trees (231 live vs 117 teacher nodes); lower T only with the certified gap reported | `solver/subgame.py`, `tree.py` | the remaining decision-speed lever after item 12 | identical policies on unpruned-equivalent spots; solve time before and after | 6-10 h | open |

## Appendix C. Where the auditors disagreed, and how it was resolved

The rule: pinned, measured numbers beat estimates, and every "replace" needs a technical necessity.

**D1. Does batching help on CPU, and what does a river iteration cost?** The solver auditor's first run said 63.7 ms per iteration at B = 1 against 22.8 ms per spot at B = 32 (batching helps 2.8x); the engine profile, pinned with `taskset`, said 14.5 / 18.7 / 20.4 / 21.3 ms at B = 1 / 4 / 16 / 32 (batching hurts). Resolution: the engine profile. The solver auditor's own pinned rerun gives 13.3 ms at B = 1 and its B = 32 figure matches. On CPU at this layout, run B = 1-4 per pinned process; confirm on the E5 (AVX, no AVX-512) before changing the loop.

**D2. Is `TEACHER_T` too high or too low?** Both: it overspends on low-SPR spots (the short tree reaches E 1e-5 at T 800; up to 4x to gain) and underspends at SPR 90 (E 1.9 % of pot, label drift 0.017 pot, worst 0.30). The iterations to E < 0.001 rise from 100 at SPR 0.5 to 2 000 at SPR 30. Resolution: T per SPR band, certificate-stopped solving above SPR 16, labels gated on E (plan item 10).

**D3. Is the missing gadget blocking?** The spec auditor said yes; the live auditor ranked it behind the match-killing failures. Resolution: both are blocking in different senses. The gadget for correctness of play (air gaining +1.0 pot per hand); the runner bugs for measuring anything at all. Both were fixed in this stage.

**D4. Live decision time on CPU.** Estimates of 15-25 s for flop and turn against measurements of 10.5 s (flop, warm), about 13 s (turn, extrapolated), 7.9 s (preflop), 4.9 s (river). Resolution: the measured numbers; the conclusion is unchanged: CPU play cannot meet 4 s without lowering T, and lowering T needs the certified gap reported (now on in play).

**D5. Where the flop iteration goes.** 77 % in the per-leaf check-down products (live auditor, live tree, B = 1) against 13-18 % leaf check-down plus 11-17 % all-in GEMMs (engine profile, B = 4). Resolution: different batch sizes and trees; at B = 1 the engine profile's terminal values are 75.5 %, consistent. Both point at one check-down GEMM per spot per iteration (plan item 12).

**D6. Label price and where to generate labels.** "10^6 river labels cost $40-140 rented, which breaks $10-15" against "run the generator on the home 1070 at a recorded 100 k river and 51 k turn/flop labels per day and rent only the overflow". Resolution: both rest on GPU speeds recorded in the code, measured by nobody here. At the recorded speeds the home 1070 gives 10^6 river labels in about 10 days and 10^6 turn labels in about 20, which conflicts with "not willing to wait two weeks"; the gate needs 10^3-10^4 exact plus 10^4-10^5 single-card labels, not 10^6. Decision: measure the 1070 (item 14), fix throughput (items 10-12), run the gate (item 15) before any rental.

**D7. rank7 build time.** The code says about 10 min on a 1070; measured 184 s on 2 contended cores (90 s projected), 2.3 min single-thread extrapolated. Resolution: 1.5-3 CPU-minutes; the docstring is stale; it matters only because the loop builds the table for river-only runs.

**D8. The rules engine says nothing is blocking, yet an illegal step silently corrupts state.** No reachable corrupting path was found (the live path checks legality, the arena asserts it, the Slumbot replay cross-checks contributions). Resolution: not blocking, fixed anyway because it costs hours and removes a class of silent errors (done, section 4.1).

**D9. Critic size.** The spec's default is 3.3 M parameters and 6.5 MFLOP; the engine stores 18.9 M and uses 4.64 M (9.3 MFLOP) because the exact F doubles the first layer's input. Not a correctness disagreement; pick the width by accuracy per FLOP inside the gate; the forward pass is 12-21 % of an iteration.

**D10. Turn tree size in cost models.** The spec priced a 28-node turn with 13 leaves; the engine's teacher turn has 40 decisions and 19 leaves, the live turn 78 and 38. Resolution: price with the engine's tree (section 6.3): about 1.5x the spec's per-iteration figure before the kernel fix.

**D11. What V error means.** The brief and the toy README called 0.15 x-units "15 % of pot"; the X1 experiment shows it is 7.5 % of the leaf pot. X1 also found that 0.3 % leaf-pot iid error adds about 0.15 % of the initial pot per hand while learned nets cost 1.4-2.5x more than iid noise of the same RMSE. Resolution: no RMSE target is accepted as a quality criterion; only the decision-unit gate decides, which is also the user's rule that regression error is not exploitability.

**D12. Commit-merge rule** (25 % of pot in the engine vs 67 % of stack in the spec). Neither has evidence. Keep the code value and measure both (plan item 17).

**Replace decisions under the user's rule.** One component was marked "replace": the opponent-range carrying with a likelihood floor in the live player. The necessity is that the construction is unsafe by design, shown by measurement (demo +1.0 pot per air hand; local loss 226-584 mbb/hand in the river-floor probe; toy 520 vs 23.7 mA/h; section 2.3 on the exact river game: +16 to +942 mbb/hand with the right range, +1 100 to +7 700 with the wrong one). The PCFR+ layout rewrite, the self-play duplicate and the critic-averaged end labels are deliberately not "replace": the first is an internal rewrite with the same algorithm and an equivalence test, the second stays as a smoke test, the third is kept until the gate measures the alternatives.

**Root causes of "the result sucked", as far as the audits can tell.** (1) The Slumbot runner ran on the CPU, where every postflop decision exceeds 4 s, and one failure stopped the match. (2) Play was exploitable by construction: no gadget. (3) The critics were never validated against exact play. (4) T was never calibrated. The engine's rules, evaluator, two-player terminals, equity matrices and Slumbot parser are not among the causes: all verified exact.

## Appendix D. Open questions only the user's hardware, logs or money can answer

- **Q1. The original card-parse crash.** Not reproduced (section 3). Needed: the crashed run's `hands/` directory, `summary.json`, stderr and the code version it ran; the cheapest test is 100-200 real Slumbot hands with the fixed runner. The old records may not show the cause because the uploaded client dropped the exception message (fixed).
- **Q2. GTX 1070 numbers.** Batch-1 decision latency per street at the play iterations (no GPU solve time is recorded anywhere; 142-194 fused kernels per iteration suggest a launch-bound GPU); label throughput at the calibrated T (the code's 0.86 s river and 1.7 s turn/flop per spot are unverified); whether the gate's footprint fits 8 GB. These set the home-vs-rent decision and the latency budget.
- **Q3. E5-2650 v2 scaling.** This box is AVX-512 with 4 cores; the E5 is AVX-only with 8 cores and 16 threads. The per-core factor (assumed 1.25-1.67x slower) and multi-process scaling at B = 1-4 need one hour on the E5.
- **Q4. The gate result.** Does any label curve (exact V4 or single-card) reach the target (about 0.01-0.02 pot per hand for the gadget; about 0.3 % of the turn pot in decision regret for the spec's gate) within the label count the budget buys? One to two days of the home PC after the throughput fixes. This single result decides whether a complete heads-up agent for $10-15 exists, or whether the deliverable is an exact turn/river engine plus a declared-regret flop/preflop map.
- **Q5. Rented GPU price at grid 13.** Measure a 3090 or 4090 for about $1 of fixed-T throughput before writing any budget line; the $40-140 for 10^6 river labels is inferred from grid-7 numbers.
- **Q6. Evaluation budget.** Resolving +-25 mbb/hand needs about 4 x 10^4 hands with AIVAT (2-7 days of the home PC per comparison at compiled speed) and competes with label generation. How many comparisons, in machine-weeks?
- **Q7. Scope.** Will stacks other than 200 BB or a rake be played before the heads-up Slumbot milestone (triggers the depth grid and the kappa estimate)? Is "heads-up plus 3-way river" acceptable as the multiway scope for now (3-way live trees exceed 6 GB; n >= 4 is refused)? The spec prices 9-max at $100-1 200 before evaluation under the full bill against the $120 ceiling: the user must pick the 3-way menu coarseness and the declared n >= 4 approximation knowing that (section 9.3).
- **Q8. Home RAM.** With fp16 k-slot rows (9.3x smaller), do 10^5 labels plus training fit 16 GB DDR3 alongside the generator? Expected yes (about 1.6 GB of rows); confirm on the machine.

## Appendix E. Sources

- Auditors' results: `results/audit/envs/`, `results/audit/solver/`, `results/audit/training/`, `results/audit/live/`, `results/audit/spec/`; the scripts that produced every number are under `tools/audit/<area>/`, each with a README giving the command lines.
- Engine profile: `results/audit/engine-profile/summary.json`.
- Gadget measurement: `results/audit/gadget/river_resolve_T800_trunk{13,34,32}.json` (right range) and `crushed_trunk{13,34,32}.json` (wrong range), with the `.log` of each run.
- Exact end-of-turn labels: `results/audit/v4/validate_v4.json`.
- Toy curve: `toy/hu/results/X1_NOISE_CURVE.md`, `toy/hu/results/x1_*.json`.
- The specification the engine was audited against: `docs/poker-core-mechanism.md`; the toy references: `toy/hu/README.md`, `toy/TARGETED.md`.
