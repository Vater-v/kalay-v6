# Poker at Its Roots: The Difficulty Is Not Where You Think

*A foundation for building a poker agent from zero on a 2015 PC plus $10–15 of rented compute: no blueprint, no card buckets, no snapping of opponent bet sizes postflop, one mechanism for robustness and exploitation, and an honest account of the places where the plan can fail, led by the one that decides heads-up.*

Everything here was either derived, measured in the two toy games under `toy/` (exact best-response numbers, pure numpy), or checked by an adversarial review that tried to refute each load-bearing claim and then reviewed the finished text again. Where a claim did not survive, the corrected statement is what appears. Where the evidence is against the plan, that is said in the same font as everything else.

**Units.** Toys: mA/h is milli-antes per hand (an exploitability or EV rate; the big-blind analogue), mA is milli-antes (a per-hand value), and "toy % pot" is per cent of the initial two-ante pot (1 mA/h = 0.05 %). Hold'em: mbb/hand, and "% of pot" means per cent of the pot at the decision being solved. A "core-hour" is one modern rented core for one hour; your $10–15 is 100–300 of them ($0.033–0.15 each) or 10–20 GPU-hours ($0.5–1.5 each). The home PC's four 2015 cores give about 670 core-hours a week, worth roughly 300–450 modern ones.

---

## 0. The answer on one page

**The architecture is not new; what people get wrong about it is.** Public belief states, depth-limited re-solving, street-boundary value networks and the ceiling gadget are DeepStack (2017) and ReBeL (2020). What gets overlooked, and what this document is about, is where the difficulty actually sits and what the solver should carry.

**Your three layers are one array.** Hidden cards, reactions and the future are three coordinates of a single object, and in the right coordinates none of them is an inference problem:

1. **The hidden layer costs nothing.** Attach to every public node (betting sequence plus board) one vector per dealt player over that player's possible private hands: the range. Along a betting line the range is a running product of the probabilities the attributed strategy gave to the actions actually taken. Bayes' rule in poker is a componentwise multiply. The belief about nine players' hands is 9 × 1326 numbers, never 1326⁹. Nothing is inferred and nothing is bucketed, at any player count.
2. **Reactions are a fixed-point condition, not a prediction.** The opponents' strategies are either what a regret-minimizing learner would do in their seat (robust) or a fitted low-dimensional model of what they actually do (exploitative). Either way the solver carries the strategy as the input to the ranges; layer 2 is the same array as layer 1.
3. **The future is a boundary condition, not a tree.** Given the ranges, the public state is a sufficient statistic for the rest of the hand, so the hand is cut at the next public chance node and the remainder replaced by a value map $V$. $V$ is learned from the bottom of the game up, river first, which is why no blueprint is needed.

**The idea people miss: never carry a belief about the opponent; carry what each of their hands is owed.** Between decisions the only state is your own range (known exactly, it is your own strategy) and one number per opponent hand: the value your last plan conceded to it, their *ceiling*. Inside each local solve the opponent's range is an adversarial variable bounded by those ceilings; it is an output of the solve, never an input. Carrying their range forward from your own equilibrium solve assumes they play your equilibrium, which is exactly what "opponents are weather" forbids. This is not a nicety: in the toy, re-solving with exact leaf values was exploitable by 23.7 mA/h when ceilings were carried and by 520 when only ranges were carried, with identical leaves. Hand reading, opponent prediction and lookahead then reduce to one regret-matching loop on a street tree of tens to a few hundred nodes heads-up (a forest of them at a street start).

**The real wall is public, not hidden.** A no-abstraction heads-up 100 bb strategy with a three-size menu needs about 0.9 PB of solver state (0.2 PB for the strategy alone). Boards multiply betting lines; hidden hands only add. So the cut at the chance node is forced by arithmetic, and the only thing a machine must *learn* is what a pair of ranges is worth when the next card comes. That regression, and how many solved examples it needs, is the whole risk of the heads-up project. The solver is not.

**What the evidence predicts.** In the toy, the best learned $V$ (20 000 exactly solved turn situations and at most 2 000 bootstrapped flop situations, range-weighted loss) was 18× more exploitable than exact leaves (417 vs 24 mA/h) and only 22 % better than a leaf with no strategic content (533); with 200 labels it was 2 % better. Going from 200 to 20 000 labels moved play within seed noise. Published systems used 10⁶–10⁷ solved situations per street. Priced honestly (an exact end-of-turn label is 48 river solves), $10–15 of CPU buys about 10⁴–10⁵ such labels per rung, or about 10⁶ single-card labels whose value per solve the toy measured at 0.25×. **Verdict for demand 5:** $10–15 buys a certified turn-and-river engine. A full heads-up agent costs 320–790 core-hours (1–3 weeks of the home PC, or $11–120 rented) and works only if the one-to-two-day gate in §6.3 shows the label curve reaching 0.3 % of pot; the prior is that it will not. A batched GPU kernel cuts the network-bound rungs by about 10× and is the one engineering item that could flip that.

**Exploitation has a free part and a priced part.** Among the many strategies with the same worst-case value, pick the one best against your forecast: in the toy that captured 5–20 % of the whole Nash-to-best-response gain at zero measurable cost. Beyond that, every gain against a forecast costs worst-case value, and the price is set by one declared risk parameter that no data can choose. The dial is a two-line change in the solver; it was verified against exactly known opponent models. Fitting the model from HUD data was not tested.

**What is exact** (heads-up): hand representation (every combo); card removal between two card-holding players, with fold and river showdown values in closed-form $O(F)$ sums and all-in leaves before the river through one exact per-board equity matrix; opponent bet sizes postflop (inserted into the tree and re-solved, never snapped; the toy residual was 1.4–18.8 mA/h, up to 10× a full augmented solve and 20–36× below snapping); the regret dynamics (no learning rate, no critic; a fixed published averaging scheme and an iteration budget); the arithmetic of ranges and ceilings.

**What is approximated.** Heads-up: the value map $V$ (dominant), the off-menu re-solve residual (measured), any ceiling infeasibility the gadget audit reports, and the preflop stack grid if the preflop map is not built. Beyond heads-up: card removal with 5+ live players (sampled, or a logged approximation), leaves for 4+-way pots (none are trained), and the folded players' dead cards wherever $V$ does not take them as input. In exploitation mode: the opponent is a 30–60-parameter model. And one thing you choose: your own bet-size menu, whose cost is measurable (§3).

**Beyond two players** the solver runs unchanged and the design does not. With no gadget theorem, multiway re-solving carries opponent ranges, the construction that was 22× worse heads-up. Each re-solve selects its own equilibrium, and recombining independent self-play runs raised exploitability up to 100× in the 3-player toy. No toy has measured a multiway re-solving agent or a learned 3-way map. Pots with 4+ live players have no trained leaf. What can be promised at 3–9 players is measured, not guaranteed: per-player regret, a measured best-response gap, and a computable coalition-security number. "Robust" at a full table has no content until you name a coalition.

---

## 1. Why the three layers are one object

### 1.1 Coordinates: a public tree with a private fiber

Textbook game theory draws poker as a tree over *information sets*. That picture is why the problem looks like nested enumeration: every one of your information sets contains one node per opponent hand, so every value appears to need a sum over their hands, and their reasoning appears to need a sum over yours.

Change coordinates. Let $\mathcal T$ be the **public tree**: nodes $x$ are public histories (every action with its exact chip amount, plus the board). Nothing private is in $\mathcal T$. Over every node sits a **fiber** for each player $i$: the set $H_i(x)$ of private hands $i$ could hold, $F = \binom{52}{2} = 1326$ combos preflop, 1176 on the flop, 1128 on the turn, 1081 on the river after removing board conflicts. A strategy for player $i$ is a *section*

$$\sigma_i : \{x : P(x) = i\} \times H_i(x) \to \Delta(A(x)),$$

one probability vector over the legal actions per public node per private hand. In memory it is an array `[public node, hand, action]`. Everything below is arithmetic on such arrays.

### 1.2 Layer 1 is a product, not a posterior

Define player $i$'s **reach** at public node $x$ with hand $h$ as the product of the probabilities $i$'s strategy assigned to $i$'s actions on the path:

$$\pi_i(x, h) = \prod_{(x', a)\ \text{on the path},\ P(x') = i} \sigma_i(x', h, a).$$

**Belief factorization (exact).** Under a uniform deal and strategies that depend only on (public history, own hand),

$$\Pr(h_1,\dots,h_N \mid x) = \frac{1}{Z(x)} \prod_{i=1}^{N} \pi_i(x, h_i)\cdot \mathbb{1}[h_1,\dots,h_N,\text{board pairwise disjoint}],$$

where $N$ counts **every player who was dealt in, including those who have folded** (their reach vectors are frozen at the fold). The product over players is the range tuple; the indicator is card removal.

Three consequences change the computational structure:

- Beliefs are **not** an inference problem. They are a by-product of whatever strategy you attribute to each player. "What are they holding?" is meaningless without "what are they playing?", and once that is fixed the answer is a multiply. A trained "hand reader" is ill-posed for the same reason.
- Beliefs are a **vector**, not a branch. One $F$-vector per dealt player, carried down the public tree. The joint over nine players has about 5 × 10²⁶ card-disjoint points (1326⁹ ≈ 10²⁸ before removal); its sufficient statistic has 11 934 numbers.
- Beliefs are exact **only relative to the attributed strategy**. If the opponent takes an action your attributed strategy gave probability zero, the update is undefined; this is the case the re-solve in §2.4 handles.

The folded players are not a detail. In the 3-player toy, replacing a folder's frozen range by a uniform one raised the solved strategy's exploitability (NashConv) from 1.1 to 144–200 mA. In hold'em, seven early folders with ordinary folding ranges raise the big blind's probability of holding an ace by about 10 % relative and the probability of an ace on the flop by about 12 % relative. The deck a late-position player faces is not uniform.

### 1.3 Layer 2 is a fixed-point condition, not a prediction

"How will they react" is $\sigma_{-i}$. There are two coherent sources for it, and the design uses both at different times:

- **Robust.** $\sigma_j$ is whatever a regret-minimizing learner in seat $j$ would play, computed in the same loop as your own strategy, as a joint fixed point. You assume nothing about them except that they are not systematically throwing money away, and §2.3 shows even that assumption is not load-bearing for the guarantee.
- **Exploitative.** $\sigma_j = \hat\sigma_j$, a low-dimensional model fitted to observed behaviour (§2.6).

Either way $\sigma_j$ feeds §1.2 and the ranges follow. Layers 1 and 2 are one array read at different indices.

### 1.4 Layer 3 is a boundary condition, not a tree

Because strategies depend only on (public node, own hand) and beliefs are determined by the ranges, the pair

$$s = (x,\ r_1(x), \dots, r_N(x))$$

is a **sufficient statistic** for the remainder of the hand (the public belief state of DeepStack and ReBeL; the fact is older than both). Two histories reaching the same public node with the same ranges have identical continuation games. So the hand can be cut anywhere. Cut it at the next street boundary and replace everything below by a map

$$V(y, r_1, \dots, r_N) \in \mathbb{R}^{N\times F},$$

the counterfactual value of every hand of every player if the rest of the hand is played by regret-minimizing players from $y$ with those ranges. For two players the hand-summed value is unique across equilibria; per-hand values are a selection (§2.2), and the training procedure in §2.5 selects what the solver would have selected.

In heads-up play, $V$ is the dominant approximation in this design; §0 and §3 list the smaller ones.

### 1.5 What was hiding behind the three layers

Two things the three-layer framing has no slot for turned out to be where the difficulty sits.

**The public tree.** Fibers add across players; boards and betting lines multiply. Heads-up 100 bb with a tapered three-size menu has 16 / 172 / 1 396 / 8 780 betting lines on the four streets and about 3.8 × 10¹⁰ public river decision nodes across all boards. With fibers, regrets and averages that is roughly 0.9 PB. A one-size menu still needs 33 TB. Preflop alone grows about 5–7× per added seat: 16 nodes heads-up, 3.9 × 10⁴ at 6-max, 7.8 × 10⁶ at 9-max with a 4-bet size, 4.5 × 10⁵ at 9-max if a 4-bet is all-in only (no open-limps in either; allowing limps exceeds 1.7 × 10⁷ already at 8 seats). No representation shrinks this; it is public chance times public action.

**Selection.** With two players all equilibria share a value but not per-hand counterfactual values, so the label $V$ must learn is set-valued at kinks (§2.2). With three or more players equilibria differ in value and nothing says which one the table is "in" (§5). An opponent model is also a selection: many strategies reproduce the same HUD counts. "Which fixed point?" is a question the user's framing never asks, and it decides stability and reproducibility.

### 1.6 The single coupled computation

At your decision at public node $x$:

- the past enters as your own range at $x$ (exact) and, for each opponent, either their ceilings (robust, §2.4) or their modelled range (exploitative, §2.6);
- the future enters only through $V$ at the street boundaries below $x$;
- the present is the current street's public betting subtree from $x$: with a human menu, tens to a few hundred nodes.

Run regret minimization on that subtree, all players at once, strategies as `[node, hand, action]` arrays, and read off your action for your actual hand. Nothing is nested and nothing is enumerated over hidden states. The only precomputed objects are the preflop table and the flop-start ceilings per on-menu line and stack depth, both built bottom-up from $V$; every later decision is re-solved from your range and the carried ceilings, so no full-game solution exists and no "delta from a full solution" is ever needed. The three layers were never three problems; they were three views of one array and one boundary map.

---

## 2. Minimal mathematical formulation

### 2.1 Objects

- Public tree $\mathcal T$ with node types *decision* ($P(x)$ acts, legal actions $A(x)$ from the menu, which shrinks with depth in the street), *chance* (a public card) and *terminal* (fold or showdown).
- Fibers $H_i(x)$, hand–card incidence $M \in \{0,1\}^{F\times 52}$, hand rank $\rho(h, \text{board})$.
- Payoff $u_i(z, h)$ at terminal $z$ given the joint hand $h$. At a fold terminal it does not depend on $h$. At a showdown it depends on $h$ only through ranks and the pot.
- Strategy profile $\sigma$, reach $\pi_i$, ranges $r_i(x) = \pi_i(x,\cdot)$ with board-conflicting hands zeroed.

### 2.2 Counterfactual values: linear at two-player terminals, never a matrix

Player $i$'s counterfactual value at $x$ for hand $h$ is the expected payoff if $i$ held $h$ and played to reach $x$ (own reach removed), weighted by everyone else's reach:

$$v_i^\sigma(x, h) = \sum_{h_{-i}} \Big(\prod_{j \ne i} \pi_j(x, h_j)\Big)\,\mathbb{1}[\text{disjoint}]\ \mathbb{E}^\sigma[u_i \mid x, h, h_{-i}].$$

**Two players.** At a terminal this is $v_1 = U_z\, r_2$ with $U_z$ an $F\times F$ matrix that is never built. The disjointness matrix is exactly

$$D = J - MM^\top + I$$

($J$ all-ones; $(MM^\top)_{hh'} = |h \cap h'|$ is 2 only on the diagonal), so $D$ is the identity plus a matrix whose rank is the number of unseen cards (52, 49, 48, 47 by street). Hence:

- *Fold terminal*, constant payoff $c$: $v_1 = c\,D r_2 = c\,(\mathbf 1^\top r_2 - M(M^\top r_2) + r_2)$. Cost $3F + 52$.
- *Showdown*: $W = S \circ D$ with $S_{hh'} = \operatorname{sign}(\rho(h) - \rho(h'))$. $S\circ J$ is a prefix sum along one global rank order; $S \circ MM^\top$ splits into 52 per-card prefix sums over the ≈ 46 hands containing each card; $S\circ I = 0$. The sorts depend only on the board and are done once per board. Per iteration the cost is $O(3F)$. Appendix A verifies both tricks against brute force to 10⁻¹² and gives the code; the structured showdown measured 85 µs against 5.8 ms for the dense matvec.
- *All-in before the river*: a fixed equity matrix $E_b$ per board (average over run-outs), so every all-in leaf in a street shares one matrix and all of them together are one GEMM per iteration. $E_b$ is a precomputed artifact: on the flop it averages 1 081 run-outs over 1 176² hand pairs (≈ 1.5 × 10⁹ comparisons per board), built once for the 1 755 canonical flops (≈ 2.6 × 10¹² operations, a few hours on the home PC) and stored as a 4.8 GB fp16 cache; on the turn it is built per board (46 rivers, 0.1–0.3 s in numpy) or replaced by per-river prefix-sum showdowns.
- *Decision node*: $v_i(x,\cdot) = \sum_a \sigma_{P(x)}(x,\cdot,a)\odot v_i(xa,\cdot)$ if $P(x)=i$, else $\sum_a v_i(xa,\cdot)$ with the actor's strategy folded into their reach.
- *Chance node*: a per-hand average over non-conflicting cards (weights in §2.5).

**Fiber locality.** Every nonlinear operation of regret matching (positive part, normalization) acts on one (node, hand) cell. Every operation that couples different hands is a fixed, strategy-independent map at terminals and chance nodes: linear for two card-holding players, **multilinear of degree $N-1$** in the opponents' reach vectors for $N \ge 3$ (§5.2). The depth-cut map $V$ is the one hand-coupling operation that is *not* linear: it is positively 1-homogeneous in each range, concave in your own and convex in the opponent's, and its best-response per-hand values are its super- and sub-gradients (Danskin; on-policy values of zero-reach hands can lie below them), with the exact identity

$$\langle r_1, v_1\rangle = V = -\langle r_2, v_2\rangle .$$

Per-hand values can jump. In the clairvoyance river (pot 1, one pot-sized bet, bettor holds the nuts with probability $q$ and air otherwise), the value of air is 0 for $q < 2/3$ and a full pot for $q > 2/3$, and any value in $[0,1]$ at $q = 2/3$. An $\varepsilon$-equilibrium pins it only to within $O(\varepsilon/|q - 2/3|)$. The labels $V$ must learn are therefore discontinuous and solver-dependent where ranges cross such thresholds. This is one reason $V$ is hard to learn (§6), and it is why the toy's only effective training lever was weighting the loss by the player's own range: per-hand values of hands a player rarely holds are not pinned down by the equilibrium and dominate an unweighted error.

So one pass over a public subtree with $n_{\text{nodes}}$ nodes costs $O(n_{\text{nodes}}F)$ in the interior plus the leaf kernels: $n_{\text{nodes}} \times F$, not $n_{\text{nodes}} \times F^2$ and not $F$ copies of the tree. This is the arithmetic reason the hidden state never multiplies the tree by itself.

### 2.3 The dynamics, and why "opponents are weather" is a theorem

At each decision node $x$ of player $i$, for each hand $h$ and action $a$, keep a cumulative regret $R(x,h,a)$. One iteration of regret matching⁺:

$$R \leftarrow \max\Big(0,\ R + v_i(xa, h) - \sum_{a'} \sigma_i(x,h,a')\,v_i(xa',h)\Big),\qquad \sigma_i(x,h,\cdot) \propto R(x,h,\cdot)^+ .$$

Average the iterates with linearly increasing weights (CFR⁺), or use the discounting of DCFR (published exponents 1.5 / 0 / 2, not tuned here; DCFR is the default in this document, CFR⁺ and predictive RM⁺ are alternatives). There is no learning rate, no step size, no critic. In two-player zero-sum play the only knob is the iteration count, and exploitability falls with it in trend, not strictly monotonically. At 3+ players more iterations can make things worse (the toy drifted upward after 7 000; stop at the measured plateau). After any solve, one extra pass computes the exact best-response gap *of the local game* (with $V$-valued leaves and the gadget root): a certificate of solver residual, not of real-game exploitability.

**Theorem (external regret; Zinkevich et al. 2007).** If player $i$ plays $\sigma_i^1,\dots,\sigma_i^T$ by the update above and the others play *any* sequence $\sigma_{-i}^1,\dots,\sigma_{-i}^T$, chosen with full knowledge of $i$'s past if they like, then

$$\frac{1}{T}\sum_{t} u_i(\sigma_i^t, \sigma_{-i}^t) \ \ge\ \max_{\sigma_i^*} \frac{1}{T}\sum_{t} u_i(\sigma_i^*, \sigma_{-i}^t) - \frac{\Delta\,|\mathcal I_i|\sqrt{|A|}}{\sqrt T}.$$

No assumption about the opponents appears. Against whatever the environment actually did, you did almost as well as the best single fixed strategy would have done against that same environment. Three regimes follow from the one update:

1. **Two-player self-play.** Average strategies converge to a Nash equilibrium, in practice at about $1/T$ with CFR⁺ (toy: $T^{-1.65}$ between 100 and 2 000 iterations; vanilla CFR $T^{-0.85}$, 42× more exploitable at equal time).
2. **Fixed or modelled opponent.** The best fixed strategy in hindsight is the best response; the update converges to it.
3. **Three or more players.** The per-player statement holds verbatim (toy: average regret fell as $1/T$ for all three seats). What is lost is the joint conclusion: the empirical distribution of iterates converges to a coarse correlated equilibrium, which needs the iteration index as a shared correlation device and says nothing about the product of average marginals. §5 shows what actually happens.

Two honest limits of the theorem. First, the bound's constant is numerically empty at poker scale (for 9-max push/fold it needs about 10¹⁸ iterations to certify 0.01 bb), so every certificate is *measured* by a best-response pass, never quoted. Second, the comparator is the best *fixed* strategy against the realized sequence; against an opponent who adapts to you, that is the wrong counterfactual, and policy regret against unbounded-memory adaptation cannot be made sublinear (Arora, Dekel, Tewari 2012). Weather does not change because you carry an umbrella. Opponents do. §7 states what survives.

**The guarantee consumes counterfactual feedback.** The update needs $v_i(xa, h)$ for every node and every hand, which requires the opponent's strategy at every node, including hands they did not hold and lines not taken. A simulator gives this for free. A real table gives one trajectory per hand with hidden cards revealed only at some showdowns. Learning your strategy directly from table feedback is bandit optimization in a space of dimension $d \approx 10^5$ per street; its sample complexity is of order $d L^2/\epsilon^2 \approx 4\times10^9$ hands for 1 bb/hand accuracy. The toy confirmed the sign: a model-free online regret learner was still worse than the frozen self-play strategy after 10⁶ hands against a soft-playing coalition, and lost 0.6–1.3 antes per hand while learning. **"Learn to beat the table online" is not a mechanism.** The opponent must enter through a low-dimensional model (§2.6), and the regret machinery runs against that model in the simulator, where feedback is counterfactual.

### 2.4 Cutting the hand: the local problem and its safety condition

Fix the current street and your public node $x$. Let $B(x)$ be the street-boundary nodes below $x$ (betting closed, next card not yet dealt) and $Z(x)$ the fold and all-in terminals within the street. The **local problem** is regret minimization on the subtree rooted at $x$ where every $y \in B(x)$ returns

$$v_i(y,\cdot) := V_i\big(y, r_1(y), \dots, r_N(y)\big)$$

instead of recursing. The ranges at $y$ are the ranges at $x$ times the strategies chosen inside the subtree, so $V$ is evaluated inside the loop on the current iterate's ranges.

**The past, robust version (two players).** Your own range at $x$ is your own reach, known exactly. For the opponent, carry **not** their range but the vector $\bar v_2(x, h)$ of their per-hand counterfactual values at $x$ from your previous solve. At the root of the new solve give the opponent, per hand, a choice between *Terminate* (take $\bar v_2(x,h)$) and *Follow* (play the subtree). Their range at $x$ is then a variable chosen adversarially subject to those ceilings. This is the re-solving gadget (Burch, Johanson, Bowling 2014; DeepStack). If the ceilings are attainable (the previous strategy met them in the current tree), any exact solution satisfies $u_2(h) \le \bar v_2(h)$ for every hand, which is exactly the condition under which replacing your old plan below $x$ by the new one does not increase the opponent's best-response value anywhere above $x$; otherwise the gadget minimizes the excess, which the gadget audit measures. Exploitability of the combined strategy is bounded by the starting strategy's plus the solver residual plus the ceiling error, linearly over re-solves.

Why this is not optional, in one example. River, pot 1. The opponent holds the nuts or air; all your hands are bluff-catchers. Suppose your carried range for them says "nuts only". A range-only re-solve folds every hand to every bet. If air is in fact present (they deviated upstream, or the carried range was wrong), air bets and takes the pot every time. The gadget sets air's ceiling at 0 (its value under your previous plan, which called at the minimum defence frequency) and forces the re-solve to keep calling half the time however improbable air is believed to be. The toy's whole-game measurement is in §8.1; the naive agent's out-of-position seat was exploitable by 757 mA/h.

**The past, exploitative version.** When a model $\hat\sigma_j$ is in play, the opponent's modelled range is $r_j^{\text{model}} = b_0 \odot \prod_k \hat\sigma_j(a_k \mid \cdot)$ along the hand, and it enters as a *second type* of opponent alongside the adversarial one (§2.6). The gadget is still present for the adversarial share.

**Off-menu opponent bets.** If the opponent bets a size not in your menu at node $p$, re-solve from the *opponent's* node $p$ with the observed size added to their action set and the ceilings $\bar v_2(p)$ that held at $p$. The solve decides which of their hands would prefer that size given your replies. Carry forward your own range at the reached child and the opponent's per-hand values there as the new ceilings; their range stays an adversarial variable (their reach after the bet is readable from the solve, as §4.2 shows, but it is not what is carried). Nothing is snapped. Cost: one extra re-solve of a tree with a handful more nodes. Toy: inserting the size cost 1.4–18.8 mA/h against an exploiter, versus 1.8–2.2 for the responder's strategy from a full solve of the augmented game and 51–389 for snapping to the nearest menu size. Snapping was not even monotone in quality: for a 1.5-pot bet, snapping to the nearest size lost 51 and snapping up lost 236. The residual (up to 10× the augmented-Nash figure) was measured with exact leaves; with learned leaves it inherits their error.

### 2.5 The blueprint-free ladder, and where exactly the cut goes

$V$ is defined by a recursion that terminates at the river, where there is no boundary. So:

- **River.** Exact. A river re-solve has no $V$.
- **End of turn (before the river card).** Needs $V^{(4)}$. Label by sampling boundary situations (4-card board, pot, stacks, two ranges) and solving the river subgames exactly.
- **End of flop (before the turn card).** Needs $V^{(3)}$. Label by solving turn subgames with $V^{(4)}$ at their boundaries.
- **Preflop.** Either a third map $V^{(2)}$ at the end of preflop, or solve the preflop tree offline with *sampled* flops: each iteration deals one flop per boundary leaf, solves that flop subgame with $V^{(3)}$ leaves, and feeds the value back. This is public-chance-sampled CFR, unbiased only with the flop weight: draw $f$ uniformly from all 22 100 flops and feed back $\frac{22\,100}{17\,296}\,\mathbb 1[f \cap h = \emptyset]\,v_f(h)$ (17 296 = $\binom{48}{3}$ flops are disjoint from both hands), with the opponent's $f$-blockers zeroed in $v_f$. The preflop tree is 16 nodes on a 169-class fiber; at about 10⁴ iterations × 7–9 flop leaves × 0.3–0.6 core-s per flop solve this is 6–15 core-hours per stack depth, so 60–150 core-hours for a 10-point log grid from 20 to 500 bb.

  **Price of dropping $V^{(2)}$.** Preflop becomes a precomputed table per stack-grid point. An off-menu preflop size, a limp, a stack depth between grid points, or (at 9-max) unequal stacks cannot be re-solved at play time: they are mapped to the nearest solved configuration, which is snapping, with the cost snapping showed in the toy's E3 experiment (51–389 mA/h, §8.1). $V^{(2)}$ (one more learned rung and one more compounding step) removes the preflop part of this: the 16-node preflop tree can then be re-solved in under a second with the actual sizes and stacks. It does not remove the flop-start part below, whose forests stay precomputed per on-menu line and grid depth. Choose, and log every snap.

**Where the cut sits relative to the card.** Cut *before* the chance node. The boundary value is then the chance average of independent per-card subgames,

$$v_{\text{pre}}(h) = \frac{1}{N_{\text{left}} - 4}\sum_{c \notin h} v_c(h),$$

where $N_{\text{left}}$ is the number of cards not on the board (49 after the flop, 48 after the turn) and $v_c$ is the unnormalized counterfactual value of the post-card subgame with the opponent's $c$-blockers zeroed. The divisor is 45 from flop to turn and 44 from turn to river for two dealt players (with $N_{\text{dealt}}$ dealt players it is $N_{\text{left}} - 2N_{\text{dealt}}$, and folded players' ranges make the weights hand-dependent). Dividing by all 49 / 48 unseen cards shrinks every label by 0.92 per cut, 0.84 over two cuts; averaging over the 47 / 46 cards not in $h$ shrinks by 0.96 per cut. (The toy confirmed the identity exact to machine precision with the right weights; with uniform weights its single cut shrank labels to 0.84. The hold'em figures are derived, not measured.) An unbiased single-card label is $\frac{N_{\text{left}}}{N_{\text{left}}-4}\,\mathbb 1[c\notin h]\,v_c(h)$ with $c$ uniform over the $N_{\text{left}}$ cards. Inside a lookahead, one net call per boundary leaf per iteration instead of 47–49.

**Two kinds of label, and what each costs.** An *exact* (all-card) end-of-turn label is 48 river solves (one per unseen card; 44 and 45 are the per-hand weights, not the solve counts); an exact end-of-flop label is 49 turn solves. A *single-card* label is one solve but carries the whole next-card variance. In the toy, at equal compute, exact all-card labels beat single-card labels (range-weighted error 0.153 vs 0.180 at 18 000 solves); the nominal 9× saving came out as 0.25×. The toy is a worst case (one-card hands make the next card decide everything), and a per-card control variate was not tested, so the hold'em exchange rate is open and is one of the two things the gate in §6.3 measures.

**Carrying the ceilings across a card.** A pre-chance $V$ outputs *chance-averaged* opponent values, and those are not valid per-card ceilings: a card that favours you leaves the averaged ceiling above the true per-card value, and the re-solve may concede up to it. In the toy, averaged ceilings scored 608 mA/h, worse than having no strategic leaf at all. A separate "start-of-street solve to produce per-card ceilings" also fails, structurally (62–93 at any iteration count), because it selects a different equilibrium than the one the previous solve committed to. What works is the chance-rooted gadget: root the gadget re-solve at the *pre-chance* node with its averaged ceilings, so all next cards sit in one forest and the opponent's Terminate/Follow choice is made per hand before the card. Toy: 25.2 vs 23.7 for exact per-card ceilings. Its cost depends on the street:

- *Turn start*: 49 turn cards × the 28-node street × 13 $V^{(4)}$ leaves = 637 net calls plus 49 copies of the street arithmetic and 49 all-in GEMMs per iteration: 45–75 s in numpy; compiled 20–35 s with a 3.3M-parameter net, 5–10 s with a 1M-parameter net and per-river prefix-sum all-in leaves. Affordable with a time bank.
- *River start*: 48 exact river subgames in one forest, no network: about 48 s in numpy, about 2 s compiled.
- *Flop start*: the pre-chance node spans 22 100 flops (1 755 classes) × 13 $V^{(3)}$ leaves ≈ 23 000 net calls per iteration, about 44 TFLOP at 300 iterations. **Not affordable at play time.** The ceilings for the first flop decision must therefore be precomputed: for each on-menu preflop line and each stack-grid point, run the chance-rooted flop-start solve once offline with the preflop solve's averaged ceilings (10–30 min each; about 70 of them for 7 lines × 10 depths) and store the opponent's per-hand values at each flop-root child (≈ 8 MB per line per depth). After an off-menu preflop line there is no precomputed forest; the fallback is a start-of-street solve, declared unsafe at the toy's measured 62–93 vs 24 mA/h, or a snap to the nearest precomputed line. Whether precomputed forest ceilings behave like the toy's chance-rooted ceilings at hold'em scale is untested.

**What makes $V$ learnable, and what does not.**

- *Symmetry.* Suit permutation reduces 22 100 flops to 1 755 classes and 4-card boards to 16 432 classes (the unordered set is the right input object for $V$; the ordered flop-plus-turn count of 63 193 is the right object for counting public nodes). Postflop fibers compress only 1.1–1.4×, so symmetry canonicalizes $V$'s inputs and augments its data; it does not shrink the solver.
- *An exact baseline.* The check-down value, each hand's equity against the opponent's range times the pot, is exact, bilinear in the ranges, commutes with chance, and equals $V$ at stack-to-pot ratio zero. Learn the residual. In the toy the residual target lowered regression error (0.230 vs 0.315) but **did not improve play** (499 vs 506 mA/h). Treat it as architecture, not as a data multiplier.
- *Homogeneity and the zero-sum identity.* Exact constraints you can impose on the architecture (normalize ranges; project outputs onto $\langle r_1, v_1\rangle + \langle r_2, v_2\rangle = 0$).
- *Per-hand weight sharing.* Each solved situation yields about 2 × 1128 per-hand targets for one shared per-hand model over exact showdown-profile features. Plausible as a data lever; unmeasured.
- *Range-weighted loss.* The one lever that helped in the toy (417 vs 499).
- *Smoothed targets.* Replacing Nash by an entropy-regularized equilibrium makes the labels unique and Lipschitz at a declared exploitability price $\tau(\Psi_1^{\max}+\Psi_2^{\max})$. Examined and demoted: the standard own-reach-weighted regularizer breaks concavity and homogeneity of $V$, the constant $\Psi^{\max}$ is of order hundreds on a single river decision rather than $O(1)$, and at the $\tau$ a label budget can resolve the learnability gain is negligible. An optional priced dial, not part of the core.

**Interface, so a builder can run the gate.** Input: the two ranges over the 1 326 combo slots (board-blocked combos zero), each normalized to sum 1, after suit canonicalization of the board; the board as a 52-bit vector; SPR. Output: 2 × 1 326 per-hand counterfactual values in pot units for unit opponent mass; multiply player $i$'s output by the opponent's unnormalized mass (degree-1 homogeneity), then project so that $\langle r_1,v_1\rangle + \langle r_2,v_2\rangle = 0$. Default net: an MLP with 3 × 512 hidden units (≈ 3.3M parameters, 6.5 MFLOP per call); try a 1M-parameter net at the gate, since the street-start forests are network-bound. Loss: $\sum_i\sum_h \bar r_i(h)\,(\hat V_i(h) - v_i(h))^2$ with $\bar r_i$ player $i$'s own normalized range (Huber optional). Data: half DeepStack-style recursive random ranges, half ranges harvested from your own solves, plus perturbed ranges; SPR stratified log-uniformly over 0.1–100; label solves at 200 DCFR iterations; 24× suit augmentation on the fly; about 10.6 GB per 10⁶ examples in fp16, streamed from disk.

### 2.6 Exploitation is the same loop with one input swapped

Let $\hat\sigma_j$ be a model of opponent $j$. Inside §2.3's loop, replace $j$'s iterate by the mixture

$$\sigma_j^t \leftarrow p\,\hat\sigma_j + (1-p)\,\sigma_j^{t,\text{regret}},\qquad p\in[0,1].$$

In sequence form the opponent now lies in the Huber set $\{(1-\varepsilon)\hat y + \varepsilon y\}$ with $\varepsilon = 1-p$, and because your payoff is linear in their sequence-form vector the max-min over that set is exactly the restricted Nash response (Johanson, Zinkevich, Bowling 2008). $p = 0$ is equilibrium, $p = 1$ is the best response to the model, and $p$ sweeps the Pareto frontier of EV-against-model versus your own exploitability. In the toy this is a two-line change to the opponent reach in the solver; "RNR" below means this restricted Nash response.

**The frontier has a free part.** Maxmin strategies form a face, not a point. Choosing among them the one best against your forecast costs no worst-case value. The toy measured this at $p \to 0^+$ (experiment T5, §8.1): +179 mA/h against the maniac at $p = 0.001$, +179 against the station and +138 against the nit at $p = 0.01$, with exploitability at the solver's noise floor (3.4–4.4 vs 3.6–4.2). That is 5–20 % of the whole Nash-to-best-response gain for free. Only beyond the face does every gain cost worst-case value.

**The priced part needs a declared number.** A Bayesian posterior mean over opponent strategies produces no caution at all: payoff is linear in the predictive strategy, so a wide posterior and a point estimate give the same best response. Caution must come from a declared quantity: $\varepsilon$ (the mass of behaviour the model is allowed to be wrong about), or a confidence level on a constraint set, or a budget of banked gifts (Ganzfried, Sandholm 2015: risk only what the opponent's past mistakes have already paid). This contradicts the user's "no arbitrary parameters" exactly here, and it cannot be engineered away: no finite sample bounds how an opponent will deviate from their past in the future. The gift-budget rule, risking only what the opponent's past mistakes have paid, is the one parameter-free choice, and its price is slow exploitation of opponents who make few mistakes.

**Where the model comes from (the HUD, done honestly).** You never observe hands except at some showdowns, so you cannot fit a free $F$-dimensional strategy per node. What you observe is action frequencies per public context. Two principled forms, both convex inside the same solve:

1. *Frequency constraints.* A HUD statistic is a ratio of linear functionals of the opponent's sequence-form vector. With your own historical play held fixed, "fold to flop c-bet $= \hat f \pm \varepsilon$ over $n$ observations" is two linear inequalities on their realization plan. The local problem becomes $\max_x \min_{y \in Y,\ Ay \le b} x^\top U y$, solved by adding one Lagrange multiplier per constraint on your side (Davis, Waugh, Bowling 2019). The opponent is then the worst case consistent with what was seen, and still chooses adversarially *which* hands take which action. $\varepsilon$ from counting plus the across-board dispersion of that statistic in your own solves. The confidence level is the declared parameter.
2. *An exponential tilt of the solver's own strategy.* $\hat\sigma_j(a \mid x, h) \propto \sigma_j^{\text{ref}}(a\mid x,h)\,e^{\lambda_{c(x),a}}$, with one parameter per (context class, action), 30–60 parameters per opponent. This is the information projection of the reference strategy onto the observed frequencies: HUD counts are its sufficient statistics; $\lambda = 0$ is the equilibrium, so with no data you play robustly; and a frequency deviation is attributed to the hands the reference *mixes* with, so "bets the river too often" is read as "more bluffs", not "more value", without anyone designing that rule. Fit by EM with hands latent (the E-step is a range computation you already have). Non-stationarity: discount the counts, and run a small mixture over half-lives weighted by predictive likelihood (log-loss regret $\le \ln K$ against the best half-life in hindsight, for any sequence). Showdown data is not missing at random (players muck losers), so fit from action likelihoods unless showdowns are forced.

Sample complexity is the honest limit of exploitation. Detecting a 10-point frequency deviation at 2σ needs about 100 opportunities per parameter: roughly 400–1 000 hands heads-up, 3 000–10 000 hands against one specific opponent at a full table.

**Untested.** None of the model-fitting machinery above was run; the toy's E4 and T5 used exactly known opponents. The 30–60 context classes are a chosen coarsening of the opponent, and the EV it leaves on the table is unmeasured. **Exploitation also moves $V$ off its training distribution**: exploitative play produces ranges unlike equilibrium ranges, where $V$ has seen less data. Train $V$ on random, harvested and perturbed ranges, and increase $p$ slowly.

---

## 3. The honest ledger

| Component | Status | Notes |
|---|---|---|
| Hand representation | **Exact** | All combos, no buckets, ever. |
| Beliefs | **Exact given the attributed strategy** | A product of known numbers over all dealt players, folded ones included. Undefined for zero-probability actions; handled by re-solving. |
| Card removal, 2 card-holding players, nobody folded | **Exact** | $D = J - MM^\top + I$; $O(3F)$ per terminal per iteration. |
| Card removal, 3 live players, nobody folded | **Exact** | Fold terminals verified (10⁻¹³) at $O(52^3+F)$; the $O(52F)$ showdown sweep is derived, not yet implemented (ties and side pots need extra accumulators). |
| Card removal, 4 live players | Exact offline, $O(52^3 F)$ | Too slow for play-time; sample there. |
| Card removal, 5+ live players | Sampled (unbiased) or a **declared approximation** | Independence costs 0.3–3 % of pot mean error and 10–15 pp in blocker-heavy spots; in a solver it was a 24–65× exploitability error in the toy (§5.2). |
| Folded players' dead cards | Carried in the belief state; **approximated** at terminals and in $V$ unless $V$ takes them as input | Toy: dropping them cost 128–178× (§1.2, §5.3). |
| Opponent bet sizes, postflop | **Exact** | Inserted and re-solved. Measured residual 1.4–18.8 vs 1.8–2.2 mA/h for the augmented-game solution; snapping 51–389. |
| Opponent bet sizes, preflop; stack depth | **Exact only with $V^{(2)}$** | Without it, a precomputed table per stack-grid point: off-menu preflop sizes, limps and off-grid stacks are snapped (declared). |
| Your own bet sizes | **Chosen** | A restriction of your actions. Measured on three river spots against a 6-size menu: a tapered 3-size menu loses less than the solver can resolve (≲ 0.25–0.4 % of pot); 1 size loses 0.07–2.6 %, growing with SPR; jam-only 0.6–6 %. Multi-street and continuous-size losses unmeasured. |
| Regret dynamics | **Exact arithmetic; fixed published averaging scheme; iteration budget** | Certified gap of the local game after every solve. |
| Past (earlier streets), 2 players | **Exact summary** | Your range plus opponent ceilings; ceilings must be attainable (audited). |
| Street-start ceilings | Turn and river: chance-rooted gadget at play time. Flop: **precomputed offline per on-menu preflop line**; off-menu preflop lines fall back to a declared-unsafe start-of-street solve | §2.5. |
| Past, 3+ players | **Tracked ranges, no safety theorem** | §5.3. |
| Future (later streets) | **Approximated: $V$** | The dominant approximation. Error measurable on held-out exact solves. Its certified bound is useless (≈ 250 mbb/hand at 1 % sup-norm error); its practical effect must be measured in decision units. |
| Opponent model | **Model**, exploitation mode only | 30–60 parameters, likelihood-fit, prior at equilibrium, discounted. Misspecification bounded by $p$. Fitting untested. |
| Safety-exploitation trade-off | **Declared parameter** | Free part costs nothing; priced part needs $\varepsilon$, a confidence level, or a gift budget. |
| 3+ players: equilibrium | **No guarantee** | Per-player regret holds; NashConv measured; coalition security computable. |

---

## 4. A real decision, step by step

### 4.1 Heads-up no-limit, 100 bb, with a human menu

**Menu** (fixed per street, shrinking with depth; any size committing ≥ 67 % of the remaining stack becomes all-in):

| Street | First bet | Raise (depth 1) | Depth ≥ 2 | Cap |
|---|---|---|---|---|
| Preflop | open 2.5 bb; 3-bet 3× | 4-bet 2.25× | all-in | 4 |
| Flop / turn / river | 33 %, 75 %, 150 % of pot | 75 % of pot-after-call | all-in | 3 |

With this menu one heads-up postflop street has **28 decision nodes, 26 fold terminals, 13 street-boundary leaves and 14 all-in leaves** at 100 bb, and the same counts at 500 bb because the taper ends in all-in. (The count follows the recursion $Q(\text{cap}) = 1$, $Q(d) = 2 + m_d Q(d+1)$, $n_{\text{street}} = 2 + 2(m_0 Q(1) + 1)$, with $Q(d)$ the number of decision nodes below "facing the $d$-th bet" and $m_d$ the number of non-all-in sizes at depth $d$; three sizes at every depth with cap 4 gives up to 322, 206 enumerated at 100 bb and 320 at 500 bb because all-in collapses prune the tree.)

**Hand.** Button opens 2.5 bb, you call in the big blind. Pot 5, stacks 97.5. Flop K♠ 7♦ 2♣ (rainbow, distinct ranks, so all 1 176 live combos are distinct fiber entries). You check. Button bets 3.75 (75 %). You hold 8♦ 7♣.

**Step 1: the public node.** $x$ = [BTN 2.5, BB call | K♠7♦2♣ | check, bet 3.75]. Pot 8.75, you face 3.75, button has 93.75 behind.

**Step 2: what you carry.** Your range: your preflop probability of calling with each combo (a lookup in the offline preflop solve) times your probability of checking this flop with it (from the re-solve you ran one node ago). Two numbers multiplied per hand. The button's *ceilings*: their per-hand counterfactual values at $x$ from that same re-solve (which solved the whole flop subtree, both players, so their reaction to your check is already in hand). The re-solve one node ago was the first flop decision, so its own ceilings came from the precomputed flop-start forest for this preflop line and stack depth (§2.5). You do not carry a button range. If a model were active, the button would in addition get a model type with range $r^{\text{model}} = \text{open prob} \times \hat\sigma(\text{bet 75 \%} \mid \cdot)$.

**Step 3: the local tree.** From $x$: fold, call, raise to 75 % of the pot after a call (call 3.75, pot 12.5, raise 9.375 on top), or jam. After a raise the button folds, calls or jams; after either jam the other player folds or calls. That is 4 decision nodes ($Q(1) = 4$ in the recursion), 4 fold terminals, 2 street-boundary leaves (call; raise–call), each valued by $V^{(3)}(y, r_{BB}(y), r_{BTN}(y))$, and 2 all-in leaves (raise–jam–call; jam–call) served exactly by the run-out equity matrix. One net call per boundary leaf per iteration because the cut is before the turn card.

**Step 4: the fixed point.** Run 300 iterations of DCFR with the gadget at the root. Per iteration: 4 nodes × 1 176 hands × ≤ 3 actions of regret arithmetic (trivial), 4 fold terminals in $O(3F)$ (trivial), 2 network calls (≈ 13 MFLOP with the 3.3M-parameter net) and one all-in GEMM over 2 leaves for both players (≈ 11 MFLOP). About 25 MFLOP per iteration, 7.5 GFLOP per decision: **under a second in batched numpy on a 2015 quad-core.** A full 28-node street from its root is about 85 MFLOP of network plus 77 MFLOP of all-in GEMM per iteration and takes 1.5–3 s in numpy, 0.6–1 s compiled. River decisions have no network (under 1 s numpy, 40 ms compiled). The *first* decision of the turn, which must carry the ceilings across the card, runs the chance-rooted forest: 45–75 s in numpy, 20–35 s compiled with this net, 5–10 s with a 1M-parameter net.

**Step 5: the action.** Read $\sigma_{BB}(x, 8♦7♣,\cdot)$ from the averaged strategy and sample. (Frequencies are not quoted here because this solve was not run at hold'em scale; §4.2 shows a complete decision with real numbers from the toy.) Note what did not happen: nobody asked "what does he have?". The button's hands entered as an adversarial variable bounded by ceilings, exactly, in every value computation.

**Off-menu.** Had the button bet 2.2 bb (44 %), add that branch at the button's node, re-solve the flop street from your check decision with the ceilings that held there, carry forward your range and the button's per-hand values after the 2.2 bet as the new ceilings, and proceed to Step 1. One extra re-solve of a tree with four or five more nodes.

**Exploitation.** Suppose 150 observed spots say this button folds to flop check-raises 62 % of the time, against roughly 45 % for the robust solve. Add the constraint $f_{\text{fold}} \in [0.62 - \varepsilon, 0.62 + \varepsilon]$ at the button's node after your raise with $\varepsilon = 1.96\sqrt{0.62 \cdot 0.38/150} + \varepsilon_{\text{pool}} \approx 0.08 + 0.05$, or fit the tilt parameter for that context and run the mixture at some $p$. Your raise frequency with 8♦7♣ rises, by an amount bounded by what the worst case inside the band could punish. Same tree, same fiber, same network, same loop.

### 4.2 A complete decision with real numbers (toy game)

The same mechanism, run end to end in `toy/hu` (`python3 experiments/walkthrough.py`, 23 s on one core; full output in `toy/hu/results/walkthrough.txt`). The game: 10 cards (Ts Th Js Jh Qs Qh Ks Kh As Ah), one private card each, antes of 1, stacks of 12, menu {½ pot, pot, all-in} → {pot-raise, all-in} → {all-in}. The agent is P0, out of position. Oracle leaves (the turn is solved exactly below the root), 300 CFR⁺ iterations per re-solve. The single-root chain here reproduces the batched E2c agent's strategy to a maximum difference of 0.0 at every step.

**The chain so far.** [1] Preflop root (the whole game, 54 957 nodes, 20 s; this is the only solve the oracle agent pays once per hand). [2] Flop street root after check–check and the flop card Ks (2 082 nodes, 0.57 s), with the gadget fed the opponent's values from [1]. Now:

**(a) The public node.** Preflop: P0 checks, P1 checks. Flop Ks: P0 checks, P1 bets 2 (one pot). Pot 4; P0 has 11 behind, P1 has 9. P0 to act, facing 2 into 2 (needs 33.3 % equity). Legal: fold, call, raise (pot), all-in.

**(b) The agent's own range, carried.** Product of its own preflop-check and flop-check probabilities from [1] and [2]:

| hand | Ts | Th | Js | Jh | Qs | Qh | Ks | Kh | As | Ah |
|---|---|---|---|---|---|---|---|---|---|---|
| preflop check | 0.749 | 0.749 | 0.726 | 0.725 | 0.447 | 0.448 | 0.403 | 0.403 | 0.324 | 0.323 |
| flop check | 0.720 | 0.720 | 0.669 | 0.669 | 0.470 | 0.471 | — | 0.502 | 0.189 | 0.198 |
| **range** | 0.539 | 0.539 | 0.485 | 0.485 | 0.210 | 0.211 | board | 0.202 | 0.061 | 0.064 |

**(c) The opponent's ceilings, from [2]** (what each opponent hand is worth at this node under the previous solution; in the re-solve each hand may take this instead of entering):

| hand | Ts | Th | Js | Jh | Qs | Qh | Ks | Kh | As | Ah |
|---|---|---|---|---|---|---|---|---|---|---|
| ceiling $\bar v$ (mA) | −1.92 | −1.92 | −0.34 | −0.34 | 1.57 | 1.57 | — | 8.83 | 4.79 | 4.79 |

No opponent range is passed in. These two vectors are the only state carried between decisions.

**(d) Local subtree.** With oracle leaves: 390 public nodes (148 decision, 2 chance, 112 fold, 128 showdown), 1 188 information sets. The same root cut at the end of the flop, as a value network would see it: 12 public nodes (4 decision, 4 fold, 2 showdown, **2 street-boundary leaves**: after our call, and after our raise and P1's call).

**(e) The re-solved strategy, every hand** (rows with larger range matter more; last row is range-weighted):

| hand | range | fold | call | raise | all-in |
|---|---|---|---|---|---|
| Ts, Th | 0.539 | 1.000 | 0 | 0 | 0 |
| Js, Jh | 0.485 | 0.659 | 0.267 | 0.072 | 0.003 |
| Qs, Qh | 0.210 | 0.006 | 0.989 | 0.004 | 0.001 |
| Kh | 0.202 | 0 | 0.566 | 0.413 | 0.020 |
| As | 0.061 | 0 | 0.837 | 0.153 | 0.010 |
| Ah | 0.064 | 0 | 0.846 | 0.144 | 0.010 |
| **frequency** | | 0.615 | 0.320 | 0.062 | 0.003 |

The full-game solve [1] at the same node (a different equilibrium, so only roughly comparable): 0.610 / 0.312 / 0.058 / 0.020. **Safety check**: the opponent's exact best response against this strategy sits at the ceiling for every hand (largest excess 0.071 mA against ceilings up to 8.83 mA; CFR⁺ residual at 300 iterations). The re-solve gives the opponent what [2] allowed and no more.

**(f) The opponent's range, as a by-product.** The gadget player's own Follow probability per hand, read off the solution, is the opponent's reach at this node: 0.736 / 0.741 / 0.257 / 0.257 / 0.419 / 0.415 / — / 0.580 / 0.273 / 0.283 for Ts…Ah, i.e. a range of 18.6 / 18.7 / 6.5 / 6.5 / 10.6 / 10.5 / — / 14.6 / 6.9 / 7.2 %. It was never an input. The naive agent, handed its own carried range instead (19.7 / 19.5 / 6.4 / 6.4 / 10.2 / 10.1 / — / 15.2 / 6.3 / 6.3 %) and no gadget, folds Js and Jh only 40 % instead of 66 % and calls 42 % overall instead of 32 %; over the whole game that agent is exploitable by 520 mA/h against 23.7 for the safe one (§8.1).

**(g) Wall-clock.** This decision 212 ms (390 nodes, 300 iterations). The flop root 565 ms. The preflop root 20.1 s. (Timings vary by a few per cent between runs; the strategies do not.)

**Off-menu.** P1 instead bets 0.75 pot after our check, a size not in the menu. The agent has no values for that branch, so it re-solves from P1's node right after our check with the 0.75 size added to P1's actions (1 958 public nodes, 521 ms including building the augmented tree), our range after the check as the root range, and P1's ceilings at that node from [2]. The opponent's reach after the new bet is readable from the solve (P(bet 0.75 | hand) = 0.268 / 0.267 / 0.121 / 0.120 / 0.200 / 0.199 / — / 0.285 / 0.476 / 0.468), and our strategy facing it is folds 0.565, calls 0.349, raises 0.086: Js/Jh now call 38 % instead of 27 % (the price is 1.5 into 3.5, 30 % equity), As/Ah raise 37 % instead of 15 %. The opponent's best response against this strategy, including the new size, exceeds its ceilings by at most 0.058 mA. What is carried on is our range and P1's values at the reached node. Snapping the 0.75 bet to the pot-sized branch would have played the (e) column instead.

**Exploitation, one input swapped.** Same node, same range, same ceilings; the only change is the opponent reach the traversal sees, `q = p · reach(station) + (1−p) · reach(free CFR⁺ opponent)`, two lines in `cfr.py`; $p$ = 0 is the safe solve. Against the calling station at $p$ = 0.01 the strategy barely moves (largest range-weighted shift 0.003). At $p$ = 0.25: Js/Jh fold 84 % (the station never folds, so thin raises lose their bluff value), Qs/Qh raise 50 %, Kh raises 99 %, As/Ah raise 97 %, overall 0.677 / 0.131 / 0.190 / 0.002. Re-solve time unchanged (220 ms). The caveat the script prints: the station never bets, so this node has probability zero under the model; the hook applies the model's behaviour (calls every raise, never folds) below the root.

### 4.3 Three players to the flop

Three ranges (plus the folded players' frozen vectors), boundary values from a three-player map $V_3$ with three range inputs and three output vectors (§5.3), fold terminals by the inclusion-exclusion sums of §5.2, 3-way showdowns at $O(52F)$ once implemented (exact only when nobody has folded; with folders present the dead-card measure enters as in §5.3), and the same loop. The tree is larger (300 decision nodes for the full menu at 3 players, 42 for a one-size menu), the equilibrium guarantee and the gadget theorem are gone, and the opponents' ranges must be tracked from the attributed strategies. When one player folds, the leaves switch to $V_2$ and the menu switches to the heads-up menu at the next street. No toy has measured this agent.

---

## 5. Three to nine players, and other variants

### 5.1 What survives, what breaks

Survives unchanged: carried ranges (dimension additive in players), the public-tree-plus-fiber representation, the per-player no-regret guarantee, the chance-node cut and the value map, off-menu insertion, the exploitation dial, and the entire variant plug-in structure.

Breaks, for mathematical reasons rather than engineering ones:

- **No Nash target.** CFR⁺ self-play in the 3-player toy reached about 1 mA of NashConv at 2 000 iterations, plateaued, and drifted upward after 7 000; the 2-player control went to 0.02. The current iterate oscillated at 8–10. Eight self-play runs agreed on seat values within 1 mA but differed in strategy by 5–9 % per decision, and recombining strategies across runs raised NashConv to 4–109 mA. $V_3$ is therefore defined only as "what this procedure converges to", and leaf maps must be trained on ranges produced by the same procedure.
- **No safety theorem, and recombination at every decision.** The gadget's guarantee is two-player zero-sum. Multiway re-solving tracks ranges (the construction that was 22× worse heads-up) and has no certificate; and because each re-solve is its own self-play run, the strategy actually played is a recombination of marginals from different runs, which the toy measured as raising NashConv up to 100×.
- **No security without a coalition model.** The only opponent-independent value is the maxmin against the other seats as a card-sharing team, a 2-player zero-sum game with a product fiber. In the toy the self-play strategies' security against such a coalition was −0.55 to −1.15 antes per hand, often worse than folding every hand (−1). The coalition maxmin itself was −0.21, −0.16 and +0.006 by seat (the last seat's is positive; only the sum over seats is ≤ 0), and playing it cost 17–28 mA per hand against the self-play population. So: a computable, conservative security certificate exists; it is a dial with a price; and "weather-robust" at a full table is empty without it.

What can honestly be promised at 3–9 players: per-player hindsight regret inside self-play; a measured NashConv (one best response per player, exact in toys and abstract games, sampled lower bounds in full NLHE); a computable coalition-security number; and never "unexploitable".

### 5.2 The multiway hidden state: a product measure with a hard-core constraint

With $L$ live players at a terminal, the opponents' joint weight is a product of ranges restricted to card-disjoint tuples. Writing $C = MM^\top - I$ for the collision indicator and $D = J - C$,

$$\prod_{j<l}(1 - C_{jl}) = \sum_{G \subseteq \binom{[m]}{2}} (-1)^{|E(G)|} \prod_{(j,l)\in E(G)} C(h_j, h_l),$$

the Mayer cluster expansion of a hard-core gas on 52 sites, with one term per graph on the $m = L-1$ opponents. Each term contracts over *cards* (52-dimensional), not hands, because $C$ factors through $M$.

- **Two opponents (3-way pot): exact and cheap.** One edge. For hero hand $h$ with opponents' per-card masses $Q_j(c)$ (mass on hands containing $c$):
  $$Z(h) = Q_j Q_l - \sum_c Q_j(c) Q_l(c) + \sum_a q_j(a) q_l(a),$$
  all after masking by $h$ and the board. Fold terminals cost $O(52^3 + F)$ (one 52 × 52 matrix product; verified against brute force to 10⁻¹³); showdowns, sweeping hands in rank order with per-card accumulators, cost $O(52F)$ per player vector (derived, not yet implemented; the toy's exact evaluator is a different, partition-lattice algorithm for one-card hands, verified to 10⁻¹³).
- **Three opponents (4-way): exact at $O(52^3 F)$.** Eight graphs; the triangle term is a trace of three 52 × 52 matrices. Fine offline, too slow in play.
- **Four or more opponents:** $2^{\binom m2}$ graphs; the number of connected cluster types for $m$ = 2, 3, 4, 5, 6 opponents is 1, 4, 38, 728, 26 704 (OEIS A001187). Exact is not worth it. Sample joint deals instead: a sampled deal is card-disjoint by construction, so the error enters regret as variance, never bias.

**What ignoring opponent–opponent removal costs.** Keeping hero–opponent and board removal exact but treating opponents as independent gives mean showdown-equity errors of about 0.3–0.6 % of pot for value-heavy ranges, 0.6–1.1 % for narrow ranges and 2.5–3 % for polarized ranges at 4–6 live players, with per-hand maxima of 2–10 % and 10–15 percentage points when ranges are built around a shared blocker (a nut-flush card held as value or as bluffs). Relative errors in a bluff-catcher's equity reached 3–100×, which flips decisions. Inside a solver the damage is strategic, not cosmetic: in the 3-player toy, independence made the solved strategy 65× more exploitable (72 vs 1.1 mA) and 24× at 4 players, a bias iterations do not remove. The linear first-order inclusion–exclusion correction is numerically unstable and often *worse* than independence from 4 opponents on (it produced negative masses on small decks); the exponentiated pairwise (Kirkwood) form, at $O(m^2 \cdot 52 \cdot F)$ per terminal, cuts the error to 0.02–0.3 % mean on generic ranges but overshoots badly when ranges concentrate on one card.

**Rule.** $L \le 3$: exact vector terminals (and brute force over the small joint was actually *fastest* in the toy at $L = 3$). $L = 4$: exact offline, sampled in play. $L \ge 5$: sampled joint deals, or a declared approximation with its error logged. And the folded players' reach vectors stay in the belief state for the dead-card measure at every $L$.

### 5.3 Count live players, not seats, with one correction

After preflop a 9-max hand *is* an $L$-player game: folded players have no actions and no payoffs, only a measure on the deck. $L$ never increases, so machinery indexed by $L$ recurses downward and leaf maps are needed only for $L \le 3$ ($V_2$ is the heads-up map, widened to cover dead money and multiway range shapes); this document trains nothing for $L \ge 4$. The seat count enters combinatorially only before the flop, where the fiber is smallest (169 classes) and there is no board. Hence 9-max adds a 3-way ladder, a preflop router and evaluation to the heads-up bill rather than multiplying it by nine: 9–37× the heads-up bill in §6.5's model depending on the 3-way menu, which is the single budget dial; pots with 4+ live players get no trained leaf.

The correction: the folded players' dead-card measure is **not** small, as §1.2 showed (10–12 % relative shifts in hold'em, 128–178× exploitability in the toy when dropped). So either $V_L$ takes the dead-card measure as an input (a 52-vector of per-card dead mass is its first moment; the folded ranges themselves are exact), or dropping it is a declared, measured approximation. The 169-class preflop fiber is exact for the strategy, but preflop fold terminals with up to 8 opponents use the multiway approximation of §5.2 or sampled deals. "N enters only through the preflop trunk" is false as an exact statement and true as a cost statement.

**Where in the public tree the cost sits.** One postflop street at 100 bb:

| Live players | 3-size menu | 2-size menu | 1-size menu |
|---|---|---|---|
| 2 | 28 | 20 | 8 |
| 3 | 300 | 204 | 42 |
| 4 | 2 240 | 1 504 | 184 |
| 6 | — | 36 936 | 2 724 |
| 9 | — | 1 003 680 | 115 794 |

Two facts about multiway trees: 67 % of 3-player and 95–98 % of 9-player decision nodes are pure call/fold nodes over *caller subsets* (the $2^{k-1}$ fan-out after a bet), so the number of sizes per bet matters less than the caller-subset fan-out, while the raise cap and the 3-way menu remain the budget dial (300 vs 42 nodes per street at three players, $66 vs $268 at 9-max in §6.5). The preflop router at 9-max (4.5 × 10⁵ to 7.8 × 10⁶ nodes without limps, about 10⁸ with them, on the 169 fiber; 13–25 GB of tables with a 4-bet size) is solved offline by public-action-sampled MCCFR: at an opponent's node sample one action with its range-weighted probability and divide the child's leaf value by it, which is unbiased because $V$ is degree-1 homogeneous in each opponent's reach. A limping opponent is off that tree; under the no-snapping rule they need either the 10⁸-node tree or an online sampled re-solve.

**Vectorized versus sampled.** On the 3-player toy, vectorized CFR⁺ reached 35× to 10 877× lower NashConv than external-sampling MCCFR at equal wall-clock (roughly the square of that in work to tie), the ratio shrinking as the deck grows (855 → 442 → 219 for decks 8 → 12 → 16) and as players are added (10 877 → 855 → 35 for 2 → 3 → 4 players). The hold'em fiber is about 100× the toy's, so the crossover for hold'em is unmeasured and may sit at $L \le 4$. Measure vectorized versus sampled at $L$ = 3 and 4 on hold'em river spots before fixing the boundary; the default here (vectors for $L \le 3$, sampling for the preflop router and $L \ge 5$) is a hypothesis.

### 5.4 Variants: three plug-ins

Variant-specific: the **deal schedule** (deck, hole cards per player, which cards are public and to whom, private draws), the **showdown order** (a total preorder on the fiber, or two for hi/lo, plus qualifiers), and the **betting grammar** (forced bets, legal-size function, the menu). Shared: everything else. Card removal for $k$-card hands generalizes exactly by inclusion–exclusion over shared subsets, $D = \sum_{s=0}^{k} (-1)^s M_s M_s^\top$ ($k = 2$ recovers §2.2).

| Variant | Fiber (preflop / river) | Collision probability of two random hands | What changes |
|---|---|---|---|
| NLHE | 1 326 / 1 081 | 0.076 (0.082 on the turn) | baseline |
| Short-deck (36) | 630 / 465 | higher | evaluator swap; cheaper |
| Limit hold'em | 1 326 / 1 081 | 0.076 | no menu: the public tree *is* the game, no action restriction at all |
| PLO4 | 270 725 / 178 365 | **0.281** (0.302 on the turn) | 16-term inclusion–exclusion; independence unacceptable multiway; **$V$'s input width (2 × 270k) is unaffordable** |
| 7-card stud | depends on upcards (≈ 1–2k) | — | upcards move into the public tree; fits better than hold'em |
| Draw games | $\binom{52}{5}$ | — | a draw is a linear stochastic operator on the fiber; exact but large |

The mechanism transfers to all of these. The budget does not transfer to PLO: every exact component scales with the fiber (about 200× hold'em per node, still linear), but a learned $V$ over 2 × 270 725 inputs needs a compressed range representation, and that is abstraction under another name. Say so rather than hide it.

### 5.5 Coalitions

A coalition is a change in the weather players' *utility aggregation* (a matrix $W$ summing payoffs over the team) and, if they share cards, a merge of their fibers into the card-disjoint product. The same engine runs. With card sharing, "hero versus coalition" is 2-player zero-sum, CFR converges to its value, and the coalition best response against any fixed hero strategy is one traversal: that is the computable security certificate of §5.1, feasible on river subgames in hold'em (the merged fiber is about 1.1 × 10⁶ per river node, 1.6 × 10⁶ preflop) and everywhere in toys. Without card sharing, team optimality needs a correlation device (TMECor) and team-maxmin without one is NP-hard in general; alternating team best responses give a lower bound on the damage. Toy: a soft-playing coalition cost the frozen hero only 4 mA per hand but left 74 on the table; a "whipsaw" coalition that always raised into the hero was worth +5.7 antes per hand to a hero who knew it; a model-free online learner reached about 91 % of that only after 10⁶ hands, and against the soft-play coalition it was still at −0.48 to −0.58 after 10⁶ hands, far worse than keeping the frozen strategy (−0.106).

---

## 6. Cost and feasibility

### 6.1 Calibration

All dollar figures use your own calibration: $0.033–0.15 per core-hour, $0.5–1.5 per GPU-hour. Vector CFR is memory-bandwidth-bound in the interior and matmul-bound at the network leaves, and the GPU's advantage depends on which: by the unit prices in §6.3, a bare river solve costs about the same on a GPU as on a rented core (0.1–7× across your price range, 0.8× at mid prices), while the network-bound turn solves and street-start forests are 3–70× cheaper on the GPU (about 10× at $1 per GPU-hour against $0.07 per core-hour). Both need a batched CFR kernel (CuPy or torch on the rented box, about 300–500 lines), which is a Sprint 2 artifact in §10, not an assumption. Pure numpy is 3–10× slower than a compiled kernel and does not meet the label budget; the compiled kernel (a few hundred lines of C) is the largest single engineering item.

### 6.2 Per-decision latency (heads-up, 2015 quad-core)

| Decision | numpy | compiled C |
|---|---|---|
| River, mid-street (exact, no network) | ≈ 1 s | 0.04 s |
| Turn or flop, mid-street, 13 pre-chance network calls, 300 iterations | 1.5–3 s | 0.6–1 s |
| First turn decision (chance-rooted forest: 637 $V^{(4)}$ calls, 49 × street arithmetic, 49 all-in GEMMs) | 45–75 s | 20–35 s with a 3.3M-parameter net; 5–10 s with a 1M net and per-river prefix-sum all-in leaves |
| First river decision (chance-rooted, 48 exact river subgames) | ≈ 48 s | ≈ 2 s |
| First flop decision | not solvable at play time; ceilings precomputed per on-menu preflop line (§2.5) | same |
| Exact turn-plus-river (no $V^{(4)}$, 3-size menu, 213 MB of state) | 1–3 min | 10–40 s wall on 4 cores |
| Preflop, on-menu line (table lookup) | < 1 ms | < 1 ms |
| Preflop with $V^{(2)}$ (16-node re-solve) | < 1 s | < 0.1 s |
| 3-way flop, 2-size menu | ≈ 6 s | ≈ 5 s |
| 6-way flop, 1-size menu, *with a 4+-way value net that is not part of this design* | ≈ 1 min | ≈ 50 s |
| 6-way flop, this design (sampled joint deals, full depth) | minutes | minutes |

Solver convergence on real river boards with exact card removal: 300 DCFR iterations leave 0.035 % of pot at SPR 2.4 and 0.24–0.28 % at SPR 20; CFR⁺ 0.095 % and 0.94 %. Deep stacks need about 3× the iterations for the same percentage of pot. 300 is the decision-time default and 200 is enough for training labels.

Two consequences. The exact turn-plus-river solve that DeepStack ran on a GPU is viable at play time *only compiled* and only with a time bank; it is the right held-out validator for $V^{(4)}$ either way. And 4+-way pots have no play-time solution under a minute in this design.

### 6.3 Offline: the ladder, the dominant unknown, and the gate

**Unit prices with a compiled CPU kernel** (numpy 3–10× slower; GPU figures assume the batched kernel):

| Item | What it is | CPU core-seconds | GPU |
|---|---|---|---|
| one river solve (single-card $V^{(4)}$ label) | one 5-card board, exact terminals, 200 iterations | 0.05–0.2 (0.64 numpy) | 8.5 ms |
| exact $V^{(4)}$ label | all 48 unseen river cards solved | 2.4–9.6 (≈ 31 numpy) | ≈ 0.41 s |
| one turn solve (single-card $V^{(3)}$ label) | one turn card, 13 $V^{(4)}$ calls per iteration | 0.5–1 | 4.3 ms |
| exact $V^{(3)}$ label | all 49 unseen turn cards | 25–49 | ≈ 0.21 s |
| exact validator | one turn-plus-river solve | 40–160 (10–40 s wall on 4 cores) | ≈ 6 s |
| flop-start forest | one chance-rooted solve over 1 755 flop classes | ≈ 3 000–7 000 (network-bound) | ≈ 20–60 s |

**What the budget buys.** 100–300 CPU core-hours are 1.8–22 × 10⁶ river solves. Spent entirely on one rung that is 4 × 10⁴–5 × 10⁵ exact $V^{(4)}$ labels (or 10⁶–10⁷ single-card ones) and 7 × 10³–4 × 10⁴ exact $V^{(3)}$ labels; shared across the bill below, about 10⁴–10⁵ exact labels per rung, or about 10⁶ single-card labels. Published systems used 10⁷ solved turn situations (DeepStack, on bucketed inputs), 10⁶–10⁷ flop situations, and more (Supremus: 5 × 10⁷ river, 2 × 10⁷ turn, 5 × 10⁶ flop subgames, with Huber losses of 0.016 and 0.028 in fractions of pot; a 2020 reimplementation of DeepStack at its original scale lost to Slumbot; OpenHoldem's flop network "suffers from severe over-fitting" at 1–3 × 10⁶). The toy's best learned $V$ (2 × 10⁴ exact turn labels, at most 2 × 10³ bootstrapped flop labels, range-weighted loss) was 18× worse than exact leaves and 22 % better than no strategy; with 200 labels it was 2 % better. The data-efficiency ideas of §2.5 (per-hand weight sharing, residual target, equivariance, range weighting) are the reasons a 10–100× smaller requirement is *possible*; none is proven, and one (the residual) was already measured not to move play in the toy. Single-card labels buy DeepStack-scale counts only if their variance penalty in hold'em is small, and the toy says the opposite.

**The HU bill, compiled CPU, at 10⁶ single-card labels per rung** (the lean plan; exact-label plans cost 48–49× per label and buy 10⁴–10⁵ labels instead):

| Line | Core-hours |
|---|---|
| gate (below) | 11–45 |
| $V^{(4)}$: 10⁶ single-card labels | 14–56 |
| $V^{(3)}$: 10⁶ single-card labels | 140–280 |
| 2 000 exact validators | 22–89 |
| sampled-flop preflop, 10-depth grid | 60–150 |
| flop-start forests, 7 lines × 10 depths | 60–140 |
| $E_b$ cache for 1 755 flops | 10–20 |
| network training | 3–10 |
| **total** | **≈ 320–790** |

That is $11–120 at your prices, or 1–3 weeks of the home PC. $10–15 holds if label generation runs at home and only the overflow is rented, or if the batched GPU kernel exists (the two label rungs are then about 4 GPU-hours, $2–6). It does not hold for CPU rental of everything. Evaluation (§6.4) is on top, in home-PC weeks.

**The gate, pre-committed.** Before renting anything beyond about $2: on the home PC with the compiled kernel, generate end-of-turn situations in canonical suit form and label them two ways: 10³, 3 × 10³ and 10⁴ situations with exact all-card labels (≈ 6.7 × 10⁵ river solves, 9–37 core-hours), and 10⁴, 3 × 10⁴ and 10⁵ single-card labels (2–8 core-hours). Train the §2.5 network on each set with the range-weighted loss, two seeds per point (12 networks). Score every network in **decision units** on 100 held-out turn situations (turn card dealt, canonical board; 50 from the same generator, 50 harvested from Sprint-1 river play): (1) solve the turn street (28 nodes, 300 iterations) with the learned $V^{(4)}$ at its 13 pre-river leaves; call the resulting turn strategy $\hat\sigma$. (2) Fix $\hat\sigma$ on the turn, let the river be played by the exact river solve at the ranges $\hat\sigma$ induces, and compute the opponent's best response in that exact turn-plus-river game (one exact solve, 10–40 s wall); do the same for the exact-leaf turn solve at the same iteration count. (3) Report the difference in best-response gain, in percent of the turn pot. Scoring is about 12 × 100 × 10–40 s, 3–13 hours of the 4-core PC, so the whole gate is one to two days. Target: below the solver's own residual of about 0.3 %. Fit a power law to the seed means on each curve, read off the exchange rate between single-card and exact labels, and extrapolate to the label count the budget buys. Then decide:

- *A curve reaches the target inside the budget:* buy the labels, build $V^{(3)}$ the same way (validated against exact turn-plus-river solves), solve preflop by sampled flops, precompute the flop-start forests.
- *Neither curve reaches it:* within $10–15 you then have no full heads-up NLHE agent. What you do have: (i) an exact turn-and-river engine, compiled, 10–40 s per turn decision, whose per-solve gap is certified relative to the ranges and ceilings it is handed, not against the whole game; a complete trainer and analyser for the last two streets. (ii) For flop and preflop, buy GPU time at the measured per-label price, or play with the best map the budget bought and state its measured decision regret; the floor is the check-down leaf (equity only, no strategic content), which scored 533 mA/h in the toy against 24 for exact leaves. The exploitability bound for depth-limited play is linear in the leaf error (about $2\varepsilon \times \sum \text{reach}\cdot\text{pot} \approx 250$ mbb/hand at 1 % sup-norm error), so there is no certified route to a small number; there is only the measured one.

### 6.4 Evaluation is a budget line

Heads-up NLHE has a per-hand standard deviation of about 8 bb. Resolving ±25 mbb/hand at 95 % confidence needs about 4 × 10⁵ raw hands, or about 4 × 10⁴ with AIVAT variance reduction (an unbiased estimator that subtracts the luck of the cards using the agent's own value estimates). At 0.04–1 s compiled (1–4 s numpy) per mid-street decision plus 2–35 s compiled (20–75 s numpy) per street-start re-solve, about four mid-street decisions and 0.6 street starts per hand, 4 × 10⁴ hands take about 2–7 days compiled and about 3 weeks in numpy per comparison. Evaluating a 9-max agent to ±1 bb/100 takes about 10⁶ hands, at least 1 650 core-hours ($55–250, or 4–6 weeks of the home PC). Evaluation, not training, is the slowest loop in the project, and it competes with training for the same machine.

### 6.5 9-max

With the live-count structure, 9-max offline cost is the heads-up ladder plus a 3-way ladder plus the preflop router plus evaluation. A separate multiway cost model built for this document (post-card street-root maps, CPU at $0.05 per core-hour, ×2 margin for retries, a lean heads-up unit of about 70 core-hours with 10⁵ river, 3 × 10⁴ turn and 2 × 10⁴ flop labels per live count) gives about **$66** with a one-size-plus-all-in 3-way menu, **$148** with one size plus a raise size, and **$268** with the heads-up menu in 3-way pots; at your own price range multiply by 0.67–3. Its robust output is the ratio: 9-max ≈ 9–10× heads-up at the coarse menu, ≈ 20× at the medium one, ≈ 37× at the full one. Its heads-up unit is 100–500× below DeepStack's label counts and 5–10× below §6.3's bill of 320–790 core-hours; scaling the ratios by that bill gives 9-max ≈ 3 000–8 000 core-hours at the coarse menu, $100–1 200 at your prices. Both versions exclude evaluation, which alone can exceed $65–100 (§6.4). The router at 9-max is a few to 30 % of the total and needs a rented 32–64 GB box for its tables. Pots with 4+ live players get a declared approximation (sampled full-depth solving is exact in expectation but minutes per decision on a 2015 PC; check-down leaves are a named heuristic whose use is logged). Carrying the folded players' dead-card measure into $V_2$ and $V_3$ is an input-width cost unpriced here.

Judgement: **$65–100 for 9-max holds only under the lean model: a 70 core-hour heads-up unit, single-card labels, a coarse 3-way menu and toy-exact evaluation.** With §6.3's heads-up bill it is $100–1 200 before evaluation. The ratios are the reliable output of the cost model; the dollar figures are not.

---

## 7. Failure modes, and where the requirements contradict each other

### 7.1 Failure modes with their detectors

1. **$V$ under-trained.** Detector: the §6.3 learning curve in decision units; held-out error against exact turn-plus-river solves; the gadget audit (the opponent's realized value exceeding their ceiling after a re-solve measures ceiling drift online for free).
2. **$V$ off-distribution under exploitation or against strange opponents.** Detector: ceiling drift; Bellman residual $\|V^{(3)}(\beta) - \text{solve}(\beta; V^{(4)})\|$ on reached situations. Mitigation: random plus harvested plus perturbed training ranges; raise $p$ slowly.
3. **Error compounding through the ladder.** $V^{(3)}$ inherits $V^{(4)}$'s error; the toy measured bootstrapped flop labels already 0.07–0.21 range-weighted RMSE off exact before the flop network added its own 0.23–0.57. Removing one rung (sampled-flop preflop) helps; exact validators at each rung are the only check.
4. **Per-hand label discontinuities** at range thresholds (§2.2). Mitigation: range-weighted loss; optional smoothing at a declared price.
5. **Flop-start ceilings.** Precomputed forests exist only for on-menu preflop lines and grid stacks; off-menu lines fall back to a solve measured unsafe in the toy. Detector: the gadget audit on the first flop decision; log every fallback.
6. **Multiway: no safety, no unique target, non-interchangeable runs** (§5.1). Detector: measured NashConv per solve; never mix leaf maps and routers from different runs.
7. **Multiway card removal and folded players** (§5.2–5.3). Detector: compare exact vs approximate terminals on logged spots.
8. **Opponent model misspecification.** The tilt family expresses frequency deviations with hand selection inherited from the reference; it cannot express "bluffs with showdown-value hands" or size tells the menu merges. Detector: prequential log-loss against a saturated per-class model; response: raise $\varepsilon$.
9. **Reference drift.** Tilt parameters are defined relative to the solver's reference; when $V$ improves, recompute the stored reference matrices or the HUD model silently decays.
10. **Showdown selection bias** (mucking is not missing at random). Fit from action likelihoods.
11. **Adaptive opponents.** External regret is the wrong counterfactual; a baiting opponent can recover at most what the gift budget risked, in expectation. Mutual adaptation has no convergence guarantee; the $\varepsilon$ floor bounds the loss per cycle, not the cycling.
12. **Latency.** Street-start forests run from 2 s (river, compiled) to 75 s (turn, numpy, large net) on a 2015 CPU; 4+-way pots are minutes. Fewer iterations raise a *certified* local gap.
13. **Engineering.** Numpy alone does not meet the label budget; the compiled kernel is the largest single engineering item, a batched GPU kernel is the second if the curve demands more than 10⁶ labels per rung, and array layout alone is a 7× factor (action-major `(A, boards, F)` vs hand-minor). Running several numpy processes with multithreaded BLAS oversubscribes the machine; set one BLAS thread per process.
14. **Deep stacks (500 bb).** The tapered menu keeps tree size flat, but $V$ must cover SPR up to about 100, deep stacks need 3× the iterations, and menu thresholds (the 67 % all-in rule) put kinks in $V$ as a function of SPR. Stratify training by SPR; without $V^{(2)}$, stacks between grid points are snapped.

### 7.2 Where the requirements contradict, exactly

1. **"Independent of the opponents" vs "maximum profit from observed behaviour."** The free part of the frontier (selection on the equilibrium face) costs nothing. Beyond it, the minimax theorem makes every gain against a forecast a loss against some opponent; the trade-off is set by a declared risk quantity ($\varepsilon$, confidence level, or gift budget). "No arbitrary parameters" and "maximal exploitation" cannot both hold. The gift-budget rule is parameter-free and exploits slowly.
2. **"Opponents are weather" vs opponents who adapt.** The guarantee is against the realized sequence; policy regret against unbounded-memory adaptation cannot be sublinear. Weather that reacts to your umbrella has no analogue in the theorem.
3. **"Guarantees, no OOPS" vs 3–9 players.** No unique value, no gadget theorem, range-carrying re-solves, recombination of runs at every decision, self-play marginals with no security (−0.55 to −1.15 antes per hand against colluders in the toy), and the only guaranteed strategy is the coalition maxmin, which is passive and loses against ordinary tables. What can be promised is measured: per-player regret, NashConv, a conservative security number.
4. **"No abstraction" vs "$10–15 on a 2015 PC."** The full tree is 0.9 PB and an exact flop decision is hours, so a learned $V$ is forced. Its certified bound is useless, its practical quality is measurable, and the toy measured exactly the failure the user calls an OOPS at a small label budget. Priced honestly, the budget buys about 10⁴–10⁵ exact labels per rung (or 10⁶ single-card ones) against the 10⁶–10⁷ published systems used. "No OOPS" holds for everything except $V$, and $V$ is where the project's risk is.
5. **"No snapping" vs a finite own menu and a finite preflop table.** Postflop opponent sizes are exact at a measured residual; your own menu costs EV (≲ 0.25–0.4 % of pot at three tapered sizes on river spots, 0.07–2.6 % at one) and never safety. Preflop and the flop start are where this design snaps: off-menu preflop opens, limps, off-grid stacks and unequal 9-max stacks map to the nearest precomputed line and depth, and so do the flop-start ceilings. $V^{(2)}$ would remove only the preflop part, because the flop-start forests stay precomputed per line and depth.
6. **"9-max at $65–100" vs heads-up-quality multiway.** Only a coarse 3-way menu fits; the heads-up menu in 3-way pots costs about $270 at $0.05 per core-hour with margin ($180–800 across your price range), before evaluation.
7. **"Exact card handling" vs 5+ live players.** Exact opponent–opponent removal is super-exponential in live opponents; the choices are unbiased sampling (variance, time) or a declared approximation whose error the toy shows is strategic, not cosmetic.
8. **"Stability, no trial and error" vs learned maps.** The label count and the network's hyperparameters are measurements, not derivations. The only principled substitute is one learning curve with a pre-committed extrapolation rule, and the prior is that it comes back with "not enough budget".
9. **"One mechanism for all variants" vs budget.** PLO's fiber breaks $V$'s input width; compressing it is abstraction.
10. **"Demonstrated strength" vs budget.** ±25 mbb/hand needs about 4 × 10⁴ AIVAT hands, one to five weeks of the home PC per comparison.

---

## 8. Evidence from the toy games

Both games are in `toy/` with READMEs, test suites and result logs. Every number is an exact best-response measurement, not an estimate.

### 8.1 Heads-up mini hold'em (`toy/hu`)

10-card deck (5 ranks × 2 suits), one private card, three streets (preflop, one flop card, one turn card), stacks of 12 antes, menu {½ pot, pot, all-in} → {pot-raise, all-in} → {all-in}, raise cap 2. 54 957 public nodes, 171 920 information sets, fiber of 10. Vectorized EV matches a brute-force walk over all deals to 10⁻¹².

**E1, exact solve.** CFR⁺ reaches 0.53 toy % pot in 300 iterations (20 s, one core, numpy), 0.071 % in 1 000 and 0.020 % in 2 000; vanilla CFR is 8.6× and 42× more exploitable at equal wall-clock.

**E2, continual re-solving, 300 iterations per re-solve, complete strategies extracted and best-responded exactly:**

| Agent | Leaves at the street boundary | Exploitability mA/h | toy % pot |
|---|---|---|---|
| full solve, 300 / 2 000 it. | — | 10.5 / 0.41 | 0.53 / 0.02 |
| safe (gadget), exact leaves | oracle | **23.7** | 1.18 |
| naive (ranges only), exact leaves | oracle | 520.5 | 26.0 |
| safe, check-down leaf | equity × pot, no strategy | 532.5 | 26.6 |
| safe, learned leaf, 200 / 2 000 / 20 000 exact turn labels (flop net on ≤ 2 000 bootstrapped labels) | MLP, residual over check-down | 521 / 512 / 499 | ≈ 25 |
| safe, learned leaf, 20 000 labels, range-weighted loss | same | 417 | 20.9 |
| naive, learned leaf, 20 000 labels | same | 824 | 41 |
| safe, exact at one boundary and learned (rw) at the other | mixed | 216 / 259 | 11 / 13 |

Label noise was not the limit (range-weighted label noise 0.015 vs network error 0.15–0.47). The limit was data and capacity.

**E3, off-menu opening sizes** (loss to an exploiter, mA/h): snapping to the nearest size 389 / 227 / 51 for bets of 0.25 / 0.75 / 1.5 pot; safe re-solve with the size inserted 18.8 / 7.2 / 1.4; the responder's strategy from a full solve of the augmented game 1.8 / 2.2 / 1.9.

**E4, one solver for robustness and exploitation** (seat-averaged mA/h; RNR at 600 iterations):

| Model | Nash EV | $p$ = 0.25: EV / exploitability | $p$ = 0.5 | $p$ = 0.75 | Best response: EV / exploitability |
|---|---|---|---|---|---|
| station | 812 | 2 862 / 154 | 3 086 / 278 | 3 123 / 342 | 3 143 / 1 570 |
| maniac | 541 | 1 178 / 47 | 1 298 / 138 | 1 429 / 334 | 1 429 / 1 498 |
| nit | 422 | 1 393 / 108 | 2 901 / 1 115 | 3 296 / 1 591 | 3 344 / 9 039 |

The pure best response to the nit is exploitable by 9 antes per hand because of arbitrary play at nodes the model never reaches. The free part (T5): at $p$ = 0.001–0.01 the gain was +179 / +179 / +138 with exploitability at the 3.4–4.4 noise floor. The models were known exactly; fitting them from data was not tested.

**Timing.** Full solve 0.067 s per iteration; a last-street (turn) re-solve 0.1 s; a flop-root re-solve 0.15–0.6 s; an off-menu re-solve 0.6–0.7 s; a turn-oracle label 11 ms; an exact flop label 255 ms.

### 8.2 Three-player Tri-Leduc (`toy/multi`)

12-card deck, one private card each, two streets, stacks of 15 antes, menu {pot, all-in} → {pot-raise, all-in}, cap 2. 6 295 public nodes, 30 888 information sets. Exact-vs-brute-force to 10⁻¹²; the Möbius card-removal evaluator to 7 × 10⁻¹³.

| Experiment | Result |
|---|---|
| M1 card removal in the solver | NashConv 1.12 mA exact; 72.3 with opponents treated as independent (65×); pairwise correction exact at 3 players; at 4 players (deck 8) exact 47, pairwise 438, independent 1 127. |
| M2 convergence | 3 players: ≈ 1 mA at 2 000 iterations, plateau 0.76–0.96, upward drift after 7 000; current iterate oscillates at 8–10. 2 players: 0.02. Average external regret falls as $1/T$ for every seat. |
| M3 coalitions | Frozen self-play vs soft-play coalition −0.106 (best response −0.028); vs whipsaw −0.031 (best response +5.68). Model-free online learning after 10⁶ hands: −0.48 to −0.58 vs soft-play, losing 0.6–1.3 per hand while learning; vs whipsaw it reached 5.07–5.18 (≈ 91 % of the best response). With a model (counterfactual feedback) the best response is reached in 5–50 iterations. |
| M4 sampling vs vectors | CFR⁺ reaches 35×–10 877× lower NashConv than external-sampling MCCFR at equal wall-clock; ratio falls with deck size and player count. |
| M5 scaling | Public nodes 1 119 / 6 295 / 15 053 / 64 253 / 257 437 / 13.4 M for 2 / 3 / 4 / 5 / 6 / 9 seats (≈ 4× per seat); modelled 807 s per full-width iteration at 9 seats with independent removal. |
| T3 what can be certified | 8 self-play runs agree on values within 1 mA, differ 5–9 % per decision; recombining runs: NashConv 4.4–109 mA. Security vs card-sharing coalition −0.55 to −1.15 (folding: −1). Coalition maxmin −0.211 / −0.163 / +0.006 by seat; cost of playing it 17–28 mA vs the population. |
| T4 folded players | Uniform dead cards for folders: NashConv 1.12 → 199.5 (all terminals) or 143.9 (postflop only); at 4 players 47 → 1 141–1 406. |

### 8.3 Verified independently of the toys

- The two-player fold and showdown tricks at hold'em size (1 326 combos, 5-card board) match brute force to 10⁻¹² (Appendix A).
- Single-street betting trees under human menus: 45–300 public nodes per street; two independent enumerators agree on 28 decision nodes for the tapered 3-size menu.
- The averaging weight $1/(N_{\text{left}} - 4)$ at a pre-chance cut is exact; uniform weights are not.

---

## 9. What was examined and rejected

The user asked for "I went through twenty attempts" rather than "I tried and it didn't work." Twenty-three mechanisms were judged against five criteria: avoids nested enumeration (E), stable without tuning (S), scales to 9 players and variants (P), fits $15 for heads-up (C), yields a checkable certificate (K). The "Fails" column names the criteria a rejected mechanism violates.

| Mechanism | Verdict | Fails | Reason |
|---|---|---|---|
| Sequence-form LP | oracle for toys only | C | 10¹⁴ variables in HU NLHE; useful as a unit test |
| Scalar tabular CFR | dead | C | iterates hands one at a time; 1 326× redundant |
| **Vector CFR⁺/DCFR on the public tree** | **substrate** | — | hidden layer disappears; public layer must be cut |
| MCCFR (outcome / external sampling) | niche | E partly, K noisy | the 9-max preflop router and $L \ge 5$ terminals; 35–10 877× higher NashConv than vectors at equal time in the toy |
| Fictitious play, XFP, NFSP | dead | S, K | $O(T^{-1/2})$ or worse; needs best responses; neural version needs tuning |
| Double oracle, PSRO | dead as mechanism, alive as evaluator | C, P | exponential oracle calls; LBR is its oracle used as a test |
| Self-play policy gradient (PPO etc.) | dead | S, K | cycles in imperfect-information zero-sum games without regularization |
| NeuRD, R-NaD (DeepNash) | idea survives, implementation does not | S, C, K | GPU-scale, sample-based, schedules to tune |
| Magnetic mirror descent | alternative inner update | — | last-iterate convergence to the regularized point; same cost as CFR |
| Predictive RM⁺, PCFR⁺ | alternative fast update | — | parameter-free |
| Dual averaging, EGT smoothing | principle reused | — | smoothing as a priced dial |
| "Regret as the state" | partly | — | between decisions the state is your range plus the opponent's ceilings |
| Agent-form vs Bayesian-game form | agent form dead | S | destroys bilinear structure; type form is the public tree with type vectors |
| QRE / max-entropy equilibrium as target | optional dial | — | unique Lipschitz labels at a declared price; no free learnability win at usable $\tau$ |
| Amortized policy network (range → strategy) | dead as decision-maker | K | uncertifiable, un-re-solvable; warm start only |
| **Amortized value map at the depth limit** | **the irreducible approximation** | C at risk | known (DeepStack, ReBeL); its data cost is the project's risk |
| Direct best response to a population | circular at day 0 | C | becomes the evidence term once data exists |
| Evolutionary / population search | dead | S, C, K | 10⁶⁺ noisy evaluations per step, no certificate |
| Separate hand-reading predictor | dead | S (ill-posed) | it is layer 1 mis-learned |
| IS-MCTS / determinization | dead | E (wrong game), K | strategy fusion: solves the perfect-information game the cards would make |
| Modicum multi-valued leaves | dead from scratch | C | continuation strategies need a blueprint |
| Benders / SDDP cuts on $V$ | exact bounds, not minimal | C | tangent bounds are loose under joint range moves in 1 100 dimensions |
| CFR-D with sampled subgames (trajectory principle) | survives for trunks | — | the right offline method for the 9-max preflop and for a sampled-flop preflop; no cheaper than $V$ for heads-up flops |

---

## 10. The sprint ladder

Each sprint produces one artifact and one number in decision units. If the number is not measured the sprint is not done. No later sprint changes an earlier sprint's algorithm; later sprints add a fiber (players, cards), a boundary map (one more street), or an input (an opponent model).

| Sprint | Artifact | Exit number | Compute |
|---|---|---|---|
| **0. Rapid** (days) | The toys in `toy/`, extended: Kuhn/Leduc unit tests against closed forms; the clairvoyance kink; gadget vs range-only; RNR frontier and its free part; 3-way exact terminals vs deal sampling; coalition security. | Everything in §8 reproduced. | Laptop. |
| **1. MVP** (1–2 weeks) | HU push/fold on the real 169-class fiber; a HU **river** engine on 1 081 combos with the human menu, exact terminals, off-menu insertion, exact exploitability; the compiled CPU kernel. | Push and call decisions agree with published HU push/fold Nash charts on ≥ 99 % of the 169 classes at 1–20 bb; river exploitability $(BR_0+BR_1)/2 \le 0.05$ % of pot at SPR ≤ 3 and ≤ 0.3 % at SPR 20 at 300 DCFR iterations over 30 random boards with Dirichlet(0.3) ranges; seconds per solve logged. | Laptop. |
| **2. Base** (2–4 weeks) | Exact **turn-plus-river** engine (the validator, and a trainer and analyser for the last two streets whose certificate is relative to the ranges and ceilings it is handed); the $E_b$ cache; a batched GPU kernel if any rental is planned; then the **gate** of §6.3. | Measured ms per river solve on CPU and GPU; the two learning curves in decision units with a written go/no-go. | 1–2 days of the home PC and under $2. |
| **3. Full HU** (4–8 weeks, if go) | $V^{(4)}$ at the chosen label count; $V^{(3)}$ validated against exact turn-plus-river; preflop by sampled flops per stack depth (or $V^{(2)}$); flop-start forests; chance-rooted re-solves at turn and river starts. | Held-out decision regret per rung; local best response (LBR, a cheap lower bound on exploitability); then Slumbot's public API with AIVAT over ≥ 4 × 10⁴ hands (first target: not losing). | The rest of the $10–15 plus 1–3 weeks of the home PC; GPU for the label rungs if the kernel exists. |
| **3′. Fallback** (if no-go) | The exact turn-and-river engine as a trainer and analyser, not a player; flop and preflop with the best map the budget bought and its stated regret. | The stated number. | — |
| **4. Tuning** (ongoing) | Tilt or constraint-set model from logs; the free selection step; RNR dial; gift budget; discounting mixture. | In `toy/hu` and in river-only HUNL spots: EV gain vs each scripted bot and exact exploitability at each $p$. In full HUNL: EV gain over ≥ 10⁴ AIVAT hands per bot and an LBR lower bound at $p$ = 0 and the chosen $p$. | Laptop. |
| **5. 6–9 max** (2–4 months) | First, the multiway re-solving agent measured in Tri-Leduc against its full solve. Then the live-count engine with folded-player dead-card inputs; exact 3-way terminals (implement and verify the showdown sweep); $V_3$ at the coarse 3-way menu; preflop router by public-action MCCFR on a rented 32–64 GB box; sampled terminals for $L \ge 5$; coalition-security pass. | Per-player regret; measured NashConv in the solver's game; security number; duplicate-dealt small matches vs baseline bots. | $65–100 only at the coarse 3-way menu, lean labels and toy-exact evaluation. |
| **6. Variants** | New deal schedule, evaluator, betting grammar. Short-deck and limit first (limit needs no menu at all). PLO only with a declared range compression for $V$. | Sprints 1–3 numbers for the new game. | Fiber-dependent. |

The order is forced by the ladder (each rung's labels come from the rung below) and by the gate (nothing is rented before the curves are measured). Nothing else is forced. If sprint 2 says no, you still have an exact two-street engine with a certificate on every local solve, and you know precisely what you do not have.

---

## Appendix A. The two-player terminal tricks, verified

The code below was run against brute force on a random 5-card board with the full 1 326-combo fiber: fold values agree to 2 × 10⁻¹³ (0.6 ms vs 0.33 s for brute force), showdown values agree to 1.5 × 10⁻¹² (4.6 ms in this unoptimized per-card loop; 85 µs with sorts hoisted out of the iteration).

```python
import numpy as np, itertools
hands = np.array(list(itertools.combinations(range(52), 2)))   # F x 2
F = len(hands)                                                  # 1326
# r2: opponent reach vector (board-conflicting hands already zero); rank: hand rank on this board (-1 if dead)
# alive: mask of hands not conflicting with the board

# Fold terminal, O(F): v(h) = c * (sum r2 - mass on hands sharing card 1 - mass on hands sharing card 2 + r2(h))
S = r2.sum()
s_card = np.zeros(52); np.add.at(s_card, hands[:, 0], r2); np.add.at(s_card, hands[:, 1], r2)
v_fold = c * (S - s_card[hands[:, 0]] - s_card[hands[:, 1]] + r2) * alive

# Showdown, O(F log F) once per board then O(F) per iteration:
# mass of opponent hands with lower / higher rank, via prefix sums over the global rank order,
# minus the same quantities restricted to hands sharing a card with h (52 per-card sorted prefix sums).
order = np.argsort(rank, kind='stable'); cum = np.cumsum(r2[order])
uniq, first = np.unique(rank[order], return_index=True)
grp = np.searchsorted(uniq, rank)
below = np.concatenate([[0.0], cum[first[1:] - 1]])[grp]
above = S - cum[np.concatenate([first[1:] - 1, [F - 1]])][grp]
b_c = np.zeros((52, F)); a_c = np.zeros((52, F))
for cidx in range(52):
    idx = np.where((hands[:, 0] == cidx) | (hands[:, 1] == cidx))[0]
    o = np.argsort(rank[idx], kind='stable'); rk = rank[idx][o]; cw = np.concatenate([[0.0], np.cumsum(r2[idx][o])])
    lo = np.searchsorted(rk, rank, 'left'); hi = np.searchsorted(rk, rank, 'right')
    b_c[cidx] = cw[lo]; a_c[cidx] = cw[-1] - cw[hi]
ar = np.arange(F); c1, c2 = hands[:, 0], hands[:, 1]
v_show = ((below - b_c[c1, ar] - b_c[c2, ar]) - (above - a_c[c1, ar] - a_c[c2, ar])) * pot * alive
```

The per-card tables depend only on the board and the current $r_2$; in a solver the sorts are hoisted and only the prefix sums are recomputed per iteration.

## Appendix B. The re-solving gadget, operationally

Given your public node $x$, your range $r_1(x)$ and the opponent's ceilings $\bar v_2(x,\cdot)$ from the previous solve:

1. Build the street subtree from $x$ with your menu, the opponent's actual last size if off-menu, boundary leaves valued by $V$, and exact fold / all-in terminals.
2. Add a root node for the opponent with, per hand $h'$, two actions: *Terminate* with payoff $\bar v_2(x,h')$ to the opponent, or *Follow* into the subtree. The opponent's "range" at $x$ is the chance/card-removal prior times their Follow probability, chosen by regret matching like any other decision.
3. Run DCFR for the iteration budget; compute the best-response gap of the local game (one pass) and the maximum over hands of (opponent's realized value − ceiling), the gadget audit.
4. Act from the average strategy at your hand. Carry forward: $r_1 \leftarrow r_1 \odot \sigma_1(\cdot, a)$ after your action (exact); $\bar v_2 \leftarrow$ the opponent's values at the reached child after theirs (also after an off-menu action, from the augmented re-solve).
5. At a street boundary, do not translate averaged values into per-card ceilings. At the turn and river, root the next re-solve at the pre-chance node with the averaged ceilings and all next cards in one forest. At the flop, use the precomputed forest for this preflop line and stack depth; if there is none, the start-of-street fallback is declared unsafe.

Two-player zero-sum theorem (Burch et al. 2014; Moravčík et al. 2017): with attainable ceilings, exploitability of the combined strategy ≤ exploitability of the previous strategy + solver residual + ceiling error, summed over re-solves. For 3+ players the construction can be run with modelled ranges but has no theorem.

## Appendix C. Known versus structural

| Item | Status |
|---|---|
| Public tree × fiber, ranges as carried reach, public belief state, vector CFR | Known (Johanson et al. 2011–12; DeepStack; ReBeL) |
| $D = J - MM^\top + I$; $O(3F)$ showdowns | Trick known (Johanson 2011); closed form is a clean restatement that makes the multiway expansion mechanical |
| Belief dimension additive in players, folded players included | Known in substance; the folded-player magnitude (toy 128–178×, hold'em 10–12 %) is a new measurement |
| Fiber locality (all nonlinearity cell-local; hand coupling fixed and leaf-only except $V$) | Framing; implicit in existing code |
| Multiway removal as a cluster expansion over cards; exact 3-way algebra; independence and Kirkwood errors measured; first-order IE diverges | New quantification |
| Suit symmetry compresses boards 12–19× but postflop fibers only 1.1–1.4× | New quantification; redirects effort to $V$'s inputs |
| Pre-chance cut, $1/(N_{\text{left}}-4)$ weights, single-card labels | Known (DeepStack auxiliary net, ReBeL); the weight correction and the equal-compute measurement are new |
| Chance-rooted gadget at street starts; start-of-street solve fails; flop-start forest must be precomputed | New measurement and consequence |
| Continual re-solving with ceilings as the formal "weather"; carrying the opponent's range is unsafe and, without a blueprint, self-referential | Known machinery (DeepStack, Burch); the reframing and the 22× measurement are the point |
| $V$ concave–convex, 1-homogeneous, best-response CFVs as supergradients, clairvoyance kink | Known in substance (ReBeL); kink consequences for learning are the point |
| Regret bound as the weather theorem; dimension separation (4 × 10⁹ hands); exploitation needs a model | Theorem known; the quantitative argument and toy M3 are the contribution |
| Huber contamination ≡ RNR; Bayes gives no caution; declared risk parameter | Elementary identity; used to split estimation from misspecification |
| Free exploitation on the equilibrium face | New measurement (toy T5) |
| Tilt of the solver's own strategy as the opponent family; HUD counts as sufficient statistics | Max-entropy modelling known; anchoring at the per-hand reference is the contribution; untested |
| Live-count indexing; 3-way menu as the 9-max budget dial; coalition as utility matrix plus fiber merge | Structural cost framing; corrected by the folded-player measure |
| Cost law (product over spanned streets vs sum); depth-tapered menu law; caller-subset law; 0.9 PB; label units | Computed here |
| Menu-loss table; DCFR convergence table | Measured here |

## Appendix D. Glossary

- **Public tree**: betting sequences (exact amounts) and board cards; nothing private.
- **Fiber**: the set of private hands a player could hold at a public node; $F$ its size.
- **Range / reach**: the vector over a player's fiber of the product of their action probabilities along the line.
- **Counterfactual value (CFV)**: a hand's expected payoff weighted by everyone else's reach, own reach removed.
- **Exploitability / NashConv**: for two players, half the sum of both players' best-response gains; for $N$, the sum of per-player best-response gaps.
- **Gadget / ceilings**: the opponent's per-hand CFVs from the previous solve, offered as a Terminate option at the root of a re-solve.
- **RNR**: restricted Nash response; the opponent plays a model with probability $p$ and adversarially otherwise.
- **Pre-chance cut**: street boundary placed before the next card is dealt; one $V$ call per boundary leaf inside a lookahead.
- **Exact (all-card) label / single-card label**: a boundary label computed from every unseen next card (48 river or 49 turn solves) or from one sampled card (one solve, full next-card variance).
- **LBR**: local best response, a cheap lower bound on a full agent's exploitability. **AIVAT**: an unbiased variance-reduction estimator for match results.
- **SPR**: stack-to-pot ratio; the only geometric input $V$ needs postflop under a pot-fraction menu.
- **Live count $L$**: players still holding cards; the quantity postflop cost depends on.
