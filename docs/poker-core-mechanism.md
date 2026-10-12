# Poker at Its Roots: The Difficulty Is Not Where You Think

*A foundation for building a poker agent from zero on a 2015 PC plus $10–15 of rented compute: no blueprint, no card buckets, no snapping of opponent bet sizes, one mechanism for robustness and exploitation, and an honest account of the single place where the plan can fail.*

Everything in this document was either derived here, measured in the two toy games under `toy/` (exact best-response numbers, pure numpy), or checked by an adversarial review that tried to refute each load-bearing claim. Where a claim did not survive, the corrected statement is given instead. Where the evidence is against the plan, that is said in the same font as everything else.

---

## 0. The answer on one page

**Your three layers are not three problems.** Hidden cards, reactions and the future are three coordinates of one array. In the right coordinates:

1. **The hidden layer costs nothing.** Attach to every public node (betting sequence plus board) one vector per dealt player over that player's possible private hands: the *range*. Along any betting line the range is a running product of the probabilities the attributed strategy gave to the actions actually taken. Bayes' rule in poker is a componentwise multiply. The belief about nine players' hands is 9 × 1326 numbers, never 1326⁹. Nothing is inferred and nothing is bucketed, at any player count.
2. **Reactions are a fixed-point condition, not a prediction.** The opponents' strategies are either what a regret-minimizing learner would do in their seat (robust) or a fitted model of what they actually do (exploitative). Either way the solver carries the strategy as the input to the ranges, so layer 2 is the same array as layer 1.
3. **The future is a boundary condition, not a tree.** Given the ranges, the public state is a sufficient statistic for the rest of the hand, so the hand can be cut at the next public chance node (the next card) and the remainder replaced by a value map $V$. $V$ is learned from the bottom of the game up, river first, which is why no blueprint is needed.

**The three counterintuitive points, ranked by how much they change the computation:**

- **For robust heads-up play, do not even carry the opponent's range.** Carry *your own* range (you know your own strategy exactly) and a vector of *ceilings* on the opponent's per-hand counterfactual values from your previous solve, and let the opponent's range be an adversarial variable inside the local solve. That is the formal content of "opponents are weather", and it is not a nicety. In the toy, re-solving with exact leaf values scored 23.7 milli-antes per hand of exploitability when the ceilings were carried and 520 when only ranges were carried, with the same exact leaves. Carrying the opponent's range forward from your own equilibrium solve assumes they play your equilibrium, which is exactly what the weather principle forbids.
- **The real combinatorial wall is public, not hidden.** A no-abstraction heads-up 100 bb strategy with a three-size menu needs about 0.9 petabytes to store. Boards multiply betting lines; hidden hands only add. So the cut at the chance node is forced by arithmetic, not taste, and everything that is approximated in this design lives at that one cut.
- **Exploitation has a free part and a priced part.** Among the many strategies with the same worst-case value, pick the one best against your forecast of the opponent. In the toy this captured 5–20 % of the whole Nash-to-best-response gain with no measurable rise in exploitability. Beyond that, every further gain against a forecast costs worst-case value, and the price is set by one declared risk parameter that no data can choose for you.

**What is exact.** Hand representation (every combo). Card removal, including that of players who have folded. Opponent bet sizes (an off-menu size is inserted into the tree and the street is re-solved; nothing is snapped). Terminal values (closed-form sums, never a 1326 × 1326 matrix). The regret dynamics (no learning rate, no critic, nothing to tune except the iteration budget, and a certified gap after every solve).

**What is approximated.** One thing: the value map $V$ at the street boundaries. Also one thing you *choose*: your own bet-size menu, a restriction of your actions whose cost is measurable (about 0.25 % of pot for a tapered three-size menu, 0.4–2.6 % for one size).

**Where it lives or dies.** $V$. The toy's learned $V$, trained on 200 to 20 000 exactly solved situations, played no better than a value function with zero strategic content, while the same agent with exact leaves was near equilibrium. Published systems needed 10⁶–10⁷ solved situations per street. The hidden-information machinery, the safety construction and the exploitation dial are all verified; whether $V$ can be learned well enough inside $10–15 is a hypothesis with a measured learning curve as its only test. §6 gives the gate and the fallback.

**Beyond two players** the mechanism runs unchanged, but the certificate does not: per-player no-regret survives, Nash and the safety construction do not, self-play runs are not interchangeable, and self-play strategies have almost no security against colluders. What can be promised at 3–9 players is measured, not guaranteed: per-player regret, measured best-response gaps, and a computable coalition-security number.

---

## 1. Why the three layers are one object

### 1.1 Coordinates: a public tree with a private fiber

Textbook game theory draws poker as a tree over *information sets*. That picture is why the problem looks like nested enumeration: every one of your information sets contains one node per opponent hand, so every value appears to need a sum over their hands, and their reasoning appears to need a sum over yours.

Change coordinates. Let $T$ be the **public tree**: nodes $x$ are public histories (every action with its exact chip amount, plus the board). Nothing private is in $T$. Over every node sits a **fiber** for each player $i$: the set $H_i(x)$ of private hands $i$ could hold, $F = \binom{52}{2} = 1326$ combos preflop, 1176 on the flop, 1128 on the turn, 1081 on the river after removing board conflicts. A strategy for player $i$ is a *section*

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
- Beliefs are a **vector**, not a branch. One $F$-vector per dealt player, carried down the public tree. The joint over nine players has about 10²⁸ points; its sufficient statistic has 11 934 numbers.
- Beliefs are exact **only relative to the attributed strategy**. If the opponent takes an action your attributed strategy gave probability zero, the update is undefined; this is precisely the case the re-solve in §2.4 handles.

The folded players are not a detail. In the 3-player toy, replacing a folder's frozen range by a uniform one raised the solved strategy's exploitability (NashConv) from 1.1 to 144–200 milli-antes per hand. In hold'em, seven early folders with ordinary folding ranges raise the big blind's probability of holding an ace by about 10 % relative and the probability of an ace on the flop by about 12 % relative. The deck a late-position player faces is not uniform.

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

$V$ is the only object in this design that is approximated.

### 1.5 What was hiding behind the three layers

Two things the three-layer framing has no slot for turned out to be where the difficulty sits.

**The public tree.** Fibers add across players; boards and betting lines multiply. Heads-up 100 bb with a tapered three-size menu has 16 / 172 / 1 396 / 8 780 betting lines on the four streets and about 3.8 × 10¹⁰ public river decision nodes across all boards. With fibers, regrets and averages that is roughly 0.9 PB. A one-size menu still needs 33 TB. Preflop alone grows about 5–7× per added seat: 16 nodes heads-up, 3.9 × 10⁴ at 6-max, 7.8 × 10⁶ at 9-max with a 4-bet size, 4.7 × 10⁵ at 9-max if a 4-bet is all-in only. No representation shrinks this; it is public chance times public action.

**Selection.** With two players all equilibria share a value but not per-hand counterfactual values, so the label $V$ must learn is set-valued at kinks (§2.2). With three or more players equilibria differ in value and nothing says which one the table is "in" (§5). An opponent model is also a selection: many strategies reproduce the same HUD counts. "Which fixed point?" is a question the user's framing never asks, and it decides stability and reproducibility.

### 1.6 The single coupled computation

At your decision at public node $x$:

- the past enters as your own range at $x$ (exact) and, for each opponent, either their ceilings (robust, §2.4) or their modelled range (exploitative, §2.6);
- the future enters only through $V$ at the street boundaries below $x$;
- the present is the current street's public betting subtree from $x$: with a human menu, tens to a few hundred nodes.

Run regret minimization on that subtree, all players at once, strategies as `[node, hand, action]` arrays, and read off your action for your actual hand. Nothing is nested, nothing is enumerated over hidden states, no full-game solution exists anywhere, and so no "delta from a full solution" is ever needed. The three layers were never three problems; they were three views of one array and one boundary map.

---

## 2. Minimal mathematical formulation

### 2.1 Objects

- Public tree $T$ with node types *decision* ($P(x)$ acts, legal actions $A(x)$ from the menu, which shrinks with depth in the street), *chance* (a public card) and *terminal* (fold or showdown).
- Fibers $H_i(x)$, hand–card incidence $M \in \{0,1\}^{F\times 52}$, hand rank $\rho(h, \text{board})$.
- Payoff $u_i(z, h)$ at terminal $z$ given the joint hand $h$. At a fold terminal it does not depend on $h$. At a showdown it depends on $h$ only through ranks and the pot.
- Strategy profile $\sigma$, reach $\pi_i$, ranges $r_i(x) = \pi_i(x,\cdot)$ with board-conflicting hands zeroed.

### 2.2 Counterfactual values: linear at two-player terminals, never a matrix

Player $i$'s counterfactual value at $x$ for hand $h$ is the expected payoff if $i$ held $h$ and played to reach $x$ (own reach removed), weighted by everyone else's reach:

$$v_i^\sigma(x, h) = \sum_{h_{-i}} \Big(\prod_{j \ne i} \pi_j(x, h_j)\Big)\,\mathbb{1}[\text{disjoint}]\ \mathbb{E}^\sigma[u_i \mid x, h, h_{-i}].$$

**Two players.** At a terminal this is $v_1 = U_z\, r_2$ with $U_z$ an $F\times F$ matrix that is never built. The disjointness matrix is exactly

$$D = J - MM^\top + I$$

($J$ all-ones; $(MM^\top)_{hh'} = |h \cap h'|$ is 2 only on the diagonal), so $D$ is the identity plus a matrix of rank at most 48. Hence:

- *Fold terminal*, constant payoff $c$: $v_1 = c\,D r_2 = c\,(\mathbf 1^\top r_2 - M(M^\top r_2) + r_2)$. Cost $3F + 52$.
- *Showdown*: $W = S \circ D$ with $S_{hh'} = \operatorname{sign}(\rho(h) - \rho(h'))$. $S\circ J$ is a prefix sum along one global rank order; $S \circ MM^\top$ splits into 52 per-card prefix sums over the ≈ 46 hands containing each card; $S\circ I = 0$. The sorts depend only on the board and are done once per board. Per iteration the cost is $O(3F)$. Appendix A verifies both tricks against brute force to 10⁻¹² and gives the code; the structured showdown measured 85 µs against 5.8 ms for the dense matvec.
- *All-in before the river*: a fixed equity matrix $E_b$ per board (average over run-outs), so every all-in leaf in a street shares one matrix and all of them together are one GEMM per iteration.
- *Decision node*: $v_i(x,\cdot) = \sum_a \sigma_{P(x)}(x,\cdot,a)\odot v_i(xa,\cdot)$ if $P(x)=i$, else $\sum_a v_i(xa,\cdot)$ with the actor's strategy folded into their reach.
- *Chance node*: a per-hand average over non-conflicting cards (weights in §2.5).

**Fiber locality.** Every nonlinear operation of regret matching (positive part, normalization) acts on one (node, hand) cell. Every operation that couples different hands is a fixed, strategy-independent map at terminals and chance nodes: linear for two card-holding players, **multilinear of degree $N-1$** in the opponents' reach vectors for $N \ge 3$ (§5.2). The depth-cut map $V$ is the one hand-coupling operation that is *not* linear: it is positively 1-homogeneous in each range, concave in your own and convex in the opponent's, and its per-hand values are its super- and sub-gradients (Danskin), with the exact identity

$$\langle r_1, v_1\rangle = V = -\langle r_2, v_2\rangle .$$

Per-hand values can jump. In the clairvoyance river (pot 1, one pot-sized bet, bettor holds the nuts with probability $p$ and air otherwise), the value of air is 0 for $p < 2/3$ and a full pot for $p > 2/3$, and any value in $[0,1]$ at $p = 2/3$. An $\varepsilon$-equilibrium pins it only to within $O(\varepsilon/|p - 2/3|)$. The labels $V$ must learn are therefore discontinuous and solver-dependent where ranges cross such thresholds. This is one reason $V$ is hard to learn (§6), and it is why the toy's only effective training lever was weighting the loss by the player's own range: per-hand values of hands a player rarely holds are not pinned down by the equilibrium and dominate an unweighted error.

So one pass over a public subtree with $n$ nodes costs $O(nF)$ in the interior plus the leaf kernels: $n \times F$, not $n \times F^2$ and not $F$ copies of $n$. This is the arithmetic reason the hidden state never multiplies the tree by itself.

### 2.3 The dynamics, and why "opponents are weather" is a theorem

At each decision node $x$ of player $i$, for each hand $h$ and action $a$, keep a cumulative regret $R(x,h,a)$. One iteration of regret matching⁺:

$$R \leftarrow \max\Big(0,\ R + v_i(xa, h) - \sum_{a'} \sigma_i(x,h,a')\,v_i(xa',h)\Big),\qquad \sigma_i(x,h,\cdot) \propto R(x,h,\cdot)^+ .$$

Average the iterates with linearly increasing weights (CFR⁺), or with the discounting of DCFR. There is no learning rate, no step size, no critic. The only knob is the iteration count, and more is monotonically better. After any solve, one extra pass computes the exact best-response gap, so every solve emits a certificate.

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

**The past, robust version (two players).** Your own range at $x$ is your own reach, known exactly. For the opponent, carry **not** their range but the vector $\bar v_2(x, h)$ of their per-hand counterfactual values at $x$ from your previous solve. At the root of the new solve give the opponent, per hand, a choice between *Terminate* (take $\bar v_2(x,h)$) and *Follow* (play the subtree). Their range at $x$ is then a variable chosen adversarially subject to those ceilings. This is the re-solving gadget (Burch, Johanson, Bowling 2014; DeepStack). Any solution satisfies $u_2(h) \le \bar v_2(h)$ for every hand, which is exactly the condition under which replacing your old plan below $x$ by the new one does not increase the opponent's best-response value anywhere above $x$. Exploitability of the combined strategy is bounded by the starting strategy's plus the solver residual plus the ceiling error, linearly over re-solves.

Why this is not optional, in one example. River, pot 1. The opponent holds the nuts or air; all your hands are bluff-catchers. Suppose your carried range for them says "nuts only". A range-only re-solve folds every hand to every bet. If air is in fact present (they deviated upstream, or the carried range was wrong), air bets and takes the pot every time. The gadget sets air's ceiling at 0 (its value under your previous plan, which called at the minimum defence frequency) and forces the re-solve to keep calling half the time however improbable air is believed to be. The toy measured the difference at 23.7 vs 520 milli-antes per hand with identical exact leaves; the naive agent's out-of-position seat was exploitable by 757.

**The past, exploitative version.** When a model $\hat\sigma_j$ is in play, the opponent's modelled range is $r_j^{\text{model}} = b_0 \odot \prod_k \hat\sigma_j(a_k \mid \cdot)$ along the hand, and it enters as a *second type* of opponent alongside the adversarial one (§2.6). The gadget is still present for the adversarial share.

**Off-menu opponent bets.** If the opponent bets a size not in your menu at node $p$, re-solve from the *opponent's* node $p$ with the observed size added to their action set and the ceilings $\bar v_2(p)$ that held at $p$. The solve decides which of their hands would prefer that size given your replies; read their range after the bet from the solved strategy and proceed. Nothing is snapped. Cost: one extra re-solve of a tree with a handful more nodes. Toy: inserting the size cost 1.4–18.8 milli-antes per hand against an exploiter, versus 1.5–2.2 for a full solve of the augmented game and 51–389 for snapping to the nearest menu size. Snapping was not even monotone in quality: for a 1.5-pot bet, snapping to the nearest size lost 51 and snapping up lost 236.

### 2.5 The blueprint-free ladder, and where exactly the cut goes

$V$ is defined by a recursion that terminates at the river, where there is no boundary. So:

- **River.** Exact. A river re-solve has no $V$.
- **End of turn (before the river card).** Needs $V^{(4)}$. Label by sampling boundary situations (4-card board, pot, stacks, two ranges) and solving river subgames exactly.
- **End of flop (before the turn card).** Needs $V^{(3)}$. Label by solving turn subgames with $V^{(4)}$ at their boundaries.
- **Preflop.** Either a third map $V^{(2)}$ at the end of preflop, or solve the preflop tree offline with *sampled* flops: each iteration deals one flop per boundary leaf, solves that flop subgame with $V^{(3)}$ leaves, and feeds the values back (public-chance-sampled CFR, unbiased). The preflop tree is 16 nodes on a 169-class fiber, so this is hours per stack depth on the home PC and removes one learned map and one compounding step.

**Where the cut sits relative to the card.** Cut *before* the chance node. The boundary value is then the chance average of independent per-card subgames,

$$v_{\text{pre}}(h) = \frac{1}{N_{\text{left}} - 4}\sum_{c \notin h} v_c(h),$$

where $N_{\text{left}}$ is the number of cards not on the board (49 after the flop, 48 after the turn) and $v_c$ is the unnormalized counterfactual value of the post-card subgame with the opponent's $c$-blockers zeroed. The divisor is 45 from flop to turn and 44 from turn to river, **not** a uniform average over 47 or 49 legal cards; the uniform form shrinks every label by 0.92 per cut and compounds to 0.84 over two cuts. (The toy confirmed the identity exact to machine precision with the right weights, and the 0.84 shrinkage with the wrong ones.) An unbiased single-card label is $\frac{N_{\text{left}}}{N_{\text{left}}-4}\,\mathbb 1[c\notin h]\,v_c(h)$ with $c$ uniform over the $N_{\text{left}}$ cards. One net call per boundary leaf per iteration instead of 47–49.

**Chance-sampled labels are an option to measure, not a lever to assume.** Each single-card label costs 1/48 of an all-cards label, but it carries the whole next-card variance. In the toy, at equal compute, exact all-card labels beat single-card labels (range-weighted error 0.153 vs 0.180 at 18 000 solves); the nominal 9× saving came out as 0.25×. The toy is a worst case (one-card hands make the next card decide everything), and a per-card control variate was not tested, so the hold'em answer is open. Measure it before relying on it.

**Carrying the ceilings across a card.** A pre-chance $V$ outputs *chance-averaged* opponent values, and those are not valid per-card ceilings: a card that favours you leaves the averaged ceiling above the true per-card value, and the re-solve may concede up to it. In the toy, averaged ceilings scored 608 milli-antes per hand, worse than having no strategic leaf at all. A "start-of-street solve to produce per-card ceilings" also fails, structurally (62–93 at any iteration count), because it selects a different equilibrium than the one the previous solve committed to. What works is the chance-rooted gadget: at the first decision of each street, root the gadget re-solve at the *pre-chance* node with its averaged ceilings, so all next cards sit in one forest and the opponent's Terminate/Follow choice is made per hand before the card. Toy: 25.2 vs 23.7 for exact per-card ceilings. Cost: that one re-solve per street evaluates $V$ once per card, about 49× the net calls of a mid-street re-solve.

**What makes $V$ learnable, and what does not.**

- *Symmetry.* Suit permutation reduces 22 100 flops to 1 755 classes and 4-card boards to 16 432 classes (the unordered set is the right input object for $V$; the ordered flop-plus-turn count of 63 193 is the right object for counting public nodes). Postflop fibers compress only 1.1–1.4×, so symmetry canonicalizes $V$'s inputs and augments its data; it does not shrink the solver.
- *An exact baseline.* The check-down value, each hand's equity against the opponent's range times the pot, is exact, bilinear in the ranges, commutes with chance, and equals $V$ at stack-to-pot ratio zero. Learn the residual. In the toy the residual target lowered regression error (0.230 vs 0.315) but **did not improve play** (499 vs 506 milli-antes per hand). Treat it as architecture, not as a data multiplier.
- *Homogeneity and the zero-sum identity.* Exact constraints you can impose on the architecture (normalize ranges; project outputs onto $\langle r_1, v_1\rangle + \langle r_2, v_2\rangle = 0$).
- *Per-hand weight sharing.* Each solved situation yields about 2 × 1128 per-hand targets for one shared per-hand model over exact showdown-profile features. Plausible as a data lever; unmeasured.
- *Range-weighted loss.* The one lever that helped in the toy (417 vs 499).
- *Smoothed targets.* Replacing Nash by an entropy-regularized equilibrium makes the labels unique and Lipschitz at a declared exploitability price $\tau(D_1+D_2)$. Examined and demoted: the standard own-reach-weighted regularizer breaks concavity and homogeneity of $V$, the constant $D$ is of order hundreds on a single river decision rather than $O(1)$, and at the $\tau$ a label budget can resolve the learnability gain is negligible. An optional priced dial, not part of the core.

### 2.6 Exploitation is the same loop with one input swapped

Let $\hat\sigma_j$ be a model of opponent $j$. Inside §2.3's loop, replace $j$'s iterate by the mixture

$$\sigma_j^t \leftarrow p\,\hat\sigma_j + (1-p)\,\sigma_j^{t,\text{regret}},\qquad p\in[0,1].$$

In sequence form the opponent now lies in the Huber set $\{(1-\varepsilon)\hat y + \varepsilon y\}$ with $\varepsilon = 1-p$, and because your payoff is linear in their sequence-form vector the max-min over that set is exactly the restricted Nash response (Johanson, Zinkevich, Bowling 2008). $p = 0$ is equilibrium, $p = 1$ is the best response to the model, and $p$ sweeps the Pareto frontier of EV-against-model versus your own exploitability. In the toy this is a three-line change to the solver.

**The frontier has a free part.** Maxmin strategies form a face, not a point. Choosing among them the one best against your forecast costs no worst-case value. The toy measured this at $p \to 0^+$: +179 milli-antes per hand against the maniac at $p = 0.001$, +179 against the station and +138 against the nit at $p = 0.01$, with exploitability at the solver's noise floor (3.4–4.4 vs 3.6–4.2). That is 5–20 % of the whole Nash-to-best-response gain for free. Only beyond the face does every gain cost worst-case value.

**The priced part needs a declared number.** A Bayesian posterior mean over opponent strategies produces no caution at all: payoff is linear in the predictive strategy, so a wide posterior and a point estimate give the same best response. Caution must come from a declared quantity: $\varepsilon$ (the mass of behaviour the model is allowed to be wrong about), or a confidence level on a constraint set, or a budget of banked gifts (Ganzfried, Sandholm 2015: risk only what the opponent's past mistakes have already paid). This contradicts the user's "no arbitrary parameters" exactly here, and it cannot be engineered away: no finite sample bounds how an opponent will deviate from their past in the future. The gift-budget rule is the one parameter-free choice, and its price is slow exploitation of opponents who make few mistakes.

**Where the model comes from (the HUD, done honestly).** You never observe hands except at some showdowns, so you cannot fit a free $F$-dimensional strategy per node. What you observe is action frequencies per public context. Two principled forms, both convex inside the same solve:

1. *Frequency constraints.* A HUD statistic is a ratio of linear functionals of the opponent's sequence-form vector. With your own historical play held fixed, "fold to flop c-bet $= \hat f \pm \varepsilon$ over $n$ observations" is two linear inequalities on their realization plan. The local problem becomes $\max_x \min_{y \in Y,\ Ay \le b} x^\top U y$, solved by adding one Lagrange multiplier per constraint on your side (Davis, Waugh, Bowling 2019). The opponent is then the worst case consistent with what was seen, and still chooses adversarially *which* hands take which action. $\varepsilon$ from counting plus the across-board dispersion of that statistic in your own solves. The confidence level is the declared parameter.
2. *An exponential tilt of the solver's own strategy.* $\hat\sigma_j(a \mid x, h) \propto \sigma_j^{\text{ref}}(a\mid x,h)\,e^{\lambda_{c(x),a}}$, with one parameter per (context class, action), 30–60 parameters per opponent. This is the information projection of the reference strategy onto the observed frequencies: HUD counts are its sufficient statistics; $\lambda = 0$ is the equilibrium, so with no data you play robustly; and a frequency deviation is attributed to the hands the reference *mixes* with, so "bets the river too often" is read as "more bluffs", not "more value", without anyone designing that rule. Fit by EM with hands latent (the E-step is a range computation you already have). Non-stationarity: discount the counts, and run a small mixture over half-lives weighted by predictive likelihood (log-loss regret $\le \ln K$ against the best half-life in hindsight, for any sequence). Showdown data is not missing at random (players muck losers), so fit from action likelihoods unless showdowns are forced.

Sample complexity is the honest limit of exploitation. Detecting a 10-point frequency deviation at 2σ needs about 100 opportunities per parameter: roughly 400–1 000 hands heads-up, 3 000–10 000 hands against one specific opponent at a full table.

**Exploitation moves $V$ off its training distribution.** Exploitative play produces ranges unlike equilibrium ranges, and $V$ has seen less data there. This is a real trade-off: train $V$ on random ranges (DeepStack) and on ranges harvested from your own play (ReBeL), include perturbed ranges, and increase $p$ slowly.

---

## 3. The honest ledger

| Component | Status | Notes |
|---|---|---|
| Hand representation | **Exact** | All combos, no buckets, ever. |
| Beliefs | **Exact given the attributed strategy** | A product of known numbers over all dealt players, folded ones included. Undefined for zero-probability actions; handled by re-solving. |
| Card removal, 2 card-holding players | **Exact** | $D = J - MM^\top + I$; $O(3F)$ per terminal per iteration. |
| Card removal, 3 live players | **Exact and cheap** | $O(52F)$ per showdown, $O(52^3+F)$ per fold terminal (§5.2). |
| Card removal, 4 live players | Exact offline, $O(52^3 F)$ | Too slow for play-time; sample there. |
| Card removal, 5+ live players | Sampled (unbiased) or a **declared approximation** | Independence costs 0.3–3 % of pot mean error and 10–15 pp in blocker-heavy spots; in a solver it was a 24–65× exploitability error in the toy (§5.2). |
| Opponent bet sizes | **Exact** | Inserted and re-solved. Measured cost 1.4–18.8 vs 51–389 mA/h for snapping. |
| Your own bet sizes | **Chosen** | A restriction of your actions. Measured: tapered 3-size menu ≤ 0.25 % of pot vs a 6-size menu; 1 size 0.4–2.6 %; jam-only 0.6–6 %. |
| Regret dynamics | **Exact, parameter-free** | Iteration count is a budget. Certified gap after every solve. |
| Past (earlier streets), 2 players | **Exact summary** | Your range plus opponent ceilings. |
| Past, 3+ players | **Tracked ranges, no safety theorem** | §5.3. |
| Future (later streets) | **Approximated: $V$** | The one approximation. Error measurable on held-out exact solves. Its certified bound is useless (≈ 250 mbb/hand at 1 % sup-norm error); its practical effect must be measured in decision units. |
| Opponent model | **Model**, exploitation mode only | Low-dimensional, likelihood-fit, prior at equilibrium, discounted. Misspecification bounded by $p$. |
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

With this menu one heads-up postflop street has **28 decision nodes, 26 fold terminals, 13 street-boundary leaves and 14 all-in leaves** at 100 bb, and the same counts at 500 bb because the taper ends in all-in. (The count follows the recursion $D(\text{cap}) = 1$, $D(d) = 2 + m_d D(d+1)$, $N = 2 + 2(m_0 D(1) + 1)$ with $m_d$ the menu size at depth $d$; three sizes at every depth with cap 4 would give 322.)

**Hand.** Button opens 2.5 bb, you call in the big blind. Pot 5, stacks 97.5. Flop K♠ 7♦ 2♣ (rainbow, distinct ranks, so all 1 176 live combos are distinct fiber entries). You check. Button bets 3.75 (75 %). You hold 8♦ 7♣.

**Step 1: the public node.** $x$ = [BTN 2.5, BB call | K♠7♦2♣ | check, bet 3.75]. Pot 8.75, you face 3.75, button has 93.75 behind.

**Step 2: what you carry.** Your range: your preflop probability of calling with each combo (a lookup in the offline preflop solve) times your probability of checking this flop with it (from the re-solve you ran one node ago). Two numbers multiplied per hand. The button's *ceilings*: their per-hand counterfactual values at $x$ from that same re-solve (which solved the whole flop subtree, both players, so their reaction to your check is already in hand). You do not carry a button range. If a model were active, the button would in addition get a model type with range $r^{\text{model}} = \text{open prob} \times \hat\sigma(\text{bet 75 \%} \mid \cdot)$.

**Step 3: the local tree.** From $x$: fold, call, raise to 75 % of the pot after a call (call 3.75, pot 12.5, raise 9.375 on top), or jam. After a raise the button folds, calls or jams; after a jam you fold or call. About 5 decision nodes, 5 fold terminals, and 5 street-boundary leaves (call; raise–call; raise–jam–call; jam–call; and the all-in lines that run out the board with no further betting, which can be served exactly by the run-out equity GEMM). Each boundary leaf $y$ returns $V^{(3)}(y, r_{BB}(y), r_{BTN}(y))$, one call per leaf per iteration because the cut is before the turn card.

**Step 4: the fixed point.** Run 300 iterations of DCFR with the gadget at the root. Per iteration: 5 nodes × 1 176 hands × 3 actions of regret arithmetic (trivial), 5 fold terminals in $O(3F)$ (trivial), 5 network calls (about 6.5 MFLOP each for a DeepStack-sized 3.3M-parameter MLP, 33 MFLOP), and one all-in GEMM (about 77 MFLOP shared across the all-in leaves). About 110 MFLOP per iteration, 33 GFLOP per decision: **1–3 s in batched numpy on a 2015 quad-core, under a second compiled.** A full 28-node street from its root is about 85 MFLOP of network per iteration and takes 1.5–3 s in numpy. River decisions have no network (under 1 s numpy, 40 ms compiled). The *first* decision of a street, which must carry the ceilings across the card, runs the chance-rooted gadget with one network call per turn card: 13 × 49 = 637 calls per iteration, 5–20 s on a 2015 CPU depending on network size, which is why the network should be as small as its measured accuracy allows.

**Step 5: the action.** Read $\sigma_{BB}(x, 8♦7♣,\cdot)$ from the averaged strategy and sample. (Frequencies are not quoted here because this solve was not run at hold'em scale; §4.2 shows a complete decision with real numbers from the toy.) Note what did not happen: nobody asked "what does he have?". The button's hands entered as an adversarial variable bounded by ceilings, exactly, in every value computation.

**Off-menu.** Had the button bet 2.2 bb (44 %), add that branch at the button's node, re-solve the flop street from your check decision with the ceilings that held there, read the button's range after the 2.2 bet from the solved strategy, and proceed to Step 1 with it. One extra re-solve of a tree with four or five more nodes.

**Exploitation.** Suppose 150 observed spots say this button folds to flop check-raises 62 % of the time, against roughly 45 % for the robust solve. Add the constraint $f_{\text{fold}} \in [0.62 - \varepsilon, 0.62 + \varepsilon]$ at the button's node after your raise with $\varepsilon = 1.96\sqrt{0.62 \cdot 0.38/150} + \varepsilon_{\text{pool}} \approx 0.08 + 0.05$, or fit the tilt parameter for that context and run the mixture at some $p$. Your raise frequency with 8♦7♣ rises, by an amount bounded by what the worst case inside the band could punish. Same tree, same fiber, same network, same loop.

### 4.2 A complete decision with real numbers (toy game)

[[WALKTHROUGH: inserted from toy/hu/results/walkthrough.txt]]

### 4.3 Three players to the flop

Three ranges (plus the folded players' frozen vectors), boundary values $V_3$ with three range inputs and three output vectors, fold terminals by the inclusion-exclusion sums of §5.2, exact 3-way showdowns at $O(52F)$, and the same loop. The tree is larger (300 decision nodes for the full menu at 3 players, 42 for a one-size menu), the equilibrium guarantee and the gadget theorem are gone, and the opponents' ranges must be tracked from the attributed strategies. When one player folds, the leaves switch to $V_2$ and the menu switches to the heads-up menu at the next street.

---

## 5. Three to nine players, and other variants

### 5.1 What survives, what breaks

Survives unchanged: carried ranges (dimension additive in players), the public-tree-plus-fiber representation, the per-player no-regret guarantee, the chance-node cut and the value map, off-menu insertion, the exploitation dial, and the entire variant plug-in structure.

Breaks, for mathematical reasons rather than engineering ones:

- **No Nash target.** CFR⁺ self-play in the 3-player toy reached about 1 milli-ante of NashConv at 2 000 iterations, plateaued, and drifted upward after 7 000; the 2-player control went to 0.02. The current iterate oscillated at 8–10. Eight self-play runs agreed on seat values within 1 mA but differed in strategy by 5–9 % per decision, and recombining strategies across runs raised NashConv to 4–109 mA. $V_3$ is therefore defined only as "what this procedure converges to", and leaf maps must be trained on ranges produced by the same procedure.
- **No safety theorem.** The gadget's guarantee is two-player zero-sum. Multiway re-solving tracks ranges and has no certificate.
- **No security without a coalition model.** The only opponent-independent value is the maxmin against the other seats as a card-sharing team, a 2-player zero-sum game with a product fiber ($1326 \times \binom{50}{2}$ for two colluders). In the toy the self-play strategies' security against such a coalition was −0.55 to −1.15 antes per hand, often worse than folding every hand (−1). The coalition maxmin itself was −0.21, −0.16 and +0.006 by seat (the last seat's is positive; only the sum over seats is ≤ 0), and playing it cost 17–28 mA per hand against the self-play population. So: a computable, conservative security certificate exists; it is a dial with a price; and "weather-robust" at a full table is empty without it.

What can honestly be promised at 3–9 players: per-player hindsight regret inside self-play; a measured NashConv (one best response per player, exact in toys and abstract games, sampled lower bounds in full NLHE); a computable coalition-security number; and never "unexploitable".

### 5.2 The multiway hidden state: a product measure with a hard-core constraint

With $L$ live players at a terminal, the opponents' joint weight is a product of ranges restricted to card-disjoint tuples. Writing $K = MM^\top - I$ for the collision indicator and $D = J - K$,

$$\prod_{j<l}(1 - K_{jl}) = \sum_{G \subseteq \binom{[m]}{2}} (-1)^{|E(G)|} \prod_{(j,l)\in E(G)} K(h_j, h_l),$$

the Mayer cluster expansion of a hard-core gas on 52 sites, with one term per graph on the $m = L-1$ opponents. Each term contracts over *cards* (52-dimensional), not hands, because $K$ factors through $M$.

- **Two opponents (3-way pot): exact and cheap.** One edge. For hero hand $h$ with opponents' per-card masses $Q_j(c)$ (mass on hands containing $c$):
  $$Z(h) = Q_j Q_l - \sum_c Q_j(c) Q_l(c) + \sum_a q_j(a) q_l(a),$$
  all after masking by $h$ and the board. Fold terminals cost $O(52^3 + F)$ (one 52 × 52 matrix product; verified against brute force to 10⁻¹³); showdowns, sweeping hands in rank order with per-card accumulators, cost $O(52F)$ per player vector (derived; the toy's 1-card version verified to 10⁻¹³).
- **Three opponents (4-way): exact at $O(52^3 F)$.** Eight graphs; the triangle term is a trace of three 52 × 52 matrices. Fine offline, too slow in play.
- **Four or more opponents:** $2^{\binom m2}$ graphs; the number of connected cluster types is 1, 4, 38, 728, 26 704, … (OEIS A001187). Exact is not worth it. Sample joint deals instead: a sampled deal is card-disjoint by construction, so the error enters regret as variance, never bias.

**What ignoring opponent–opponent removal costs.** Keeping hero–opponent and board removal exact but treating opponents as independent gives mean showdown-equity errors of about 0.3–0.6 % of pot for value-heavy ranges, 0.6–1.1 % for narrow ranges and 2.5–3 % for polarized ranges at 4–6 live players, with per-hand maxima of 2–10 % and 10–15 percentage points when ranges are built around a shared blocker (a nut-flush card held as value or as bluffs). Relative errors in a bluff-catcher's equity reached 3–100×, which flips decisions. Inside a solver the damage is strategic, not cosmetic: in the 3-player toy, independence made the solved strategy 65× more exploitable (72 vs 1.1 mA) and 24× at 4 players, a bias iterations do not remove. The linear first-order inclusion–exclusion correction is numerically unstable and often *worse* than independence from 4 opponents on (it produced negative masses on small decks); the exponentiated pairwise (Kirkwood) form, at $O(m^2 \cdot 52 \cdot F)$ per terminal, cuts the error to 0.02–0.3 % mean on generic ranges but overshoots badly when ranges concentrate on one card.

**Rule.** $L \le 3$: exact vector terminals (and brute force over the small joint was actually *fastest* in the toy at $L = 3$). $L = 4$: exact offline, sampled in play. $L \ge 5$: sampled joint deals, or a declared approximation with its error logged. And the folded players' reach vectors stay in the belief state for the dead-card measure at every $L$.

### 5.3 Count live players, not seats, with one correction

After preflop a 9-max hand *is* an $L$-player game: folded players have no actions and no payoffs, only a measure on the deck. $L$ never increases, so machinery indexed by $L$ recurses downward and leaf maps are needed only for $L \le 3$ ($V_2$ is the heads-up map, widened to cover dead money and multiway range shapes). The seat count enters combinatorially only before the flop, where the fiber is smallest (169 classes, exact) and there is no board. Hence the cost of 9-max is roughly the cost of 3-way pots, not nine times heads-up, and the 3-way menu is the single budget dial.

The correction: the folded players' dead-card measure is **not** small, as §1.2 showed (10–12 % relative shifts in hold'em, 128–178× exploitability in the toy when dropped). So either $V_L$ takes the dead-card measure as an input (a 52-vector of per-card dead mass, or the folded ranges themselves), or dropping it is a declared, measured approximation. "N enters only through the preflop trunk" is false as an exact statement and true as a cost statement.

**Where in the public tree the cost sits.** One postflop street at 100 bb:

| Live players | 3-size menu | 2-size menu | 1-size menu |
|---|---|---|---|
| 2 | 28 | 20 | 8 |
| 3 | 300 | 204 | 42 |
| 4 | 2 240 | 1 504 | 184 |
| 6 | — | 36 936 | 2 724 |
| 9 | — | 1 003 680 | 115 794 |

Two facts about multiway trees: 67 % of 3-player and 95–98 % of 9-player decision nodes are pure call/fold nodes over *caller subsets* (the $2^{k-1}$ fan-out after a bet), so bet-size menus are not the multiway cost driver; the raise cap and the per-$L$ menu are. And the preflop router at 9-max (4.7 × 10⁵ to 7.8 × 10⁶ nodes on the 169 fiber, 13–25 GB of tables with a 4-bet size) is solved offline by public-action-sampled MCCFR: at an opponent's node sample one action with its range-weighted probability and divide the child's leaf value by it, which is unbiased because $V$ is degree-1 homogeneous in each opponent's reach.

**Vectorized versus sampled.** On the 3-player toy, vectorized CFR⁺ beat external-sampling MCCFR at equal wall-clock by 35× to 10 877× in NashConv, the ratio shrinking as the deck grows (855 → 442 → 219 for decks 8 → 12 → 16) and as players are added (10 877 → 855 → 35 for 2 → 3 → 4 players). The crossover is beyond $L = 4$ for hold'em-sized fibers; sampling is for the preflop router and for $L \ge 5$ terminals.

### 5.4 Variants: three plug-ins

Variant-specific: the **deal schedule** (deck, hole cards per player, which cards are public and to whom, private draws), the **showdown order** (a total preorder on the fiber, or two for hi/lo, plus qualifiers), and the **betting grammar** (forced bets, legal-size function, the menu). Shared: everything else. Card removal for $k$-card hands generalizes exactly by inclusion–exclusion over shared subsets, $D = \sum_{s=0}^{k} (-1)^s M_s M_s^\top$ ($k = 2$ recovers §2.2).

| Variant | Fiber (preflop / river) | Collision probability of two random hands | What changes |
|---|---|---|---|
| NLHE | 1 326 / 1 081 | 0.079 | baseline |
| Short-deck (36) | 630 / 465 | higher | evaluator swap; cheaper |
| Limit hold'em | 1 326 / 1 081 | 0.079 | no menu: the public tree *is* the game, no action restriction at all |
| PLO4 | 270 725 / 178 365 | **0.302** | 16-term inclusion–exclusion; independence unacceptable multiway; **$V$'s input width (2 × 270k) is unaffordable** |
| 7-card stud | depends on upcards (≈ 1–2k) | — | upcards move into the public tree; fits better than hold'em |
| Draw games | $\binom{52}{5}$ | — | a draw is a linear stochastic operator on the fiber; exact but large |

The mechanism transfers to all of these. The budget does not transfer to PLO: every exact component scales with the fiber (about 200× hold'em per node, still linear), but a learned $V$ over 2 × 270 725 inputs needs a compressed range representation, and that is abstraction under another name. Say so rather than hide it.

### 5.5 Coalitions

A coalition is a change in the weather players' *utility aggregation* (a matrix $W$ summing payoffs over the team) and, if they share cards, a merge of their fibers into the card-disjoint product. The same engine runs. With card sharing, "hero versus coalition" is 2-player zero-sum, CFR converges to its value, and the coalition best response against any fixed hero strategy is one traversal: that is the computable security certificate of §5.1, feasible on river subgames in hold'em (the merged fiber is about 1.6 × 10⁶ per node) and everywhere in toys. Without card sharing, team optimality needs a correlation device (TMECor) and team-maxmin without one is NP-hard in general; alternating team best responses give a lower bound on the damage. Toy: a soft-playing coalition cost the frozen hero only 4 mA per hand but left 74 on the table; a "whipsaw" coalition that always raised into the hero was worth +5.7 antes per hand to a hero who knew it, and a model-free online learner never found that in 10⁶ hands.

---

## 6. Cost and feasibility

### 6.1 Calibration

The user's own figures: $10–15 ≈ 100–300 core-hours (so $0.033–0.15 per core-hour) or ≈ 10–20 GPU-hours (so $0.5–1.5 per GPU-hour). All dollar figures below use these; essays that priced a GPU at $0.20/h were re-priced. The home PC: 4 cores of 2015 vintage, DDR3; a week of it is about 670 core-hours worth roughly 300–450 modern core-hours. Vector CFR is memory-bandwidth-bound in the interior and matmul-bound at the network leaves; a GPU is about 25× cheaper per unit of this work *if a kernel exists*, which no estimate here costs.

### 6.2 Per-decision latency (heads-up, 2015 quad-core)

| Decision | numpy | compiled C |
|---|---|---|
| River (exact, no network) | ≈ 1 s | 0.04 s |
| Turn or flop, mid-street, 13 pre-chance network calls, $T = 300$ | 1.5–3 s | 0.6–1 s |
| First decision of a street (chance-rooted gadget, 637 network calls) | 20–40 s with a 3.3M-parameter net; 5–10 s with a 1M-parameter net | 7–15 s |
| Exact turn-plus-river (no $V^{(4)}$, 3-size menu, 213 MB of state) | 1–3 min | 10–40 s |
| Preflop (169 fiber, lookup or 16-node re-solve) | < 1 s | < 0.1 s |
| 3-way flop, 2-size menu | ≈ 6 s | ≈ 5 s |
| 6-way flop, 1-size menu | ≈ 1 min | ≈ 50 s |
| 6-way flop, 2-size menu | ≈ 15 min | ≈ 12 min |

Solver convergence on real river boards with exact card removal: $T = 300$ DCFR leaves 0.035 % of pot at SPR 2.4 and 0.25 % at SPR 20; CFR⁺ 0.095 % and 0.94 %. Deep stacks need about 3× the iterations for the same percentage of pot. $T = 300$ is the decision-time default and $T = 200$ is enough for training labels.

Two consequences. The exact turn-plus-river solve that DeepStack ran on a GPU is viable at play time *only compiled* and only with a time bank; it is the right held-out validator for $V^{(4)}$ either way. And 4+-way pots must drop to a one-size menu to stay under a minute.

### 6.3 Offline: the ladder, and the one unknown

Per-label costs with a compiled kernel (numpy is 3–10× slower; GPU figures assume batched kernels):

| Label | What one label solves | CPU core-seconds | GPU ms |
|---|---|---|---|
| $V^{(4)}$ (end of turn) | one river subgame, exact terminals | 0.05–0.2 (0.64 numpy) | 8.5 |
| $V^{(3)}$ (end of flop) | one turn subgame with 13 pre-chance $V^{(4)}$ calls per iteration | 0.5–1 | 4.3 |
| exact validator | one turn-plus-river solve | 10–40 | ≈ 6 000 |

So 10⁶ river labels are 15–55 core-hours ($0.5–8); 10⁶ turn-subgame labels are 140–280 core-hours ($5–40) on CPU or about 1.2 GPU-hours ($1–2); 2 000 exact validation solves are 6–22 core-hours. With GPU kernels, label volume up to a few million per rung fits inside $10–15. With CPU only, about 10⁵–10⁶ labels per rung fit. Training the networks is a few CPU-hours per rung.

**The one unknown is how many labels $V$ needs, and the prior is against the budget.** DeepStack trained its turn network on 10⁷ solved turn situations (on bucketed inputs) and its flop network on 10⁶; Supremus used 5 × 10⁷ river, 2 × 10⁷ turn and 5 × 10⁶ flop subgames and reports Huber losses of 0.016 (turn) and 0.028 (flop) in fractions of pot; a 2020 reimplementation of DeepStack at its original scale lost to Slumbot; OpenHoldem's flop network "suffers from severe over-fitting" at 1–3 × 10⁶ situations. The toy's learned $V$ at 2 × 10² to 2 × 10⁴ labels played no better than the check-down leaf (417–550 vs 533 mA/h; exact leaves 24). The data-efficiency ideas in §2.5 (per-hand weight sharing, residual target, equivariance, range weighting) are the reasons a 10–100× smaller requirement is *possible*; none is proven, and one (the residual) was already measured not to move play in the toy.

**The gate, pre-committed.** Before spending beyond about $1: generate 10⁴, 3 × 10⁴ and 10⁵ end-of-turn situations with exact all-card labels in canonical suit form; train with range-weighted loss; score each in **decision units**: on 20–50 held-out flop situations, solve the flop street with the learned $V^{(4)}$ at its leaves, then compute the exact turn-plus-river values of the leaf situations those strategies induce, and report the best-response gap caused by the leaves in percent of pot (target: below the solver's own residual of about 0.3 %). Fit a power law to the three points and extrapolate to the budget's label count. Then decide:

- *Curve reaches the target inside the budget:* buy the labels, build $V^{(3)}$ the same way (validated against exact turn-plus-river solves), and solve preflop by sampled flops.
- *Curve does not reach it:* the honest products are (i) an exact river and turn engine (compiled, no learned component, exploitability certified per solve), which is a complete sound player for the last two streets, and (ii) flop and preflop play with a $V^{(3)}$ whose measured decision regret is stated, not hidden; or buy GPU time at the measured per-label price. The exploitability bound for depth-limited play is linear in the leaf error (about $2\varepsilon \times \sum \text{reach}\cdot\text{pot} \approx 250$ mbb/hand at 1 % sup-norm error), so there is no certified route to a small number; there is only the measured one.

### 6.4 Evaluation is a budget line

Heads-up NLHE has a per-hand standard deviation of about 8 bb. Resolving ±25 mbb/hand at 95 % confidence needs about 4 × 10⁵ raw hands, or about 4 × 10⁴ with AIVAT-style variance reduction. At 1–4 s per decision on the home PC that is 2–14 days per comparison. 9-max self-play evaluation to 1 bb/100 is of order 10⁶ hands. Evaluation, not training, is the slowest loop in the project, and it competes with training for the same machine.

### 6.5 9-max

With the live-count structure, 9-max offline cost is the heads-up ladder plus a 3-way ladder plus the preflop router plus evaluation. Ignoring the HU uncertainty above and pricing at $0.05 per core-hour with a ×2 margin, the 3-way menu sets the bill: about **$66** with a one-size-plus-all-in 3-way menu, **$148** with one size plus a raise size, **$268** with the heads-up menu in 3-way pots. At the user's own price range ($0.033–0.15) multiply by 0.67–3. The router at 9-max is a few to 30 % of the total and needs a rented 32–64 GB box for its tables. Pots with 4+ live players get a declared approximation (sampled full-depth solving is exact in expectation but minutes per decision on a 2015 PC; check-down leaves are a named heuristic whose use is logged). Carrying the folded players' dead-card measure into $V_2$ and $V_3$ is an input-width cost no essay priced.

Judgement: **$65–100 for 9-max holds only with a coarse 3-way menu, and only if the heads-up label count turns out lean.** If heads-up needs 10× the lean label count, heads-up alone is about $35 and 9-max is $300+. The robust output of the cost model is the ratio (9-max ≈ 9–10× heads-up at the coarse menu, ≈ 20× at the medium one, ≈ 37× at the full one), not the dollar figures.

---

## 7. Failure modes, and where the requirements contradict each other

### 7.1 Failure modes with their detectors

1. **$V$ under-trained.** Detector: the §6.3 learning curve in decision units; held-out error against exact turn-plus-river solves; the gadget audit (the opponent's realized value exceeding their ceiling after a re-solve measures ceiling drift online for free).
2. **$V$ off-distribution under exploitation or against strange opponents.** Detector: ceiling drift; Bellman residual $\|V^{(3)}(\beta) - \text{solve}(\beta; V^{(4)})\|$ on reached situations. Mitigation: random plus harvested plus perturbed training ranges; raise $p$ slowly.
3. **Error compounding through the ladder.** $V^{(3)}$ inherits $V^{(4)}$'s error; the toy measured bootstrapped flop labels already 0.07–0.21 range-weighted RMSE off exact before the flop network added its own 0.23–0.57. Removing one rung (sampled-flop preflop) helps; exact validators at each rung are the only check.
4. **Per-hand label discontinuities** at range thresholds (§2.2). Mitigation: range-weighted loss; optional smoothing at a declared price.
5. **Multiway: no safety, no unique target, non-interchangeable runs** (§5.1). Detector: measured NashConv per solve; never mix leaf maps and routers from different runs.
6. **Multiway card removal and folded players** (§5.2–5.3). Detector: compare exact vs approximate terminals on logged spots.
7. **Opponent model misspecification.** The tilt family expresses frequency deviations with hand selection inherited from the reference; it cannot express "bluffs with showdown-value hands" or size tells the menu merges. Detector: prequential log-loss against a saturated per-class model; response: raise $\varepsilon$.
8. **Reference drift.** Tilt parameters are defined relative to the solver's reference; when $V$ improves, recompute the stored reference matrices or the HUD model silently decays.
9. **Showdown selection bias** (mucking is not missing at random). Fit from action likelihoods.
10. **Adaptive opponents.** External regret is the wrong counterfactual; a baiting opponent can recover at most what the gift budget risked, in expectation. Mutual adaptation has no convergence guarantee; the $\varepsilon$ floor bounds the loss per cycle, not the cycling.
11. **Latency.** Chance-rooted re-solves at street starts are 5–40 s on a 2015 CPU; 4+-way pots approach minutes. Fewer iterations raise a *certified* gap.
12. **Engineering.** Numpy alone does not meet the label budget; a compiled kernel (a few hundred lines) is the largest single engineering item, and array layout alone is a 7× factor (action-major `(A, boards, F)` vs hand-minor). Running several numpy processes with multithreaded BLAS oversubscribes the machine; set one BLAS thread per process.
13. **Deep stacks (500 bb).** The tapered menu keeps tree size flat, but $V$ must cover SPR up to about 100, deep stacks need 3× the iterations, and menu thresholds (the 67 % all-in rule) put kinks in $V$ as a function of SPR. Stratify training by SPR.

### 7.2 Where the requirements contradict, exactly

1. **"Independent of the opponents" vs "maximum profit from observed behaviour."** The free part of the frontier (selection on the equilibrium face) costs nothing. Beyond it, the minimax theorem makes every gain against a forecast a loss against some opponent; the trade-off is set by a declared risk quantity ($\varepsilon$, confidence level, or gift budget). "No arbitrary parameters" and "maximal exploitation" cannot both hold. The gift-budget rule is parameter-free at $\rho = 0$ and exploits slowly.
2. **"Opponents are weather" vs opponents who adapt.** The guarantee is against the realized sequence; policy regret against unbounded-memory adaptation cannot be sublinear. Weather that reacts to your umbrella has no analogue in the theorem.
3. **"Guarantees, no OOPS" vs 3–9 players.** No unique value, no gadget theorem, self-play marginals carry no security (−0.55 to −1.15 antes per hand against colluders in the toy), and the only guaranteed strategy is the coalition maxmin, which is passive and loses against ordinary tables. What can be promised is measured: per-player regret, NashConv, a conservative security number.
4. **"No abstraction" vs "$10–15 on a 2015 PC."** The full tree is 0.9 PB and an exact flop decision is hours, so a learned $V$ is forced. Its certified bound is useless, its practical quality is measurable, and the toy measured exactly the failure the user calls an OOPS at a small label budget. "No OOPS" holds for everything except $V$, and $V$ is where the project's risk is.
5. **"No snapping" vs a finite own menu.** One-sided. Opponent sizes are exact at a small measured cost; your own menu costs EV (≤ 0.25 % of pot at three tapered sizes, 0.4–2.6 % at one) and never safety.
6. **"9-max at $65–100" vs heads-up-quality multiway.** Only a coarse 3-way menu fits; the heads-up menu in 3-way pots costs $130–270 at CPU prices.
7. **"Exact card handling" vs 5+ live players.** Exact opponent–opponent removal is super-exponential in live opponents; the choices are unbiased sampling (variance, time) or a declared approximation whose error the toy shows is strategic, not cosmetic.
8. **"Stability, no trial and error" vs learned maps.** The label count and the network's hyperparameters are measurements, not derivations. The only principled substitute is one learning curve with a pre-committed extrapolation rule, and it may come back with "not enough budget".
9. **"One mechanism for all variants" vs budget.** PLO's fiber breaks $V$'s input width; compressing it is abstraction.
10. **"Demonstrated strength" vs budget.** ±25 mbb/hand needs about 4 × 10⁴ AIVAT hands, days to weeks of the home PC.

---

## 8. Evidence from the toy games

Both games are in `toy/` with READMEs, test suites and result logs. Every number is an exact best-response measurement, not an estimate. Units: mA/h is milli-antes per hand (the big-blind analogue); % pot is exploitability over the initial pot of two antes.

### 8.1 Heads-up mini hold'em (`toy/hu`)

10-card deck (5 ranks × 2 suits), one private card, three streets (preflop, one flop card, one turn card), stacks of 12 antes, menu {½ pot, pot, all-in} → {pot-raise, all-in} → {all-in}, raise cap 2. 54 957 public nodes, 171 920 information sets, fiber of 10. Vectorized EV matches a brute-force walk over all deals to 10⁻¹².

**E1, exact solve.** CFR⁺ reaches 0.53 % of pot in 300 iterations (20 s, one core, numpy), 0.071 % in 1 000 and 0.020 % in 2 000; vanilla CFR is 8.6× and 42× more exploitable at equal wall-clock.

**E2, continual re-solving, 300 iterations per re-solve, complete strategies extracted and best-responded exactly:**

| Agent | Leaves at the street boundary | Exploitability mA/h | % pot |
|---|---|---|---|
| full solve, 300 / 2 000 it. | — | 10.5 / 0.41 | 0.53 / 0.02 |
| safe (gadget), exact leaves | oracle | **23.7** | 1.18 |
| naive (ranges only), exact leaves | oracle | 520.5 | 26.0 |
| safe, check-down leaf | equity × pot, no strategy | 532.5 | 26.6 |
| safe, learned leaf, 200 / 2 000 / 20 000 labels | MLP, residual over check-down | 521 / 512 / 499 | ≈ 25 |
| safe, learned leaf, 20 000 labels, range-weighted loss | same | 417 | 20.9 |
| naive, learned leaf, 20 000 labels | same | 824 | 41 |
| safe, exact at one boundary and learned (rw) at the other | mixed | 216 / 259 | 11 / 13 |

Label noise was not the limit (range-weighted label noise 0.015 vs network error 0.15–0.47). The limit was data and capacity.

**E3, off-menu opening sizes** (loss to an exploiter, mA/h): snapping to the nearest size 389 / 227 / 51 for bets of 0.25 / 0.75 / 1.5 pot; safe re-solve with the size inserted 18.8 / 7.2 / 1.4; a full solve of the augmented game 1.8 / 2.2 / 1.9.

**E4, one solver for robustness and exploitation** (seat-averaged mA/h; RNR at 600 iterations):

| Model | Nash EV | $p$ = 0.25: EV / exploitability | $p$ = 0.5 | $p$ = 0.75 | Best response: EV / exploitability |
|---|---|---|---|---|---|
| station | 812 | 2 862 / 154 | 3 086 / 278 | 3 123 / 342 | 3 143 / 1 570 |
| maniac | 541 | 1 178 / 47 | 1 298 / 138 | 1 429 / 334 | 1 429 / 1 498 |
| nit | 422 | 1 393 / 108 | 2 901 / 1 115 | 3 296 / 1 591 | 3 344 / 9 039 |

The pure best response to the nit is exploitable by 9 antes per hand because of arbitrary play at nodes the model never reaches. The free part (T5): at $p$ = 0.001–0.01 the gain was +179 / +179 / +138 with exploitability at the 3.4–4.4 noise floor.

**Timing.** Full solve 0.067 s per iteration; a river re-solve 0.1 s; a flop re-solve 0.15–0.6 s; an off-menu re-solve 0.6–0.7 s; a turn-oracle label 11 ms; an exact flop label 255 ms.

### 8.2 Three-player Tri-Leduc (`toy/multi`)

12-card deck, one private card each, two streets, stacks of 15 antes, menu {pot, all-in} → {pot-raise, all-in}, cap 2. 6 295 public nodes, 30 888 information sets. Exact-vs-brute-force to 10⁻¹²; the Möbius card-removal evaluator to 7 × 10⁻¹³.

| Experiment | Result |
|---|---|
| M1 card removal in the solver | NashConv 1.12 mA exact; 72.3 with opponents treated as independent (65×); pairwise correction exact at 3 players; at 4 players (deck 8) exact 47, pairwise 438, independent 1 127. |
| M2 convergence | 3 players: ≈ 1 mA at 2 000 iterations, plateau 0.76–0.96, upward drift after 7 000; current iterate oscillates at 8–10. 2 players: 0.02. Average external regret falls as $1/T$ for every seat. |
| M3 coalitions | Frozen self-play vs soft-play coalition −0.106 (best response −0.028); vs whipsaw −0.031 (best response +5.68). Model-free online learning after 10⁶ hands: −0.48 to −0.58 (soft-play), losing 0.6–1.3 per hand while learning. With a model (counterfactual feedback) the best response is reached in 5–50 iterations. |
| M4 sampling vs vectors | CFR⁺ beats external-sampling MCCFR 35×–10 877× at equal wall-clock; ratio falls with deck size and player count. |
| M5 scaling | Public nodes 1 119 / 6 295 / 15 053 / 64 253 / 257 437 / 13.4 M for 2 / 3 / 4 / 5 / 6 / 9 seats (≈ 4× per seat); modelled 807 s per full-width iteration at 9 seats with independent removal. |
| T3 what can be certified | 8 self-play runs agree on values within 1 mA, differ 5–9 % per decision; recombining runs: NashConv 4.4–109 mA. Security vs card-sharing coalition −0.55 to −1.15 (folding: −1). Coalition maxmin −0.211 / −0.163 / +0.006 by seat; cost of playing it 17–28 mA vs the population. |
| T4 folded players | Uniform dead cards for folders: NashConv 1.12 → 199.5 (all terminals) or 143.9 (postflop only); at 4 players 47 → 1 141–1 406. |

### 8.3 Verified independently of the toys

- The two-player fold and showdown tricks at hold'em size (1 326 combos, 5-card board) match brute force to 10⁻¹² (Appendix A).
- Single-street betting trees under human menus: 45–300 public nodes per street; the panel's enumerators agree on 28 decision nodes for the tapered 3-size menu.
- The averaging weight $1/(N_{\text{left}} - 4)$ at a pre-chance cut is exact; uniform weights are not.

---

## 9. What was examined and rejected

The user asked for "I went through twenty attempts" rather than "I tried and it didn't work." Twenty-three mechanisms were judged against five criteria: avoids nested enumeration (E), stable without tuning (S), scales to 9 players and variants (P), fits $15 for heads-up (C), yields a checkable certificate (K).

| Mechanism | Verdict | Reason |
|---|---|---|
| Sequence-form LP | oracle for toys only | 10¹⁴ variables in HU NLHE; useful as a unit test |
| Scalar tabular CFR | dead | iterates hands one at a time; 1 326× redundant |
| **Vector CFR⁺/DCFR on the public tree** | **substrate** | hidden layer disappears; public layer must be cut |
| MCCFR (outcome / external sampling) | niche | the 9-max preflop router and $L \ge 5$ terminals; 35–10 877× slower than vectors elsewhere |
| Fictitious play, XFP, NFSP | dead | $O(T^{-1/2})$ or worse; needs best responses; neural version needs tuning |
| Double oracle, PSRO | dead as mechanism, alive as evaluator | exponential oracle calls; LBR is its oracle used as a test |
| Self-play policy gradient (PPO etc.) | dead | cycles in imperfect-information zero-sum games without regularization |
| NeuRD, R-NaD (DeepNash) | idea survives, implementation does not | GPU-scale, sample-based, schedules to tune |
| Magnetic mirror descent | alternative inner update | last-iterate convergence to the regularized point; same cost as CFR |
| Predictive RM⁺, PCFR⁺ | fast default | parameter-free |
| Dual averaging, EGT smoothing | principle reused | smoothing as a priced dial |
| "Regret as the state" | partly | between decisions the state is your range plus the opponent's ceilings |
| Agent-form vs Bayesian-game form | agent form dead | destroys bilinear structure; type form is the public tree with type vectors |
| QRE / max-entropy equilibrium as target | optional dial | unique Lipschitz labels at a declared price; no free learnability win at usable $\tau$ |
| Amortized policy network (range → strategy) | dead as decision-maker | uncertifiable, un-re-solvable; warm start only |
| **Amortized value map at the depth limit** | **the one irreducible approximation** | known (DeepStack, ReBeL); its data cost is the project's risk |
| Direct best response to a population | circular at day 0 | becomes the evidence term once data exists |
| Evolutionary / population search | dead | 10⁶⁺ noisy evaluations per step, no certificate |
| Separate hand-reading predictor | dead | ill-posed; it is layer 1 mis-learned |
| IS-MCTS / determinization | dead | strategy fusion: solves the wrong game |
| Modicum multi-valued leaves | dead from scratch | continuation strategies need a blueprint |
| Benders / SDDP cuts on $V$ | beautiful, not minimal | exact tangent bounds, but loose under joint range moves in 1 100 dimensions |
| CFR-D with sampled subgames (trajectory principle) | survives for trunks | the right offline method for the 9-max preflop and for a sampled-flop preflop; no cheaper than $V$ for heads-up flops |

---

## 10. The sprint ladder

Each sprint produces one artifact and one number in decision units. If the number is not measured the sprint is not done. No later sprint changes an earlier sprint's algorithm; later sprints add a fiber (players, cards), a boundary map (one more street), or an input (an opponent model).

| Sprint | Artifact | Exit number | Compute |
|---|---|---|---|
| **0. Rapid** (days) | The toys in `toy/`, extended: Kuhn/Leduc unit tests against closed forms; the clairvoyance kink; gadget vs range-only; RNR frontier and its free part; 3-way exact terminals vs deal sampling; coalition security. | Everything in §8 reproduced. | Laptop. |
| **1. MVP** (1–2 weeks) | HU push/fold on the real 169-class fiber; then a HU **river** engine on 1 081 combos with the human menu, exact terminals, off-menu insertion, exact exploitability. A compiled kernel for the interior. | Push/fold tables match published Nash tables within 1 %; river exploitability < 0.25 % of pot at 300 iterations; seconds per solve logged. | Laptop. |
| **2. Base** (2–4 weeks) | Exact **turn-plus-river** engine (the validator and a complete sound player for two streets). Then the **gate**: $V^{(4)}$ learning curve at 10⁴ / 3 × 10⁴ / 10⁵ labels scored as leaf-exact decision regret. | Power-law extrapolation vs budget; a written go/no-go. | ≈ $1 so far. |
| **3. Full HU** (4–8 weeks, if go) | $V^{(4)}$ at the chosen label count; $V^{(3)}$ validated against exact turn-plus-river; preflop by sampled flops per stack depth; chance-rooted gadget at street starts. | Held-out decision regret per rung; LBR; then Slumbot's public API with AIVAT over ≥ 4 × 10⁴ hands (first target: not losing). | The rest of the $10–15, GPU preferred for the upper rungs. |
| **3′. Fallback** (if no-go) | Exact river and turn engine as the product; flop/preflop with a stated $V^{(3)}$ regret. | The stated number. | — |
| **4. Tuning** (ongoing) | Tilt or constraint-set model from logs; the free selection step; RNR dial; gift budget; discounting mixture. | Against scripted leaky bots: EV gain and exploitability at each $p$, both exact. | Laptop. |
| **5. 6–9 max** (2–4 months) | Live-count engine with folded-player dead-card inputs; exact 3-way terminals; $V_3$ at the coarse 3-way menu; preflop router by public-action MCCFR on a rented 32–64 GB box; sampled terminals for $L \ge 5$; coalition-security pass. | Per-player regret; measured NashConv in the solver's game; security number; match results vs baseline bots. | $65–100 only at the coarse 3-way menu. |
| **6. Variants** | New deal schedule, evaluator, betting grammar. Short-deck and limit first (limit needs no menu at all). PLO only with a declared range compression for $V$. | Sprints 1–3 numbers for the new game. | Fiber-dependent. |

The order is forced by the ladder (each rung's labels come from the rung below) and by the gate (nothing is spent before the curve is measured). Nothing else is forced. If sprint 2 says no, you still have an exact two-street player with a certificate on every decision, and you know precisely what you do not have.

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

1. Build the street subtree $\mathcal T(x)$ with your menu, the opponent's actual last size if off-menu, boundary leaves valued by $V$, and exact fold / all-in terminals.
2. Add a root node for the opponent with, per hand $h'$, two actions: *Terminate* with payoff $\bar v_2(x,h')$ to the opponent, or *Follow* into $\mathcal T(x)$. The opponent's "range" at $x$ is the chance/card-removal prior times their Follow probability, chosen by regret matching like any other decision.
3. Run CFR⁺/DCFR for $T$ iterations; compute the best-response gap (one pass) and the maximum over hands of (opponent's realized value − ceiling), the gadget audit.
4. Act from the average strategy at your hand. Carry forward: $r_1 \leftarrow r_1 \odot \sigma_1(\cdot, a)$ after your action (exact); $\bar v_2 \leftarrow$ the opponent's values at the reached child after theirs.
5. At a street boundary, do not translate averaged values into per-card ceilings. Root the next re-solve at the pre-chance node with the averaged ceilings and all next cards in one forest.

Two-player zero-sum theorem (Burch et al. 2014; Moravčík et al. 2017): exploitability of the combined strategy ≤ exploitability of the previous strategy + solver residual + ceiling error, summed over re-solves. For 3+ players the construction can be run with modelled ranges but has no theorem.

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
| Chance-rooted gadget at street starts; start-of-street solve fails | New measurement |
| Continual re-solving with ceilings as the formal "weather"; carrying the opponent's range is unsafe and, without a blueprint, self-referential | Known machinery (DeepStack, Burch), reframing is the point; toy 23.7 vs 520 |
| $V$ concave–convex, 1-homogeneous, CFVs as supergradients, clairvoyance kink | Known in substance (ReBeL); kink consequences for learning are the point |
| Regret bound as the weather theorem; dimension separation (4 × 10⁹ hands); exploitation needs a model | Theorem known; the quantitative argument and toy M3 are the contribution |
| Huber contamination ≡ RNR; Bayes gives no caution; declared risk parameter | Elementary identity; used to split estimation from misspecification |
| Free exploitation on the equilibrium face | New measurement (toy T5) |
| Tilt of the solver's own strategy as the opponent family; HUD counts as sufficient statistics | Max-entropy modelling known; anchoring at the per-hand reference is the contribution |
| Live-count indexing; 3-way menu as the 9-max budget dial; coalition as utility matrix plus fiber merge | Structural cost framing; corrected by the folded-player measure |
| Cost law (product over spanned streets vs sum); depth-tapered menu law; caller-subset law; 0.9 PB | Computed here |
| Menu-loss table; DCFR convergence table | Measured here |

## Appendix D. Glossary

- **Public tree**: betting sequences (exact amounts) and board cards; nothing private.
- **Fiber**: the set of private hands a player could hold at a public node; $F$ its size.
- **Range / reach**: the vector over a player's fiber of the product of their action probabilities along the line.
- **Counterfactual value (CFV)**: a hand's expected payoff weighted by everyone else's reach, own reach removed.
- **Exploitability / NashConv**: for two players, half the sum of both players' best-response gains; for $N$, the sum of per-player best-response gaps. mA/h: milli-antes per hand (toys); mbb/hand: milli-big-blinds per hand (hold'em).
- **Gadget / ceilings**: the opponent's per-hand CFVs from the previous solve, offered as a Terminate option at the root of a re-solve.
- **RNR**: restricted Nash response; the opponent plays a model with probability $p$ and adversarially otherwise.
- **Pre-chance cut**: street boundary placed before the next card is dealt; one $V$ call per boundary leaf.
- **SPR**: stack-to-pot ratio; the only geometric input $V$ needs postflop under a pot-fraction menu.
- **Live count $L$**: players still holding cards; the quantity postflop cost depends on.
