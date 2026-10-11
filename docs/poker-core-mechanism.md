> **DRAFT IN PROGRESS** — sections marked [[...]] are being filled from the verification workflow (design panel, refuters, toy-game measurements). Numbers in those sections are not final.

# Poker at Its Roots: Ranges Are Carried, Not Inferred

*A foundation document for building a poker agent from zero, on a 2015 PC and $10–15 of rented compute, with no blueprint, no card buckets, no snapping of opponent bet sizes, and one mechanism for both robustness and exploitation.*

---

## 0. The answer on one page

**The missing principle.** Your three layers (what is hidden, how others react, what happens next) look like three coupled inference problems. In the right coordinates none of them is an inference problem:

1. **Hidden cards are not inferred; they are carried.** For every opponent, keep a vector over their possible private hands, the *range*. Along any betting line that vector is just a running product of the probabilities their strategy assigns to the actions they actually took. Bayes' rule in poker is a componentwise multiply. There is no "hand predictor" to train, and if you trained one it would be ill-defined, because beliefs are only meaningful relative to a strategy.

2. **Reactions are not predicted; they are a fixed-point condition.** The opponent's strategy is either (a) what a regret-minimizing learner would do in their seat (the robust assumption), or (b) a model fitted to what they have actually done (exploitation). In both cases the solver *carries* that strategy as input to the ranges, so layer 2 is the same object as layer 1.

3. **The future is not a tree; it is a boundary condition.** Once you carry ranges, the pair (public history, all ranges) is a sufficient statistic for the rest of the hand. So the hand can be cut at the next street boundary and the remainder replaced by a value map $V(\text{public state}, \text{ranges})$. $V$ is learned from the bottom of the game up (river first, where it is exact), which is why no blueprint is needed.

The three layers collapse to a single small computation per decision: **regret minimization on the current street's public betting tree, with the private information as a vector attached to every node, the past summarized by the ranges, and the future summarized by $V$.** The tree you solve has hundreds to a few thousand nodes. The 1326-dimensional hidden state is never branched on; it is a vector you multiply.

**Why "opponents are weather" is a theorem, not a hope.** Regret minimization guarantees that your average payoff is within $o(1)$ of the best *fixed* strategy's payoff against the *same realized* opponent behavior, whatever that behavior is (adversarial, random, drifting, colluding). In self-play this yields equilibrium (2 players) or a per-player no-regret guarantee (3+). Against a modeled opponent the same update converges to a best response. Robustness and exploitation are one algorithm with one input swapped: the opponent's carried strategy. The safety dial between them is a single scalar $p$ (restricted Nash response).

**What is exact.** Hand representation (all 1326 combos, no buckets). Card removal (handled in the terminal sums, in $O(F \log F)$ per terminal for two players). Opponent bet sizes (an off-menu size is added to the tree and re-solved, never snapped). The regret dynamics (no learning-rate, no critic, nothing to tune except the iteration budget).

**What is approximated, and the only thing.** The value map $V$ at street boundaries. Its error is the single quantity to measure; exploitability is bounded by it plus the solver's residual regret. Everything else, including your own bet-size menu, is a design *choice* (a restriction of your own actions), not an approximation of the opponent.

**What it costs.** [[F: per-decision re-solve seconds; ladder data-generation core-hours; $ for HU; $ for 9-max; from Lens F and toys]]

**Where it genuinely breaks.** Multiway showdowns with 4+ live players (exact card-removal sums grow as $F^{k-1}$; sample there). Three-plus players have no equilibrium guarantee (the per-player regret guarantee survives). Exploitation drives ranges off the distribution $V$ was trained on. HUD identifies only a few parameters per opponent per hundred hands. PLO's 270 725-combo fiber makes $V$'s input too wide for a cheap network. Each is quantified in §7.

---

## 1. Why the three layers are one object

### 1.1 Coordinates: a public tree with a private fiber

Standard game theory draws poker as a tree over *information sets*. That representation is why the problem looks like nested enumeration: every one of your information sets contains one node per opponent hand, so values appear to require a sum over the opponent's hands at every node, and the opponent's node-level reasoning seems to require a sum over yours.

Change coordinates. Let $T$ be the **public tree**: its nodes $x$ are public histories (the betting sequence so far plus the board). Nothing private is in $T$. Attach to every node a **fiber**: the set $H_i$ of private hands player $i$ could hold (for hold'em, the $F = \binom{52}{2} = 1326$ two-card combos; the board rules some out). A strategy for player $i$ is a function

$$\sigma_i : \{x : P(x) = i\} \times H_i \to \Delta(A(x)),$$

one probability vector over the legal actions $A(x)$ per public node per private hand. That is the whole strategy. It is an array indexed by `[public node, hand, action]`. Everything below is array arithmetic on it.

### 1.2 Layer 1 is a product, not a posterior

Define player $i$'s **reach** at public node $x$ with hand $h$:

$$\pi_i(x, h) = \prod_{(x', a) \text{ on the path to } x,\ P(x') = i} \sigma_i(x', h, a).$$

This is a product of numbers you already have. The **range** $r_i(x) \in \mathbb{R}^{F}$ is $\pi_i(x,\cdot)$ with hands conflicting with the board zeroed and, if you like, normalized. The joint belief over everyone's hands at $x$ is

$$P(h_1,\dots,h_n \mid x) \;\propto\; \prod_i \pi_i(x, h_i)\cdot \mathbb{1}[h_1,\dots,h_n,\text{board pairwise disjoint}].$$

That *is* Bayes' rule, exactly. Two facts follow that change the computational structure:

- Beliefs are **not** an inference problem. They are a by-product of whatever strategy you attribute to the opponent. The question "what are they holding?" is meaningless without "what strategy are they playing?", and once you fix that strategy the answer is a multiply.
- Beliefs are a **vector**, not a branch. You carry one $F$-vector per player down the public tree. You never create $F$ copies of the tree.

Note the joint belief is a product measure *restricted to disjoint tuples*. For two players the restriction costs nothing (§2.2). For three or more it is the one place where multiway poker is genuinely harder (§5).

### 1.3 Layer 2 is a fixed-point condition, not a prediction

"How will they react" is $\sigma_{-i}$, the opponents' strategies. There are only two coherent sources for it:

- **Robust:** $\sigma_j$ is whatever a regret-minimizing learner in seat $j$ would play. You do not predict it; you compute it, in the same loop that computes your own strategy, as a joint fixed point. This is the "weather" assumption in its mathematical form: you assume nothing about them except that they are not systematically throwing away money, and even that assumption is not load-bearing (§2.3).
- **Exploitative:** $\sigma_j = \hat\sigma_j$ is a model fitted to observed behavior (§2.6).

Either way $\sigma_j$ feeds §1.2 and the ranges follow. Layers 1 and 2 are the same array.

### 1.4 Layer 3 is a boundary condition, not a tree

Because the strategy is a function of (public node, own hand), and the belief at a public node is determined by the ranges, the pair

$$s = (x,\; r_1(x), \dots, r_n(x))$$

is a **sufficient statistic** for the remainder of the hand: two histories that reach the same public node with the same ranges have identical continuation games. (This is the public belief state of DeepStack and ReBeL; the fact is older than both.) It means the hand can be cut anywhere. Cut it at the next street boundary $y$ (the chance node that deals the next card). Replace everything below $y$ by a map

$$V(y, r_1(y), \dots, r_n(y)) \in \mathbb{R}^{n \times F},$$

the counterfactual value of every hand of every player if the rest of the hand is played by regret-minimizing players from $y$ with those ranges. $V$ is a well-defined function (for two players its hand-summed value is unique across equilibria; per-hand values depend on which equilibrium, and we use the one our own solver would find, which is what the training procedure below does automatically).

$V$ is the **only** thing in this document that is approximated.

### 1.5 The single coupled computation

Put §1.2–1.4 together. At your decision at public node $x$:

- the past enters only as the ranges $r_1(x),\dots,r_n(x)$ (plus one safety datum per opponent, §2.4);
- the future enters only through $V$ at the street boundaries below $x$;
- the present is the current street's public betting subtree rooted at $x$: with a human bet-size menu it has tens to a few thousand nodes.

Run regret minimization on that subtree, all players at once, each player's strategy an array `[node, hand, action]`, and read off your action for your actual hand. Nothing nested, nothing enumerated over hidden states, no full-game solution anywhere, and therefore no "delta from a full solution". The three layers were never three problems; they were three views of one array and one boundary map.

---

## 2. Minimal mathematical formulation

### 2.1 Objects

- Public tree $T$ with node types: *decision* (player $P(x)$ acts, legal actions $A(x)$ from the menu, which may shrink with depth in the street), *chance* (public card), *terminal* (fold or showdown).
- Fibers $H_i$, $|H_i| = F$. Hand evaluator $\rho(h, \text{board}) \in \mathbb{N}$ (rank). Payoff at terminal $z$ for player $i$ given the joint hand $h$: $u_i(z, h)$; at a fold terminal it does not depend on $h$; at a showdown it depends on $h$ only through the ranks and the pot.
- Strategy profile $\sigma = (\sigma_1,\dots,\sigma_n)$ as in §1.1. Reach $\pi_i$ as in §1.2; chance reach $\pi_c(x)$ for the board.

### 2.2 Counterfactual value is a linear map, and you never form the matrix

Player $i$'s counterfactual value at $x$ for hand $h$ is the expected payoff if $i$ held $h$ and played to reach $x$ (so $i$'s own reach is removed), weighted by everyone else's reach:

$$v_i^\sigma(x, h) = \sum_{h_{-i}} \Big(\prod_{j \ne i} \pi_j(x, h_j)\Big)\, \mathbb{1}[\text{disjoint}]\; \mathbb{E}^\sigma\big[u_i \mid x, h, h_{-i}\big].$$

For two players this is $v_1(x,\cdot) = U_x\, r_2(x)$ with $U_x$ an $F\times F$ matrix. The matrix is never built:

- **Fold terminal.** $u$ is a constant $c$ (who won the pot). $v_1(h) = c\cdot\big(S - s(c_1) - s(c_2) + r_2(h)\big)$ where $S = \sum_{h'} r_2(h')$ and $s(c) = \sum_{h' \ni c} r_2(h')$. Cost $O(F)$.
- **Showdown.** Sort hands by rank. Prefix sums over rank give the mass of opponent hands that lose, tie, and win; card-removal is corrected with the same per-card sums $s(c)$ restricted to each rank class. Cost $O(F \log F)$ (the sort) and $O(F)$ per terminal thereafter.
- **Decision node.** $v_i(x,\cdot) = \sum_a \sigma_{P(x)}(x,\cdot,a)\, v_i(x a,\cdot)$ if $P(x) = i$, else $v_i(x, \cdot) = \sum_a v_i(xa,\cdot)$ with the acting player's $\sigma$ folded into their reach. Cost $O(F \cdot |A|)$.
- **Chance node.** $v_i(x,\cdot) = \sum_{\text{card}} \pi_c \cdot v_i(x\,\text{card},\cdot)$ with conflicting hands zeroed.

So one pass over a public subtree with $N$ nodes costs $O(N F \log F)$ at worst and $O(NF)$ in practice. **This is the arithmetic reason the hidden state never multiplies the tree by itself.** It is $N \times F$, not $N \times F^2$ and not $F$ copies of $N$.

### 2.3 The dynamics: regret matching+, and why it is the "weather" principle

At each decision node $x$ of player $i$, for each hand $h$ and action $a$, maintain a cumulative regret $R(x, h, a)$. One iteration:

$$R(x,h,a) \leftarrow \max\Big(0,\; R(x,h,a) + v_i(xa, h) - \sum_{a'} \sigma_i(x,h,a')\, v_i(xa',h)\Big), \qquad \sigma_i(x,h,\cdot) \propto R(x,h,\cdot)^+ \;(\text{uniform if all zero}).$$

Average the strategies with linearly increasing weights. That is CFR+. It has no learning rate, no step size, no critic, no target network, no temperature. The only "parameter" is how many iterations you run, and more is monotonically better.

**Theorem (external regret; Zinkevich et al. 2007, Tammelin 2014 for the + variant).** Let player $i$ play $\sigma_i^1, \dots, \sigma_i^T$ produced by the update above, and let the opponents play *any* sequence $\sigma_{-i}^1,\dots,\sigma_{-i}^T$, possibly chosen with full knowledge of $i$'s past play. Then

$$\frac{1}{T}\sum_{t=1}^T u_i(\sigma_i^t, \sigma_{-i}^t) \;\ge\; \max_{\sigma_i^*} \frac{1}{T}\sum_{t=1}^T u_i(\sigma_i^*, \sigma_{-i}^t) \;-\; \frac{\Delta\, |\mathcal{I}_i| \sqrt{|A|}}{\sqrt{T}},$$

where $\Delta$ is the payoff range and $|\mathcal{I}_i|$ the number of (node, hand) pairs where $i$ acts.

Read it as a sentence: **against whatever the environment actually did, you did almost as well as the best single fixed strategy would have done against that same environment.** No assumption on the opponents appears in the statement. That is exactly "opponents are weather". Three corollaries, all from the one theorem:

1. **Two-player self-play.** If both players run the update, the average strategies converge to a Nash equilibrium at rate $O(1/\sqrt{T})$ in theory and about $O(1/T)$ in practice for CFR+. Exploitability (how much a perfect adversary could win per hand) goes to zero.
2. **Fixed opponent.** If the opponent plays a fixed $\hat\sigma$, the best fixed strategy in hindsight is the best response to $\hat\sigma$; the update converges to it. That is exploitation.
3. **Three or more players.** The per-player statement is unchanged. What is lost is only the *joint* conclusion: the average profile converges to a coarse correlated equilibrium, not necessarily a Nash equilibrium, and the per-player exploitability need not vanish. §5 and the toy results (§8) show what actually happens.

One update, three regimes. Everything called "GTO solver", "exploitation module", and "multiway heuristic" in practice is this one update with different inputs.

### 2.4 Cutting the hand: depth-limited solving and its safety condition

Fix the current street and your current public node $x$. Let $B(x)$ be the set of street-boundary nodes below $x$ (chance nodes dealing the next card) and $Z(x)$ the terminals below $x$ within the street. The local problem is regret minimization on the subtree rooted at $x$ where, in the value recursion of §2.2, every $y \in B(x)$ returns

$$v_i(y, \cdot) := V_i\big(y, r_1(y), \dots, r_n(y)\big)$$

instead of recursing. The ranges at $y$ are the ranges at $x$ times the strategies chosen inside the subtree, so $V$ is evaluated inside the loop on the *current iterate's* ranges. For a hold'em flop decision, $B(x)$ has one entry per (betting line that reaches the turn) $\times$ 47 turn cards; a few thousand $V$ evaluations per iteration.

**Inputs from the past.** Your own range at $x$ is your own strategy's reach, which you know exactly. Each opponent's range at $x$ is the reach of the strategy you attributed to them in the previous re-solves along this hand. **Safety.** Carrying only ranges forward is not enough in theory: the re-solved continuation might be one the opponent would have avoided from an earlier node, making the earlier node's solution inconsistent (the classic unsafe-subgame-solving problem). The fix is one extra datum per opponent hand: their counterfactual value $\bar v_j(x, h)$ at $x$ from the previous solve. Give the opponent, at the root of the re-solve, a gadget choice per hand between "enter the subgame" and "take $\bar v_j(x, h)$ instead". Then (Burch, Johanson, Bowling 2014; Moravčík et al. 2016) the exploitability of your combined strategy is at most that of the strategy you started from plus the solver residual. For two players this is a theorem. For three or more it is the same construction without the theorem; it is still the right thing to do and its effect is measurable (§8).

**Off-menu opponent bets.** If the opponent bets a size not in your menu, add that size as an action at the node where they bet, re-solve the parent node's subtree with it included, read off the opponent's range after the action from the re-solved strategy, and continue. Nothing is snapped. Your own menu limits only you.

### 2.5 The blueprint-free ladder: learning $V$ from the bottom up

$V$ is defined by a recursion that terminates at the river, where there is no boundary: a river subtree's leaves are all terminals. So:

- **River.** Exact. No $V$ needed. A river re-solve is the full local problem of §2.4 with $B(x) = \emptyset$.
- **Turn.** Needs $V^{(4)}(y, r_1, r_2)$ at turn→river boundaries. Generate training data by sampling boundary situations (board of 5 cards, pot and stacks, two ranges) and solving each river subgame exactly with CFR+; the solver's counterfactual values are the labels. Fit $V^{(4)}$.
- **Flop.** Needs $V^{(3)}$ at flop→turn boundaries. Sample turn situations, solve each turn subgame with $V^{(4)}$ at its leaves, record labels, fit $V^{(3)}$.
- **Preflop.** Same one level up, then solve preflop once offline (169 strategically distinct starting hands, suit-canonical flops).

Each level's data is produced by the level below. **No blueprint, no previously solved game, no human strategy anywhere.** The accuracy of each level is measurable at the next: label a held-out set of situations with the exact lower-level solve and compare.

Three structural facts make $V$ cheap to learn:

1. **Symmetry.** Suit isomorphism reduces 22 100 flops to 1 755 classes, turns and rivers proportionally; permuting the two players is another symmetry; permuting hands consistently in input and output is a third (the map is equivariant). Canonicalize before the network.
2. **An exact baseline absorbs most of the signal.** The *check-down value* (each hand's equity against the opponent's range times the pot, with card removal, computed in $O(F\log F)$ by the §2.2 trick) is exact, bilinear in the ranges, and already the right order of magnitude. Learn the residual $V - V_{\text{checkdown}}$, which is the value of being allowed to bet, a much smoother function.
3. **The output is a hand-indexed vector, so each solved situation is $2F$ labels, not one.**

[[C/F/TOY: data requirement estimates; the toy's learned-leaf exploitability vs N; comparison to DeepStack's 10M flop / ReBeL]]

### 2.6 Exploitation is the same loop with one input swapped

Let $\hat\sigma_j$ be a model of opponent $j$. Inside the §2.3 loop, replace $j$'s regret-driven iterate by

$$\sigma_j^t \leftarrow p\,\hat\sigma_j + (1-p)\,\sigma_j^{t,\text{regret}}, \qquad p \in [0, 1].$$

- $p = 0$: equilibrium (robust).
- $p = 1$: full best response to the model (maximal exploitation, maximal exposure to model error).
- $0 < p < 1$: the *restricted Nash response* (Johanson, Zinkevich, Bowling 2008): the solution is optimal against an opponent who plays the model with probability $p$ and plays adversarially otherwise. As $p$ sweeps $[0,1]$ you trace the Pareto frontier between EV-against-model and your own exploitability. This is the entire safety-exploitation dial: one scalar, one line of code (§8 shows the diff on the toy is a handful of lines).

**Where $\hat\sigma_j$ comes from (the HUD, done honestly).** You never observe opponents' hands except at showdown, so you cannot estimate a free $F$-dimensional strategy per node. What you observe is their action frequencies per public situation. The principled model is therefore a **low-dimensional tilt of the climatology**: take the equilibrium strategy $\sigma_j^{\text{eq}}(x, h, \cdot)$ (what the robust solve says they should do, known for every hand) and tilt it by a few parameters $\theta$,

$$\hat\sigma_j(x, h, a;\theta) \propto \sigma_j^{\text{eq}}(x, h, a)\, \exp\!\big(\theta^\top \phi(x, a)\big),$$

with features $\phi$ such as "this action is a call", "this action is a raise", "street", "facing a bet". Fit $\theta$ by maximizing the likelihood of the opponent's *observed actions* under the ranges the model itself induces (an EM-style update because hands are latent; the expected-count step is a range computation you already have). With a handful of parameters this is identified from a few hundred hands, which is precisely the information content of a HUD (VPIP, PFR, 3-bet %, c-bet %, fold-to-c-bet). Set a prior at $\theta = 0$ (= equilibrium): with no data you play robustly; with data you tilt. Non-stationarity: exponentially discount the sufficient statistics. "Climatology" is $\sigma^{\text{eq}}$; "nowcasting" is $\theta$ on a sliding window. Both are the same solver.

**The cost of exploitation that nobody can remove.** $V$ was trained on situations drawn from some distribution of ranges. Exploitative play pushes ranges away from equilibrium shapes, into regions where $V$ has seen less data. This is a real and fundamental trade-off, not an implementation detail: it is why the training set for $V$ must be drawn from *random* ranges (as DeepStack did) rather than self-play ranges (as ReBeL did), and it is why $p$ should be increased slowly. §8 measures this on the toy.

---

## 3. The honest ledger: exact vs approximate vs chosen

| Component | Status | Notes |
|---|---|---|
| Hand representation | **Exact** | All $F$ combos, no buckets, ever. |
| Beliefs (ranges) | **Exact given the attributed strategy** | A product of known numbers. |
| Card removal, 2 players | **Exact** | §2.2, $O(F\log F)$ per terminal. |
| Card removal, $k$ live players at showdown | Exact for $k \le 3$ at acceptable cost; sampled or pairwise-corrected beyond | §5.2; the error of pairwise correction is quantified in the toy (§8). |
| Opponent bet sizes | **Exact** | Observed size added to the tree; nothing snapped. |
| Your own bet sizes | **Chosen** | A restriction of your action set, not an approximation of the game. Its cost is measurable: solve with a richer menu offline and compare. |
| Regret dynamics | **Exact, parameter-free** | Iteration count is a budget, not a tuning knob. |
| Past (earlier streets) | **Exact summary** | Ranges plus opponent root CFVs (gadget). |
| Future (later streets) | **Approximated: $V$** | The one approximation. Error measurable on held-out solved situations; exploitability $\lesssim$ solver residual $+ O(\varepsilon_V)$. |
| Opponent model $\hat\sigma$ | **Model** (exploitation mode only) | Low-dimensional, likelihood-fit, prior at equilibrium, discounted. Misspecification cost is bounded by the choice of $p$. |
| 3+ players: equilibrium | **No guarantee** | Per-player no-regret guarantee holds; exploitability does not provably vanish. |

---

## 4. A real decision, step by step

**Setting.** Heads-up no-limit hold'em, 100 bb effective, blinds 0.5/1. Your bet-size menu (chosen by you, fixed per street, shrinking with depth):

| Street | First bet / open | Raise (depth 1) | Re-raise (depth 2) | Cap |
|---|---|---|---|---|
| Preflop | open 2.5 bb; 3-bet 3× the open | 4-bet 2.5× | all-in | 3 raises |
| Flop / turn / river | 33 %, 75 %, 150 % of pot, all-in | 50 % of pot-after-call, all-in | all-in | 3 raises |

**Hand so far.** Button opens 2.5 bb, you call in the big blind. Pot 5 bb, stacks 97.5 bb. Flop K♠ 7♦ 2♣. You check, button bets 75 % (3.75 bb). You hold 8♦ 7♣. Decide.

**Step 1: the public node.** $x$ = [BTN 2.5, BB call | K♠7♦2♣ | check, bet 3.75]. Pot 8.75, you face 3.75, button has 93.75 behind.

**Step 2: the ranges (layer 1, a multiply).**
- Your range $r_{BB}(x)$: for each of the 1 081 combos not blocked by the board, your preflop probability of calling 2.5 bb with it (from the offline preflop solve, a lookup) times your probability of checking this flop with it (from the re-solve you ran one node ago, where you decided to check). Two numbers multiplied, per hand.
- Button's range $r_{BTN}(x)$: their open probability per hand (preflop solve) times their probability of betting 75 % with it at the check node, read off the *same* re-solve you ran one node ago (that solve computed both players' strategies on the whole flop subtree, so the opponent's reaction to your check is already in hand). No inference happened. If the button were a modeled opponent, you would use $\hat\sigma_{BTN}$ instead of the solved strategy; same multiply.

**Step 3: the local tree (layer 3, a boundary).** From $x$ you can fold, call, raise to 50 % of the pot after a call (≈ 6.25 bb on top), or jam. After a raise the button can fold, call, or jam; after a jam you can fold or call. Enumerating with the menu above gives **5 decision nodes, 5 fold terminals and 5 street-boundary nodes** (the lines that reach the turn: call; raise–call; raise–jam–call; jam–call; and the all-in call lines, which run out the board with no further betting). That is the whole tree you solve. Each boundary node $y$ returns $V(y, r_{BB}(y), r_{BTN}(y))$, a vector of 1 081 counterfactual values per player, from the turn-level network trained in §2.5. Cut at the chance node (before the turn card is dealt) so each boundary line is one network call per iteration; cutting after the card would cost 47× more for no gain in principle. The fold terminals are exact in $O(F)$ (§2.2). The all-in lines are exact in $O(F \log F)$ with the run-out averaged over the 990 turn-river pairs, or served by $V$ at the all-in boundary, your choice.

**Step 4: the safety datum (layer 2, a constraint).** From the previous re-solve, take the button's counterfactual value per hand at $x$, $\bar v_{BTN}(x, h)$. At the root of the new solve, the button chooses per hand between entering this subtree and taking $\bar v_{BTN}(x,h)$. This is what makes "solve only the current street" sound rather than merely convenient.

**Step 5: the fixed point.** Run CFR+ (§2.3) on the 5-node tree with 1 081-dimensional fibers for, say, 1 000 iterations. Per iteration: 5 nodes × 1 081 hands × 3 actions of regret arithmetic (trivial), 5 fold terminals in $O(F)$ (trivial), and 5 network evaluations. With a network of about 2 700 inputs → 512 → 512 → 2 162 outputs, that is about 5 MFLOP per call, 25 MFLOP per iteration, 25 GFLOP per decision: **about one second on a 2015 quad-core.** River decisions have no network and take well under a second. Preflop decisions are a lookup in the offline solve.

**Step 6: the action.** Read $\sigma_{BB}(x, 8♦7♣, \cdot)$ from the averaged strategy. It might say fold 0.00, call 0.72, raise 0.28, jam 0.00. Sample from it. You have acted on your middle pair without ever asking "what does he have?": the answer was the vector $r_{BTN}(x)$, and it was used, exactly, in every value computation.

**The off-menu case.** Suppose the button had bet 63 % instead of 75 %. Nothing is snapped. Add a 63 % branch at the button's bet node, re-solve the tree rooted at your check decision with that branch included (a handful more nodes), read off the button's range after the 63 % bet from the re-solved strategy, and proceed to Step 1 with that range. The cost of handling an arbitrary size is one extra re-solve of a tree with a few more nodes. (The toy measures what snapping would have cost instead, §8.)

**The exploitation case.** Suppose your HUD says this button folds to check-raises 70 % of the time, versus roughly 45 % for the robust strategy. Fit the tilt parameter $\theta_{\text{fold-vs-raise}}$ (§2.6) from their observed actions, set $p = 0.5$, and re-run Step 5 with the button's iterate replaced by $0.5\,\hat\sigma_{BTN} + 0.5\,\sigma_{BTN}^{\text{regret}}$. Your raise frequency with 8♦7♣ rises; the amount it rises is bounded by what a half-adversarial button could punish. Same tree, same fibers, same network, same loop; one input swapped.

**The multiway case.** Three players to the flop: three ranges, boundary values $V$ with three range inputs and three output vectors, fold terminals computed by the inclusion-exclusion sums of §5.2, and the same loop. The tree is larger (more betting orders) and the equilibrium guarantee is gone, but the per-player no-regret property, the carried ranges and the boundary cut are unchanged.

---

## 5. Three to nine players and other variants

[[TODO: from Lens D + toy multi: what survives; product measure with disjointness; fold terminal normalizer via inclusion–exclusion; showdown k-way; MCCFR crossover; public-tree growth; V input n×F; coalitions as a joint fiber for the colluding pair (F² = 1.76M, still a vector); variant plug-in points: deal structure (fiber), evaluator (rank), betting grammar (public tree). PLO fiber]]

---

## 6. Cost and feasibility

[[TODO: Lens F table; toy timings; $10–15 HU plan; $65–100 9-max verdict]]

---

## 7. Failure modes and where the requirements contradict each other

[[TODO: V off-distribution; multiway no-Nash; multiway showdown k≥4; HUD sample complexity; PLO fiber; own menu; non-stationarity bias/variance; the contradiction list]]

---

## 8. Evidence from the toy games

[[TODO: toy/hu and toy/multi results]]

---

## 9. The sprint ladder

Every sprint below produces one artifact and one number. The number is always an exploitability or a held-out error; if it is not measured the sprint is not done. Nothing in a later sprint changes the algorithm of an earlier one; later sprints add a fiber (more players, more cards), a boundary map (one more street), or an input (an opponent model).

| Sprint | Artifact | Acceptance number | Compute |
|---|---|---|---|
| **0. Rapid** (days) | The toy in `toy/`: public tree × fiber arrays, CFR+, exact exploitability, street-boundary re-solving with oracle / check-down / learned leaves, off-menu re-solve, RNR dial, 3-player card-removal sums. | Full-solve exploitability < 1 % of the ante; re-solved agent with oracle leaves within solver residual of the full solve; the diff between "Nash" and "exploit" is a handful of lines. | Laptop. |
| **1. MVP** (1–2 weeks) | HU push/fold with the real 169-class fiber and the real evaluator; then a HU **river** solver on 1 326 combos with your menu. | Push/fold tables match published Nash tables to ±1 %; river exploitability < 0.25 % of pot in 1 000 iterations; seconds per river solve logged. | Laptop. |
| **2. Base** (2–4 weeks) | $V^{(4)}$ (turn→river boundary map) from random river situations; HU **turn** re-solver. | Held-out $V^{(4)}$ error in % of pot; turn re-solve exploitability vs an exact turn+river solve on a reduced menu (brute force is still feasible there). | [[F: river labels: situations × seconds → core-hours → $]] |
| **3. Full HU** (4–8 weeks) | $V^{(3)}$, $V^{(2)}$, preflop solve; the agent plays whole hands. | Held-out errors per level; match result vs Slumbot's public API over ≥ 50k hands with confidence intervals (first target: not losing; second: positive). | [[F: turn and flop labels → $; total vs the $10–15 budget]] |
| **4. Tuning** (ongoing) | Opponent tilt model fit from logs; RNR dial; discounting. | On a pool of scripted exploitable bots: EV gain over the robust strategy at each $p$, and the measured exploitability at each $p$ (the frontier, as in §8 E4). | Laptop. |
| **5. 6–9 max** (2–4 months) | Multiway fibers, inclusion-exclusion fold sums, $k \le 3$ exact showdowns, sampled beyond, multiway $V$ with seat-equivariant inputs. | Per-player regret against self-play; per-player best-response gain; match results vs baseline bots. | [[D/F: $ estimate vs the $65–100 budget]] |
| **6. Variants** | New fiber + evaluator + betting grammar; same loop. | Same numbers as sprints 1–3 for the new game. | Depends on fiber size; PLO is the hard one (§7). |

The order is forced by the ladder in §2.5: each level's training data comes from the solved level below it. Nothing else is forced. If sprint 3's budget runs out, you have a sound turn-and-river agent and a preflop/flop strategy with a larger $V$ error, and you know that error. There is no point at which the plan says "and now train a sufficiently strong model."
