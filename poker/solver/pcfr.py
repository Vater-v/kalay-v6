"""The teacher: vector PCFR+ on public subgames with exact terminal values. One solver for every
street and supported table size; no switching by result.

Everything below is written for ONE spot and batched with jax.vmap. A spot is a board
(rank tables), ranges r[n, 1326], and the replayed tree data (legal, contributions, pot).
Values are in units of the pot at the start of the street.

One iteration: reach forward; exact terminal values for every hand of every player; values
backward; instantaneous counterfactual regrets of the acting poker.player. PCFR+ (Farina, Kroer,
Sandholm 2021) plays regret matching on (clipped cumulative regret + last regret) and averages
the policy with quadratic weights. The average policy is the solved strategy and
the stopping metric: a spot stops once the exact exploitability of the average (best response,
pot units) is <= e_target, checked every check_every iterations, or at the cap. Regret matching
is invariant to the payoff scale, so deep spots (SPR 50, all-in = 50 pots) need no smaller step.

Teacher memory per spot (float32): regret, average and last regret, each [internal nodes, 1326,
actions], plus reach and values [nodes, players, 1326] inside an iteration. HU river, grid 7,
cap 2 + all-in: 84 internal nodes -> 3 * 84 * 1326 * 7 * 4 B = 9.4 MB; 100 spots per batch ~1 GB.

PCFR+ is the update rule for the teacher and live subgame solves.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.solver import checkdown as CD
from poker.solver import critics as CR
from poker.solver import showdown as sd
from poker.solver.cards import HANDS
from poker.solver.cards import NUM_HANDS as H
from poker.solver.tree import Topology


@dataclass(frozen=True)
class SolverConfig:
    cap: int = 3000
    terminal_batch: int = 16         # lax.map batch over terminals (memory bound for n=3)
    runout_chunk: int = 16           # 3-player runouts before the river: terminals per vectorised call
    e_target: float = 0.0            # a spot stops once its E (pot units) <= e_target; 0 = off
    check_every: int = 50            # iterations between exploitability checks


class Static:
    """Topology as index arrays (host numpy, static under jit).

    Boundaries (topo.terminals) split by kind: folds and river showdowns (exact, grouped by live
    pattern), all-in runouts before the river (exact two-player equity, `allins`), and street
    ends before the river (`leaves`, the next street's critic). Critic slots follow the
    postflop action order: seat 1 first, the button (seat 0) last.
    """

    def __init__(self, topo: Topology):
        if topo.n > 3:
            raise NotImplementedError(
                f"the solver values terminals exactly for 2 and 3 players (_two, _three, mass_three: one or two "
                f"opponents); a {topo.n}-player tree would be valued with two of its {topo.n - 1} opponents "
                f"(task #35, edge_cases B7; brief M2 item 3)")
        self.topo = topo
        self.n, self.V = topo.n, topo.size
        self.street = getattr(topo, "street", 3)
        self.internal = topo.internal
        self.terminals = topo.terminals
        self.to_internal = np.full(self.V, -1, np.int32)
        self.to_internal[self.internal] = np.arange(len(self.internal))
        self.to_terminal = np.full(self.V, -1, np.int32)
        self.to_terminal[self.terminals] = np.arange(len(self.terminals))
        self.levels = [lvl for lvl in topo.levels]
        self.actor = topo.actor
        leaf = topo.leaf if getattr(topo, "leaf", None) is not None else np.zeros(self.V, bool)
        groups, allins = {}, []
        for t in self.terminals:
            if leaf[t]:
                continue
            live = tuple(bool(x) for x in topo.live[t])
            if sum(live) >= 2 and self.street < 3:
                allins.append(t)
                continue
            groups.setdefault(live, []).append(t)
        self.term_groups = {k: np.asarray(v, np.int32) for k, v in groups.items()}
        self.allins = np.asarray(allins, np.int32)
        self.leaves = np.flatnonzero(leaf).astype(np.int32)
        if len(self.allins) and (self.n > 3 or (self.n == 3 and self.street not in (1, 2))):
            raise NotImplementedError("runouts before the river: two players, or three on the turn (exact) and the "
                                      "flop (sampled pairs) - brief M2, item 3")
        allin_groups = {}
        for t in self.allins:
            allin_groups.setdefault(tuple(bool(x) for x in topo.live[t]), []).append(t)
        self.allin_groups = {k: np.asarray(v, np.int32) for k, v in allin_groups.items()}
        self.slot = (np.arange(self.n) - 1) % self.n                      # seat -> critic slot
        by_actor = {}
        for v in self.internal:
            by_actor.setdefault(int(topo.actor[v]), []).append(v)
        self.internal_by_actor = {k: np.asarray(v, np.int32) for k, v in by_actor.items()}


def policy(z, legal):
    """Masked softmax over actions: z [Vi, H, A], legal [Vi, A]."""
    mask = legal[:, None, :]
    shifted = jnp.where(mask, z, -jnp.inf)
    top = jnp.max(shifted, axis=-1, keepdims=True)
    shifted = shifted - jnp.where(jnp.isfinite(top), top, 0.0)
    e = jnp.where(mask, jnp.exp(shifted), 0.0)
    # Dead nodes (no legal action on this spot) get an all-zero row instead of 0/0.
    return e / jnp.maximum(jnp.sum(e, axis=-1, keepdims=True), 1e-30)


def forward(pi, ranges, st: Static):
    """reach [V, n, H]: initial range times own action probabilities along the path."""
    reach = jnp.zeros((st.V, st.n, H), ranges.dtype).at[0].set(ranges)
    players = np.arange(st.n)
    for level in st.levels[1:]:
        parents = st.topo.parent[level]
        labels = st.topo.label[level]
        acting = st.actor[parents][:, None] == players[None, :]                       # [L, n]
        factor = pi[st.to_internal[parents], :, labels]                                # [L, H]
        reach = reach.at[level].set(reach[parents] * jnp.where(acting[:, :, None], factor[:, None, :], 1.0))
    return reach


def _two(reach_t, contrib, pot, tables, live):
    """Folds need only the compatible mass (any street); showdowns need the river's tables."""
    both_live = all(live)
    # A full shove may exceed the other seat's cover. The engine returns that
    # uncalled excess regardless of the showdown result; only matched chips compete.
    matched = jnp.minimum(contrib[0], contrib[1]) if both_live else None
    contested = pot - jnp.abs(contrib[0] - contrib[1]) if both_live else pot
    out = []
    for i in range(2):
        if both_live:
            beats, ties, comp = sd.apply_all(reach_t[1 - i], tables)
            share = beats + 0.5 * ties
        else:
            # A fold needs the compatible mass only: the board-free j_mass (one 52 x 1326 matmul and gathers) equals
            # apply_all's J to 1e-14 on board-restricted reach (solver audit 2026-10-12, terminals.py B) and skips the
            # prefix-sum showdown kernel that 38 of the 77 river boundaries were running for nothing.
            comp = j_mass(reach_t[1 - i])
            share = comp if live[i] else jnp.zeros_like(comp)
        out.append(contested * share - (matched if both_live else contrib[i]) * comp)
    return jnp.stack(out)


def _three(reach_t, contrib, pot, tables, live):
    """Exact three-player terminal values. Per-card [52, 1326] tables are built only when both
    opponents of a live hero are live (three-way showdown); every term with a folded opponent
    uses the O(52^3) forms in showdown.py."""
    out = []
    for i in range(3):
        b, c = [j for j in range(3) if j != i]
        wb, wc = reach_t[b], reach_t[c]
        mass = sd.mass_three(wb, wc)
        if not live[i]:
            share = jnp.zeros_like(mass)
        elif live[b] and live[c]:
            ab, eb, _ = sd.apply_all(wb, tables)
            ac, ec, _ = sd.apply_all(wc, tables)
            sab, seb, _ = sd.per_card(wb, tables)
            sac, sec, _ = sd.per_card(wc, tables)
            a_bc, e_bc, _ = sd.apply_all(wb * wc, tables)
            share = (sd.couple(ab, sab, ac, sac, a_bc)
                     + 0.5 * (sd.couple(eb, seb, ac, sac, 0.0) + sd.couple(ab, sab, ec, sec, 0.0))
                     + sd.couple(eb, seb, ec, sec, e_bc) / 3.0)
        elif live[b] or live[c]:
            f_a, f_e = sd.couple_live_folded(wb, wc, tables) if live[b] else sd.couple_live_folded(wc, wb, tables)
            share = f_a + 0.5 * f_e
        else:
            share = mass
        out.append(pot * share - contrib[i] * mass)
    return jnp.stack(out)


def payout_pot(spot):
    """[V] what the pot pays out at every node (pot units): the pot minus the engine's rake at
    terminals. Values stay per player: with rake HU is not zero-sum, sum over players = -rake."""
    return spot["pot"] - spot["rake"] if "rake" in spot else spot["pot"]


def j_mass(w, tables=None):
    """Compatible mass of one opponent range for every hero hand (board-free when no tables:
    blocked hands carry zero weight, so only hand-hand card removal remains)."""
    if tables is not None:
        return sd.apply_all(w, tables)[2]
    c = sd.card_sums(w)
    return jnp.sum(w) - c[HANDS[:, 0]] - c[HANDS[:, 1]] + w


def mass_of(reach_t, i, n, tables=None):
    """[..., H] opponents' compatible mass for player i at boundary/internal reach [..., n, H]."""
    if n > 3:
        raise NotImplementedError(f"compatible mass of {n - 1} opponents: exact for one or two (task #35, edge_cases B7)")
    opp = [j for j in range(n) if j != i]
    if n == 2:
        return jax.vmap(lambda w: j_mass(w, tables))(reach_t[:, opp[0]])
    return jax.vmap(sd.mass_three)(reach_t[:, opp[0]], reach_t[:, opp[1]])


def allin_values(reach_t, contrib_t, pot_t, equity, st: Static):
    """Two players all-in before the river: exact runout values [L, 2, H] (cfv)."""
    matched = jnp.minimum(contrib_t[:, 0], contrib_t[:, 1])
    contested = pot_t - jnp.abs(contrib_t[:, 0] - contrib_t[:, 1])
    out = []
    for i in range(2):
        w = reach_t[:, 1 - i]
        win = w @ equity.T                                                          # (W @ w)(h)
        mass = jax.vmap(j_mass)(w)
        out.append(contested[:, None] * win - matched[:, None] * mass)
    return jnp.stack(out, axis=1)


RIVERS_PER_TRIPLE = 42.0             # 52 - 4 turn board cards - 6 hole cards: rivers of a compatible triple


def turn_runout_three(reach_t, contrib_t, pot_t, river_cards, river_tables, live, chunk=16):
    """Three players with nothing left to decide on the turn (an all-in runout; `live` static, a
    folded player's range still blocks cards): exact runout values [L, 3, H] (cfv, pot units). For
    every river card c (river_cards [48], the cards not on the turn board) the three-player showdown
    of board + c (river_tables: rank tables of the 48 river boards, leading axis) with every hand
    holding c removed from every range; summed over c and divided by 42, the rivers that every
    compatible triple of hands sees. A hero hand holding c takes no value from c.
    Vectorised over the 48 rivers and over `chunk` terminals at a time (memory ~ chunk x 48 x the
    per-card tables of _three); the chunks run one after another."""
    keep = ~jnp.asarray(sd.MEMBER)[river_cards]                                       # [48, H]

    def river(r, c, p, k, tables):
        return jnp.where(k[None, :], _three(r * k[None, :], c, p, tables, live), 0.0)

    def leaf(args):
        r, c, p = args
        per = jax.vmap(river, in_axes=(None, None, None, 0, 0))(r, c, p, keep, river_tables)   # [48, 3, H]
        return jnp.sum(per, axis=0) / RIVERS_PER_TRIPLE

    return jax.lax.map(leaf, (reach_t, contrib_t, pot_t), batch_size=chunk)


FLOP_COMPLETIONS = 1176              # C(49, 2) turn-river pairs of a flop; a compatible triple sees C(43, 2) = 903


def _completion_terms(r, keep, tables, live):
    """Per board completion (rows of keep [k, H] / tables): (S, M) [k, 3, H] - every hero's share sum
    sum_{hb, hc} w w share and joint compatible mass with the completion's cards removed (_three's
    value is pot * S - contribution * M; pot 1 and no contribution give S)."""
    zero = jnp.zeros(3, r.dtype)

    def one(k_mask, tab):
        rr = r * k_mask[None, :]
        s = _three(rr, zero, 1.0, tab, live)
        m = jnp.stack([sd.mass_three(rr[1], rr[2]), sd.mass_three(rr[0], rr[2]), sd.mass_three(rr[0], rr[1])])
        return jnp.where(k_mask[None, :], s, 0.0), jnp.where(k_mask[None, :], m, 0.0)

    return jax.vmap(one)(keep, tables)


def flop_runout_three(reach_t, contrib_t, pot_t, pairs, pair_tables, live, chunk=16):
    """Three players with nothing left to decide on the flop: runout values [L, 3, H] (cfv) from k
    sampled turn-river pairs (pairs [k, 2], pair_tables: rank tables of flop + pair;
    48 pairs per spot, seed from the spot): the ratio estimate of the share,
    sum_p S_p / sum_p M_p (_completion_terms), times the exact joint mass of the flop. With all 1176
    pairs it is the exact runout (every compatible triple sees 903 of them, each counted once)."""
    keep = ~(jnp.asarray(sd.MEMBER)[pairs[:, 0]] | jnp.asarray(sd.MEMBER)[pairs[:, 1]])     # [k, H]

    def leaf(args):
        r, c, p = args
        s, m = _completion_terms(r, keep, pair_tables, live)
        share = CD.ratio(jnp.sum(s, axis=0), jnp.sum(m, axis=0))
        mass = jnp.stack([sd.mass_three(r[1], r[2]), sd.mass_three(r[0], r[2]), sd.mass_three(r[0], r[1])])
        return (p * share - c[:, None]) * mass

    return jax.lax.map(leaf, (reach_t, contrib_t, pot_t), batch_size=chunk)


def flop_runout_three_se(reach_t, pairs, pair_tables, live, chunk=16):
    """Standard error [L, 3, H] of the sampled share of flop_runout_three per hero hand: the ratio
    estimator over the k pairs, sqrt((1 - k / 1176) k / (k - 1) sum_p (S_p - R M_p)^2) / sum_p M_p,
    R = sum S / sum M (the k values of the same estimate; finite population of 1176 pairs)."""
    keep = ~(jnp.asarray(sd.MEMBER)[pairs[:, 0]] | jnp.asarray(sd.MEMBER)[pairs[:, 1]])
    k = pairs.shape[0]

    def leaf(r):
        s, m = _completion_terms(r, keep, pair_tables, live)
        den = jnp.sum(m, axis=0)
        ratio = CD.ratio(jnp.sum(s, axis=0), den)
        resid = jnp.sum((s - ratio[None] * m) ** 2, axis=0)
        var = (1.0 - k / FLOP_COMPLETIONS) * k / (k - 1) * resid
        return jnp.where(den > 1e-30, jnp.sqrt(var) / jnp.where(den > 1e-30, den, 1.0), 0.0)

    return jax.lax.map(leaf, reach_t, batch_size=chunk)


def leaf_values(reach_t, nodes, spot, st: Static, aux):
    """Street ends: the next street's critic -> cfv [L, n, H]. aux is a pytree callable
    (jax.tree_util.Partial, e.g. critics.net_critic(params)) taking (board1h [L, 52],
    pub [L, PUB], ranges [L, n, H] and F [L, n, H] of the first n slots, spot) -> V [L, n, H] in
    pot units of the leaf, relative accounting. F is the exact check-down value of every hand at
    the leaf (poker/solver/checkdown.py): share (W w_opp) / J(w_opp) from the all-in matrix W the spot
    carries for its all-in nodes, times the leaf pot net of the rake. Here value = V * pot -
    contribution (pot units of this street's start), times the opponents' compatible mass."""
    if st.n != 2:
        raise NotImplementedError("street ends with 3+ players need their check-down values (brief M2, item 3)")
    post = reach_t / jnp.maximum(jnp.sum(reach_t, axis=-1, keepdims=True), 1e-30)
    L = reach_t.shape[0]
    live = jnp.asarray(st.topo.live[nodes], reach_t.dtype)                       # [L, n]
    slots = jnp.zeros((L, st.n, H), reach_t.dtype).at[:, st.slot].set(post * live[:, :, None])
    present = jnp.zeros((L, CR.P_MAX), reach_t.dtype).at[:, st.slot].set(live)
    pot = spot["pot"][nodes]
    contrib = spot["contrib"][nodes]
    behind = spot["behind"][nodes]
    pad = lambda x: jnp.zeros((L, CR.P_MAX), reach_t.dtype).at[:, st.slot].set(x)
    pub = jax.vmap(CR.public_row)(spot["pot_bb"] * pot, jnp.broadcast_to(spot["rake_level"], (L,)),
                                  jnp.broadcast_to(spot["rake_rate"], (L,)), spot["rake_cap"] / pot,
                                  present, pad(behind / pot[:, None]), pad(contrib / pot[:, None]))
    board = jnp.broadcast_to(spot["board1h"], (L, 52))
    share = jax.vmap(lambda r: CD.two(lambda w: CD.share_matrix(w, spot["equity"]), r))(slots)
    gross_chips = spot["pot_chips"][nodes]
    net = CD.net_fraction(gross_chips, spot["rake_bps"], spot["rake_cap_chips"], reach_t.dtype)
    f = share * net[:, None, None] * spot["valid"][None, None, :]
    # A canonical street-end node can replay as a real all-in. Its exact gross
    # payout includes the uncalled refund, and rake applies only to matched chips.
    runout = jnp.any((behind <= 0) & (live > 0), axis=1)                                        # [L]
    refund = contrib - jnp.min(contrib, axis=1, keepdims=True)
    returned_chips = jnp.rint(jnp.sum(refund, axis=1) * spot["pot_bb"] * config.BB).astype(jnp.int32)
    contested_chips = gross_chips - returned_chips
    runout_rake = config.rake_from_terms(spot["rake_bps"], spot["rake_cap_chips"], contested_chips, 1)
    contested_net = (contested_chips - runout_rake).astype(reach_t.dtype) / gross_chips.astype(reach_t.dtype)
    refund_slots = jnp.zeros_like(contrib).at[:, st.slot].set(refund)
    runout_f = (share * contested_net[:, None, None] + refund_slots[:, :, None] / pot[:, None, None])
    runout_f = runout_f * spot["valid"][None, None, :]
    f = jnp.where(runout[:, None, None], runout_f, f)
    ev = aux(board, pub, slots, f, spot).astype(reach_t.dtype)                                # [L, n, H]
    ev = jnp.where(runout[:, None, None], f, ev)
    out = []
    for i in range(st.n):
        value = ev[:, st.slot[i]] * pot[:, None] - contrib[:, i, None]
        out.append(value * mass_of(reach_t, i, st.n) * live[:, i, None])
    return jnp.stack(out, axis=1)


def terminal_values(reach, contrib, pot, tables, st: Static, cfg: SolverConfig, spot=None, aux=None):
    """Counterfactual values [T, n, H] at every boundary (pot units of the street start)."""
    out = jnp.zeros((len(st.terminals), st.n, H), reach.dtype)
    fn = _two if st.n == 2 else _three
    for live, nodes in st.term_groups.items():
        def one(args, live=live):
            r, c, p = args
            return fn(r, c, p, tables, live)
        args = (reach[nodes], contrib[nodes], pot[nodes])
        if st.n == 2:
            vals = jax.vmap(one)(args)
        else:
            vals = jax.lax.map(one, args, batch_size=cfg.terminal_batch)
        out = out.at[st.to_terminal[nodes]].set(vals)
    if len(st.allins) and st.n == 2:
        nodes = st.allins
        out = out.at[st.to_terminal[nodes]].set(allin_values(reach[nodes], contrib[nodes], pot[nodes],
                                                             spot["equity"], st))
    elif len(st.allins):                         # three players: exact rivers on the turn, sampled pairs on the flop
        for live, nodes in st.allin_groups.items():
            args = (reach[nodes], contrib[nodes], pot[nodes])
            if st.street == 2:
                vals = turn_runout_three(*args, spot["river_cards"], spot["river_tables"], live, cfg.runout_chunk)
            else:
                vals = flop_runout_three(*args, spot["flop_pairs"], spot["pair_tables"], live, cfg.runout_chunk)
            out = out.at[st.to_terminal[nodes]].set(vals)
    if len(st.leaves):
        if aux is None:
            raise ValueError("street ends before the river need the next street's critic (aux)")
        nodes = st.leaves
        out = out.at[st.to_terminal[nodes]].set(leaf_values(reach[nodes], nodes, spot, st, aux))
    return out


def opponent_mass(reach, tables, st: Static):
    """den [Vi, H]: compatible reach mass of the actor's opponents at every internal node."""
    den = jnp.zeros((len(st.internal), H), reach.dtype)
    for p, nodes in st.internal_by_actor.items():
        opp = [j for j in range(st.n) if j != p]
        if st.n == 2:
            vals = jax.vmap(lambda w: j_mass(w, tables))(reach[nodes, opp[0]])
        else:
            vals = jax.vmap(sd.mass_three)(reach[nodes, opp[0]], reach[nodes, opp[1]])
        den = den.at[st.to_internal[nodes]].set(vals)
    return den


def backward(term, pi, legal, st: Static, best_response=False):
    """Values [V, n, H]. best_response: each player maximises at its own nodes (others fixed).

    Dead nodes: a child through an action the engine forbids in this spot (legal False at the
    parent) and its whole subtree have zero reach of that actor; their `legal` rows are replayed
    from an engine state stepped with the forbidden action and can be empty. Both passes mask them
    out. On-policy values would need no mask while every dead value is finite (pi is 0 on the
    illegal action, a dead child adds exactly 0 to the other players by zero reach), but nothing
    forbids a non-finite dead terminal, and masked they cannot poison the sums (bitwise the same
    values on finite trees: 0 for 0). The best response must mask: the actor maximises over legal children only, the others
    sum legal children only (exact, a dead child adds 0), and an empty legal row (dead nodes only)
    is worth 0. Without the mask the -inf of a max over an empty row climbed through the sums of
    the players who do not act into alive nodes (seen with three players: a player who acts at
    neither of two consecutive levels; HU was spared by its tree, not by the code), and a max above
    dropped that branch: the best response came out too small, NashConv negative (solver audit
    2026-10-10, F1)."""
    vals = jnp.zeros((st.V + 1, st.n, H), term.dtype).at[st.terminals].set(term)
    kids_all = np.where(st.topo.children >= 0, st.topo.children, st.V)
    players = np.arange(st.n)
    for level in reversed(st.levels):
        nodes = level[~st.topo.terminal[level]]
        if len(nodes) == 0:
            continue
        idx = st.to_internal[nodes]
        child = vals[kids_all[nodes]]                                                 # [L, A, n, H]
        acting = (st.actor[nodes][:, None] == players[None, :])[:, None, :, None]     # [L, 1, n, 1]
        probs = jnp.transpose(pi[idx], (0, 2, 1))[:, :, None, :]                      # [L, A, 1, H]
        ok = legal[idx][:, :, None, None]                                             # [L, A, 1, 1] legal children
        child = jnp.where(ok, child, 0.0)                                             # a dead child adds exactly 0
        mixed = jnp.sum(child * jnp.where(acting, probs, 1.0), axis=1)                # [L, n, H]
        if best_response:
            best = jnp.max(jnp.where(ok, child, -jnp.inf), axis=1)
            best = jnp.where(jnp.any(ok, axis=1), best, 0.0)
            mixed = jnp.where(acting[:, 0], best, jnp.sum(child, axis=1))
        vals = vals.at[nodes].set(mixed)
    return vals[:st.V]


_COMPILED = {}


def cached(name, st: Static, cfg: SolverConfig, build):
    """One compiled function per (name, topology, config): no recompilation per batch."""
    key = (name, id(st), cfg)
    if key not in _COMPILED:
        _COMPILED[key] = (st, build())
    return _COMPILED[key][1]


def regret_policy(x, legal):
    """Regret matching: positive part normalised; uniform over legal actions if none."""
    mask = legal[:, None, :]
    pos = jnp.where(mask, jnp.maximum(x, 0.0), 0.0)
    total = jnp.sum(pos, axis=-1, keepdims=True)
    count = jnp.sum(mask, axis=-1, keepdims=True)
    uniform = jnp.where(mask, 1.0 / jnp.maximum(count, 1), 0.0)
    return jnp.where(total > 0, pos / jnp.maximum(total, 1e-30), uniform)


def instant_regrets(pi, spot, st: Static, cfg: SolverConfig, aux=None):
    """Counterfactual regrets [Vi, H, A] of the actor and its own reach [Vi, H] at pi, and the root's
    counterfactual values [n, H] (the live path checks their finiteness, poker/solver/subgame.py)."""
    legal = spot["legal"][st.internal]
    reach = forward(pi, spot["ranges"], st)
    term = terminal_values(reach, spot["contrib"], payout_pot(spot), spot.get("tables"), st, cfg, spot, aux)
    vals = backward(term, pi, legal, st)
    kids = np.where(st.topo.children[st.internal] >= 0, st.topo.children[st.internal], 0)
    actors = st.actor[st.internal]
    child = jnp.transpose(vals[kids, actors[:, None], :], (0, 2, 1))                  # [Vi, H, A]
    node = vals[st.internal, actors, :]                                               # [Vi, H]
    regret = jnp.where(legal[:, None, :], child - node[:, :, None], 0.0)
    return regret, reach[st.internal, actors, :], vals[0]


def average_policy(s, legal):
    """Normalised average-strategy accumulator; uniform where never reached."""
    mask = legal[:, None, :]
    total = jnp.sum(s, axis=-1, keepdims=True)
    count = jnp.sum(mask, axis=-1, keepdims=True)
    return jnp.where(total > 0, s / jnp.maximum(total, 1e-30), jnp.where(mask, 1.0 / jnp.maximum(count, 1), 0.0))


# ------------------------------------------------------------------ the re-solving gadget (CFR-D / DeepStack)
#
# A live solve of a street that is not the hand's first carries OUR range and, for every other seat, a per-hand
# CEILING: the counterfactual value that seat was given for the hand at the start of this street by the previous
# street's solve (the post-card street critic evaluated at the leaf we came through, poker/player/live.py). At the
# root the seat chooses per hand between Terminate (the ceiling, in the same CFV units as the root values) and
# Follow (playing the subgame); Follow scales his deal range. The hero's values see the scaled range, the seat's own
# counterfactual values do not depend on his own reach, so nothing else changes. The hero's strategy of the gadget
# game is safe against ANY opponent whose values at the street start are at most the ceilings (Burch, Johanson,
# Bowling 2014); carrying his Bayesian range instead was measured 22x more exploitable in toy/hu (E2, T1).
# Spot keys: gadget_ev [n, H] whole-hand EV per hand in street-start pot units (0 on the hero's row), gadget_mask
# [n] bool (the seats with a gadget). Their presence selects the compiled variant; the teacher never has them.

def gadget_follow(g):
    """RM+ over the gadget's two actions [..., 2] = (Follow, Terminate): the probability of Follow (1/2 without
    positive regret)."""
    pos = jnp.maximum(g, 0.0)
    total = pos[..., 0] + pos[..., 1]
    return jnp.where(total > 0, pos[..., 0] / jnp.maximum(total, 1e-30), 0.5)


def gadget_ceiling(spot, st: Static):
    """[n, H] Terminate values in CFV units: gadget_ev times the compatible mass of the other seats' deal ranges
    (the hero's range at the root, unscaled), like every root counterfactual value of pcfr."""
    ranges = spot["ranges"]
    mass = jnp.stack([mass_of(ranges[None], i, st.n, spot.get("tables"))[0] for i in range(st.n)])
    return spot["gadget_ev"] * mass


def gadget_scale(spot, follow):
    """The deal ranges of the gadget game: ranges x follow on the masked seats."""
    return jnp.where(spot["gadget_mask"][:, None], follow, 1.0)


def init_cfr_state(st: Static, dtype=jnp.float32):
    vi, a = len(st.internal), st.topo.children.shape[1]
    zeros = jnp.zeros((vi, H, a), dtype)
    g = jnp.zeros((st.n, H, 2), dtype)
    return {"regret": zeros, "avg": zeros, "last": zeros, "u": jnp.int32(0),
            "done": jnp.bool_(False), "e": jnp.asarray(jnp.inf, dtype), "root": jnp.zeros((st.n, H), dtype),
            "g_regret": g, "g_last": g, "g_avg": jnp.zeros((st.n, H), dtype), "g_norm": jnp.zeros((), dtype)}


def make_cfr_chunk(st: Static, cfg: SolverConfig):
    return cached("cfr", st, cfg, lambda: _make_cfr_chunk(st, cfg))


def _make_cfr_chunk(st: Static, cfg: SolverConfig):
    def step(state, spot, aux):
        legal = spot["legal"][st.internal]
        t = (state["u"] + 1).astype(state["regret"].dtype)                             # 1-based
        # PCFR+: play RM+ of (clipped cumulative regret + last regret).
        pi = regret_policy(state["regret"] + state["last"], legal)
        if "fixed_mask" in spot:
            # The live path (poker/solver/subgame.py): at the nodes of our own played actions the policy is the one
            # we played (fixed_pi [Vi, H, A], fixed_mask [Vi]), so the solve's reach at the current node is our
            # exact range; the regrets there are still accumulated but never used. Absent keys: the teacher's path.
            pi = jnp.where(spot["fixed_mask"][:, None, None], spot["fixed_pi"], pi)
        gadget = "gadget_ev" in spot
        if gadget:
            # The gadget seats' deal ranges are scaled by their current Follow probability (RM+ of the gadget's
            # regrets, predictive like the nodes'); their own counterfactual values below are reach-free.
            follow = gadget_follow(state["g_regret"] + state["g_last"])                 # [n, H]
            spot_eff = {**spot, "ranges": spot["ranges"] * gadget_scale(spot, follow)}
        else:
            spot_eff = spot
        r, own, root = instant_regrets(pi, spot_eff, st, cfg, aux)
        regret = jnp.maximum(state["regret"] + r, 0.0)
        weight = t * t
        avg = state["avg"] + weight * own[:, :, None] * pi
        hold = state["done"]
        keep = lambda new, prev: jnp.where(hold, prev, new)
        out = {"regret": keep(regret, state["regret"]), "avg": keep(avg, state["avg"]),
               "last": keep(r, state["last"]), "u": keep(state["u"] + 1, state["u"]),
               "done": hold, "e": state["e"], "root": keep(root, state["root"]),
               "g_regret": state["g_regret"], "g_last": state["g_last"], "g_avg": state["g_avg"], "g_norm": state["g_norm"]}
        if gadget:
            ceiling = gadget_ceiling(spot, st)                                          # [n, H] CFV units
            value = follow * root + (1.0 - follow) * ceiling
            g_r = jnp.stack([root - value, ceiling - value], axis=-1)                   # (Follow, Terminate)
            g_r = jnp.where(spot["gadget_mask"][:, None, None], g_r, 0.0)
            out["g_regret"] = keep(jnp.maximum(state["g_regret"] + g_r, 0.0), state["g_regret"])
            out["g_last"] = keep(g_r, state["g_last"])
            out["g_avg"] = keep(state["g_avg"] + weight * follow, state["g_avg"])
            out["g_norm"] = keep(state["g_norm"] + weight, state["g_norm"])
        return out

    def chunk(states, spots, aux=None):
        batched = jax.vmap(step, in_axes=(0, 0, None))
        return jax.lax.fori_loop(0, cfg.check_every, lambda _, s: batched(s, spots, aux), states)

    return jax.jit(chunk)


E_TOL = 1e-5     # pot units: E below -E_TOL or not finite is a solver bug, never convergence


def solve_cfr(batch, st: Static, cfg: SolverConfig, log=None, aux=None):
    """Regret matching until every spot has E <= e_target (checked every check_every
    iterations) or the cap. Returns a state with z = log(average policy) like solve(). An E that
    is not finite or below -E_TOL raises: the stopping test "E <= e_target" would otherwise
    certify it (solver audit 2026-10-10, F1: the dead-subtree best response gave negative E on
    3-player spots, which passed e_target and cut the solves short)."""
    count = batch["ranges"].shape[0]
    dtype = batch["ranges"].dtype
    state = jax.vmap(lambda _: init_cfr_state(st, dtype))(jnp.arange(count))
    chunk = make_cfr_chunk(st, cfg)
    ex = cached("ex_avg", st, cfg, lambda: jax.jit(jax.vmap(
        lambda s, spot, aux: exploitability(average_logits(s["avg"], spot, st), spot, st, cfg, aux,
                                            follow=average_follow(s))["nash_conv"] / st.n,
        in_axes=(0, 0, None))))
    while True:
        state = chunk(state, batch, aux)
        e = ex(state, batch, aux)
        e_host = np.asarray(e)
        bad = np.flatnonzero(~np.isfinite(e_host) | (e_host < -E_TOL))
        if len(bad):
            raise RuntimeError(f"solve_cfr: E not finite or below -{E_TOL:g} on {len(bad)} spots, first {bad[:8].tolist()} "
                               f"with E {e_host[bad[:8]].tolist()}: a solver bug (backward or terminals), not convergence")
        state["e"] = jnp.where(state["done"], state["e"], e)
        state["done"] = state["done"] | (e <= cfg.e_target) | (state["u"] >= cfg.cap)
        if log is not None:
            log(state)
        if bool(np.asarray(state["done"]).all()):
            break
    z = jax.vmap(lambda s, spot: average_logits(s, spot, st))(state["avg"], batch)
    return {"z": z, "u": state["u"], "e": state["e"], "residual": state["e"],
            "reached": jnp.where(state["e"] <= cfg.e_target, state["u"], -1),
            "alpha": jnp.zeros(count, dtype), "done": state["done"]}


def average_follow(state):
    """[B?, n, H] the average Follow probability of the gadget seats (1 where there is no gadget / no weight)."""
    norm = state["g_norm"]
    return jnp.where(norm[..., None, None] > 0, state["g_avg"] / jnp.maximum(norm[..., None, None], 1e-30), 1.0)


def average_logits(s, spot, st: Static):
    legal = spot["legal"][st.internal]
    return jnp.log(jnp.maximum(average_policy(s, legal), 1e-30))


def solve(batch, st: Static, cfg: SolverConfig, log=None, aux=None):
    """Solve a prepared batch until every spot stops; returns the solver state (z = log policy).
    aux: the next street's critic params when the tree has street ends (turn, flop)."""
    return solve_cfr(batch, st, cfg, log, aux)


def solve_fixed(batch, st: Static, cfg: SolverConfig, iterations: int, aux=None):
    """Exactly `iterations` PCFR+ iterations on every spot of a batch, without any
    exploitability evaluation: the fixed-T teacher in poker/solver/loop.py, whose E is the
    caller's diagnostic, computed once after the solve. The iterations run in chunks of cfg.check_every (the
    compiled chunk of solve_cfr; `iterations` must be a multiple of it); cfg.cap and cfg.e_target are not used.
    Returns {"z": log of the average policy [B, Vi, H, A], "u": iterations done [B]}."""
    if iterations <= 0 or iterations % cfg.check_every:
        raise ValueError(f"iterations {iterations} must be a positive multiple of the chunk {cfg.check_every}")
    count = batch["ranges"].shape[0]
    dtype = batch["ranges"].dtype
    state = jax.vmap(lambda _: init_cfr_state(st, dtype))(jnp.arange(count))
    chunk = make_cfr_chunk(st, cfg)
    for _ in range(iterations // cfg.check_every):
        state = chunk(state, batch, aux)
    z = jax.vmap(lambda s, spot: average_logits(s, spot, st))(state["avg"], batch)
    return {"z": z, "u": state["u"]}


def exploitability(z, spot, st: Static, cfg: SolverConfig, aux=None, nodes=False, follow=None):
    """Per-player on-policy value, best-response value and NashConv (pot units), one spot; nodes=True
    adds the values at every node, "on" and "br" [V, n, H] (checks: one trace of the terminals).
    Traceable (jit, vmap); check the results with check_exploitability once they are on the host.
    A gadget spot (gadget_ev / gadget_mask, follow = the average Follow probabilities [n, H]) is the gadget
    game: the gadget seats enter with ranges x follow, their on-policy root value is follow x Follow value +
    (1 - follow) x ceiling and their best response takes the better of the two per hand; Z is the deal's."""
    legal = spot["legal"][st.internal]
    pi = policy(z, legal)
    ranges = spot["ranges"]
    gadget = "gadget_ev" in spot and follow is not None
    deal = ranges * gadget_scale(spot, follow) if gadget else ranges
    reach = forward(pi, deal, st)
    term = terminal_values(reach, spot["contrib"], payout_pot(spot), spot.get("tables"), st, cfg, spot, aux)
    on = backward(term, pi, legal, st)
    br = backward(term, pi, legal, st, best_response=True)
    root_on, root_br = on[0], br[0]
    if gadget:
        ceiling = gadget_ceiling(spot, st)
        mask = spot["gadget_mask"][:, None]
        root_on = jnp.where(mask, follow * root_on + (1.0 - follow) * ceiling, root_on)
        root_br = jnp.where(mask, jnp.maximum(root_br, ceiling), root_br)
    # Z: total probability of a compatible deal (same for every player).
    mass0 = opponent_mass_root(jnp.zeros((1, st.n, H), ranges.dtype).at[0].set(ranges), spot.get("tables"), st)
    z_norm = jnp.sum(ranges[0] * mass0)
    value = jnp.sum(ranges * root_on, axis=-1) / z_norm
    best = jnp.sum(ranges * root_br, axis=-1) / z_norm
    out = {"value": value, "best": best, "nash_conv": jnp.sum(best - value)}
    return {**out, "on": on, "br": br} if nodes else out


def check_exploitability(res, tol=1e-6):
    """Invariants of exploitability() results, one spot or a vmapped batch, on the host (not
    traceable): values and best responses finite, every player's best response >= its value - tol,
    NashConv >= -tol (pot units; tol absorbs rounding). A violation is a solver bug (solver audit
    2026-10-10, F1: dead subtrees in the best response gave negative NashConv on solved 3-player
    spots), so it raises. Every caller that reads exploitability() results on the host should pass
    them through here (the solve stage included). Returns res."""
    value = np.atleast_2d(np.asarray(res["value"], np.float64))                      # [B, n]
    best = np.atleast_2d(np.asarray(res["best"], np.float64))
    nash = np.atleast_1d(np.asarray(res["nash_conv"], np.float64))                  # [B]
    errors = [f"spot {s}: value {value[s]}, best response {best[s]}, NashConv {nash[s]} not finite"
              for s in np.flatnonzero(~(np.isfinite(value).all(-1) & np.isfinite(best).all(-1) & np.isfinite(nash)))]
    errors += [f"spot {s} player {i}: best response {best[s, i]:.6g} < value {value[s, i]:.6g}"
               for s, i in np.argwhere(best - value < -tol)]
    errors += [f"spot {s}: NashConv {nash[s]:.6g} < 0" for s in np.flatnonzero(nash < -tol)]
    if errors:
        raise AssertionError(f"exploitability invariants violated ({len(errors)}, tol {tol:g}): " + "; ".join(errors[:6]))
    return res


def root_values(z, spot, st: Static, cfg: SolverConfig, aux=None):
    """Expected value [n, H] of every hand of every player at the street root (pot units, whole
    hand accounting: chips received minus chips put in over the whole hand).

    These are the critic targets: V(board, pot, stacks, ranges) -> EV per hand per player
    (the critic stores them in relative accounting, + own contribution before the street).
    Hands that cannot occur (blocked or incompatible with every opponent hand) get 0.
    """
    legal = spot["legal"][st.internal]
    pi = policy(z, legal)
    reach = forward(pi, spot["ranges"], st)
    term = terminal_values(reach, spot["contrib"], payout_pot(spot), spot.get("tables"), st, cfg, spot, aux)
    cfv = backward(term, pi, legal, st)[0]
    out = []
    for i in range(st.n):
        opp = [j for j in range(st.n) if j != i]
        if st.n == 2:
            mass = j_mass(spot["ranges"][opp[0]], spot.get("tables"))
        else:
            mass = sd.mass_three(spot["ranges"][opp[0]], spot["ranges"][opp[1]])
        out.append(jnp.where(mass > 1e-30, cfv[i] / jnp.maximum(mass, 1e-30), 0.0) * spot["valid"])
    return jnp.stack(out)


def opponent_mass_root(reach, tables, st: Static):
    """Compatible mass of players 1.. for player 0's hands at the root."""
    if st.n == 2:
        return j_mass(reach[0, 1], tables)
    return sd.mass_three(reach[0, 1], reach[0, 2])


