"""Residual street critics with exact check-down baselines.

V(board, public numbers, ranges of the live players) -> EV of every hand of every player slot,
in pot units of the street start and in relative accounting: the pot share a hand can expect
minus its future investment (= whole-hand value + own contribution before the street, divided by
the street-start pot). Targets are the root EVs of the teacher's solves of that street. An
auxiliary "end of street" critic is the next street's critic averaged over the next card with
card removal (DeepStack's auxiliary network); its targets come from the next street's critic.

Residual critics: V = network(board, pub, ranges, F) + CD. CD is the exact check-down value of
every hand of every slot (poker/solver/checkdown.py: showdown share against the other slot's range over
every completion of the board, times the pot net of the rake); F = CD is also an input. The network
learns only the value of the betting (measured on the k0b river: MAE 0.063 against 0.103 for the
plain critic on the same data). Wherever a critic is called the
caller supplies F: the solver's street ends from the all-in matrix the spot already carries
(pcfr.leaf_values), the end-of-street targets from the next street's boards (end_of_turn,
end_of_flop, end_of_preflop_classes).

Chip conservation (zero-sum audit 2026-10-10): the values of a situation must pay out what its pot
holds minus the rake, S(V) = sum_i sum_h w_i(h) V_i(h) = 1 - E[rake] / pot with the solver's deal
weights; the raw networks did not, and the solver amplified it. Every critic call (net_critic)
and the training loss (_batch_loss) project the network's values onto the target total T = net0 -
min(kappa, headroom) (project_values, per-hand constant per slot); errors are reported for the
projected values and the raw network (row_metrics).

Rows: one per spot. Player slots follow the street's action order (slot 0 acts first); P_MAX = 9
slots (the input shape stays fixed as more players become live), absent players have zero
ranges, zero F and no loss. Zero slots contribute nothing, so the network is evaluated and trained
on the first k slots only (k = players of the spot; apply_net slices the weights): exactly the
zero-padded network, at the cost of k slots. The 24 suit permutations of board, ranges, F and
targets are exact copies of a solved spot: training draws one per row (augmentation).
"""

import itertools
import json
import logging
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from poker.envs import config
from poker.solver import checkdown as CD
from poker.solver import flops as FL
from poker.solver import showdown as sd
from poker.solver.cards import HANDS, MEMBER, NUM_HANDS, PAIR_INDEX

P_MAX = 9
# log(pot / BB), rake level, rake rate (1 = 5%), log1p(rake cap / pot); per slot: present,
# log1p(behind / pot), contribution before the street / pot.
PUB = 4 + 3 * P_MAX
D_IN = 52 + PUB + P_MAX * NUM_HANDS
D_F = P_MAX * NUM_HANDS               # input F: the check-down value of every hand of every slot
D_OUT = P_MAX * NUM_HANDS

_SUITS = list(itertools.permutations(range(4)))
CARD_PERM = np.asarray([[p[c // 13] * 13 + c % 13 for c in range(52)] for p in _SUITS], np.int32)   # [24, 52]
HAND_PERM = PAIR_INDEX[CARD_PERM[:, HANDS[:, 0]], CARD_PERM[:, HANDS[:, 1]]].astype(np.int32)       # [24, 1326]
INV_CARD = np.argsort(CARD_PERM, axis=1).astype(np.int32)
INV_HAND = np.argsort(HAND_PERM, axis=1).astype(np.int32)


def public_row(pot_bb, rake_level, rake_rate, rake_cap_pot, present, behind_pot, contrib_pot):
    """[PUB] public numbers of one spot; per-slot arrays have length P_MAX. At a called HU
    boundary (two present slots with equal contributions), only the smaller behind stack can
    enter a future pot. Represent both slots by that exact effective stack; the engine's chips
    are unchanged. Unequal contributions and 3+ present slots retain their original features.
    Equal deep stacks remain unbounded. payout_terms recovers the pot, rake and effective chips
    for the chip-conservation projection."""
    live = present > 0
    called_hu = ((jnp.sum(live) == 2)
                 & (jnp.min(jnp.where(live, contrib_pot, jnp.inf))
                    == jnp.max(jnp.where(live, contrib_pot, -jnp.inf))))
    effective = jnp.min(jnp.where(live, behind_pot, jnp.inf))
    behind_pot = jnp.where(called_hu & live, effective, behind_pot)
    head = jnp.stack([jnp.log(pot_bb), rake_level, rake_rate, jnp.log1p(rake_cap_pot)])
    slots = jnp.concatenate([present, jnp.log1p(behind_pot) * present, contrib_pot * present])
    return jnp.concatenate([head, slots]).astype(jnp.float32)


def board_onehot(board):
    """board int32[k] (k = 3..5, -1 = empty) -> [52]."""
    return jnp.sum(jax.nn.one_hot(board, 52, dtype=jnp.float32), axis=0)


def permute(sigma, board1h, ranges, *per_hand):
    """Apply suit permutation sigma (0..23) to one row: card c -> CARD_PERM[sigma, c]; ranges and
    every further per-hand array [..., 1326] are permuted alike."""
    b = board1h[jnp.asarray(INV_CARD)[sigma]]
    inv = jnp.asarray(INV_HAND)[sigma]
    return (b, ranges[..., inv]) + tuple(x[..., inv] for x in per_hand)


def init_net(key, hidden, d_in=D_IN, d_out=D_OUT, d_f=0, fan_in0=None, fan_out_last=None):
    """MLP parameters (Glorot uniform). d_f > 0 adds a first-layer block "wf" [d_f, h0] for the
    input F, kept apart so that both blocks slice by slot (apply_net). fan_in0 / fan_out_last: the
    first layer's fan-in and the last layer's fan-out the network actually uses (the slots it is
    evaluated on, apply_net slices the rest away); default: the full shapes (d_in + d_f, d_out)."""
    shapes = list(zip((d_in,) + tuple(hidden), tuple(hidden) + (d_out,)))
    keys = jax.random.split(key, len(shapes) + 1)
    params = []
    for i, (k, (rows, cols)) in enumerate(zip(keys, shapes)):
        fan_in = (fan_in0 or d_in + d_f) if i == 0 else rows
        fan_out = (fan_out_last or cols) if i == len(shapes) - 1 else cols
        limit = sqrt(6.0 / (fan_in + fan_out))
        layer = {"w": jax.random.uniform(k, (rows, cols), jnp.float32, -limit, limit),
                 "b": jnp.zeros(cols, jnp.float32)}
        if i == 0 and d_f:
            layer["wf"] = jax.random.uniform(keys[-1], (d_f, cols), jnp.float32, -limit, limit)
        params.append(layer)
    return params


def init_critic(key, hidden, k=P_MAX):
    """Critic parameters for P_MAX slots, Glorot limits by the fan-in and fan-out of the first k
    slots (the slots of the training rows: the part of the network a row actually uses). Slots beyond k get the same limits."""
    return init_net(key, hidden, D_IN, D_OUT, D_F, fan_in0=52 + PUB + 2 * k * NUM_HANDS,
                    fan_out_last=k * NUM_HANDS)


def apply_net(params, board1h, pub, ranges, f):
    """V [..., k, 1326] = network + F from rows: board1h [..., 52], pub [..., PUB], ranges and F
    [..., k, 1326] of the first k <= P_MAX slots (the others zero): only those slots' weights are
    used. Ranges enter as they are (each sums to 1): fed as densities (x 1326)
    the network stops learning - observed in a training comparison on k0b: with Glorot
    limits by the used slots test MAE 0.1188 for both seeds, best epoch 0, against 0.0632 / 0.0620
    with the ranges as they are."""
    k = ranges.shape[-2]
    lead = ranges.shape[:-2]
    x = jnp.concatenate([board1h, pub, ranges.reshape(lead + (k * NUM_HANDS,))], axis=-1)
    fx = f.reshape(lead + (k * NUM_HANDS,))
    x = jax.nn.relu(x @ params[0]["w"][:x.shape[-1]] + fx @ params[0]["wf"][:k * NUM_HANDS] + params[0]["b"])
    for layer in params[1:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    x = x @ params[-1]["w"][:, :k * NUM_HANDS] + params[-1]["b"][:k * NUM_HANDS]
    return x.reshape(lead + (k, NUM_HANDS)) + f


# ------------------------------------------------------------------ chip conservation (zero-sum)
#
# Every deal and every line pays out exactly the pot minus the rake, so for ANY strategy profile
# the values of one situation satisfy (zero-sum audit 2026-10-10, REPORT section 0)
#     S(V) := sum_i sum_h w_i(h) V_i(h) = 1 - E[rake_final] / pot0,   lo <= S <= net0,
# w_i = r_i M_i / Z the solver's joint deal weights (M_i: compatible mass of the other live slots;
# pcfr.exploitability / root_values weigh hands this way), net0 = 1 - R(pot0) / pot0, lo = 1 -
# R(pot_max) / pot0 (R the engine's rake once the flop is dealt), S = 1 without rake. F satisfies
# S(F) = net0 exactly. The raw networks do not: the solver steers reach into the street ends whose
# total is inflated (bet-call lines +0.6 pot, check-check -0.1), 59% of the k0 flop solves and 21%
# of the preflop ones paid out more than the pot (preflop +0.13 - 0.30 pot), and those totals became
# the next critic's targets. project_values removes the violation: V' = V - (S(V) - T) / n on every
# hand of the n live slots (per-hand constant, equal split: the minimum change in the solver's own
# metric sum_i sum_h w_i d_i^2; a range-proportional split changes the hands' order and gained
# nothing on the river, audit F6), so S(V') = T exactly and the hand-dependent shape stays. Target
# T = net0 - min(kappa, net0 - lo): kappa is the expected extra rake of the betting the values still
# contain, capped by the headroom (0 once the cap is reached or nothing is behind; 0 without rake).
# Every critic call (net_critic: the solver's street ends, the end-of-street targets, Delta E) and
# the training loss (_batch_loss) apply the same projection; apply_net stays raw.

# kappa by the first street of the betting a critic's values contain, pot units of its situation.
# river: the mean of net0 - S(target) over the 45,000 exact river_big targets (recorded HU calibration:
# mean 0.0176, median 0.0209, p99 0.033, exactly 0 on
# the 14% of spots where the cap is reached, flat in SPR). With the cap-aware min() the projected
# total misses the exact one by 0.0058 pot on the 4500 river_big test spots (mean |.|; the k0
# critic's own total 0.024, audit F5); the mean over the spots below the cap, 0.0205, would miss by
# 0.0049 (results/fix_zerosum/C_river45k.json, totals) - the consistent constant for the cap-aware rule.
# turn / flop / preflop: PLACEHOLDERS (0 = the projection onto net0), to be estimated from
# regenerated solves (audit F7 / fix step 2: the mean of net0 - S(target) of the stage's rows, free
# in every rows file; critic reports carry it as test_target_total) - the k0 targets of those
# streets are biased by the very violation this removes.
KAPPA = {"river": 0.0176, "turn": 0.0, "flop": 0.0, "preflop": 0.0}
# The river critic and the end-of-turn critic (the river critic averaged over the river card) owe
# the river's betting; the turn critic and the end of the flop the turn's and the river's; ...
# The card average of totals is the total of the average, so an end critic's targets sum to the
# next critic's T and both project with the same kappa.
KAPPA_STREET = {"river_critic": "river", "turn_end": "river", "turn_critic": "turn", "flop_end": "turn",
                "flop_critic": "flop", "pre_end": "flop"}
ZMIN = 1e-6                  # a situation without compatible deals (normalised ranges) is not projected


def critic_kappa(name, kappa=None):
    """kappa of a loop critic by its stage name (KAPPA_STREET); `kappa` overrides KAPPA (a run's own
    values, loop layout "critic_projection")."""
    return (KAPPA if kappa is None else kappa)[KAPPA_STREET[name]]


def payout_terms(pub, k, dtype=jnp.float32):
    """(present [..., k], net0 [...], lo [...]) of critic rows from their public numbers alone
    (public_row; pure, jittable): the pot in whole chips round(exp(pub[0]) * BB), basis points
    round(500 * rate), the cap round(expm1(pub[3]) * pot), chips behind expm1(.) * pot of the first k
    slots; net0 = 1 - R(pot) / pot, lo = 1 - R(pot_max) / pot with pot_max = pot + sum over the
    present slots of min(b_i, max_{j != i} b_j) (what the deepest other player can match) and
    R = config.rake_from_terms of the flop dealt. No rake: net0 = lo = 1."""
    head = pub[..., :4].astype(jnp.float32)
    pot = jnp.round(jnp.exp(head[..., 0]) * config.BB)
    bps = jnp.round(head[..., 2] * 500.0).astype(jnp.int32)
    cap = jnp.round(jnp.expm1(head[..., 3]) * pot).astype(jnp.int32)
    present = pub[..., 4:4 + k]
    behind = jnp.expm1(pub[..., 4 + P_MAX:4 + P_MAX + k].astype(jnp.float32)) * pot[..., None] * (present > 0)
    other = jnp.max(jnp.where(jnp.eye(k, dtype=bool), 0.0, behind[..., None, :]), axis=-1)
    pot_max = jnp.round(pot + jnp.sum(jnp.minimum(behind, other), axis=-1))
    rake = lambda p: config.rake_from_terms(bps, cap, p.astype(jnp.int32), 1).astype(dtype)
    potd = pot.astype(dtype)
    return present, 1.0 - rake(pot) / potd, 1.0 - rake(pot_max) / potd


def target_total(net0, lo, kappa):
    """T = net0 - min(kappa, net0 - lo): what a situation pays out after the expected extra rake."""
    return net0 - jnp.minimum(kappa, jnp.maximum(net0 - lo, 0.0))


def _j_mass(w):
    """checkdown.j_mass over leading axes: compatible mass [..., H] of ranges [..., H]."""
    c = w @ jnp.asarray(MEMBER.T, w.dtype)                                          # [..., 52] card sums
    return jnp.sum(w, axis=-1, keepdims=True) - c[..., HANDS[:, 0]] - c[..., HANDS[:, 1]] + w


def deal_weights(ranges, present):
    """The solver's joint deal weights w_i(h) = r_i(h) M_i(h) / sum_h r_i M_i of the live slots
    (present, with a range), 0 elsewhere; M_i = compatible mass of the other live slots (one:
    checkdown.j_mass; two: showdown.mass_three, as opponent_mass), board-free (blocked hands must be
    zero in the ranges). ranges [..., k, H] (any scale), present [..., k]. Returns (w, live [..., k],
    ok [...]): ok where two or three slots are live, every live slot has compatible deals (sum r_i
    M_i > ZMIN for normalised ranges; the joint mass is the same for every slot) and the weights are
    finite. Four and more live slots (exact compatible mass is not implemented) are not ok."""
    k = ranges.shape[-2]
    live = (present > 0) & (jnp.sum(ranges, axis=-1) > 0)
    r = jnp.where(live[..., None], ranges, 0.0)
    r = r / jnp.maximum(jnp.sum(r, axis=-1, keepdims=True), 1e-30)
    if k == 2:
        mass = jnp.stack([_j_mass(r[..., 1, :]), _j_mass(r[..., 0, :])], axis=-2)
    elif k >= 3:
        on = live.astype(r.dtype).reshape((-1, k))
        each = lambda rr, pp: jnp.stack([opponent_mass(rr, pp, i) for i in range(k)])
        mass = jax.vmap(each)(r.reshape((-1, k, NUM_HANDS)), on).reshape(r.shape)
    else:
        mass = jnp.zeros_like(r)
    num = r * mass
    z = jnp.sum(num, axis=-1)                                                         # [..., k]
    ok = ((jnp.sum(live, axis=-1) >= 2) & jnp.all(jnp.where(live, z > ZMIN, True), axis=-1)
          & jnp.all(jnp.isfinite(num), axis=(-1, -2)))
    w = jnp.where(ok[..., None, None] & live[..., None], num / jnp.where(z > ZMIN, z, 1.0)[..., None], 0.0)
    return w, live, ok


def _shift(v, w, live, ok, t):
    n = jnp.maximum(jnp.sum(live, axis=-1), 1).astype(v.dtype)
    gap = (jnp.sum(w * v, axis=(-1, -2)) - t) / n
    return jnp.where(ok[..., None, None], v - gap[..., None, None] * live[..., None].astype(v.dtype), v)


def project_values(v, ranges, present, net0, lo, kappa):
    """V' = V - (S(V) - T) / n on every hand of the n live slots, T = target_total(net0, lo, kappa):
    S(V') = T exactly with the solver's weights (deal_weights), each slot's hands move alike. Rows
    that are not ok (no two live slots with compatible deals, four and more live) pass unchanged.
    v, ranges [..., k, H]; present [..., k]; net0, lo [...]; kappa a scalar. Pure, jittable."""
    w, live, ok = deal_weights(ranges, present)
    return _shift(v, w, live, ok, target_total(net0, lo, jnp.asarray(kappa, v.dtype)))


def project(v, pub, ranges, kappa):
    """project_values with present, net0 and lo from the public row (payout_terms)."""
    present, net0, lo = payout_terms(pub, ranges.shape[-2], v.dtype)
    return project_values(v, ranges, present, net0, lo, kappa)


def _net_call(params, kappa, board1h, pub, ranges, f, spot=None):
    v = apply_net(params, board1h, pub, ranges, f)
    return v if kappa is None else project(v, pub, ranges, kappa)


def net_critic(params, kappa=0.0):
    """A critic as a pytree callable (board1h, pub, ranges, F, spot) -> V', so jitted solvers take
    it as an ordinary argument (poker/solver/pcfr.py leaf_values): the network's values projected onto
    chip conservation (project) with the critic's kappa - critic_kappa(name) for the loop's critics,
    0 = onto net0. kappa=None returns the raw network (apply_net): audits and comparisons only."""
    return jax.tree_util.Partial(_net_call, params, kappa)


def loss_weights(ranges, valid, present):
    """Per (slot, hand) weights: in-range hands weigh ~1 each on average (range * #valid), every
    valid hand keeps a floor of 0.05 so off-range values stay sane; absent slots weigh 0."""
    count = jnp.maximum(jnp.sum(valid, axis=-1, keepdims=True), 1.0)
    w = (ranges * count[..., None, :] + 0.05) * valid[..., None, :]
    return w * present[..., :, None]


@dataclass
class Rows:
    """Critic dataset (host numpy): board [S, 52], pub [S, PUB], ranges/target/f [S, P_MAX, 1326],
    valid [S, 1326] (hands not blocked by the board), present [S, P_MAX]. f = the check-down value
    CD of every hand of every slot (input F and the residual's baseline)."""
    board: np.ndarray
    pub: np.ndarray
    ranges: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    present: np.ndarray
    f: np.ndarray

    def __len__(self):
        return len(self.pub)

    def take(self, idx):
        return Rows(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))

    def active(self):
        """Number of leading slots any row uses (rows store P_MAX slots)."""
        used = np.flatnonzero(np.any(self.present > 0, axis=0))
        return int(used[-1]) + 1 if len(used) else 1

    def first_slots(self, k):
        return Rows(self.board, self.pub, self.ranges[:, :k], self.target[:, :k], self.valid, self.present[:, :k],
                    self.f[:, :k])

    @staticmethod
    def concat(parts):
        return Rows(*(np.concatenate([getattr(p, f) for p in parts]) for f in Rows.__dataclass_fields__))


FIELDS = tuple(Rows.__dataclass_fields__)


def rows_from_npz(data, prefix=""):
    """Rows stored by the loop (field names with an optional prefix, "allin_" for the exact SPR-0
    rows). Rows of the river written before the residual critics (k0b) carry no f: there the
    check-down value of a spot is exactly the target of its SPR-0 row (same board, ranges, pot and
    rake), so f = allin_target; any other file without f is an error."""
    fields = {f: data[prefix + f] for f in FIELDS if prefix + f in data.files}
    if "f" not in fields:
        if "allin_target" not in data.files:
            raise ValueError("critic rows without the check-down input f and without SPR-0 rows to derive it from")
        fields["f"] = data["allin_target"]
    return Rows(**fields)


def pad_rows(rows: Rows) -> Rows:
    """Rows of a run with fewer slots (k0b: P_MAX = 3) in the current layout: zero slots and the
    per-slot blocks of pub (present, behind, contribution) padded to P_MAX."""
    p = rows.ranges.shape[1]
    if p == P_MAX:
        return rows
    pub = np.zeros((len(rows), PUB), np.float32)
    pub[:, :4] = rows.pub[:, :4]
    for b in range(3):
        pub[:, 4 + b * P_MAX:4 + b * P_MAX + p] = rows.pub[:, 4 + b * p:4 + (b + 1) * p]
    pad = lambda x: np.concatenate([x, np.zeros((len(x), P_MAX - p) + x.shape[2:], x.dtype)], axis=1)
    return Rows(rows.board, pub, pad(rows.ranges), pad(rows.target), rows.valid, pad(rows.present), pad(rows.f))


def pad_slots(x, axis=1):
    """[..., k, ...] -> P_MAX slots along `axis` (zeros)."""
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, P_MAX - x.shape[axis])
    return np.pad(np.asarray(x), widths)


def spot_checkdown(batch, st):
    """[B, n, 1326] check-down value of every hand of every seat at the root of prepared spots (seat
    order, like the root EVs): river - rank tables of the board; turn, flop, preflop - the all-in
    matrix the batch carries; times the root pot net of the rake. Two players."""
    if st.n != 2:
        raise NotImplementedError("check-down values for 3+ players come with brief M2, item 3")
    ranges = batch["ranges"]
    if "tables" in batch:
        share = jax.vmap(lambda r, t: CD.two(lambda w: CD.share_river(w, t), r))(ranges, batch["tables"])
    else:
        share = jax.vmap(lambda r, e: CD.two(lambda w: CD.share_matrix(w, e), r))(ranges, batch["equity"])
    # CD.two is in "slot of the other seat" order: row i is seat i against seat 1 - i.
    net = CD.net_fraction(batch["pot_chips"][:, 0], batch["rake_bps"], batch["rake_cap_chips"], share.dtype)
    return share * net[:, None, None] * batch["valid"][:, None, :]


def rows_from_solves(spots, batch, root_ev, st, cd):
    """Critic rows of solved spots (one street, n players): root EVs [B, n, 1326] in whole-hand
    accounting (pot units) -> relative accounting (+ own contribution before the street), placed
    in critic slots (postflop action order, poker.solver.pcfr.Static.slot); cd [B, n, 1326] the
    check-down values of the seats (spot_checkdown)."""
    count = len(spots["pot"])
    root_ev = np.asarray(root_ev, np.float32)
    cd = np.asarray(cd, np.float32)
    contrib0 = np.asarray(batch["contrib"])[:, 0]                       # [B, n], pot units
    behind0 = np.asarray(batch["behind"])[:, 0]
    ranges = np.zeros((count, P_MAX, NUM_HANDS), np.float32)
    target = np.zeros_like(ranges)
    f = np.zeros_like(ranges)
    present = np.zeros((count, P_MAX), np.float32)
    behind = np.zeros((count, P_MAX), np.float32)
    contrib = np.zeros((count, P_MAX), np.float32)
    for i in range(st.n):
        j = int(st.slot[i])
        ranges[:, j] = spots["ranges"][:, i]
        target[:, j] = root_ev[:, i] + contrib0[:, i, None]
        f[:, j] = cd[:, i]
        present[:, j], behind[:, j], contrib[:, j] = 1.0, behind0[:, i], contrib0[:, i]
    pub = jax.vmap(public_row)(jnp.asarray(np.asarray(spots["pot"]) / config.BB, jnp.float32),
                               jnp.asarray(spots.get("rake_level", np.zeros(count)), jnp.float32),
                               jnp.asarray(np.asarray(spots.get("rake_bps", np.zeros(count))) / 500.0, jnp.float32),
                               jnp.asarray(np.asarray(spots.get("rake_cap", np.zeros(count))) / np.asarray(spots["pot"]),
                                           jnp.float32),
                               jnp.asarray(present), jnp.asarray(behind), jnp.asarray(contrib))
    board = np.asarray(jax.vmap(board_onehot)(jnp.asarray(spots["board"])))
    valid = spots["valid"][:, None, :].astype(np.float32)
    return Rows(board, np.asarray(pub), ranges, target * valid, np.asarray(spots["valid"], np.float32), present,
                f * valid)


def allin_rows(spots, batch, st, cd):
    """Exact river rows with nothing behind (SPR 0, two players): no betting is left, so the EV
    of a hand is its check-down value (relative accounting) and the residual target is zero. They
    teach the river critic the all-in edge that street-end leaves approach; no solve is needed."""
    if st.n != 2 or "tables" not in batch:
        raise ValueError("all-in rows are built for two-player river spots")
    zero = {**batch, "behind": jnp.zeros_like(batch["behind"])}
    rows = rows_from_solves(spots, zero, np.zeros_like(np.asarray(cd)), st, cd)
    rows.target = rows.f.copy()                          # EV at SPR 0 = the check-down value, exactly
    return rows


def _batch_loss(params, b, p, r, t, v, s, f, kappa=0.0):
    """Huber loss of the critic's values as every caller sees them: the network's output projected
    onto chip conservation (project, the critic's kappa), so the network learns the hand-dependent
    part and the projection is the same in training and in use (kappa=None: the raw network)."""
    pred = apply_net(params, b, p, r, f)
    if kappa is not None:
        pred = project(pred, p, r, kappa)
    w = loss_weights(r, v, s)
    return jnp.sum(w * optax.huber_loss(pred, t, delta=1.0)) / jnp.maximum(jnp.sum(w), 1e-9)


@jax.jit
def _augment(key, b, r, t, v, f):
    sig = jax.random.randint(key, (b.shape[0],), 0, len(_SUITS))
    b2, r2, t2, f2 = jax.vmap(permute)(sig, b, r, t, f)
    v2 = jax.vmap(lambda s_, x: x[jnp.asarray(INV_HAND)[s_]])(sig, v)
    return b2, r2, t2, v2, f2


METRICS = ("mae", "offset", "centred", "worst", "worst_substantial")


def prediction_metrics(pred, target, ranges, valid, present):
    """Per-row errors (pot units) of predicted EVs pred [B, k, H] against targets [B, k, H]; ranges
    [B, k, H], valid [B, H], present [B, k]. Per slot weighted by the slot's range, then the mean over
    the present slots (critic audit 2026-10-10, F4: one MAE hides a per-spot offset and heavy tails):
      mae                sum_h r_h |e_h| / sum_h r_h, e = pred - target on the valid hands
      offset             |sum_h r_h e_h| / sum_h r_h: the error of the slot's range EV
      centred            sum_h r_h |e_h - o| / sum_h r_h, o = the signed offset: the hand-dependent part
      worst              max |e_h| over in-range hands (r_h > 1e-3 / #valid) of the present slots
      worst_substantial  max |e_h| over hands with r_h >= 10% of their slot's largest weight
    """
    e = (pred - target) * valid[:, None, :]
    mass = jnp.maximum(ranges.sum(-1), 1e-12)                                       # [B, k]
    signed = (ranges * e).sum(-1) / mass
    on = (present > 0)[..., None]
    over_slots = lambda x: (x * present).sum(-1) / jnp.maximum(present.sum(-1), 1.0)
    inrange = (ranges > 1e-3 / jnp.maximum(valid.sum(-1), 1.0)[:, None, None]) & on
    substantial = (ranges >= 0.1 * ranges.max(-1, keepdims=True)) & (ranges > 0) & on
    return {"mae": over_slots((ranges * jnp.abs(e)).sum(-1) / mass),
            "offset": over_slots(jnp.abs(signed)),
            "centred": over_slots((ranges * jnp.abs(e - signed[..., None])).sum(-1) / mass),
            "worst": jnp.where(inrange, jnp.abs(e), 0.0).max(axis=(1, 2)),
            "worst_substantial": jnp.where(substantial, jnp.abs(e), 0.0).max(axis=(1, 2))}


def _device_chunks(rows: Rows, chunk=512):
    """Rows on the device as chunks [n_chunks, size, ...] (size = min(chunk, rows)); zero rows pad
    the last chunk (no range, no present slot: no weight; results are cut to len(rows))."""
    size = min(chunk, len(rows))
    pad = -len(rows) % size
    out = {}
    for name in FIELDS:
        x = getattr(rows, name)
        if pad:
            x = np.concatenate([x, np.zeros((pad,) + x.shape[1:], x.dtype)])
        out[name] = jnp.asarray(x.reshape((-1, size) + x.shape[1:]))
    return out


# row_metrics beyond METRICS: the same errors of the raw network (apply_net, before the projection),
# its chip-conservation violation S(V_raw) - T and the targets' own S(target) - T (pot units).
RAW_METRICS = tuple("raw_" + name for name in METRICS)
TOTALS = ("raw_total", "target_total")


@jax.jit
def _chunk_metrics(params, dev, kappa):
    def one(c):
        raw = apply_net(params, c["board"], c["pub"], c["ranges"], c["f"])
        present, net0, lo = payout_terms(c["pub"], raw.shape[-2], raw.dtype)
        w, live, ok = deal_weights(c["ranges"], present)
        t = target_total(net0, lo, jnp.asarray(kappa, raw.dtype))
        pred = _shift(raw, w, live, ok, t)                                    # = project(raw, pub, ranges, kappa)
        gap = lambda v: jnp.where(ok, jnp.sum(w * v, axis=(-1, -2)) - t, 0.0)
        out = prediction_metrics(pred, c["target"], c["ranges"], c["valid"], c["present"])
        out.update({"raw_" + name: x for name, x in
                    prediction_metrics(raw, c["target"], c["ranges"], c["valid"], c["present"]).items()})
        out["raw_total"], out["target_total"] = gap(raw), gap(c["target"])
        return out
    return jax.lax.map(one, dev)


def _metrics_on_device(params, dev, count, kappa=0.0):
    return {name: np.asarray(x).reshape(-1)[:count] for name, x in _chunk_metrics(params, dev, kappa).items()}


def row_metrics(params, rows: Rows, chunk=512, kappa=0.0):
    """Per-row errors of a critic: {name: [S]} for name in METRICS (prediction_metrics of the values
    callers use: the network projected with the critic's kappa, net_critic), RAW_METRICS (the same
    of the raw network) and TOTALS. kappa: critic_kappa(name) of a loop critic."""
    if not len(rows):
        return {name: np.zeros(0) for name in METRICS + RAW_METRICS + TOTALS}
    rows = rows.first_slots(rows.active())
    return _metrics_on_device(params, _device_chunks(rows, chunk), len(rows), kappa)


def errors(params, rows: Rows, batch=512, kappa=0.0):
    """Per spot: range-weighted mean |EV error| over present players (pot units), and the max
    over in-range hands (weight > 1e-3 / #valid) of the projected critic: row_metrics' "mae" and
    "worst" (the raw network's: "raw_mae", "raw_worst")."""
    m = row_metrics(params, rows, batch, kappa)
    return m["mae"], m["worst"]


def checkdown_errors(rows: Rows):
    """errors() of the baseline V = CD (no network): what the network has to learn."""
    rows = rows.first_slots(rows.active())
    err = np.abs(rows.f - rows.target) * rows.valid[:, None, :]
    per_player = (rows.ranges * err).sum(-1) / np.maximum(rows.ranges.sum(-1), 1e-12)
    return (per_player * rows.present).sum(-1) / np.maximum(rows.present.sum(-1), 1)


# The one training protocol of every critic and auxiliary critic: a step
# budget, not passes over the rows - epochs of 25 batches of 256, 400 of them (10,000 steps), Adam
# at lr 5e-4 with the cosine decay over all of them, the gradient clipped to global norm 0.05, no
# early stop: the whole budget runs and the parameters of the epoch with the best validation MAE
# (solved rows only, loop.critic_split) are kept; loop.train_best trains the fixed seeds 0 and 1,
# with no substitute seeds on failure. The learning curves (poker.solver.learning_curve) measure
# the critics with exactly this protocol.
# Why lr 5e-4 and clip 0.05 - the protocol comparison of 2026-10-10 (the code of d3f48ee,
# the projected loss at the river kappa, the
# 45k HU river rows, seeds 0-4, 77-103 s per run on the 1070). lr 1e-3 / clip 1.0 (the protocol
# before) collapsed in 1 of 5 runs (seed 2: one gradient spike of 0.08 at epoch 120, then the
# validation MAE back at the untrained network's 0.113, the loss at 2.5-3.1x its running minimum
# and the gradient norms at 0.0005, a tenth of normal), and its 4 finished runs gave test MAE
# 0.0440 on average (0.0429-0.0458); lr 3e-4 / clip 1.0: 0 of 5, 0.0450; lr 1e-3 / clip 0.02: 0 of
# 5, 0.0432; lr 5e-4 / clip 0.05: 0 of 5, 0.0426 (0.0423-0.0431, the smallest spread), collapse-free
# and the best mean of the four (seeds 0 / 1, the pair train_best trains: 0.0428 / 0.0431); its
# seeds 5-9, run after the choice: 0 of 5 again, 0.0427 (0.0422-0.0432). The clip acts
# on spikes only: 0.04-0.12% of the steps are clipped (the median of the per-epoch gradient-norm
# maxima is 0.006, the largest norm 0.16; the largest loss spike, 1.85x the running minimum at
# epoch 60 of seed 6, had 4 clipped steps and was back under 1.21x the next epoch). A collapse is a
# chance event of a run, not of a seed: the GPU trajectories of one seed differ from epoch 0
# (reductions are not bitwise reproducible), and seed 2 of lr 1e-3 / clip 1.0 run again collapsed
# at epoch 84 in the repeat run. Ten collapse-free runs bound the rate only loosely (below 26% at
# 95% confidence). A finite loss rise alone does not establish a collapse.
# History. Before af5295a the protocol stopped early after 25 epochs without a better validation
# MAE and collapsed at lr 1e-3 in the training measurements of 2026-10-10:
# on the 45k HU river rows (seed 0) the training loss
# fell to 0.0036 by epoch 56, jumped to 0.0067 at epoch 57 and stayed at the level of an untrained
# network (0.0075-0.0081, initial 0.0083); the early stop then kept the epoch-56 iterate at lr
# 9.5e-4 of a cosine that never annealed, test MAE 0.0717 (lr 3e-4 or clip 1.0 alone on that seed:
# 0.0445). Every k0 street critic stopped at lr 5e-4 - 9.5e-4 and no k0/k0b learning-curve run
# reached the cap. A validation half made of SPR-0 copies (fitted to ~0.001) read half the real
# error. af5295a ran the whole annealed budget at lr 1e-3 with clip 1.0 instead (raw loss): test MAE
# 0.0443 / 0.0438 on seeds 0 / 1 (recorded training comparison; best epochs
# 381 / 385 at lr 5e-6 / 3e-6), one recovering loss spike each (1.6x / 2.3x the running minimum at
# epochs 109 / 134), and the clip never acted (largest gradient norm 0.050 / 0.074, 13x below 1.0).
# Two seeds were a lucky draw: on the same rows the zero-sum fix (d3f48ee) saw that protocol
# collapse in 5 of 7 raw-loss runs and 1 of 4 projected ones in the protocol comparison,
# and one collapse stopped a whole loop run (train_best had no retry).
PROTOCOL = {"epochs": 400, "epoch_steps": 25, "patience": None, "lr": 5e-4, "clip_norm": 0.05}
# Finite loss spikes are diagnostic, not a convergence test: each epoch samples new batches,
# and their running minimum is biased downwards. The old 1.5x / 3-epoch stop falsely rejected
# e358b35 pre_end seed 0, which then completed the unchanged 400-epoch budget with finite weights
# and validation MAE 0.003923 (best epoch 360). Keep that threshold for a recorded warning when
# a run first reaches three consecutive spike epochs; do not stop or substitute a seed. The
# largest ratio and number of spike epochs retain their reporting meaning. Non-finite loss,
# unclipped gradient norm, updated parameters or validation MAE are hard failures.
LOSS_SPIKE_RATIO, LOSS_SPIKE_EPOCHS = 1.5, 3


class TrainingDiverged(RuntimeError):
    """train() stopped on a non-finite numerical signal; .history = the epochs run."""

    def __init__(self, message, history):
        super().__init__(message)
        self.history = history


def train(rows: Rows, val: Rows, hidden=(512, 512, 512), seed=0, lr=PROTOCOL["lr"], batch=256,
          epochs=PROTOCOL["epochs"], patience=PROTOCOL["patience"], augment=True, log=None,
          epoch_steps=PROTOCOL["epoch_steps"], clip_norm=PROTOCOL["clip_norm"], kappa=0.0):
    """Adam + cosine decay over the step budget epochs * epoch_steps, the gradient clipped to global
    norm `clip_norm` first (None or 0: not clipped), Huber loss with loss_weights of the projected
    values (_batch_loss: chip conservation with the critic's kappa, critic_kappa(name); None = the
    raw network, the protocol before the zero-sum fix: comparisons only); an epoch is
    `epoch_steps` batches (passes over the rows repeat as needed; None = one pass). After every
    epoch, non-finite training loss, gradient norm or updated parameters raise TrainingDiverged.
    Validation MAE on `val` (the caller's choice: solved rows only, loop.critic_split) is checked
    for finiteness too (row_metrics "mae" with this kappa; kappa None: "raw_mae"). Finite loss
    spikes only produce recorded warnings (LOSS_SPIKE_RATIO, LOSS_SPIKE_EPOCHS). Returns the
    parameters of the epoch with the best validation MAE (without validation rows, toy runs: the
    last epoch) and the training record. patience=None runs the whole budget (PROTOCOL); an integer
    stops after that many epochs without a better validation MAE, the cosine then cut hot (audit
    F2). One jitted scan per epoch (gather, augmentation and update on the device). Glorot limits
    by the slots the rows use (init_critic)."""
    for name, part in (("rows", rows), ("val", val)):
        live = (np.asarray(part.present) > 0).sum(axis=1)
        for i in np.flatnonzero(live >= 4)[:1]:
            raise NotImplementedError(f"critics.train: {name}[{i}] has {int(live[i])} live slots; the chip-conserving "
                                      f"projection (deal_weights, opponent_mass) is exact for 2 and 3 and would pass "
                                      f"4+ unchanged on a NaN mass (task #35, edge_cases B7)")
    select = "mae" if kappa is not None else "raw_mae"
    key = jax.random.key(seed)
    k = max(rows.active(), val.active())
    params = init_critic(key, hidden, k)
    rows, val = rows.first_slots(k), val.first_slots(k)              # zero slots: no input, no loss
    bsz = min(batch, len(rows))
    steps_per_epoch = epoch_steps or max(1, len(rows) // bsz)
    schedule = optax.cosine_decay_schedule(lr, steps_per_epoch * epochs)
    opt = optax.adam(schedule)
    if clip_norm:
        opt = optax.chain(optax.clip_by_global_norm(clip_norm), opt)
    state = opt.init(params)

    @jax.jit
    def run_epoch(params, state, dev, keys, order):
        def body(carry, x):
            params, state = carry
            sub, idx = x
            b, p, r, t, v, s, f = (dev[name][idx] for name in FIELDS)
            if augment:
                b, r, t, v, f = _augment(sub, b, r, t, v, f)
            loss, grad = jax.value_and_grad(_batch_loss)(params, b, p, r, t, v, s, f, kappa)
            upd, state = opt.update(grad, state, params)
            return (optax.apply_updates(params, upd), state), (loss, optax.global_norm(grad))
        (params, state), (losses, norms) = jax.lax.scan(body, (params, state), (keys, order))
        params_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(params)]))
        return params, state, losses, norms, params_finite

    dev = {name: jnp.asarray(getattr(rows, name)) for name in FIELDS}
    dval = _device_chunks(val) if len(val) else None
    best, best_params, best_epoch, bad, history = np.inf, params, None, 0, []
    low, above, warnings = np.inf, 0, []      # diagnostics: previous minimum, consecutive spikes, warning episodes

    def stop_nonfinite(signal, record):
        raise TrainingDiverged(
            f"critic training diverged (seed {seed}, epoch {record['epoch']}, lr {record['lr']:.2e}, "
            f"clip_norm {clip_norm}): non-finite {signal}; epoch losses so far "
            f"{[round(x['train_loss'], 6) for x in history[-8:]]}", history)

    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        passes = -(-steps_per_epoch * bsz // len(rows))
        order = np.concatenate([rng.permutation(len(rows)) for _ in range(passes)])[:steps_per_epoch * bsz]
        key, sub = jax.random.split(key)
        params, state, losses, norms, params_finite = run_epoch(
            params, state, dev, jax.random.split(sub, steps_per_epoch),
            jnp.asarray(order.reshape(steps_per_epoch, bsz)))
        losses, norms = np.asarray(losses), np.asarray(norms)
        loss = float(np.mean(losses))
        spike = bool(np.isfinite(low) and loss > LOSS_SPIKE_RATIO * low)
        h = {"epoch": epoch, "train_loss": loss, "val_mae": None,
             "lr": float(schedule(steps_per_epoch * (epoch + 1) - 1)),       # of the epoch's last step
             "loss_ratio": loss / low if np.isfinite(low) and low > 0 else None,
             "loss_spike": spike, "grad_norm_max": float(norms.max()),
             "clipped_steps": int(np.sum(norms > clip_norm)) if clip_norm else 0}
        history.append(h)
        if not np.isfinite(loss) or not np.isfinite(losses).all():
            stop_nonfinite("training loss", h)
        if not np.isfinite(norms).all():
            stop_nonfinite("gradient norm", h)
        if not bool(params_finite):
            stop_nonfinite("updated parameters", h)
        above = above + 1 if spike else 0
        low = min(low, loss)
        mae = None
        if dval is not None:
            mae = float(np.mean(_metrics_on_device(params, dval, len(val), 0.0 if kappa is None else kappa)[select]))
            h["val_mae"] = mae
            if not np.isfinite(mae):
                stop_nonfinite("validation MAE", h)
            if log is not None:
                log(f"  critic epoch {epoch}: loss {loss:.5f} val MAE {mae:.5f} lr {h['lr']:.2e}")
        if above == LOSS_SPIKE_EPOCHS:
            warnings.append({"kind": "loss_spike", "epoch": epoch, "train_loss": loss, "running_min": low,
                             "loss_ratio": h["loss_ratio"], "consecutive_epochs": above, "val_mae": mae})
            message = (f"critic seed {seed}, epoch {epoch}: finite training loss {loss:.6f} has stayed "
                       f"above {LOSS_SPIKE_RATIO} x its running minimum {low:.6f} for {above} epochs; "
                       "continuing the fixed training budget")
            if log is not None:
                log(f"WARNING: {message}")
            else:
                logging.getLogger(__name__).warning(message)
        if dval is None:                             # nothing to select on (toy runs): keep the last
            best_params, best_epoch = params, epoch
            continue
        if mae < best - 1e-6:
            best, best_params, best_epoch, bad = mae, params, epoch, 0
        else:
            bad += 1
            if patience is not None and bad >= patience:
                break
    ratios = [x["loss_ratio"] for x in history if x["loss_ratio"] is not None]
    steps = len(history) * steps_per_epoch
    return best_params, {
        "history": history, "val_mae": best if np.isfinite(best) else None, "best_epoch": best_epoch,
        "epochs_run": len(history), "steps": steps, "lr": lr,
        "lr_at_best": history[best_epoch]["lr"] if best_epoch is not None else None,
        "lr_final": history[-1]["lr"], "clip_norm": clip_norm or None, "early_stop": patience,
        "stopped_early": len(history) < epochs,
        "clipped_fraction": sum(x["clipped_steps"] for x in history) / steps,
        "loss_ratio_max": max(ratios) if ratios else None,
        "loss_spike_epochs": int(sum(x["loss_spike"] for x in history)), "warnings": warnings,
        "kappa": kappa, "projected": kappa is not None}


# ------------------------------------------------------------------ chance nodes with card removal

def opponent_mass(ranges, present, slot):
    """[1326] compatible mass of the other present slots for every hand of `slot` (board-free:
    only hand-hand card removal; blocked hands must already be zero in the ranges). ranges
    [k, 1326], present [k]. Exact for one or two opponents; three or more give NaN (the exact
    mass for 4+ live players is not implemented)."""
    def one(w):
        c = sd.card_sums(w)
        return jnp.sum(w) - c[HANDS[:, 0]] - c[HANDS[:, 1]] + w

    others = [j for j in range(ranges.shape[0]) if j != slot]
    if len(others) == 1:
        return one(ranges[others[0]] * present[others[0]])
    on = present[jnp.asarray(others)] > 0
    first = jnp.asarray(others)[jnp.argsort(jnp.where(on, 0, 1), stable=True)]   # present first
    wa, wb = ranges[first[0]] * present[first[0]], ranges[first[1]] * present[first[1]]
    count = jnp.sum(on)
    return jnp.where(count <= 1, one(wa), jnp.where(count == 2, sd.mass_three(wa, wb), jnp.nan))


def _normalise(r):
    return r / jnp.maximum(jnp.sum(r, axis=-1, keepdims=True), 1e-30)


def _next_card_average(critic, boards, pub, ranges, shares, mass, net):
    """The next street's critic on every next card c (two players): boards int32 [52, 5] (row c =
    the board with c dealt; rows of cards on the board are masked by mass = 0), ranges [2, H] of the
    situation, shares and card-removal weights [52, 2, H] (checkdown.river_cards / turn_cards).
    The critic sees the ranges with the hands holding c removed and F = shares * net. Returns the
    card averages of its network part V - F and of F [2, H]; by the identity the second equals the
    situation's own check-down value."""
    keep = ~jnp.asarray(MEMBER)                                                       # [52, H]
    r_c = _normalise(ranges[None] * keep[:, None, :])
    f = shares * net
    v = critic(jax.vmap(board_onehot)(boards), jnp.broadcast_to(pub, (52, pub.shape[-1])), r_c, f)
    return CD.card_average(v - f, mass), CD.card_average(f, mass)


def end_of_turn(river_critic, board4, pub, ranges, net, table):
    """The chance node of the river for one situation at the end of the turn (two players): board4
    int32 [4], pub [PUB], ranges [2, H] (normalised, zero on board-blocked hands), net = the pot's
    payout share. Averages over the river cards (card removal between the card, the hero hand and
    the opponent's hands) of the river critic's network part and of its F."""
    shares, mass = CD.river_cards(board4, ranges, table)
    boards = jnp.concatenate([jnp.broadcast_to(board4, (52, 4)), jnp.arange(52, dtype=board4.dtype)[:, None]], axis=1)
    return _next_card_average(river_critic, boards, pub, ranges, shares, mass, net)


def end_of_flop(turn_critic, board3, pub, ranges, net, table):
    """The chance node of the turn at the end of the flop: as end_of_turn with the turn critic and
    F = the turn check-down of every turn board (checkdown.turn_cards)."""
    shares, mass = CD.turn_cards(board3, ranges, table)
    cards = jnp.arange(52, dtype=board3.dtype)[:, None]
    boards = jnp.concatenate([jnp.broadcast_to(board3, (52, 3)), cards, jnp.full((52, 1), -1, board3.dtype)], axis=1)
    return _next_card_average(turn_critic, boards, pub, ranges, shares, mass, net)


@jax.jit
def _preflop_class_step(acc, flop, weight, mv, critic, pub, ranges, net):
    """One flop class of end_of_preflop_classes for every situation (arguments, not constants:
    the critic and the batch are not baked into the program)."""
    member = jnp.asarray(MEMBER)
    keep = ~(member[flop[0]] | member[flop[1]] | member[flop[2]])
    w = ranges * keep
    opp = w[:, ::-1]                                                                  # [S, 2, H]
    mass = jax.vmap(jax.vmap(CD.j_mass))(opp)
    f = CD.ratio(mv(opp), mass) * keep * net[:, None, None]
    board = jnp.concatenate([flop, jnp.full(2, -1, flop.dtype)])
    board1h = jnp.broadcast_to(board_onehot(board), (w.shape[0], 52))
    v = critic(board1h, pub, _normalise(w), f)
    m = jnp.where(keep, mass, 0.0) * weight
    return acc + jnp.stack([m * (v - f), m * f, m])


def end_of_preflop_classes(flop_critic, pub, ranges, net, matvec_for, log=None):
    """The chance node of the flop for a batch of S situations at the flop start (two players,
    ranges [S, 2, H] suit-symmetric: one weight per 169-class), over all 22,100 flops through the
    1755 suit-isomorphism classes. For flop f: w_f = the ranges with the hands touching f removed,
    F_f = (W_f w_f,opp) / J(w_f,opp) * net (W_f = the flop's exact all-in matrix; matvec_for(f) is a
    pytree callable x [..., H] -> W_f x [..., H]), m_f = J(w_f,opp), V_f = critic(f, w_f, F_f).
    Exact for symmetric ranges and an equivariant critic: a class sums to (size / 24) times the
    sum over the 24 suit permutations s of the representative's terms at s(h), so
        avg(h) = sum_s A(s(h)) / sum_s B(s(h)),  A = sum_classes size/24 * m_f * X_f,  B = ... m_f.
    Returns the averages of the network part V_f - F_f and of F_f [S, 2, H] (the second is the
    preflop check-down value: the identity)."""
    acc = jnp.zeros((3,) + ranges.shape, ranges.dtype)
    weights = FL.SIZES / 24.0
    for i, (flop, weight) in enumerate(zip(FL.FLOPS, weights)):
        acc = _preflop_class_step(acc, jnp.asarray(flop, jnp.int32), jnp.asarray(weight, ranges.dtype),
                                  matvec_for(flop), flop_critic, pub, ranges, net)
        if log is not None and i % 500 == 0:
            log(f"  end of preflop: flop class {i}/{len(FL.FLOPS)}")
    sym = jnp.zeros_like(acc)
    for perm in jnp.asarray(HAND_PERM):
        sym = sym + acc[..., perm]
    den = sym[2]
    ok = den > 1e-30
    safe = jnp.where(ok, den, 1.0)
    return jnp.where(ok, sym[0] / safe, 0.0), jnp.where(ok, sym[1] / safe, 0.0)


def config_payload():
    """What a saved critic must match when loaded (with config.check_scheme)."""
    return {"scheme": config.scheme(), "p_max": P_MAX, "pub": PUB, "form": "residual_checkdown_f"}


def load_critic(path: Path, *, name=None):
    """Load a critic of this exact game. Missing provenance is not an implicit legacy mode."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        scheme = meta.get("scheme")
        if not isinstance(scheme, dict) or scheme.get("regime") != config.regime_record():
            raise ValueError(f"{path}: explicit matching table regime required; train a new checkpoint")
        config.check_scheme(scheme, str(path))
        config.check_game(meta, str(path))
        if meta.get("game_hash") != config.game_hash():
            raise ValueError(f"{path}: an exact current game record is required")
        from poker.solver.situations import training_domain_record
        if meta.get("training_domain") != training_domain_record():
            raise ValueError(f"{path}: missing or different training domain; train a new checkpoint")
        want = config_payload()
        for key in ("p_max", "pub", "form"):
            if meta.get(key) != want[key]:
                raise ValueError(f"{path}: critic {key} {meta.get(key)!r} != {want[key]!r}")
        if name is not None and meta.get("name") != name:
            raise ValueError(f"{path}: expected {name}, recorded {meta.get('name')!r}")
        if "kappa" not in meta or not np.isfinite(meta["kappa"]) or meta["kappa"] < 0:
            raise ValueError(f"{path}: explicit finite nonnegative projection kappa required")
        count = sum(key.endswith("_w") for key in data.files)
        if count < 2:
            raise ValueError(f"{path}: critic requires at least one hidden layer and an output layer")
        layers = [{key.split("_", 1)[1]: np.asarray(data[key]) for key in data.files
                   if key.startswith(f"{i}_")} for i in range(count)]
        width = D_IN
        for i, layer in enumerate(layers):
            expected = {"w", "b", "wf"} if i == 0 else {"w", "b"}
            if set(layer) != expected or layer["w"].ndim != 2 or layer["w"].shape[0] != width:
                raise ValueError(f"{path}: invalid critic layer {i}")
            width = layer["w"].shape[1]
            if (width <= 0 or layer["b"].shape != (width,)
                    or (i == 0 and layer["wf"].shape != (D_F, width))
                    or not all(np.isfinite(value).all() for value in layer.values())):
                raise ValueError(f"{path}: invalid shape or nonfinite weights in layer {i}")
        if width != D_OUT:
            raise ValueError(f"{path}: critic output {width} != {D_OUT}")
        params = [{key: jnp.asarray(value) for key, value in layer.items()} for layer in layers]
    return params, meta


def load_live_critics(directory):
    """The live player's three boundary critics, using their recorded projection parameters."""
    result = {}
    for street, name in enumerate(("pre_end", "flop_end", "turn_end")):
        params, meta = load_critic(Path(directory) / f"{name}.npz", name=name)
        result[street] = net_critic(params, meta["kappa"])
    return result
