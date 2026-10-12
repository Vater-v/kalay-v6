"""(3) Are the auxiliary end-of-street labels exact card averages with the right card-removal weights?

A. Turn -> river (turn_end): engine CD.river_cards + CD.card_average vs an INDEPENDENT brute force
   (own pure-python 7-card evaluator, explicit enumeration over opponent hands and river cards).
B. The weights m_c(h) = J(r_opp minus c)(h) satisfy sum_c m_c(h) = 44 * J(r_opp)(h)  (the spec's
   1/(N_left-4) identity in normalised form); uniform 1/44 or 1/48 weights are measured for contrast.
C. critics.end_of_turn with a synthetic card-dependent "critic" against the explicit weighted average.
D. Flop -> turn (flop_end weights): CD.turn_cards card average vs brute force for a few hero hands
   (engine hand ranks here, independent averaging).
Output: results/audit/training/aux_identity.json
"""
import sys, itertools, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402

import jax, jax.numpy as jnp, numpy as np  # noqa: E402
from poker.solver import checkdown as CD, critics as CR, situations as SI  # noqa: E402
from poker.solver import cards as C  # noqa: E402
from poker.solver.cards import HANDS, NUM_HANDS, MEMBER  # noqa: E402

rng = np.random.default_rng(11)
out = {}


def eval7(cards):
    """Independent 7-card evaluator: tuple, larger is better. card = rank + 13 * suit (rank 0 = deuce)."""
    ranks = [c % 13 for c in cards]; suits = [c // 13 for c in cards]
    best = None
    for five in itertools.combinations(range(7), 5):
        r = sorted((ranks[i] for i in five), reverse=True)
        s = {suits[i] for i in five}
        cnt = {}
        for x in r:
            cnt[x] = cnt.get(x, 0) + 1
        groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        uniq = sorted(set(r), reverse=True)
        straight = None
        if len(uniq) == 5:
            if uniq[0] - uniq[4] == 4:
                straight = uniq[0]
            elif uniq == [12, 3, 2, 1, 0]:
                straight = 3
        flush = len(s) == 1
        if straight is not None and flush:
            v = (8, straight)
        elif groups[0][1] == 4:
            v = (7, groups[0][0], groups[1][0])
        elif groups[0][1] == 3 and groups[1][1] == 2:
            v = (6, groups[0][0], groups[1][0])
        elif flush:
            v = (5, *r)
        elif straight is not None:
            v = (4, straight)
        elif groups[0][1] == 3:
            v = (3, groups[0][0], *[g[0] for g in groups[1:]])
        elif groups[0][1] == 2 and groups[1][1] == 2:
            v = (2, groups[0][0], groups[1][0], groups[2][0])
        elif groups[0][1] == 2:
            v = (1, groups[0][0], *[g[0] for g in groups[1:]])
        else:
            v = (0, *r)
        best = v if best is None or v > best else best
    return best


# ---------------- A/B: turn board
board4 = rng.choice(52, 4, replace=False).astype(np.int32)
valid = C.board_valid(board4[None])[0]
ranges = np.stack([SI.random_range(rng, valid), SI.random_range(rng, valid)]).astype(np.float32)   # [2, H]
t0 = time.perf_counter()
shares, mass = (np.asarray(x) for x in CD.river_cards(jnp.asarray(board4), jnp.asarray(ranges), None))  # [52, 2, H]
avg = np.asarray(CD.card_average(jnp.asarray(shares), jnp.asarray(mass)))                               # [2, H]
t_engine = time.perf_counter() - t0
rivers = [c for c in range(52) if c not in board4]
t0 = time.perf_counter()
rank_of = {}                                     # (river, hand) -> rank tuple
for c in rivers:
    for h in np.flatnonzero(valid):
        a, b = HANDS[h]
        if c in (a, b):
            continue
        rank_of[(c, h)] = eval7([int(a), int(b), *map(int, board4), c])
t_eval = time.perf_counter() - t0
hero_sample = rng.choice(np.flatnonzero(valid), 60, replace=False)
member = MEMBER  # [52, H] bool
diffs, diffs_uniform44, diffs_uniform48, ident = [], [], [], []
for slot in (0, 1):
    opp = ranges[1 - slot]
    for h in hero_sample:
        a, b = HANDS[h]
        num = den = 0.0
        per_card = {}
        for o in np.flatnonzero(opp > 0):
            if set(HANDS[o]) & {a, b}:
                continue
            for c in rivers:
                if c in (a, b) or c in HANDS[o]:
                    continue
                rh, ro = rank_of[(c, h)], rank_of[(c, o)]
                sc = 1.0 if rh > ro else 0.5 if rh == ro else 0.0
                num += opp[o] * sc; den += opp[o]
                pc = per_card.setdefault(c, [0.0, 0.0]); pc[0] += opp[o] * sc; pc[1] += opp[o]
        brute = num / den
        diffs.append(abs(brute - avg[slot, h]))
        cards_ok = [c for c in rivers if c not in (a, b)]
        sh = [shares[c, slot, h] for c in cards_ok]
        diffs_uniform44.append(abs(brute - np.mean(sh)))
        diffs_uniform48.append(abs(brute - np.sum(sh) / 48.0))
        # B: sum_c m_c(h) = 44 * J(h) with J the compatible opponent mass
        J = sum(opp[o] for o in np.flatnonzero(opp > 0) if not (set(HANDS[o]) & {a, b}))
        ident.append(abs(mass[:, slot, h].sum() - 44.0 * J) / max(J, 1e-12))
out["turn_to_river"] = {"board4": board4.tolist(), "hero_hands_checked": int(2 * len(hero_sample)),
                        "max_abs_diff_engine_vs_bruteforce": float(np.max(diffs)),
                        "max_abs_diff_uniform_over_46_cards": float(np.max(diffs_uniform44)),
                        "mean_abs_diff_uniform_over_46_cards": float(np.mean(diffs_uniform44)),
                        "max_abs_diff_divide_by_48": float(np.max(diffs_uniform48)),
                        "max_rel_err_sum_m_c_eq_44J": float(np.max(ident)),
                        "engine_seconds": t_engine, "independent_eval_seconds": t_eval}
print(out["turn_to_river"], flush=True)

# ---------------- C: end_of_turn with a card-dependent synthetic critic
vec = jnp.asarray(rng.normal(0, 1, 52), jnp.float32)
hw = jnp.asarray(rng.normal(0, 1, (2, NUM_HANDS)), jnp.float32)


def synth(board1h, pub, r, f, spot=None):
    # value = F + 0.1 * <board1h, vec> * hw + 0.05 * r * 1326   (depends on the card and on the ranges)
    return f + 0.1 * (board1h @ vec)[:, None, None] * hw[None] + 0.05 * r * NUM_HANDS


critic = jax.tree_util.Partial(synth)
pub = jnp.zeros(CR.PUB, jnp.float32)
res, favg = (np.asarray(x) for x in CR.end_of_turn(critic, jnp.asarray(board4), pub, jnp.asarray(ranges), 1.0, None))
# explicit: weights J(r_opp minus c)(h), ranges with c removed and renormalised
expl = np.zeros((2, NUM_HANDS)); wsum = np.zeros((2, NUM_HANDS))
for c in rivers:
    keep = ~member[c]
    rc = ranges * keep; rc = rc / rc.sum(-1, keepdims=True)
    b1h = np.zeros(52, np.float32); b1h[list(board4) + [c]] = 1
    netpart = 0.1 * float(b1h @ np.asarray(vec)) * np.asarray(hw) + 0.05 * rc * NUM_HANDS
    for slot in (0, 1):
        o = ranges[1 - slot] * keep
        cs = MEMBER.T.astype(np.float64).T @ o if False else (MEMBER.astype(np.float64) @ o)   # [52] card sums
        J = o.sum() - cs[HANDS[:, 0]] - cs[HANDS[:, 1]] + o
        m = np.where(keep & valid, J, 0.0)
        expl[slot] += m * netpart[slot]; wsum[slot] += m
expl = np.where(wsum > 0, expl / np.maximum(wsum, 1e-30), 0)
ok = valid[None] & (wsum > 1e-9)
out["end_of_turn_synthetic"] = {"max_abs_diff_network_part": float(np.max(np.abs(np.where(ok, res - expl, 0)))),
                                "max_abs_diff_F_avg_vs_turn_share": float(np.max(np.abs(np.where(ok, favg - avg, 0))))}
print(out["end_of_turn_synthetic"], flush=True)

# ---------------- D: flop -> turn weights (engine ranks, independent averaging), a few hero hands
board3 = rng.choice(52, 3, replace=False).astype(np.int32)
valid3 = C.board_valid(board3[None])[0]
r3 = np.stack([SI.random_range(rng, valid3), SI.random_range(rng, valid3)]).astype(np.float32)
t0 = time.perf_counter()
tsh, tm = (np.asarray(x) for x in CD.turn_cards(jnp.asarray(board3), jnp.asarray(r3), None))
flop_avg = np.asarray(CD.card_average(jnp.asarray(tsh), jnp.asarray(tm)))
t_turn_cards = time.perf_counter() - t0
rest = [c for c in range(52) if c not in board3]
pairs = [(x, y) for i, x in enumerate(rest) for y in rest[i + 1:]]
rank_fn = jax.jit(jax.vmap(lambda b: C.hand_ranks(b)))
ranks = np.asarray(rank_fn(jnp.asarray([list(board3) + [x, y] for x, y in pairs], jnp.int32)))  # [1176, H]
pidx = {p: i for i, p in enumerate(pairs)}
dd = []
for slot in (0, 1):
    opp = r3[1 - slot]
    for h in rng.choice(np.flatnonzero(valid3), 6, replace=False):
        a, b = HANDS[h]
        num = den = 0.0
        for o in np.flatnonzero(opp > 0):
            if set(HANDS[o]) & {a, b}:
                continue
            blocked = {a, b, *HANDS[o]}
            for (x, y), i in pidx.items():
                if x in blocked or y in blocked:
                    continue
                rh, ro = ranks[i, h], ranks[i, o]
                num += opp[o] * (1.0 if rh > ro else 0.5 if rh == ro else 0.0); den += opp[o]
        dd.append(abs(num / den - flop_avg[slot, h]))
out["flop_to_turn"] = {"board3": board3.tolist(), "hero_hands_checked": 12, "max_abs_diff": float(np.max(dd)),
                       "turn_cards_seconds_no_rank7_table": t_turn_cards}
print(out["flop_to_turn"], flush=True)
common.dump("aux_identity.json", out)
