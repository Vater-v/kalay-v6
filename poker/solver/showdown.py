"""Exact range-vs-range showdown sums with card removal, O(52 * 1326) per weight vector.

For one board, weights w[1326] (zero on blocked hands) and a hero hand h = {h1, h2}:
  A w (h) = sum of w(h') over h' disjoint from h with rank(h') <  rank(h)   (hero beats h')
  E w (h) = sum of w(h') over h' disjoint from h with rank(h') == rank(h)   (tie)
  J w (h) = sum of w(h') over h' disjoint from h                             (compatible mass)
Per-card tables S^X[c, h] = sum over h' that contain card c of X[h, h'] w(h') give the exact
three-player coupling (two opponents must also be disjoint) by inclusion-exclusion:
  F(X, wB; Y, wC)(h) = sum over pairwise disjoint (h, hB, hC) of wB(hB) wC(hC) X[h,hB] Y[h,hC]
                     = (X wB)(Y wC) - sum_c S^X_B[c] S^Y_C[c] + ((X*Y)(wB*wC)).
Every function works on one board; batch with jax.vmap. Values for a blocked hero hand are
meaningless and must be masked by the caller (its range weight is zero).
"""

import jax.numpy as jnp
import numpy as np

from poker.solver.cards import HANDS, MEMBER, NUM_HANDS, PAIR_INDEX

_H1 = HANDS[:, 0]
_H2 = HANDS[:, 1]
_IDX = np.arange(NUM_HANDS)
_PAIR1 = PAIR_INDEX[:, _H1]                       # [52, 1326]: hand {c, h1}
_PAIR2 = PAIR_INDEX[:, _H2]
_OUTSIDE = ~MEMBER                                # card c not in hero hand h


def _prefix(w, tables):
    total = jnp.concatenate((jnp.zeros(1, w.dtype), jnp.cumsum(w[tables["order"]])))
    card = jnp.cumsum(w[tables["card_order"]], axis=1)
    card = jnp.concatenate((jnp.zeros((52, 1), w.dtype), card), axis=1)            # [52, 52]
    return total, card


def apply_all(w, tables):
    """(A w, E w, J w), each [1326]."""
    total, card = _prefix(w, tables)
    klt, kle = tables["klt"], tables["kle"]
    beats = total[tables["lt"]] - card[_H1, klt[_H1, _IDX]] - card[_H2, klt[_H2, _IDX]]
    at_most = total[tables["le"]] - card[_H1, kle[_H1, _IDX]] - card[_H2, kle[_H2, _IDX]] + w
    compatible = total[-1] - card[_H1, -1] - card[_H2, -1] + w
    return beats, at_most - beats, compatible


def per_card(w, tables):
    """(S^A, S^E, S^J), each [52, 1326], zero where card c is in the hero hand."""
    total, card = _prefix(w, tables)
    del total
    padded = jnp.concatenate((w, jnp.zeros(1, w.dtype)))
    ranks = jnp.concatenate((tables["ranks"], jnp.full(1, -2, tables["ranks"].dtype)))
    w1, w2 = padded[_PAIR1], padded[_PAIR2]
    r1, r2, rh = ranks[_PAIR1], ranks[_PAIR2], tables["ranks"][None, :]
    below = jnp.take_along_axis(card, tables["klt"], axis=1) - w1 * (r1 < rh) - w2 * (r2 < rh)
    upto = jnp.take_along_axis(card, tables["kle"], axis=1) - w1 * (r1 <= rh) - w2 * (r2 <= rh)
    every = card[:, -1:] - w1 - w2
    zero = jnp.zeros((), w.dtype)
    return (jnp.where(_OUTSIDE, below, zero), jnp.where(_OUTSIDE, upto - below, zero),
            jnp.where(_OUTSIDE, every, zero))


def couple(x_b, s_x_b, y_c, s_y_c, xy_bc):
    """F(X, wB; Y, wC) from X wB, S^X_B, Y wC, S^Y_C and (X*Y)(wB*wC)."""
    return x_b * y_c - jnp.sum(s_x_b * s_y_c, axis=0) + xy_bc


# --- Fast exact forms for terms with a "compatible only" opponent (J): O(52^3) matmuls ---
#
# With C[c] = sum_{h' containing c} w(h') and P[a, b] = w({a, b}) (zero diagonal), for c not in
# h = {x, y}: S^J[c, h] = C[c] - P[c, x] - P[c, y]. Expanding sum_c S^X_B S^J_C over all c and
# removing c in {x, y} turns every product of per-card tables that involves J into 52x52
# matrix products plus per-hand gathers; no [52, 1326] gather tables are needed.

_MEMBER_F = MEMBER.astype(np.float32)


def card_sums(w):
    return jnp.asarray(_MEMBER_F, w.dtype) @ w


def pair_matrix(w):
    return jnp.concatenate((w, jnp.zeros(1, w.dtype)))[PAIR_INDEX]


def mass_three(w_b, w_c):
    """F(J, wB; J, wC): compatible joint mass of two opponents for every hero hand."""
    x, y = _H1, _H2
    cb, cc = card_sums(w_b), card_sums(w_c)
    pb, pc = pair_matrix(w_b), pair_matrix(w_c)
    jb = jnp.sum(w_b) - cb[x] - cb[y] + w_b
    jc = jnp.sum(w_c) - cc[x] - cc[y] + w_c
    u = w_b * w_c
    cu = card_sums(u)
    ju = jnp.sum(u) - cu[x] - cu[y] + u
    m = pb.T @ pc
    vb, vc = cb @ pc, cc @ pb
    full = (jnp.sum(cb * cc) - vb[x] - vb[y] - vc[x] - vc[y]
            + m[x, x] + m[x, y] + m[y, x] + m[y, y])
    shared = full - (cb[x] - w_b) * (cc[x] - w_c) - (cb[y] - w_b) * (cc[y] - w_c)
    return jb * jc - shared + ju


def _apply_rows(rows, tables):
    """For V[52, 1326]: (A V[x])(h), (E V[x])(h) at x = h1 and x = h2 -> four [1326] arrays."""
    order, card_order = tables["order"], tables["card_order"]
    total = jnp.concatenate((jnp.zeros((52, 1), rows.dtype), jnp.cumsum(rows[:, order], axis=1)), axis=1)
    card = jnp.cumsum(rows[:, card_order], axis=2)                                   # [x, c, 51]
    card = jnp.concatenate((jnp.zeros((52, 52, 1), rows.dtype), card), axis=2)
    lt, le, klt, kle = tables["lt"], tables["le"], tables["klt"], tables["kle"]
    out = []
    for x in (_H1, _H2):
        own = rows[x, _IDX]
        beats = total[x, lt] - card[x, _H1, klt[_H1, _IDX]] - card[x, _H2, klt[_H2, _IDX]]
        upto = total[x, le] - card[x, _H1, kle[_H1, _IDX]] - card[x, _H2, kle[_H2, _IDX]] + own
        out += [beats, upto - beats]
    return out


def couple_live_folded(w_live, w_folded, tables):
    """F(A, wL; J, wF) and F(E, wL; J, wF): beat / tie the live opponent, compatible with the folded."""
    applied = apply_all(w_live, tables)
    j_f = apply_all(w_folded, tables)[2]
    c_f, p_f = card_sums(w_folded), pair_matrix(w_folded)
    u1 = w_live * (c_f[_H1] + c_f[_H2])
    a_u1, e_u1, _ = apply_all(u1, tables)
    a_lf, e_lf, _ = apply_all(w_live * w_folded, tables)
    # v_x(h'') = wL(h'') * (P_F[h''1, x] + P_F[h''2, x]) for every card x: rows [52, 1326].
    v = w_live[None, :] * (p_f[_H1, :] + p_f[_H2, :]).T
    a_x, e_x, a_y, e_y = _apply_rows(v, tables)
    f_a = applied[0] * j_f - a_u1 + a_x + a_y + a_lf
    f_e = applied[1] * j_f - e_u1 + e_x + e_y + e_lf
    return f_a, f_e


def three_player_terms(w_b, w_c, tables):
    """Everything the three-player terminal formulas need for one hero and opponents B, C."""
    a_b, e_b, j_b = apply_all(w_b, tables)
    a_c, e_c, j_c = apply_all(w_c, tables)
    sa_b, se_b, sj_b = per_card(w_b, tables)
    sa_c, se_c, sj_c = per_card(w_c, tables)
    a_bc, e_bc, j_bc = apply_all(w_b * w_c, tables)
    zero = jnp.zeros_like(a_bc)
    f = {
        ("A", "A"): couple(a_b, sa_b, a_c, sa_c, a_bc),
        ("A", "E"): couple(a_b, sa_b, e_c, se_c, zero),
        ("E", "A"): couple(e_b, se_b, a_c, sa_c, zero),
        ("E", "E"): couple(e_b, se_b, e_c, se_c, e_bc),
        ("A", "J"): couple(a_b, sa_b, j_c, sj_c, a_bc),
        ("E", "J"): couple(e_b, se_b, j_c, sj_c, e_bc),
        ("J", "A"): couple(j_b, sj_b, a_c, sa_c, a_bc),
        ("J", "E"): couple(j_b, sj_b, e_c, se_c, e_bc),
        ("J", "J"): couple(j_b, sj_b, j_c, sj_c, j_bc),
    }
    return f
