"""Terminal exactness of the solver against independent brute force (audit, read-only w.r.t. poker/).

A  evaluator: engine ranks (cards.hand_ranks) vs an independent evaluator (indep.eval7) - same order on random boards.
B  two players, river: pcfr._two (fold both ways, showdown equal and unequal contributions) and pcfr.j_mass,
   showdown.apply_all (A, E, J) vs dense O(F^2) sums over explicit card-disjoint pairs.
C  turn all-in: runout.equity_matrix_turn (evaluator path, table=None) and pcfr.allin_values vs explicit
   enumeration of the 44 rivers per pair with the independent evaluator; checkdown identity
   (card_average of river_cards == share_matrix of the turn matrix).
D  three players, river: pcfr._three for every live pattern, showdown.mass_three and couple_live_folded
   vs brute force over all card-disjoint triples (opponent ranges restricted to 200-hand supports);
   plus a side-pot case (unequal live contributions) against the true side-pot payoff.

Usage: python3 tools/audit/solver/terminals.py [--x64] --out results/audit/solver/terminals_f32.json
"""
import argparse, json, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--x64", action="store_true")
p.add_argument("--out", required=True)
p.add_argument("--boards", type=int, default=4)
args = p.parse_args()

import jax
if args.x64:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indep
from poker.solver import cards as C, pcfr as SV, showdown as sd, runout as RO, checkdown as CD
from poker.solver import situations as SI

DT = jnp.float64 if args.x64 else jnp.float32
rng = np.random.default_rng(2026)
DIS, MEM = indep.disjoint()                     # [F, F] bool, [F, 52]
assert np.array_equal(indep.HANDS, C.HANDS)     # same hand order (indexing convention only)
out = {"dtype": str(np.dtype(DT)), "sections": {}}


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / scale), float(np.max(np.abs(a - b)))


def board(k):
    return rng.choice(52, k, replace=False).astype(np.int64)


def dense_rank(x):
    return np.unique(x, return_inverse=True)[1]


# ---------------- A: evaluator
t0 = time.time()
mism = 0
nb = 30
for _ in range(nb):
    b = board(5)
    eng = np.asarray(C.hand_ranks(jnp.asarray(b, jnp.int32)))
    ind = indep.ranks_on_board(b)
    if not np.array_equal(eng < 0, ind < 0) or not np.array_equal(dense_rank(eng[eng >= 0]), dense_rank(ind[ind >= 0])):
        mism += 1
out["sections"]["A_evaluator"] = {"boards": nb, "boards_with_order_mismatch": mism, "seconds": time.time() - t0}
print("A", out["sections"]["A_evaluator"], flush=True)


def ranges_for(valid, kind):
    if kind == "situations":
        return SI.random_range(rng, valid, 0.2)
    if kind == "dense":
        w = rng.random(indep.F) * valid
    else:                                       # sparse 60 hands
        w = np.zeros(indep.F)
        idx = rng.choice(np.flatnonzero(valid), 60, replace=False)
        w[idx] = rng.random(60) + 0.1
    return w / w.sum()


# ---------------- B: two players on the river
t0 = time.time()
rows = []
worst = {"rel": 0.0, "abs": 0.0}
for bi in range(args.boards):
    b = board(5)
    ranks_i = indep.ranks_on_board(b)
    valid = ranks_i >= 0
    tables = C.rank_tables(C.hand_ranks(jnp.asarray(b, jnp.int32)))
    S = indep.score_matrix(ranks_i)                               # [F, F]
    Dv = DIS & valid[:, None] & valid[None, :]
    ok = valid
    for kind in ("situations", "dense", "sparse"):
        w = np.stack([ranges_for(valid, kind), ranges_for(valid, kind)])
        reach = jnp.asarray(w, DT)
        # apply_all components vs brute force
        A_bf = ((ranks_i[:, None] > ranks_i[None, :]) & Dv) @ w[1]
        E_bf = ((ranks_i[:, None] == ranks_i[None, :]) & Dv) @ w[1]
        J_bf = Dv @ w[1]
        A, E, J = [np.asarray(x) for x in sd.apply_all(reach[1], tables)]
        jm_tab = np.asarray(SV.j_mass(reach[1], tables))
        jm_free = np.asarray(SV.j_mass(reach[1]))
        comp = {"apply_A": rel(A[ok], A_bf[ok]), "apply_E": rel(E[ok], E_bf[ok]), "apply_J": rel(J[ok], J_bf[ok]),
                "j_mass_tables": rel(jm_tab[ok], J_bf[ok]), "j_mass_free": rel(jm_free[ok], J_bf[ok])}
        cases = {
            "showdown_equal": ((1.0, 1.0), 2.0, (True, True)),
            "showdown_unequal": ((0.5 + 3.0, 0.5 + 1.2), 0.5 + 3.0 + 0.5 + 1.2, (True, True)),
            "fold_p0_folded": ((0.5, 1.25), 1.75, (False, True)),
            "fold_p1_folded": ((2.5, 0.5), 3.0, (True, False)),
        }
        for name, (c, pot, live) in cases.items():
            got = np.asarray(SV._two(reach, jnp.asarray(c, DT), jnp.asarray(pot, DT), tables, live))
            bf = []
            for i in range(2):
                wo = w[1 - i]
                if all(live):
                    matched = min(c)
                    contested = pot - abs(c[0] - c[1])
                    Si = S if i == 0 else S             # S[h, o] from hero's view (hero row)
                    bf.append((contested * (S * Dv) - matched * Dv) @ wo)
                else:
                    bf.append(((pot if live[i] else 0.0) - c[i]) * (Dv @ wo))
            bf = np.stack(bf)
            comp[name] = rel(got[:, ok], bf[:, ok])
        for k, (r, a) in comp.items():
            worst["rel"] = max(worst["rel"], r); worst["abs"] = max(worst["abs"], a)
        rows.append({"board": b.tolist(), "ranges": kind, **{k: v[0] for k, v in comp.items()}})
out["sections"]["B_two_player_river"] = {"cases": len(rows), "max_rel_err": worst["rel"], "max_abs_err": worst["abs"],
                                          "rows": rows, "seconds": time.time() - t0}
print("B", {k: v for k, v in out["sections"]["B_two_player_river"].items() if k != "rows"}, flush=True)

# ---------------- C: turn all-in (44 rivers) and check-down identity
t0 = time.time()
rowsC = []
for bi in range(3):
    b4 = board(4)
    t1 = time.time()
    W = np.asarray(RO.equity_matrix_turn(jnp.asarray(b4, jnp.int32), None), np.float64)
    t_eng = time.time() - t1
    total = np.zeros((indep.F, indep.F))
    rivers = [c for c in range(52) if c not in b4]
    for c in rivers:
        r = indep.ranks_on_board(np.append(b4, c))
        total += indep.score_matrix(r)                           # pairs with a hand holding c score 0 (rank -1)
    valid4 = ~np.isin(indep.HANDS, b4).any(axis=1)
    D4 = DIS & valid4[:, None] & valid4[None, :]
    # count rivers per pair explicitly (must be 44 for every disjoint valid pair)
    W_bf = np.where(D4, total / 44.0, 0.0)
    w = np.stack([ranges_for(valid4, "situations"), ranges_for(valid4, "situations")])
    contrib = np.array([[3.4, 2.1]])
    pot = np.array([5.5])
    got = np.asarray(SV.allin_values(jnp.asarray(w[None], DT), jnp.asarray(contrib, DT), jnp.asarray(pot, DT),
                                     jnp.asarray(W, DT), None))[0]
    bf = []
    for i in range(2):
        matched, contested = contrib[0].min(), pot[0] - abs(contrib[0, 0] - contrib[0, 1])
        bf.append((contested * W_bf - matched * D4) @ w[1 - i])
    bf = np.stack(bf)
    # checkdown identity: card-removal average of river shares == turn-matrix share
    shares, mass = CD.river_cards(jnp.asarray(b4, jnp.int32), jnp.asarray(w, DT), None)
    avg = np.asarray(CD.card_average(shares, mass))
    direct = np.stack([np.asarray(CD.share_matrix(jnp.asarray(w[1], DT), jnp.asarray(W, DT))),
                       np.asarray(CD.share_matrix(jnp.asarray(w[0], DT), jnp.asarray(W, DT)))])
    # the spec's unnormalised form: (1/44) * sum_c cfv_c(h), cfv_c = share_c * J_c
    unnorm = np.asarray(jnp.sum(shares * mass, axis=0)) / 44.0
    jm = np.stack([Dv_ @ w[1 - i] for i, Dv_ in enumerate([D4, D4])])
    spec_form = np.where(jm > 0, unnorm / np.where(jm > 0, jm, 1), 0)
    okh = valid4 & (jm[0] > 0)
    rowsC.append({"board4": b4.tolist(), "equity_matrix_max_abs_err": float(np.max(np.abs(W - W_bf))),
                  "allin_values": rel(got[:, valid4], bf[:, valid4]),
                  "checkdown_identity_max_abs": float(np.max(np.abs(avg - direct)[:, okh])),
                  "spec_1_over_44_form_vs_direct_max_abs": float(np.max(np.abs(spec_form - direct)[:, okh])),
                  "engine_matrix_seconds_incl_compile": t_eng})
out["sections"]["C_turn_allin"] = {"rows": rowsC, "seconds": time.time() - t0}
print("C", rowsC, flush=True)

# ---------------- D: three players on the river
t0 = time.time()


def side_pot_payoff(contrib, live, ranks3):
    """True payoff of each of 3 players (chips back minus chips in) for one deal; contrib whole-hand chips."""
    c = np.asarray(contrib, float)
    levels = sorted(set(c[np.asarray(live)]))
    pay = -c.copy()
    prev = 0.0
    for L in levels:
        layer = np.sum(np.minimum(c, L) - np.minimum(c, prev))
        elig = [j for j in range(3) if live[j] and c[j] >= L]
        best = max(ranks3[j] for j in elig)
        win = [j for j in elig if ranks3[j] == best]
        for j in win:
            pay[j] += layer / len(win)
        prev = L
    # chips above the largest live level (folded overcontribution) cannot occur here
    return pay


rowsD = []
SUP = 200
for bi in range(2):
    b = board(5)
    ranks_i = indep.ranks_on_board(b)
    valid = ranks_i >= 0
    tables = C.rank_tables(C.hand_ranks(jnp.asarray(b, jnp.int32)))
    w = np.zeros((3, indep.F))
    sup = []
    for j in range(3):
        idx = np.sort(rng.choice(np.flatnonzero(valid), SUP, replace=False))
        w[j, idx] = rng.random(SUP) + 0.1
        w[j] /= w[j].sum()
        sup.append(idx)
    reach = jnp.asarray(w, DT)
    heroes = np.flatnonzero(valid)
    patterns = [(True, True, True), (True, True, False), (True, False, True), (False, True, True),
                (True, False, False), (False, True, False), (False, False, True)]
    for live in patterns:
        contrib = np.array([2.0 if live[j] else 0.7 + 0.3 * j for j in range(3)])
        pot = float(contrib.sum())
        got = np.asarray(SV._three(reach, jnp.asarray(contrib, DT), jnp.asarray(pot, DT), tables, live))
        bf = np.zeros((3, indep.F))
        mass_bf = np.zeros((3, indep.F))
        for i in range(3):
            bb, cc = [j for j in range(3) if j != i]
            Ob, Oc = sup[bb], sup[cc]
            Dbc = DIS[np.ix_(Ob, Oc)]
            wbc = w[bb, Ob][:, None] * w[cc, Oc][None, :] * Dbc
            rb, rc = ranks_i[Ob][:, None], ranks_i[Oc][None, :]
            for h in heroes:
                m = wbc * DIS[h, Ob][:, None] * DIS[h, Oc][None, :]
                tot = m.sum()
                mass_bf[i, h] = tot
                if not live[i]:
                    bf[i, h] = -contrib[i] * tot
                    continue
                rh = ranks_i[h]
                lb, lc = live[bb], live[cc]
                best = np.full(m.shape, rh)
                if lb: best = np.maximum(best, rb)
                if lc: best = np.maximum(best, rc)
                nwin = (rh == best) * 1.0
                cnt = 1.0 + (lb * (rb == best)) + (lc * (rc == best))
                share = np.where(rh == best, 1.0 / cnt, 0.0)
                bf[i, h] = np.sum(m * (pot * share - contrib[i]))
        okh = valid
        mass_eng = np.stack([np.asarray(sd.mass_three(reach[1], reach[2])), np.asarray(sd.mass_three(reach[0], reach[2])),
                             np.asarray(sd.mass_three(reach[0], reach[1]))])
        rowsD.append({"board": b.tolist(), "live": live, "contrib": contrib.tolist(),
                      "three_values": rel(got[:, okh], bf[:, okh]), "mass_three": rel(mass_eng[:, okh], mass_bf[:, okh])})
        print("D", rowsD[-1], flush=True)
    # side-pot demonstration: player 2 all-in short (1.0), players 0 and 1 put in 3.0 each (live, equal stacks impossible)
    contrib = np.array([3.0, 3.0, 1.0]); live = (True, True, True); pot = float(contrib.sum())
    got = np.asarray(SV._three(reach, jnp.asarray(contrib, DT), jnp.asarray(pot, DT), tables, live))
    bf = np.zeros((3, indep.F))
    sample = rng.choice(heroes, 40, replace=False)
    for i in range(3):
        bb, cc = [j for j in range(3) if j != i]
        for h in sample:
            tot = 0.0
            for ob in sup[bb]:
                if not DIS[h, ob]: continue
                for oc in sup[cc]:
                    if not (DIS[h, oc] and DIS[ob, oc]): continue
                    r3 = [0, 0, 0]; r3[i], r3[bb], r3[cc] = ranks_i[h], ranks_i[ob], ranks_i[oc]
                    tot += w[bb, ob] * w[cc, oc] * side_pot_payoff(contrib, live, r3)[i]
            bf[i, h] = tot
    rowsD.append({"board": b.tolist(), "side_pot_case": True, "contrib": contrib.tolist(),
                  "three_values_vs_true_side_pots": rel(got[:, sample], bf[:, sample])})
    print("D-side", rowsD[-1], flush=True)
out["sections"]["D_three_player_river"] = {"support_per_player": SUP, "rows": rowsD, "seconds": time.time() - t0}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
print("wrote", args.out)
