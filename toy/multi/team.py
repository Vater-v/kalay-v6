"""Hero vs CARD-SHARING coalition of all other seats, as a 2-player zero-sum game
on the same public tree (used by experiments/t3_guarantees.py).

Seat `hero` keeps its fiber of D cards; the other seats form ONE player C whose
private state is the ordered tuple of their cards (fiber D^(N-1), flattened in
increasing seat order).  C chooses the actions of every non-hero seat as a
function of the joint tuple (the colluders see each other's cards).  Payoff of C
is -u_hero (the coalition's chips; transfers inside the coalition cancel).

This gives two certificate-type numbers:
  * security(sigma_hero) = min over C of u_hero  (one best-response pass for C)
      = what sigma_hero is guaranteed to earn against ANY behaviour of the others,
        including colluding, card-sharing ones;
  * coalition maxmin v_hero = max_sigma_hero security(sigma_hero), computed by CFR+
    in the 2p zero-sum game (bracketed by [security(avg hero), BR_hero(avg C)]).
"""
from __future__ import annotations

import numpy as np

from tree import PublicTree, DECISION
from terminal import TerminalEvaluator


class TeamGame:
    def __init__(self, tree: PublicTree, hero: int):
        self.t, self.h = tree, hero
        cfg = tree.cfg
        self.n, self.D = cfg.n_players, cfg.deck
        self.opps = [j for j in range(self.n) if j != hero]
        self.F = self.D ** len(self.opps)
        self.ex = TerminalEvaluator(tree, "exact")
        self.ex._exact_groups(hero)
        idx = np.indices((self.D,) * len(self.opps)).reshape(len(self.opps), -1)  # [m, F]
        self.cards = idx
        valid = np.ones(self.F, bool)
        for a in range(len(self.opps)):
            for b in range(a + 1, len(self.opps)):
                valid &= idx[a] != idx[b]
        self.valid0 = valid.astype(float)
        # boardmask_C[b, f] = 1 if no coalition card equals b
        self.bmaskC = np.stack([(idx != b).all(0).astype(float) for b in range(self.D)])
        owner = tree.edge_owner
        self.isH = owner == hero
        self.isC = (owner >= 0) & (owner != hero)
        self.ch = owner == -1
        dec = tree.dec_edges
        self.decH = dec[tree.dec_owner == hero]
        self.decC = dec[tree.dec_owner != hero]
        self.nd = cfg.n_deals

    # ------------------------------------------------------------------ passes
    def reach(self, sig, who):
        t = self.t
        F = self.D if who == "H" else self.F
        r = np.ones((t.M, F)) if who == "H" else np.tile(self.valid0, (t.M, 1))
        own = self.isH if who == "H" else self.isC
        for lev in t.levels:
            lo, hi = lev["lo"], lev["hi"]
            r[lo:hi] = r[lev["parent"]]
            o = own[lo:hi]
            r[lo:hi][o] *= sig[lo:hi][o]
            cn, cb = lev["chance_nodes"], lev["chance_boards"]
            if who == "H":
                r[cn, cb] = 0.0
            else:
                r[cn] *= self.bmaskC[cb]
        return r

    def term(self, who, rH, rC):
        """Terminal CFVs [T, fiber] of H (hero) or C (coalition) given both reaches."""
        ex, i = self.ex, self.h
        A = ex.base_utility(i)
        if who == "H":
            out = np.zeros((len(ex.nodes), self.D))
            for b, ix, sd in ex._exact_groups(i):
                P = rC[ex.nodes[ix]]
                out[ix] = A[ix, None] * (P @ ex._kernel(i, b, None))
                for key, loc, glob in sd:
                    out[glob] += ex.pot[glob, None] * (P[loc] @ ex._kernel(i, b, key))
            return out
        out = np.zeros((len(ex.nodes), self.F))
        for b, ix, sd in ex._exact_groups(i):
            Q = rH[ex.nodes[ix]]
            out[ix] = -A[ix, None] * (Q @ ex._kernel(i, b, None).T)
            for key, loc, glob in sd:
                out[glob] -= ex.pot[glob, None] * (Q[loc] @ ex._kernel(i, b, key).T)
        return out

    def backward(self, sig, tv, who, br=False):
        t = self.t
        F = tv.shape[1]
        val = np.zeros((t.M, F))
        val[t.terminals] = tv
        own = self.isH if who == "H" else self.isC
        cw = 1.0 / (self.D - self.n)
        for lev in reversed(t.levels):
            lo, hi = lev["lo"], lev["hi"]
            v = val[lo:hi]
            o = own[lo:hi]
            w = np.where(o[:, None], sig[lo:hi], 1.0)
            w = np.where(self.ch[lo:hi][:, None], cw, w)
            s = np.add.reduceat(w * v, lev["gstart"], axis=0)
            if br:
                mx = np.maximum.reduceat(v, lev["gstart"], axis=0)
                gp = lev["gparent"]
                mine = (lev["gkind"] == DECISION) & ((t.player[gp] == self.h) if who == "H"
                                                      else (t.player[gp] != self.h))
                s = np.where(mine[:, None], mx, s)
            val[lev["gparent"]] = s
        return val

    def value(self, sigH, sigC, br=None):
        """Hero EV (antes/hand) of the profile; br='H' or 'C' -> that side best-responds."""
        rH, rC = self.reach(sigH, "H"), self.reach(sigC, "C")
        if br == "C":
            v = self.backward(sigC, self.term("C", rH, rC), "C", br=True)
            return -v[0].sum() / self.nd
        v = self.backward(sigH, self.term("H", rH, rC), "H", br=(br == "H"))
        return v[0].sum() / self.nd

    # ------------------------------------------------------------------ CFR+
    def _norm(self, R, dec):
        t = self.t
        w = np.maximum(R[dec], 0)
        gs = np.flatnonzero(np.r_[True, t.parent[dec][1:] != t.parent[dec][:-1]])
        grp = np.repeat(np.arange(len(gs)), np.diff(np.r_[gs, len(dec)]))
        s = np.add.reduceat(w, gs, axis=0)[grp]
        nch = t.n_children[t.parent[dec]][:, None]
        return np.where(s > 0, w / np.where(s > 0, s, 1), 1.0 / nch)

    def solve(self, iters, sigH0=None, fixed_H=False, checkpoints=()):
        t = self.t
        sig = {"H": np.ones((t.M, self.D)), "C": np.ones((t.M, self.F))}
        R = {k: np.zeros_like(v) for k, v in sig.items()}
        avg = {k: np.zeros_like(v) for k, v in sig.items()}
        dec = {"H": self.decH, "C": self.decC}
        for k in ("H", "C"):
            sig[k][dec[k]] = self._norm(R[k], dec[k])
        if sigH0 is not None:
            sig["H"] = sigH0.copy()
        log = []
        for it in range(1, iters + 1):
            for k in (("C",) if fixed_H else ("H", "C")):
                rH, rC = self.reach(sig["H"], "H"), self.reach(sig["C"], "C")
                val = self.backward(sig[k], self.term(k, rH, rC), k)
                e = dec[k]
                par = t.parent[e]
                r = rH if k == "H" else rC
                avg[k][e] += it * r[par] * sig[k][e]
                R[k][e] = np.maximum(R[k][e] + val[e] - val[par], 0)
                sig[k][e] = self._norm(R[k], e)
            if it in checkpoints:
                a = self.average(avg, sig, fixed_H)
                log.append((it, self.value(a["H"], a["C"], br="C"), self.value(a["H"], a["C"], br="H")))
        return self.average(avg, sig, fixed_H), log

    def average(self, avg, sig, fixed_H=False):
        out = {}
        for k, e in (("H", self.decH), ("C", self.decC)):
            if k == "H" and fixed_H:
                out[k] = sig[k]
                continue
            a = np.ones_like(sig[k])
            a[e] = self._norm(avg[k], e)
            out[k] = a
        return out


def embed_hero(tree: PublicTree, sigma: np.ndarray, hero: int) -> np.ndarray:
    """A 3p sigma [M, D] -> the hero's behaviour strategy in the team game (same array)."""
    return sigma.copy()
