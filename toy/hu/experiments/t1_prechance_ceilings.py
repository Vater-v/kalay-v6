"""T1 (targeted): are chance-averaged (pre-chance) opponent CFVs valid gadget
ceilings for the specific card that was dealt?

Continual re-solving agent with ORACLE leaves (exact remaining game below the
re-solve root, as in E2c 'oracle_safe').  The only thing that changes between
variants is the opponent ceiling w(S_c, h') handed to the gadget at each
next-street root S_c (street root after card c):

  percard     w = opponent CFV at S_c from the previous re-solve (exact, per card).
              This is E2c's oracle_safe agent (DeepStack-style, per-card ceilings).
  avg         w = the chance-AVERAGED opponent CFV at the parent chance node C,
              translated to card c in the same normalised units a pre-chance value
              net would output:  u_C(h') = v_C(h') / (cs_C x M_C(h'))  applied to every c,
              i.e.  w_c(h') = v_C(h') * M_c(h') / ((N - nb - 2) * M_C(h')).
              (sum_c w_c = v_C exactly; only the split across cards is lost.)
  start_cfv   start-of-street solve: re-solve S_c UNSAFELY with the opponent range
              carried from the previous gadget solve, take the opponent's CFVs (of the
              solved average profile) at S_c as ceilings, then do the gadget re-solve.
  start_br    same, but ceilings = opponent best-response CFVs against our strategy of
              the start-of-street solve.
  start_br_mix same as start_br, opponent range mixed with 1% uniform (full support).
  chance_root the sound alternative: ONE gadget re-solve rooted at the pre-chance node C
              (all next cards c in one forest, the opponent's terminate/follow choice is
              made per hand at C, before the card), ceiling = chance-averaged CFV v_C.
              Only the strategy for the card actually dealt is used.
  start_min   ceilings = min(start_cfv ceilings, avg ceilings): the per-card solve value,
              capped by what the previous solve's (chance-averaged) leaf committed.

Every agent's COMPLETE strategy (both seats) is scored by an exact best response
on the full game.  Also reported: total positive ceiling slack sum (w_variant -
w_percard)^+ over all stored street roots (absolute CFV units = antes/hand).
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import multiprocessing as mp
import numpy as np
from common import setup, save, load, Log, mA, pct_pot
from tree import PublicTree, DEC, CHANCE
from cfr import Solver, values, forward, exploitability
from resolve import _agent_counts

K = int(os.environ.get("T1_K", 300))
MODES = os.environ.get("T1_MODES", "percard,avg,start_cfv,start_br,start_br_mix").split(",")
CFG, B, T = setup()


def avg_ceilings(F, n_list, vo, ra):
    """Chance-averaged ceilings for next-street roots n (children of chance nodes)."""
    N = F.N
    out = {}
    for n in n_list:
        C = F.parent[n]
        assert F.kind[C] == CHANCE
        nb = len(F.states[C].board)
        c = F.card[n]
        MC = ra[C].sum() - ra[C]                  # our mass excluding h'  [N]
        rc = ra[C].copy(); rc[c] = 0.0
        Mc = rc.sum() - rc                        # our mass excluding c and h'
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(MC > 0, vo[C] * Mc / ((N - nb - 2) * np.where(MC > 0, MC, 1)),
                         vo[C] / (N - nb - 1))
        w = w * (np.arange(N) != c) * B.mask[F.board_id[n]]
        out[n] = w
    return out


class ChanceRootSolver(Solver):
    """Gadget at the pre-chance node: roots = the post-chance street roots S_c,
    grouped by their chance parent; the opponent decides follow/terminate per
    (group, hand) and terminating pays the chance-averaged ceiling wC."""

    def __init__(self, tree, root_reach, gplayer, wC, group):
        super().__init__(tree, root_reach)
        self.gp, self.wC, self.group = gplayer, wC, group
        G = len(wC)
        self.cR = np.zeros((G, tree.N, 2)); self.cS = np.zeros((G, tree.N, 2))
        self.cf = np.full((G, tree.N), 0.5)
        self.gadget = dict(player=gplayer)

    def opp_reach(self, i, eff=None):
        o = 1 - i
        root = self.root_reach[o] * self.cf[self.group] if o == self.gp else None
        return self.reach(o, eff=eff, root=root)

    def _traverse(self, i):
        t = self.t
        w = float(self.it)
        q = self.opp_reach(i)
        root = self.root_reach[i] * self.cf[self.group] if i == self.gp else None
        r = self.reach(i, root=root)
        v = self.backward(i, self.terminal_values(i, q))
        E = t.E[i]
        self.Ssum[i] += w * r[E]
        inst = v[E] - v[t.E_parent[i]]
        np.maximum(self.R[i] + inst, 0.0, out=self.R[i])
        self.eff[i][E] = self._match(i, self.R[i])
        if i == self.gp:
            vf = np.zeros_like(self.cf)
            np.add.at(vf, self.group, v[: t.n_roots])
            sf = self.cf
            vg = sf * vf + (1 - sf) * self.wC
            self.cS[..., 0] += w * sf; self.cS[..., 1] += w * (1 - sf)
            self.cR[..., 0] = np.maximum(self.cR[..., 0] + vf - vg, 0)
            self.cR[..., 1] = np.maximum(self.cR[..., 1] + self.wC - vg, 0)
            tot = self.cR.sum(-1)
            self.cf = np.where(tot > 0, self.cR[..., 0] / np.where(tot > 0, tot, 1), 0.5)

    def avg_follow_roots(self):
        tot = self.cS.sum(-1)
        return np.where(tot > 0, self.cS[..., 0] / np.where(tot > 0, tot, 1), 0.5)[self.group]


def extract(a, mode, iters=K):
    N, o = T.N, 1 - a
    cnt_full = _agent_counts(T, a)
    eff_a = T.eff_template.copy()
    Ea = T.E[a]
    eff_a[Ea] = np.repeat(1.0 / T.E_cnt[a], T.E_cnt[a])[:, None]
    w_store, q_store, w_exact, w_avg, wC_store = {}, {}, {}, {}, {}
    st_full = T.states
    is_street_root = np.array([s.n_act == 0 for s in st_full]) & (T.kind == DEC)
    stats = []
    for street in range(T.cfg.n_streets):
        for k in range(3):
            if k == 0:
                roots = np.nonzero(is_street_root & (T.street == street))[0]
            else:
                roots = np.nonzero((T.kind == DEC) & (T.player == a) &
                                   (T.street == street) & (cnt_full == k))[0]
            if len(roots) == 0:
                continue
            t0 = time.time()
            pi_a = forward(T, eff_a, T.bmask[:1])[roots]
            F = PublicTree(T.cfg, B, [st_full[r] for r in roots], list(T.cscale[roots]))
            game_root = street == 0 and k == 0
            reach = [None, None]
            reach[a] = pi_a
            gadget = None
            if game_root:
                reach[o] = T.bmask[roots]
            else:
                reach[o] = T.bmask[roots]
                W = np.stack([w_store[r] for r in roots])
                if k == 0 and mode.startswith("start"):
                    # start-of-street (unsafe) solve with the carried opponent range
                    q = np.stack([q_store[r] for r in roots])
                    if mode == "start_br_mix":
                        m = q.sum(1, keepdims=True)
                        u = T.bmask[roots] / T.bmask[roots].sum(1, keepdims=True)
                        q = 0.99 * q + 0.01 * m * u
                    rr = [None, None]; rr[a] = pi_a; rr[o] = q
                    S0 = Solver(F, rr)
                    S0.iterate(iters)
                    e0 = [S0.avg_eff(0), S0.avg_eff(1)]
                    if mode in ("start_cfv", "start_min"):
                        v = values(S0, e0[0], e0[1])
                    else:
                        v = values(S0, e0[0], e0[1], best_response_for=o)
                    W = (v[0] if o == 0 else v[1])[: F.n_roots].copy()
                    if mode == "start_min":
                        W = np.minimum(W, np.stack([w_avg[r] for r in roots]))
                    for j, r in enumerate(roots):
                        w_store[r] = W[j]
                gadget = dict(player=o, w=W)
            if mode == "chance_root" and k == 0 and not game_root:
                par = T.parent[roots]
                ug, group = np.unique(par, return_inverse=True)
                S = ChanceRootSolver(F, reach, o, np.stack([wC_store[c] for c in ug]), group)
                S.iterate(iters)
                e = [S.avg_eff(0), S.avg_eff(1)]
                gf = S.avg_follow_roots()
            else:
                S = Solver(F, reach, gadget=gadget)
                S.iterate(iters)
                e = [S.avg_eff(0), S.avg_eff(1)]
                gf = S.avg_follow() if gadget is not None else None
            v0, v1, r0, r1 = values(S, e[0], e[1], g_follow=gf)
            vo, ro = (v0, r0) if o == 0 else (v1, r1)
            ra = r0 if a == 0 else r1
            cntF = _agent_counts(F, a)
            own = np.nonzero((F.kind == DEC) & (F.player == a) &
                             (F.street == street) & (cntF == 0))[0]
            if k > 0:
                own = own[own < F.n_roots]
            keys = F.states
            for n in own:
                fn = T.key_index[keys[n].key]
                cs, cc = F.child_start[n], F.n_child[n]
                fs, fc = T.child_start[fn], T.n_child[fn]
                eff_a[fs: fs + fc] = e[a][cs: cs + cc]
            nxt = np.nonzero((F.kind == DEC) & (F.player == a) &
                             (F.street == street) & (cntF == 1))[0]
            for n in nxt:
                fn = T.key_index[keys[n].key]
                w_store[fn] = vo[n].copy(); q_store[fn] = ro[n].copy()
            for C in np.nonzero((F.kind == CHANCE) & (F.street == street))[0]:
                wC_store[T.key_index[keys[C].key]] = vo[C].copy()
            nr = np.nonzero((F.kind == DEC) & (F.street == street + 1) &
                            np.array([s.n_act == 0 for s in F.states]))[0]
            wa = avg_ceilings(F, nr, vo, ra) if mode in ("avg", "start_min") else None
            for n in nr:
                fn = T.key_index[keys[n].key]
                w_exact[fn] = vo[n].copy()
                if wa is not None:
                    w_avg[fn] = wa[n]
                w_store[fn] = wa[n] if mode == "avg" else vo[n].copy()
                q_store[fn] = ro[n].copy()
            stats.append(dict(street=street, k=k, n=len(roots), s=time.time() - t0))
    # ceiling slack vs exact per-card ceilings, at street roots of streets 1,2
    slack = 0.0; under = 0.0
    for fn, w in w_store.items():
        if fn in w_exact and mode == "avg":
            d = w - w_exact[fn]
            slack += float(np.maximum(d, 0).sum()); under += float(np.maximum(-d, 0).sum())
    return eff_a, stats, slack, under


def job(args):
    mode, a = args
    t0 = time.time()
    eff, st, slack, under = extract(a, mode)
    return mode, a, eff, st, slack, under, time.time() - t0


if __name__ == "__main__":
    log = Log(os.environ.get("T1_LOG", "t1.log"))
    jobs = [(m, a) for m in MODES for a in (0, 1)]
    jobs.sort(key=lambda j: 0 if j[0].startswith("start") else 1)
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(job, jobs)
    S = Solver(T, [T.bmask[:1].copy()] * 2)
    gv = load("e1.json")["game_value_P0"]
    out = {}
    for m in MODES:
        r = {x[1]: x for x in res if x[0] == m}
        e = exploitability(S, r[0][2], r[1][2])
        out[m] = dict(expl_mA=mA(e["expl"]), pct_pot=pct_pot(e["expl"]),
                      seat0_exploitable_by_mA=mA(e["BR1"] + gv),
                      seat1_exploitable_by_mA=mA(e["BR0"] - gv),
                      ceiling_slack_up=r[0][4] + r[1][4], ceiling_slack_down=r[0][5] + r[1][5],
                      wall_s=[r[0][6], r[1][6]])
        log(f"{m:13s} expl {mA(e['expl']):8.1f} mA/h ({pct_pot(e['expl']):5.2f}% pot) "
            f"seat0 {out[m]['seat0_exploitable_by_mA']:.1f} seat1 {out[m]['seat1_exploitable_by_mA']:.1f} "
            f"slack+ {out[m]['ceiling_slack_up']:.4f} slack- {out[m]['ceiling_slack_down']:.4f} "
            f"wall {r[0][6]:.0f}/{r[1][6]:.0f}s")
    save(os.environ.get("T1_OUT", "t1.json"), dict(K=K, agents=out))
    log("done")
