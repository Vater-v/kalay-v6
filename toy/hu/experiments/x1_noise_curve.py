"""X1: how much exploitability does a given street-boundary value error cost?

Takes the ORACLE leaf of E2c (the remaining game is kept below the street
boundary and solved jointly inside each re-solve, i.e. exact boundary values)
and corrupts the boundary values in a CONTROLLED way before they enter the
re-solve.  Everything else is exactly E2c's safe (gadget) re-solving agent:
same K = 300 CFR+ iterations, same phases, same batching, same exact
best-response scoring on the full game.

Where the error is injected.  Let S be a next-street root of the re-solve's
current street (the node right after the boundary card is dealt; in a
depth-limited agent this is where V is called).  The exact CFV of player i at
S is v_i(S,h) (computed by the jointly solved subtree under S).  The re-solve
above S sees  v_i(S,h) + delta_i(S,h)  instead: the perturbation is added to
the value S contributes to its parent chance node, so the regrets of the
decisions INSIDE the subtree are untouched (the subtree only provides the
"exact V").  The opponent CFVs stored at S for the next street's gadget are
also the perturbed ones, exactly as a depth-limited agent stores V's output.
Only the boundary of the current street is perturbed (a preflop re-solve
perturbs flop roots, not the turn roots deeper in its oracle subtree); turn
re-solves have no boundary and are untouched, as in E2c.

Error models, in the normalised units of valuefn.py (u_i = value / x where x
= chips each player has in the pot at S, so the leaf pot is 2x; absolute CFV
= cscale(S) * x * M_opp(S) * u_i, M_opp = opponent reach mass at S):
  iid   : u_i(h) += 2*eps*z_i(S,h), z ~ N(0,1) fixed per (public node S,
          hand h, player i, seed): noise std = eps * (leaf pot) in chips.
  zs    : the same z, then zero-sum projected per situation like the MLP
          output (e = p0.n0 + p1.n1, n_i -= e/2, p_i = normalised range at S),
          so the perturbed values conserve chips: p0.u0 + p1.u1 unchanged.
  bias  : u_i <- u_i - lam*(u_i - u_i^checkdown): every leaf value is shrunk
          toward the check-down value by the fraction lam (lam=1 IS the
          check-down leaf; mimics an under-fit residual net).
  learned: u_i <- V_learned(board, x, p0, p1) with the E2 nets (validation of
          the harness: must reproduce E2c's depth-limited agent).
  base  : no perturbation (must reproduce E2c oracle_safe, 23.7 mA/h).

Error measurement ("in situ"): at the end of every re-solve, at every
boundary node S whose two normalised reaches are both > 1e-3 of the forest
root's mass, err_i(h) = (perturbed u_i(h) - exact u_i(h)) with the exact
value = the jointly solved subtree's value under the final (average) profile.
Range-weighted RMSE = sqrt( mean_S [ sum_h p0(h) err0(h)^2 + sum_h p1(h) err1(h)^2 ] / 2 )
(identical formula to e2_common.rmse_w), in x units ("pot-share units").
Test-set RMSE: the same error model applied to E2's exact held-out labels
(1000 random turn situations, 300 exact flop situations, on-policy roots).

Usage (each invocation < 10 min on 4 cores):
  python3 experiments/x1_noise_curve.py run base,bias_l0.1,...   (comma list)
  python3 experiments/x1_noise_curve.py run @group   (groups: see GROUPS)
  python3 experiments/x1_noise_curve.py report      (writes results/x1.json + results/x1_tables.md)
"""
import os, sys, time, json, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
from e2_common import *            # CFG, B, T, CD, K, rmse_w, on_policy, load_data, sub
from common import save, load, Log, mA, pct_pot, RES
from cfr import Solver, values, forward, exploitability
from tree import PublicTree, DEC
from valuefn import ByStreetV
from resolve import _agent_counts

EPS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
LAMS = [0.1, 0.3, 0.5, 1.0]
SEEDS = [0, 1]
GROUPS = {
    "ref": ["base"] + [f"bias_l{l}" for l in LAMS] + ["learned_20000_rw"],
    "iid0": [f"iid_e{e}_s0" for e in EPS],
    "iid1": [f"iid_e{e}_s1" for e in EPS],
    "zs0": [f"zs_e{e}_s0" for e in EPS],
    "zs1": [f"zs_e{e}_s1" for e in EPS],
    "nets": ["learned_200", "learned_2000", "learned_20000"],
}
MASS_REL = 1e-3     # boundary nodes counted in the in-situ RMSE


def parse(name):
    """setting name -> dict(mode, eps, lam, seed, net)."""
    if name == "base":
        return dict(mode="none")
    if name.startswith("bias_l"):
        return dict(mode="bias", lam=float(name[6:]))
    if name.startswith("learned_"):
        return dict(mode="learned", net=name[8:])
    kind, e, s = name.split("_")
    return dict(mode="noise", project=(kind == "zs"), eps=float(e[1:]), seed=int(s[1:]))


_Z_CACHE = {}


def noise_table(seed):
    """z[full node, player, hand] ~ N(0,1), fixed per seed (shared by iid and zs)."""
    if seed not in _Z_CACHE:
        _Z_CACHE[seed] = np.random.default_rng(1000 + seed).standard_normal((T.M, 2, T.N))
    return _Z_CACHE[seed]


def learned_V(net):
    VT = pickle.load(open(os.path.join(RES, "value_nets_turn.pkl"), "rb"))
    VF = pickle.load(open(os.path.join(RES, "value_nets_flop.pkl"), "rb"))
    return ByStreetV(B, VF[net], VT[net])


class Boundary:
    """The boundary nodes S of one re-solve forest F and the error model."""

    def __init__(self, F, street, full, cfg):
        self.cfg = cfg
        st = F.states
        S = np.nonzero((F.kind == DEC) & (F.street == street + 1) &
                       np.array([s.n_act == 0 for s in st]))[0]
        self.S = S
        self.sidx = np.full(F.M, -1, dtype=np.int64)
        self.sidx[S] = np.arange(len(S))
        self.full_idx = np.array([full.key_index[st[n].key] for n in S], dtype=np.int64)
        self.bid = F.board_id[S]
        self.x = F.contrib[S, 0]
        self.cs = F.cscale[S]
        self.bm = F.bmask[S]
        self.root = F.root_of[S]
        if cfg["mode"] == "noise":
            self.Z = noise_table(cfg["seed"])[self.full_idx] * self.bm[:, None, :]
        if cfg["mode"] == "bias":
            self.W = B.W[self.bid]
        # per level: positions (inside [lo,hi)) of S nodes among the children
        self.level_pos = {}
        for li, lv in enumerate(F.levels):
            m = (self.sidx[lv["lo"]: lv["hi"]] >= 0)
            if m.any():
                pos = np.nonzero(m)[0]
                self.level_pos[li] = (pos, lv["lo"] + pos)

    def delta(self, i, nodes, vS, r0S, r1S):
        """Perturbation of player i's CFV at boundary nodes `nodes` (forest ids)."""
        c = self.cfg
        j = self.sidx[nodes]
        bm, x, cs = self.bm[j], self.x[j], self.cs[j]
        M0, M1 = r0S.sum(1), r1S.sum(1)
        Mo = M1 if i == 0 else M0
        scale = (cs * x * Mo)[:, None]
        if c["mode"] == "none":
            return np.zeros_like(vS)
        if c["mode"] == "noise":
            n = 2.0 * c["eps"] * self.Z[j]                 # [n, 2, N] in x units
            if c["project"]:
                p0 = r0S / np.where(M0 > 0, M0, 1)[:, None]
                p1 = r1S / np.where(M1 > 0, M1, 1)[:, None]
                e = (p0 * n[:, 0]).sum(1) + (p1 * n[:, 1]).sum(1)
                n = n - 0.5 * e[:, None, None]
            return scale * n[:, i] * bm
        if c["mode"] == "bias":
            ropp = r1S if i == 0 else r0S
            vcd = (cs * x)[:, None] * np.einsum("pij,pj->pi", self.W[j], ropp) * bm
            return -c["lam"] * (vS - vcd)
        if c["mode"] == "learned":
            p0 = r0S / np.where(M0 > 0, M0, 1)[:, None]
            p1 = r1S / np.where(M1 > 0, M1, 1)[:, None]
            u = c["V"](self.bid[j], x, p0, p1)[i]
            return scale * u * bm - vS
        raise ValueError(c)


class PerturbedSolver(Solver):
    """Solver whose backward pass adds Boundary.delta to the value every
    boundary node S passes up to its parent (v[S] itself stays exact)."""

    def __init__(self, tree, root_reach, boundary, **kw):
        self.bd = boundary
        self._both = None
        super().__init__(tree, root_reach, **kw)

    def terminal_values(self, i, q, reach_both=None):
        self._both = reach_both
        return super().terminal_values(i, q, reach_both)

    def backward(self, i, v, eff=None, best_response=False):
        t, bd = self.t, self.bd
        if bd is None or not bd.level_pos:
            return super().backward(i, v, eff=eff, best_response=best_response)
        eff = self.eff[i] if eff is None else eff
        r0, r1 = self._both
        for li in range(len(t.levels) - 1, -1, -1):
            lv = t.levels[li]
            lo, hi = lv["lo"], lv["hi"]
            vc = v[lo:hi]
            if li in bd.level_pos:
                pos, nodes = bd.level_pos[li]
                vc = vc.copy()
                vc[pos] += bd.delta(i, nodes, v[nodes], r0[nodes], r1[nodes])
            if best_response:
                s = np.add.reduceat(vc * t.eff_template[lo:hi], lv["starts"], axis=0)
                m = np.maximum.reduceat(vc, lv["starts"], axis=0)
                dec = lv["is_dec"][i]
                s[dec] = m[dec]
                v[lv["nt"]] = s
            else:
                v[lv["nt"]] = np.add.reduceat(vc * eff[lo:hi], lv["starts"], axis=0)
        return v


def extract_perturbed(full, a, cfg, iters=300):
    """resolve.extract_agent(full, a, 'oracle', safe=True) with the boundary
    values perturbed by `cfg`.  Returns (eff_a, stats, err) where err holds
    the in-situ range-weighted squared errors per boundary street."""
    o = 1 - a
    cnt_full = _agent_counts(full, a)
    eff_a = full.eff_template.copy()
    Ea = full.E[a]
    eff_a[Ea] = np.repeat(1.0 / full.E_cnt[a], full.E_cnt[a])[:, None]
    w_store = {}
    st_full = full.states
    is_street_root = np.array([s.n_act == 0 for s in st_full]) & (full.kind == DEC)
    stats = []
    err = {s: [0.0, 0, 0.0, 0.0] for s in (0, 1)}   # sum e_S, count, sum w_S*e_S, sum w_S
    for street in range(full.cfg.n_streets):
        for k in range(3):
            if k == 0:
                roots = np.nonzero(is_street_root & (full.street == street))[0]
            else:
                roots = np.nonzero((full.kind == DEC) & (full.player == a) &
                                   (full.street == street) & (cnt_full == k))[0]
            if len(roots) == 0:
                continue
            t0 = time.time()
            pi_a = forward(full, eff_a, full.bmask[:1])[roots]
            F = PublicTree(full.cfg, full.B, [st_full[r] for r in roots],
                           list(full.cscale[roots]), depth_limited=False)
            if street == 0 and k == 0:
                q0, gadget = full.bmask[roots], None
            else:
                q0 = full.bmask[roots]
                gadget = dict(player=o, w=np.stack([w_store[r] for r in roots]))
            reach = [None, None]
            reach[a], reach[o] = pi_a, q0
            bd = Boundary(F, street, full, cfg) if street + 1 < full.cfg.n_streets else None
            S_ = PerturbedSolver(F, reach, bd, gadget=gadget)
            S_.iterate(iters)
            e = [S_.avg_eff(0), S_.avg_eff(1)]
            gf = S_.avg_follow() if gadget is not None else None
            v0, v1, r0, r1 = values(S_, e[0], e[1], g_follow=gf)
            vo = v0 if o == 0 else v1
            cntF = _agent_counts(F, a)
            own = np.nonzero((F.kind == DEC) & (F.player == a) &
                             (F.street == street) & (cntF == 0))[0]
            if k > 0:
                own = own[own < F.n_roots]
            keys = F.states
            for n in own:
                fn = full.key_index[keys[n].key]
                cs, cc = F.child_start[n], F.n_child[n]
                fs, fc = full.child_start[fn], full.n_child[fn]
                assert cc == fc
                eff_a[fs: fs + fc] = e[a][cs: cs + cc]
            nxt = np.nonzero((F.kind == DEC) & (F.player == a) &
                             (F.street == street) & (cntF == 1))[0]
            for n in nxt:
                w_store[full.key_index[keys[n].key]] = vo[n].copy()
            if bd is not None and len(bd.S):
                S = bd.S
                d0 = bd.delta(0, S, v0[S], r0[S], r1[S])
                d1 = bd.delta(1, S, v1[S], r0[S], r1[S])
                do = d0 if o == 0 else d1
                for jj, n in enumerate(S):
                    w_store[bd.full_idx[jj]] = vo[n] + do[jj]     # what a V-agent stores
                # in-situ error, x units
                M0, M1 = r0[S].sum(1), r1[S].sum(1)
                R0 = r0[:F.n_roots].sum(1)[bd.root]; R1 = r1[:F.n_roots].sum(1)[bd.root]
                ok = (M0 > MASS_REL * R0) & (M1 > MASS_REL * R1)
                if ok.any():
                    sc0 = (bd.cs * bd.x * M1)[ok][:, None]; sc1 = (bd.cs * bd.x * M0)[ok][:, None]
                    p0 = r0[S][ok] / M0[ok][:, None]; p1 = r1[S][ok] / M1[ok][:, None]
                    eS = (p0 * (d0[ok] / sc0) ** 2).sum(1) + (p1 * (d1[ok] / sc1) ** 2).sum(1)
                    wS = (M0 / R0)[ok] * (M1 / R1)[ok]      # joint reach of S relative to its root
                    err[street][0] += float(eS.sum()); err[street][1] += int(ok.sum())
                    err[street][2] += float((eS * wS).sum()); err[street][3] += float(wS.sum())
            dt = time.time() - t0
            stats.append(dict(street=street, k=k, n_resolves=len(roots), nodes=F.M, seconds=dt))
    return eff_a, stats, err


def _job(args):
    name, a = args
    cfg = parse(name)
    if cfg["mode"] == "learned":
        cfg["V"] = learned_V(cfg["net"])
    t0 = time.time()
    eff, stats, err = extract_perturbed(T, a, cfg, iters=K)
    return name, a, eff, stats, err, time.time() - t0


# ---------------------------------------------------------------------------
def test_set_rmse(cfg):
    """Range-weighted RMSE (x units) of the error model applied to E2's exact
    held-out labels: turn test (1000 random), flop exact test (300 random),
    and on-policy turn/flop roots of the E1 equilibrium."""
    turn = sub(load_data("e2_turn_data.npz"), 21000, 22000)
    flop = load_data("e2_flop_test.npz")
    out = {}
    for nm, d in (("turn_test", turn), ("flop_test", flop),
                  ("turn_onpolicy", on_policy(2)), ("flop_onpolicy", on_policy(1))):
        bid, x, p0, p1, u0, u1 = d
        m = B.mask[bid]
        if cfg["mode"] == "noise":
            vals = []
            for s in SEEDS:
                z = np.random.default_rng(5000 + s).standard_normal((len(bid), 2, B.W.shape[1])) * m[:, None]
                n = 2 * cfg["eps"] * z
                if cfg["project"]:
                    e = (p0 * n[:, 0]).sum(1) + (p1 * n[:, 1]).sum(1)
                    n = n - 0.5 * e[:, None, None]
                Vp = (lambda b, xx, a0, a1, n=n: (u0 + n[:, 0] * m, u1 + n[:, 1] * m))
                vals.append(rmse_w(Vp, d))
            out[nm] = float(np.mean(vals))
        elif cfg["mode"] == "bias":
            c0, c1 = CD(bid, x, p0, p1)
            lam = cfg["lam"]
            out[nm] = rmse_w(lambda *_: (u0 - lam * (u0 - c0), u1 - lam * (u1 - c1)), d)
        elif cfg["mode"] == "learned":
            out[nm] = rmse_w(cfg["V"], d)
        else:
            out[nm] = 0.0
    return out


def run(names, log):
    """Settings whose results/x1_<name>.json exists are skipped; each setting is
    scored and saved as soon as both seats are extracted (a killed run loses
    only the settings in flight)."""
    names = [n for n in names if not os.path.exists(os.path.join(RES, f"x1_{n}.json"))]
    log(f"to do: {names}")
    jobs = [(n, a) for n in names for a in (0, 1)]
    Sfull = Solver(T, [T.bmask[:1].copy()] * 2)
    pending = {}
    with mp.get_context("fork").Pool(4) as pool:
        for rr in pool.imap_unordered(_job, jobs):
            pending.setdefault(rr[0], {})[rr[1]] = rr
            if len(pending[rr[0]]) == 2:
                finish(rr[0], pending.pop(rr[0]), Sfull, log)


def finish(name, r, Sfull, log):
    gv = load("e1.json")["game_value_P0"]
    e = exploitability(Sfull, r[0][2], r[1][2])
    cfg = parse(name)
    if cfg["mode"] == "learned":
        cfg["V"] = learned_V(cfg["net"])
    ins = {}
    for s in (0, 1):
        tot = r[0][4][s][0] + r[1][4][s][0]; cnt = r[0][4][s][1] + r[1][4][s][1]
        wtot = r[0][4][s][2] + r[1][4][s][2]
        wsum = r[0][4][s][3] + r[1][4][s][3]
        ins[["preflop->flop", "flop->turn"][s]] = dict(
            rmse_rangew=float(np.sqrt(tot / max(cnt, 1) / 2)), n_nodes=cnt,
            rmse_rangew_reachweighted=float(np.sqrt(wtot / max(wsum, 1e-300) / 2)) if wsum > 0 else None)
    tot = sum(r[a][4][s][0] for a in (0, 1) for s in (0, 1))
    cnt = sum(r[a][4][s][1] for a in (0, 1) for s in (0, 1))
    cfg_out = {k: v for k, v in cfg.items() if k != "V"}
    out = dict(name=name, cfg=cfg_out, K=K, expl=e["expl"], expl_mA=mA(e["expl"]),
               expl_pct_initial_pot=pct_pot(e["expl"]),
               seat0_exploitable_by_mA=mA(e["BR1"] + gv), seat1_exploitable_by_mA=mA(e["BR0"] - gv),
               insitu=ins, insitu_rmse_rangew_all=float(np.sqrt(tot / max(cnt, 1) / 2)),
               test_rmse_rangew=test_set_rmse(cfg),
               wall_s=[r[0][5], r[1][5]],
               seconds_by_street={s: sum(p["seconds"] for p in r[0][3] + r[1][3] if p["street"] == s)
                                  for s in range(3)})
    save(f"x1_{name}.json", out)
    log(f"{name:22s} expl {out['expl_mA']:8.1f} mA/h ({out['expl_pct_initial_pot']:6.2f}% of initial pot)  "
        f"in-situ RMSE_rw(x) {out['insitu_rmse_rangew_all']:.4f}  "
        f"test RMSE_rw(x) turn {out['test_rmse_rangew']['turn_test']:.4f} flop {out['test_rmse_rangew']['flop_test']:.4f}  "
        f"wall {max(out['wall_s']):.0f}s")


# ---------------------------------------------------------------------------
def report():
    rows = {}
    for f in sorted(os.listdir(RES)):
        if f.startswith("x1_") and f.endswith(".json") and f != "x1.json":
            d = load(f)
            rows[d["name"]] = d
    e2c = load("e2c.json")["agents"]
    e2a = {r["name"]: r for r in load("e2a.json")["V_turn"]}
    e2b = {r["name"]: r for r in load("e2b.json")["V_flop"]}
    base = rows.get("base")
    b_mA = base["expl_mA"] if base else e2c["oracle_safe"]["expl_mA"]
    L = []
    P = L.append

    def ins(d, key):
        v = d["insitu"].get(key, {}).get("rmse_rangew")
        return f"{v:.4f}" if v is not None else "-"

    P("### Noise curve: iid (a) and zero-sum-projected (b) Gaussian noise\n")
    P("eps = noise std as a fraction of the LEAF pot (2x).  RMSE columns in x units "
      "(1.0 = one player's pot contribution at the boundary = half the leaf pot); "
      "'% leaf pot' = RMSE_x / 2 * 100.  Exploitability = (BR0+BR1)/2 of the agent "
      "over the whole game, mA/h = milli-antes per hand; '% init pot' = mA/h / 2000 * 100.\n")
    P("| model | eps | seed | in-situ RMSE_rw (x) | as % leaf pot | in-situ pre->flop | in-situ flop->turn | test RMSE_rw turn/flop (x) | expl mA/h | % init pot | excess over exact mA/h |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for kind in ("iid", "zs"):
        for e in EPS:
            for s in SEEDS:
                nm = f"{kind}_e{e}_s{s}"
                if nm not in rows:
                    continue
                d = rows[nm]
                r = d["insitu_rmse_rangew_all"]
                P(f"| {kind} | {e} | {s} | {r:.4f} | {50 * r:.2f} | {ins(d, 'preflop->flop')} | {ins(d, 'flop->turn')} | "
                  f"{d['test_rmse_rangew']['turn_test']:.4f} / {d['test_rmse_rangew']['flop_test']:.4f} | "
                  f"{d['expl_mA']:.1f} | {d['expl_pct_initial_pot']:.2f} | {d['expl_mA'] - b_mA:+.1f} |")
    P("\n### Seed-averaged curve\n")
    P("| eps (std, frac. of leaf pot) | iid: RMSE_rw (x) | iid: expl mA/h (seed 0, 1) | iid mean | zs: RMSE_rw (x) | zs: expl mA/h (seed 0, 1) | zs mean |")
    P("|---|---|---|---|---|---|---|")
    curve = []
    for e in EPS:
        cells = [str(e)]
        for kind in ("iid", "zs"):
            ds = [rows[f"{kind}_e{e}_s{s}"] for s in SEEDS if f"{kind}_e{e}_s{s}" in rows]
            if not ds:
                cells += ["-", "-", "-"]; continue
            r = np.mean([d["insitu_rmse_rangew_all"] for d in ds])
            ex = [d["expl_mA"] for d in ds]
            cells += [f"{r:.4f}", ", ".join(f"{v:.1f}" for v in ex), f"{np.mean(ex):.1f}"]
            curve.append(dict(kind=kind, eps=e, rmse_x=float(r), expl_mA=float(np.mean(ex)), seeds=ex))
        P("| " + " | ".join(cells) + " |")
    P("\n### Systematic bias (c): shrink toward check-down by lam; validation rows\n")
    P("| setting | in-situ RMSE_rw (x) | in-situ pre->flop | in-situ flop->turn | test RMSE_rw turn / flop (x) | on-policy RMSE_rw turn / flop (x) | expl mA/h | % init pot | E2c depth-limited agent mA/h |")
    P("|---|---|---|---|---|---|---|---|---|")
    ref = {"base": "oracle_safe", "bias_l1.0": "checkdown_safe", "learned_20000_rw": "learned20000_rw_safe",
           "learned_200": "learned200_safe", "learned_2000": "learned2000_safe", "learned_20000": "learned20000_safe"}
    for nm in GROUPS["ref"] + GROUPS["nets"]:
        if nm not in rows:
            continue
        d = rows[nm]; t = d["test_rmse_rangew"]
        e2 = f"{e2c[ref[nm]]['expl_mA']:.1f}" if nm in ref else "-"
        P(f"| {nm} | {d['insitu_rmse_rangew_all']:.4f} | {ins(d, 'preflop->flop')} | {ins(d, 'flop->turn')} | "
          f"{t['turn_test']:.4f} / {t['flop_test']:.4f} | {t['turn_onpolicy']:.4f} / {t['flop_onpolicy']:.4f} | "
          f"{d['expl_mA']:.1f} | {d['expl_pct_initial_pot']:.2f} | {e2} |")
    P("\n### E2 learned nets and check-down (from e2a/e2b/e2c.json, not re-run)\n")
    P("| leaf | turn test RMSE_rw (x) | flop test RMSE_rw (x) | turn on-pol RMSE_rw | flop on-pol RMSE_rw | E2c expl mA/h |")
    P("|---|---|---|---|---|---|")
    for nm, ag in (("200", "learned200_safe"), ("2000", "learned2000_safe"), ("20000", "learned20000_safe"),
                   ("20000_raw", "learned20000_raw_safe"), ("20000_rw", "learned20000_rw_safe"),
                   ("checkdown", "checkdown_safe")):
        P(f"| {nm} | {e2a[nm]['test_rmse_rangew']:.3f} | {e2b[nm]['exact_test_rmse_rangew']:.3f} | "
          f"{e2a[nm]['onpolicy_rmse_rangew']:.3f} | {e2b[nm]['onpolicy_rmse_rangew']:.3f} | {e2c[ag]['expl_mA']:.1f} |")
    save("x1.json", dict(rows=rows, curve=curve, base_mA=b_mA))
    with open(os.path.join(RES, "x1_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        names = []
        for tok in sys.argv[2].split(","):
            names += GROUPS[tok[1:]] if tok.startswith("@") else [tok]
        log = Log(f"x1_{sys.argv[2].replace('@', '').replace(',', '+')[:60]}.log")
        t0 = time.time()
        run(names, log)
        log(f"done in {time.time() - t0:.0f}s")
