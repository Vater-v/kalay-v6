"""E3: off-tree bet sizes.  Player 0 (opponent) may open the flop with a size
f that is NOT in our menu.  We are player 1 and respond with
  (i)  snapping: treat f as a menu size (lower neighbour / upper neighbour /
       nearest by pot fraction) and play the blueprint there, translating
       actions by label (a raise that collapsed into all-in maps to all-in);
  (ii) re-solving: safe re-solve of the flop subgame (flop + turn, exact)
       with f ADDED to the tree: gadget = player 0's blueprint CFVs at the flop
       root, our range = blueprint range; used only below the off-tree action.
Metric: player 0's best-response value in the AUGMENTED game against our
strategy minus the augmented game value  (= our exploitability in seat 1);
also minus the base game value (how much the off-tree size buys the opponent).
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import multiprocessing as mp
import numpy as np
from common import CFG, setup, save, load, mA, pct_pot, Log, RES
from cfr import Solver, exploitability, values
from tree import full_game_tree, PublicTree, DEC, CHANCE

SIZES = [float(s) for s in os.environ.get("E3_SIZES", "0.25,0.75,1.5").split(",")]
AUG_ITERS = int(os.environ.get("E3_AUG_ITERS", 1000))
K = int(os.environ.get("E3_K", 300))

cfg, B, T = setup()
bp = [np.load(os.path.join(RES, "blueprint_eff0.npy")), np.load(os.path.join(RES, "blueprint_eff1.npy"))]


def frac_of(TA, n, label):
    """Pot fraction of the bet `label` at node n (works for 'A' too)."""
    s = TA.states[n]
    cs = TA.child_start[n]
    j = TA.label[cs: cs + TA.n_child[n]].index(label)
    child = TA.states[cs + j]
    return (child.contrib[0] - s.contrib[0]) / s.pot


def base_map(TA, tag, mode):
    """For every node of the augmented tree, the base-tree node it is
    translated to.  mode in {'lo','hi','near'} picks the menu size that
    replaces the off-tree size `tag` at each flop root."""
    bmap = np.full(TA.M, -1)
    bmap[0] = 0
    for n in range(1, TA.M):
        p = TA.parent[n]
        m = bmap[p]
        if m < 0 or T.n_child[m] == 0:   # translated line ended in the base tree
            bmap[n] = -1
            continue
        lab = TA.label[n]
        if lab == tag:  # the off-tree bet: choose the menu size to snap to
            f = frac_of(TA, p, tag)
            cs = T.child_start[m]
            labs = [l for l in T.label[cs: cs + T.n_child[m]] if l.startswith("b") or l == "A"]
            fr = {l: frac_of(T, m, l) for l in labs}
            lo = [l for l in labs if fr[l] <= f]
            hi = [l for l in labs if fr[l] >= f]
            lo = max(lo, key=lambda l: fr[l]) if lo else min(labs, key=lambda l: fr[l])
            hi = min(hi, key=lambda l: fr[l]) if hi else max(labs, key=lambda l: fr[l])
            if mode == "lo":
                lab = lo
            elif mode == "hi":
                lab = hi
            else:
                lab = lo if abs(fr[lo] - f) <= abs(fr[hi] - f) else hi
        cs = T.child_start[m]
        labs = T.label[cs: cs + T.n_child[m]]
        if lab not in labs:  # e.g. a pot raise that is an all-in in the base tree
            bets = [l for l in labs if l[0] in "brA"]
            lab = "A" if "A" in labs else (bets[-1] if bets and lab[0] in "brA" else
                                           ("c" if "c" in labs else labs[-1]))
        bmap[n] = cs + labs.index(lab)
    return bmap


def translate(TA, bmap, eff_base, fallback, seat=1):
    """Our (seat) strategy on the augmented tree: at node n play the base
    strategy of bmap[n], mapping each base action to the aug action with the
    same label (bets missing in the aug node -> all-in).
    Where the translated line has already ENDED in the base tree (e.g. a
    small off-tree bet snapped to the only menu bet, an all-in, but the real
    hand continues) there is no base strategy; we then use `fallback`
    (the augmented-game equilibrium = generous to snapping)."""
    e = TA.eff_template.copy()
    for n in np.nonzero((TA.kind == DEC) & (TA.player == seat))[0]:
        m = bmap[n]
        cs, cc = TA.child_start[n], TA.n_child[n]
        labs = TA.label[cs: cs + cc]
        P = np.zeros((cc, TA.N))
        if m < 0 or T.kind[m] != DEC or T.player[m] != seat:
            e[cs: cs + cc] = fallback[cs: cs + cc]
            continue
        bs = T.child_start[m]
        for j, l in enumerate(T.label[bs: bs + T.n_child[m]]):
            if l not in labs:
                l = "A" if "A" in labs else labs[-1]
            P[labs.index(l)] += eff_base[bs + j]
        e[cs: cs + cc] = P
    return e


def run_size(f):
    log = Log(f"e3_{f:g}.log")
    tag = f"b{f:g}"
    TA = full_game_tree(cfg, B, extra_open={1: [f]})
    out = dict(size=f, aug_nodes=TA.M)
    # reference: solve the augmented game
    SA = Solver(TA, [TA.bmask[:1].copy()] * 2)
    t0 = time.time(); SA.iterate(AUG_ITERS)
    eA = exploitability(SA)
    out.update(aug_value_P0=eA["EV0"], aug_solve_expl=eA["expl"], aug_solve_s=time.time() - t0)
    log("aug solve", eA)
    strategies = {}
    fb = SA.avg_eff(1)
    for mode in ("lo", "hi", "near"):
        strategies["snap_" + mode] = translate(TA, base_map(TA, tag, mode), bp[1], fb)
    # --- safe re-solve below the off-tree action
    bm = base_map(TA, tag, "near")
    S0 = Solver(T, [T.bmask[:1].copy()] * 2)
    v0, v1, r0, r1 = values(S0, bp[0], bp[1])
    roots = np.nonzero((TA.kind == DEC) & (TA.street == 1) &
                       np.array([s.n_act == 0 for s in TA.states]))[0]
    broots = np.array([T.key_index[TA.states[n].key] for n in roots])
    t0 = time.time()
    F = PublicTree(cfg, B, [TA.states[n] for n in roots], list(TA.cscale[roots]), extra_open={1: [f]})
    gad = dict(player=0, w=v0[broots])
    SR = Solver(F, [F.bmask[: F.n_roots].copy(), r1[broots]], gadget=gad)
    SR.iterate(K)
    eR = SR.avg_eff(1)
    t_res = time.time() - t0
    e = strategies["snap_near"].copy()
    for n in range(F.M):
        if F.kind[n] == DEC and F.player[n] == 1 and tag in F.states[n].key.split("/"):
            an = TA.key_index[F.states[n].key]
            cs, cc = F.child_start[n], F.n_child[n]
            e[TA.child_start[an]: TA.child_start[an] + cc] = eR[cs: cs + cc]
    strategies["resolve"] = e
    strategies["aug_nash"] = SA.avg_eff(1)
    out["resolve_forest_s"] = t_res
    out["resolve_n_roots"] = len(roots)
    out["resolve_s_per_root"] = t_res / len(roots)
    # single-root latency
    t0 = time.time()
    F1 = PublicTree(cfg, B, [TA.states[roots[0]]], [TA.cscale[roots[0]]], extra_open={1: [f]})
    S1 = Solver(F1, [F1.bmask[:1].copy(), r1[broots[:1]]], gadget=dict(player=0, w=v0[broots[:1]]))
    S1.iterate(K)
    out["resolve_single_root_s"] = time.time() - t0
    gv = load("e1.json")["game_value_P0"]
    out["base_value_P0"] = gv
    res = {}
    for mode in ("lo", "hi", "near"):
        bm_ = base_map(TA, tag, mode)
        r1A = SA.reach(1, eff=strategies["snap_" + mode])
        # P1 decision nodes with no base counterpart that P1 reaches with prob > 0
        bad = np.nonzero((bm_ < 0) & (TA.kind == DEC) & (TA.player == 1))[0]
        out[f"unmapped_reached_P1_nodes_{mode}"] = int((r1A[bad].sum(1) > 0).sum())
        out[f"P1_decision_nodes_offtree"] = int(sum(1 for n in np.nonzero((TA.kind == DEC) & (TA.player == 1))[0]
                                                    if tag in TA.states[n].key.split("/")))
    for name, eff1 in strategies.items():
        x = exploitability(SA, eff0=SA.avg_eff(0), eff1=eff1)
        res[name] = dict(BR0=x["BR0"], loss_vs_aug_value=x["BR0"] - eA["EV0"],
                         loss_vs_base_value=x["BR0"] - gv,
                         loss_vs_aug_mA=mA(x["BR0"] - eA["EV0"]), loss_vs_base_mA=mA(x["BR0"] - gv))
        log(name, res[name])
    out["strategies"] = res
    save(f"e3_{f:g}.json", out)
    return out


if __name__ == "__main__":
    log = Log("e3.log")
    S0 = Solver(T, [T.bmask[:1].copy()] * 2)
    base = exploitability(S0, bp[0], bp[1])
    gv = load("e1.json")["game_value_P0"]
    with mp.get_context("fork").Pool(min(4, len(SIZES))) as pool:
        outs = pool.map(run_size, SIZES)
    save("e3.json", dict(base_blueprint_BR0=base["BR0"], base_loss_mA=mA(base["BR0"] - gv),
                         base_value_P0=gv, K=K, aug_iters=AUG_ITERS, sizes=outs))
    log("done")
