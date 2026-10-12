"""Walkthrough: ONE complete decision of the safe continual re-solving agent, with the
mechanism's actual numbers printed (exact / ORACLE leaves, K = 300 CFR+ iterations).

Run (from toy/hu):   python3 experiments/walkthrough.py
Output is also written to results/walkthrough.txt.  Needs nothing from results/ (no blueprint,
no value nets): the whole chain of re-solves is recomputed here, ~30 s on one core.

Scenario (see SCENARIO below).  In this game player 0 acts first on every street, so P0 is the
out-of-position seat.  The agent is P0.  "Faces a bet after checking" therefore means:
    preflop  P0 bets pot, P1 calls               (pot 6, 9 behind each)
    flop     card dealt; P0 (agent) CHECKS, P1 BETS pot; the agent must act (fold/call/raise)
The agent never stores a strategy; to act here it chains three re-solves, each using only what the
previous one handed over (this is exactly resolve.extract_agent, restricted to the one path):

   [1] preflop root   (whole game below, no gadget: the game start is common knowledge)
         -> hands over:  our range at the flop root, opponent CFVs w(flop root, h')
   [2] flop street root, gadget on the opponent with w from [1]   (flop + turn kept below = ORACLE)
         -> hands over:  our range after our check, opponent CFVs at our next decision node
   [3] the decision node itself, gadget on the opponent with w from [2]   <- the decision shown

Items printed for [3]: (a) public node, (b) our range, (c) gadget inputs w, (d) subtree size,
(e) our re-solved strategy for every hand, (f) opponent range implied (by-product) vs the range a
naive agent would carry in, (g) wall-clock.  Then the same decision (1) against an OFF-MENU bet size
and (2) under restricted Nash response (RNR) against the 'station' model.
"""
import os
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np                                     # noqa: E402
import tree as tree_mod                                # noqa: E402
from common import CFG, RES, setup, mA                 # noqa: E402
from cfr import Solver, values, forward                # noqa: E402
from tree import PublicTree, DEC                       # noqa: E402
from opponents import model_eff                        # noqa: E402

# ----------------------------------------------------------------------------------------------
# SCENARIO (all knobs in one place)
# ----------------------------------------------------------------------------------------------
K = int(os.environ.get("WT_K", 300))          # CFR+ iterations in every re-solve
AGENT = 0                                      # seat of the agent (0 = acts first = out of position)
PRE_PATH = os.environ.get("WT_PRE", "b1/c")    # preflop: P0 bets pot, P1 calls
FLOP_CARD = int(os.environ.get("WT_CARD", 5))  # card id of the flop card (rank = id // 2)
OPP_BET = os.environ.get("WT_BET", "b1")       # P1's flop bet after our check (on the menu)
OFF_MENU = float(os.environ.get("WT_OFF", 0.75))   # P1's bet size that is NOT on the menu {0.5, 1, all-in}
RNR_PS = (0.01, 0.25)                          # RNR mixing weights against the 'station' model

_LOG = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.append(s)


def hr(title=""):
    out("")
    out("=" * 100)
    if title:
        out(title)
        out("=" * 100)


# ----------------------------------------------------------------------------------------------
# small helpers: names, formatting
# ----------------------------------------------------------------------------------------------
def card(c):
    """Card id -> 'Ts','Th','Js',... (rank letter + suit; 5 ranks x 2 suits; rank = id // 2)."""
    return "TJQKA"[c // CFG.n_suits] + "sh"[c % CFG.n_suits]


def describe_key(key):
    """'b1/c/d5/k/b1' -> readable betting sequence."""
    names = {"k": "check", "c": "call", "f": "fold", "A": "ALL-IN"}
    parts, street, seat = [], "preflop", 0
    for lab in key.split("/") if key else []:
        if lab[0] == "d":
            street = "flop" if street == "preflop" else "turn"
            parts.append(f"| {street} card {card(int(lab[1:]))} |")
            seat = 0
            continue
        if lab in names:
            what = names[lab]
        elif lab[0] == "b":
            what = f"bet {float(lab[1:]):g} pot"
        else:
            what = f"raise (pot-size {float(lab[1:]):g})"
        parts.append(f"P{seat} {what}")
        seat = 1 - seat if lab not in ("c",) else seat
    return " ".join(parts)


def node_by_key(T, key):
    return T.key_index[key]


def fmt_vec(v, mask=None, nd=3, width=8):
    return " ".join(("    --  " if (mask is not None and not mask[i]) else f"{v[i]:{width}.{nd}f}")
                    for i in range(len(v)))


# ----------------------------------------------------------------------------------------------
# the re-solve primitive: the body of resolve.extract_agent for ONE root
# ----------------------------------------------------------------------------------------------
class Res:
    pass


def resolve_at(full, a, node, pi_a, *, w=None, q=None, K=K, rnr=None, tree=None, depth_limited=False):
    """Re-solve the subgame rooted at node `node` (index in the full tree).

    pi_a  our reach (range) at the root: product of our own past action probabilities  [N]
    w     opponent counterfactual-value ceilings  [N]  -> SAFE (DeepStack gadget on the opponent)
    q     opponent range carried from an earlier solve [N] -> NAIVE (no gadget)
    w = q = None: the true game start, opponent range = everything possible.
    rnr   dict(p=..., kind='station') -> restricted Nash response (opponent reach seen by us is
          p * model + (1-p) * free CFR+ player).  The ONLY change to the solver.
    tree  an already-built subtree (used for the off-menu variant), else built from `full`.
    """
    o = 1 - a
    t0 = time.perf_counter()
    F = tree if tree is not None else PublicTree(full.cfg, full.B, [full.states[node]],
                                                 [full.cscale[node]], depth_limited=depth_limited)
    reach = [None, None]
    reach[a] = pi_a[None, :].copy()
    reach[o] = (q if q is not None else full.bmask[node])[None, :].copy()
    gadget = dict(player=o, w=np.asarray(w)[None, :]) if w is not None else None
    rnr_arg = None
    if rnr is not None:
        rnr_arg = dict(player=o, p=rnr["p"], eff=model_eff(F, o, rnr["kind"]))
    S = Solver(F, reach, gadget=gadget, rnr=rnr_arg)
    S.iterate(K)
    e = [S.avg_eff(0), S.avg_eff(1)]
    gf = S.avg_follow() if gadget is not None else None
    v0, v1, r0, r1 = values(S, e[0], e[1], g_follow=gf)
    r = Res()
    r.F, r.S, r.e, r.v, r.r, r.gf = F, S, e, [v0, v1], [r0, r1], gf
    r.seconds = time.perf_counter() - t0
    return r


def sub_index(R, full, key):
    """index of the node with public key `key` inside the re-solved subtree R.F"""
    return R.F.key_index[key]


@contextmanager
def p1_offmenu_bet_after_check(fracs):
    """E3 inserts an off-menu size only for P0's OPENING bet (game.legal_actions(extra_open=...)).
    Here the opponent is P1 and bets after our check, which the existing hook cannot express, so
    for the duration of the block we wrap tree.legal_actions: at P1's node right after P0's check
    we add the extra bet sizes (pot fractions) before the all-in.  No file is modified."""
    orig = tree_mod.legal_actions

    def patched(cfg, s, extra_open=None):
        acts = orig(cfg, s, extra_open)
        if s.to_act == 1 and s.n_aggr == 0 and s.n_act == 1:       # P1, P0 just checked
            have = {round(a[1], 9) for a in acts if a[2] == "bet" and a[1] is not None}
            extra = []
            for f in fracs:
                target = s.contrib[0] + f * s.pot
                if target < cfg.stack - 1e-9 and round(target, 9) not in have:
                    extra.append((f"b{f:g}", target, "bet"))
            acts = acts[:-1] + extra + acts[-1:]                    # keep all-in last
        return acts

    tree_mod.legal_actions = patched
    try:
        yield
    finally:
        tree_mod.legal_actions = orig


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    cfg, B, full = setup()
    N, a = full.N, AGENT
    o = 1 - a
    assert a == 0, "scenario is written for the out-of-position seat (P0)"
    T_wall = time.perf_counter()

    flop_key = f"{PRE_PATH}/d{FLOP_CARD}"
    dec_key = f"{flop_key}/k/{OPP_BET}"          # the agent's decision node (it checked, P1 bet)
    n_flop, n_dec = node_by_key(full, flop_key), node_by_key(full, dec_key)
    n_chk = node_by_key(full, f"{flop_key}/k")      # P1's decision node right after our check

    hr("WALKTHROUGH: one decision of the safe continual re-solving agent "
       f"(oracle leaves, K={K} CFR+ iterations)")
    out(f"game: {cfg.n_cards} cards ({cfg.n_ranks} ranks x {cfg.n_suits} suits), ante {cfg.ante:g}, "
        f"stacks {cfg.stack:g}, menu open {{0.5 pot, 1 pot, all-in}}, raise {{1 pot, all-in}}, then all-in")
    out(f"cards: " + " ".join(f"{c}={card(c)}" for c in range(N)) + "   (strength = rank + 5*#board cards of that rank)")
    out(f"agent = P{a} (acts first on every street = out of position); opponent = P{o}")
    out("units: chips are in antes; 'mA' = milli-antes (1 mA = 0.001 ante); one hand = one private card")

    # ------------------------------------------------------------------------------------------
    # [1] preflop root: the whole game below, no gadget
    # ------------------------------------------------------------------------------------------
    hr("CHAIN STEP [1]  re-solve at the preflop root (game start: nothing to enforce, no gadget)")
    R1 = resolve_at(full, a, 0, np.ones(N), K=K)
    out(f"  tree: {R1.F.M} public nodes (the whole game); {R1.seconds:.1f} s wall-clock")
    # agent's preflop strategy at the root and the opponent's reply on the chosen path
    eff_a = full.eff_template.copy()
    eff_a[full.E[a]] = np.repeat(1.0 / full.E_cnt[a], full.E_cnt[a])[:, None]   # uniform until a re-solve writes it
    cs, cc = full.child_start[0], full.n_child[0]
    eff_a[cs:cs + cc] = R1.e[a][cs:cs + cc]                     # [1] owns the preflop-root decision
    first = PRE_PATH.split("/")[0]
    out(f"  our preflop action probabilities at the root, per hand ({first} is on the chosen path):")
    labs = full.label[cs:cs + cc]
    out("      hand    " + "  ".join(f"{card(h):>6s}" for h in range(N)))
    for j, l in enumerate(labs):
        out(f"      {l:6s}  " + "  ".join(f"{R1.e[a][cs + j, h]:6.3f}" for h in range(N)))
    # what [1] hands over to the next re-solve: opponent CFVs and (for the naive agent) range at the flop root
    w_flop_root = R1.v[o][n_flop].copy()
    q_flop_root = R1.r[o][n_flop].copy()

    # ------------------------------------------------------------------------------------------
    # [2] flop street root: gadget with w from [1]
    # ------------------------------------------------------------------------------------------
    hr(f"CHAIN STEP [2]  re-solve at the flop street root '{flop_key}'  (gadget with the CFVs handed over by [1])")
    pi_flop = forward(full, eff_a, full.bmask[:1])[n_flop]
    R2 = resolve_at(full, a, n_flop, pi_flop, w=w_flop_root, K=K)
    out(f"  tree: {R2.F.M} public nodes (flop + turn, oracle); {R2.seconds:.2f} s wall-clock")
    # [2] owns our decision at the flop root (check/bet) -> write it; it gives our range after the check
    cs0, cc0 = R2.F.child_start[0], R2.F.n_child[0]
    fs0 = full.child_start[n_flop]
    eff_a[fs0:fs0 + cc0] = R2.e[a][cs0:cs0 + cc0]
    # opponent info [2] hands over to the next re-solves on this street
    i_dec2 = sub_index(R2, full, dec_key)
    i_chk2 = sub_index(R2, full, f"{flop_key}/k")
    w_dec = R2.v[o][i_dec2].copy()           # gadget input of [3]: opponent CFV at the decision node
    q_dec = R2.r[o][i_dec2].copy()           # what a NAIVE agent would carry into [3] (this chain's range)
    w_chk = R2.v[o][i_chk2].copy()           # opponent CFV at its node right after our check (off-menu case)
    pi_chk_dec = None

    # ------------------------------------------------------------------------------------------
    # [3] the decision
    # ------------------------------------------------------------------------------------------
    pi_dec = forward(full, eff_a, full.bmask[:1])[n_dec]        # our range at the decision node
    R3 = resolve_at(full, a, n_dec, pi_dec, w=w_dec, K=K)
    R3d = PublicTree(cfg, B, [full.states[n_dec]], [full.cscale[n_dec]], depth_limited=True)
    st = full.states[n_dec]
    to_call = st.contrib[1] - st.contrib[0]
    cs3, cc3 = R3.F.child_start[0], R3.F.n_child[0]
    acts = R3.F.label[cs3:cs3 + cc3]
    board_mask = full.bmask[n_dec]

    hr("(a) THE PUBLIC NODE")
    out(f"  betting so far : {describe_key(dec_key)}")
    out(f"  key            : '{dec_key}'")
    out(f"  board          : flop card {card(FLOP_CARD)} (turn card not yet dealt)")
    out(f"  contributions  : P0 {st.contrib[0]:g}, P1 {st.contrib[1]:g}  -> pot {st.pot:g}")
    out(f"  stacks behind  : P0 {cfg.stack - st.contrib[0]:g}, P1 {cfg.stack - st.contrib[1]:g}")
    out(f"  to act         : P{st.to_act} (the agent) facing {to_call:g} into a pot of {st.pot - to_call:g}; "
        f"call needs {100 * to_call / (st.pot + to_call):.1f}% equity; legal actions {acts}")

    hr("(b) THE AGENT'S OWN RANGE at this node (carried from its earlier solves)")
    out("  pi_a(h) = product of the agent's own action probabilities on the path (preflop root [1], flop check [2]),")
    out("  0 for the flop card (card removal).  It is an INPUT (exact, never approximated).")
    out("      hand    " + "  ".join(f"{card(h):>6s}" for h in range(N)))
    out("      pi_a    " + "  ".join(("    --" if not board_mask[h] else f"{pi_dec[h]:6.3f}") for h in range(N)))
    out(f"      (preflop {first}: " + " ".join(f"{R1.e[a][cs + labs.index(first), h]:.3f}" for h in range(N))
        + ")  x  (flop check: " + " ".join(f"{eff_a[fs0 + R2.F.label[cs0:cs0 + cc0].index('k'), h]:.3f}" for h in range(N)) + ")")

    cfv_scale = full.cscale[n_dec] * np.array([pi_dec.sum() - pi_dec[h] * board_mask[h] for h in range(N)])
    # (units) absolute CFV = cscale * sum_h pi_a(h) * u(h', h);  divide by cscale * (our range mass excluding h')
    # -> average net chips the opponent hand h' makes from our range at this node (antes, incl. chips already in)
    hr("(c) THE OPPONENT'S COUNTERFACTUAL-VALUE CEILINGS w(h')  = the gadget inputs, from the previous solve [2]")
    out("  w(h') = value of the opponent holding h' at THIS node under [2]'s solution.  The gadget lets the opponent")
    out("  'terminate' with w(h') per hand instead of entering the subgame, so the new strategy cannot hold any")
    out("  hand below its old value (it can never be exploited more than [2] allowed).")
    out("      hand    " + "  ".join(f"{card(h):>8s}" for h in range(N)))
    out("      w  [mA] " + "  ".join(("      --" if not board_mask[h] else f"{mA(w_dec[h]):8.2f}") for h in range(N)))
    with np.errstate(invalid="ignore", divide="ignore"):
        w_norm = w_dec / cfv_scale
    out("      w/(cs*M)" + " " + "  ".join(("      --" if not board_mask[h] else f"{w_norm[h]:8.3f}") for h in range(N)))
    out("      ('w/(cs*M)' = w divided by the chance scale and our range mass excluding h': the average NET chips the")
    out("       opponent's hand makes against our range here, chips already in the pot included.)")
    out(f"      for contrast, the naive agent would carry the opponent RANGE instead (see (f)).")

    hr("(d) LOCAL SUBTREE SIZE (what this re-solve actually builds)")
    s_or, s_dl = R3.F.summary(), R3d.summary()
    out(f"  ORACLE tree (what is solved here; turn street kept below the flop): {s_or['public_nodes']} public nodes = "
        f"{s_or['decision']} decision, {s_or['chance']} chance, {s_or['fold']} fold terminals, "
        f"{s_or['showdown']} showdown terminals, {s_or['leaf']} street-boundary leaves, {s_or['levels']} levels")
    out(f"  DEPTH-LIMITED tree (same root, cut where the street closes; leaves would be valued by V): "
        f"{s_dl['public_nodes']} public nodes = {s_dl['decision']} decision, {s_dl['fold']} fold terminals, "
        f"{s_dl['showdown']} showdown terminals, {s_dl['leaf']} street-boundary LEAVES, {s_dl['levels']} levels")
    out(f"  (the full game has 54,957 public nodes; solver rows = nodes x {N} hands; "
        f"rows here {R3.F.M * N}; info sets here {R3.F.info_sets()})")

    # reference: what the preflop full solve [1] plays at this node (an equilibrium is not unique -> only a sanity check)
    ref_e = R1.e[a][full.child_start[n_dec]:full.child_start[n_dec] + full.n_child[n_dec]]

    hr(f"(e) THE RE-SOLVED STRATEGY of the agent at this node, EVERY hand  (K={K})")
    def table(e_rows, title, with_inputs=True, rows=acts, extra=None):
        hdr = f"      {'hand':5s} {'pi_a':>6s} " + " ".join(f"{l:>7s}" for l in rows) + f" {'raise':>7s} {'fold':>7s}"
        out("  " + title)
        out(hdr)
        for h in range(N):
            if not board_mask[h]:
                out(f"      {card(h):5s}   (on the board)")
                continue
            p = e_rows[:, h]
            rz = sum(p[j] for j, l in enumerate(rows) if l[0] in "rA")
            fo = p[rows.index("f")] if "f" in rows else 0.0
            out(f"      {card(h):5s} {pi_dec[h]:6.3f} " + " ".join(f"{x:7.3f}" for x in p) + f" {rz:7.3f} {fo:7.3f}")
    table(R3.e[a][cs3:cs3 + cc3], "agent's re-solved action probabilities (rows sum to 1; hands with pi_a=0 are not constrained by play):")
    # range-weighted action frequencies
    wgt = pi_dec * board_mask
    freq = (R3.e[a][cs3:cs3 + cc3] * wgt[None, :]).sum(1) / wgt.sum()
    out("      range-weighted frequency: " + "  ".join(f"{l} {x:.3f}" for l, x in zip(acts, freq)))

    # ---- (f) opponent range implied by the re-solve
    hr("(f) THE OPPONENT'S RANGE after the bet: a BY-PRODUCT of the safe re-solve, not an input")
    rr = R3.r[o][0]            # opponent reach at the root as computed in the re-solve = root_reach * g_follow
    out("  safe re-solve: the opponent's root reach is root_reach (all hands possible) x g_follow, where g_follow(h') is the")
    out("  gadget's learnt probability that hand h' 'enters the subgame' rather than terminating at w(h').  Nothing")
    out("  about the opponent's range was passed in; this is what the equilibrium of the gadget game says it must be.")
    def norm(x):
        x = x * board_mask
        return x / x.sum()
    out("      hand            " + "  ".join(f"{card(h):>6s}" for h in range(N)))
    out("      g_follow (raw)  " + "  ".join(("    --" if not board_mask[h] else f"{R3.gf[0][h]:6.3f}") for h in range(N)))
    out("      safe range  %   " + "  ".join(("    --" if not board_mask[h] else f"{100 * norm(rr)[h]:6.1f}") for h in range(N)) + "   <- OUTPUT of the safe re-solve")
    out("      [2]'s range  %  " + "  ".join(("    --" if not board_mask[h] else f"{100 * norm(q_dec)[h]:6.1f}") for h in range(N)) + "   <- reach of this node in the previous solve [2] (what a naive agent carries from its own last solve)")
    # naive chain: [2n] no gadget, q0 = range handed over by the naive [1]; then [3n] with q0 from [2n]
    hr("      NAIVE counterpart: same chain but every re-solve takes the opponent RANGE carried from the previous one as input")
    R2n = resolve_at(full, a, n_flop, pi_flop, q=q_flop_root, K=K)
    eff_an = eff_a.copy()
    eff_an[fs0:fs0 + cc0] = R2n.e[a][cs0:cs0 + cc0]
    pi_dec_n = forward(full, eff_an, full.bmask[:1])[n_dec]
    q_dec_n = R2n.r[o][sub_index(R2n, full, dec_key)].copy()
    R3n = resolve_at(full, a, n_dec, pi_dec_n, q=q_dec_n, K=K)
    out("      hand            " + "  ".join(f"{card(h):>6s}" for h in range(N)))
    out("      naive input  %  " + "  ".join(("    --" if not board_mask[h] else f"{100 * norm(q_dec_n)[h]:6.1f}") for h in range(N)) + "   <- INPUT of the naive agent (a range, trusted as is)")
    out("      naive our pi_a  " + "  ".join(("    --" if not board_mask[h] else f"{pi_dec_n[h]:6.3f}") for h in range(N)) + "   <- the naive agent's own range here (its [2n] check differs)")
    out("      safe our pi_a   " + "  ".join(("    --" if not board_mask[h] else f"{pi_dec[h]:6.3f}") for h in range(N)))
    table(R3n.e[a][R3n.F.child_start[0]:R3n.F.child_start[0] + R3n.F.n_child[0]], "NAIVE agent's action probabilities at the same node:")
    out("      difference safe-vs-naive in range-weighted frequency: " + "  ".join(
        f"{l} {x:+.3f}" for l, x in zip(acts, freq - (R3n.e[a][R3n.F.child_start[0]:R3n.F.child_start[0] + R3n.F.n_child[0]] * wgt[None, :]).sum(1) / wgt.sum())))

    hr("(g) WALL-CLOCK of the re-solve [3] (tree build + 300 CFR+ iterations + average strategy/values), single root, 1 core")
    out(f"  [3] this decision    : {R3.seconds * 1000:8.1f} ms   ({R3.F.M} nodes, {K} iterations)")
    out(f"  [2] flop street root : {R2.seconds * 1000:8.1f} ms   ({R2.F.M} nodes)")
    out(f"  [1] preflop root     : {R1.seconds * 1000:8.1f} ms   ({R1.F.M} nodes, whole game)")
    out("  (README E5 reports 0.593 s for a flop-root oracle re-solve; wall-clock varies with machine load)")

    # ------------------------------------------------------------------------------------------
    # off-menu bet
    # ------------------------------------------------------------------------------------------
    hr(f"OFF-MENU CASE: the opponent bets {OFF_MENU:g} pot after our check (menu has only 0.5 pot, 1 pot, all-in)")
    out("  The agent has no stored values for this node (it is not in the tree [2] solved), so it cannot continue [3].")
    out("  E3's recipe: add the size to the tree, re-solve from the root of that decision with the gadget, and use the")
    out("  result only below the new action.  E3's own hook (extra_open) only inserts P0's OPENING bet, so it cannot")
    out("  put a size into P1's reply to a check; here tree.legal_actions is wrapped for this block (no file edited).")
    out(f"  Re-solve root = P1's node right after our check ('{flop_key}/k'): our range there is pi_a after the check,")
    out("  the gadget ceilings are the opponent's CFVs at that same node from [2] (w_chk).  We have already committed to the")
    out("  check, so the street root (our own decision) is not re-opened.")
    pi_chk = forward(full, eff_a, full.bmask[:1])[n_chk]
    with p1_offmenu_bet_after_check([OFF_MENU]):
        t0 = time.perf_counter()
        Fo = PublicTree(cfg, B, [full.states[n_chk]], [full.cscale[n_chk]], depth_limited=False)
    Ro = resolve_at(full, a, n_chk, pi_chk, w=w_chk, K=K, tree=Fo)
    Ro.seconds += time.perf_counter() - t0 - Ro.seconds if False else 0.0
    so = Fo.summary()
    off_tag = f"b{OFF_MENU:g}"
    k_off = f"{flop_key}/k/{off_tag}"
    i_off = Fo.key_index[k_off]
    cso, cco = Fo.child_start[0], Fo.n_child[0]
    out(f"  augmented subtree: {so['public_nodes']} public nodes ({so['decision']} decision, {so['fold']} fold, "
        f"{so['showdown']} showdown); P1's actions at the root: {list(Fo.label[cso:cso + cco])}")
    out(f"  wall-clock of this re-solve (tree build + {K} iterations): {Ro.seconds * 1000:.1f} ms")
    st_o = Fo.states[i_off]
    out(f"  node after the off-menu bet: pot {st_o.pot:g}, we face {st_o.contrib[1] - st_o.contrib[0]:g}; "
        f"call needs {100 * (st_o.contrib[1] - st_o.contrib[0]) / (st_o.pot + st_o.contrib[1] - st_o.contrib[0]):.1f}% equity")
    ro_off = Ro.r[o][i_off]
    out("  opponent range after the off-menu bet (reach of that node in the re-solve; normalised %), a by-product:")
    out("      hand            " + "  ".join(f"{card(h):>6s}" for h in range(N)))
    out("      range %         " + "  ".join(("    --" if not board_mask[h] else f"{100 * norm(ro_off)[h]:6.1f}") for h in range(N)))
    out("      P1 bets this size with prob (per hand) " + " ".join(("   --" if not board_mask[h] else f"{ro_off[h] / max(Ro.r[o][0][h], 1e-300):5.3f}") for h in range(N)))
    cs_o, cc_o = Fo.child_start[i_off], Fo.n_child[i_off]
    acts_o = Fo.label[cs_o:cs_o + cc_o]
    out(f"  agent's strategy facing the {OFF_MENU:g}-pot bet (legal {acts_o}); pi_a = our range after the check:")
    out(f"      {'hand':5s} {'pi_a':>6s} " + " ".join(f"{l:>7s}" for l in acts_o) + f" {'raise':>7s} {'fold':>7s}")
    pi_o_node = Ro.r[a][i_off]
    for h in range(N):
        if not board_mask[h]:
            out(f"      {card(h):5s}   (on the board)")
            continue
        p = Ro.e[a][cs_o:cs_o + cc_o, h]
        rz = sum(p[j] for j, l in enumerate(acts_o) if l[0] in "rA")
        out(f"      {card(h):5s} {pi_o_node[h]:6.3f} " + " ".join(f"{x:7.3f}" for x in p) + f" {rz:7.3f} {p[acts_o.index('f')]:7.3f}")
    wo = pi_o_node * board_mask
    fo_ = (Ro.e[a][cs_o:cs_o + cc_o] * wo[None, :]).sum(1) / wo.sum()
    out("      range-weighted frequency: " + "  ".join(f"{l} {x:.3f}" for l, x in zip(acts_o, fo_)))

    # ------------------------------------------------------------------------------------------
    # RNR: the one-input swap
    # ------------------------------------------------------------------------------------------
    hr("RNR EXPLOITATION: the same decision [3] with the opponent modelled as the 'station' (never folds, never raises)")
    out("  Inputs held fixed: our range pi_a and the gadget ceilings w from the safe chain above.  The ONLY change:")
    out("  Solver(..., rnr=dict(player=1, p=p, eff=model_eff(tree, 1, 'station'))), i.e. while WE traverse, the opponent's")
    out("  reach is  p * reach(station) + (1-p) * reach(free CFR+ opponent)   (cfr.py opp_reach, 2 lines).")
    out("  Note: the station never bets, so this node has probability 0 under the model; the hook applies the model's")
    out("  behaviour BELOW the root (it calls every raise, never folds), starting from the root range.")
    for p_ in RNR_PS:
        Rr = resolve_at(full, a, n_dec, pi_dec, w=w_dec, K=K, rnr=dict(p=p_, kind="station"))
        table(Rr.e[a][cs3:cs3 + cc3], f"RNR p = {p_:g}  (re-solve {Rr.seconds * 1000:.0f} ms):")
        fr = (Rr.e[a][cs3:cs3 + cc3] * wgt[None, :]).sum(1) / wgt.sum()
        out("      range-weighted frequency: " + "  ".join(f"{l} {x:.3f}" for l, x in zip(acts, fr)))
        out("      change vs safe Nash re-solve: " + "  ".join(f"{l} {x - y:+.3f}" for l, x, y in zip(acts, fr, freq)))
    out("")
    out(f"total wall-clock of this script: {time.perf_counter() - T_wall:.1f} s")

    with open(os.path.join(RES, "walkthrough.txt"), "w") as f:
        f.write("\n".join(_LOG) + "\n")


if __name__ == "__main__":
    main()
