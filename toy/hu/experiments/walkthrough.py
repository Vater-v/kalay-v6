"""Walkthrough: ONE complete decision of the safe continual re-solving agent, with the
mechanism's actual numbers printed (exact / ORACLE leaves, K = 300 CFR+ iterations).

Run (from toy/hu):   python3 experiments/walkthrough.py        (~25 s, one core)
The output is also written to results/walkthrough.txt.  Nothing from results/ is needed (no
blueprint, no value nets): the chain of re-solves is recomputed here.  If results/ holds E2c's
agent_oracle_safe_eff0.npy the script also checks its own numbers against that stored strategy.

SCENARIO.  Player 0 acts first on every street, so P0 is the out-of-position seat; the agent is P0.
"Faces a bet after checking" is therefore:
    preflop  P0 checks, P1 checks                    (pot 2, 11 behind each)
    flop     card dealt; P0 (agent) CHECKS, P1 BETS; the agent must act (fold / call / raise)
To act here the agent chains three re-solves; each uses only what the previous one handed over
(this is resolve.extract_agent restricted to this one path, and reproduces its stored strategy):

  [1] preflop root   whole game below, no gadget (the game start is common knowledge)
        hands over:  our range at the flop root, opponent CFVs w(flop root, h')
  [2] flop street root   gadget on the opponent with w from [1]; flop + turn kept below (ORACLE leaves)
        hands over:  our range after our check, opponent CFVs at our next decision node
  [3] the decision node itself   gadget on the opponent with w from [2]          <- the decision shown

Printed for [3]: (a) public node, (b) our range, (c) gadget inputs w, (d) subtree size, (e) our
re-solved strategy for every hand, (f) the opponent range implied by the re-solve (a by-product) next
to the range a naive agent carries in, (g) wall-clock.  Then the same decision (1) when the opponent's
bet size is NOT on the menu and (2) under restricted Nash response (RNR) against the 'station' model.
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
from tree import PublicTree, CHANCE                    # noqa: E402
from opponents import model_eff                        # noqa: E402

# ----------------------------------------------------------------------------------------------
# SCENARIO KNOBS (env vars only for exploring other decisions; the saved output uses the defaults)
# ----------------------------------------------------------------------------------------------
K = int(os.environ.get("WT_K", 300))               # CFR+ iterations in every re-solve
AGENT = 0                                          # fixed: the scenario is written for the out-of-position seat
PRE_PATH = os.environ.get("WT_PRE", "k/k")         # preflop line: P0 checks, P1 checks
FLOP_CARD = int(os.environ.get("WT_CARD", 6))      # flop card id (rank = id // 2); 6 = Ks
OPP_BET = os.environ.get("WT_BET", "b1")           # P1's on-menu flop bet after our check: b1 = pot
OFF_MENU = float(os.environ.get("WT_OFF", 0.75))   # P1's off-menu bet, as a fraction of the pot
RNR_PS = (0.01, 0.25)                              # RNR weight on the 'station' model

_LOG = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.append(s)


def hr(title):
    out("")
    out("=" * 100)
    out(title)
    out("=" * 100)


# ----------------------------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------------------------
def card(c):
    """Card id -> 'Ts','Th','Js',... (rank letter + suit letter; rank = id // 2)."""
    return "TJQKA"[c // CFG.n_suits] + "sh"[c % CFG.n_suits]


def describe_path(T, key):
    """Betting sequence of public key `key`, street by street, e.g.
    'preflop: P0 checks, P1 checks  |  flop Ks: P0 checks, P1 bets 2 (= 1 pot, total in 3)'."""
    n, segs, cur, street = 0, [], [], "preflop"
    for lab in key.split("/") if key else []:
        actor = int(T.player[n])
        cs = int(T.child_start[n])
        c = cs + T.label[cs: cs + int(T.n_child[n])].index(lab)
        if T.kind[n] == CHANCE:
            segs.append(f"{street}: " + ", ".join(cur))
            street, cur = f"{'flop' if len(T.states[n].board) == 0 else 'turn'} {card(int(lab[1:]))}", []
        else:
            put = T.contrib[c][actor] - T.contrib[n][actor]
            tot = T.contrib[c][actor]
            what = {"k": "checks", "c": f"calls {put:g}", "f": "folds", "A": f"goes ALL-IN (+{put:g}, total in {tot:g})"}.get(
                lab, f"{'bets' if lab[0] == 'b' else 'raises'} {put:g} (= {lab[1:]} pot, total in {tot:g})")
            cur.append(f"P{actor} {what}")
        n = c
    segs.append(f"{street}: " + ", ".join(cur))
    return "  |  ".join(segs)


def hand_row(label, vec, mask, fmt="{:6.3f}", w=6, head=14):
    return f"      {label:<{head}s}" + " ".join((" " * (w - 2) + "--") if not mask[h] else fmt.format(vec[h])
                                                 for h in range(len(vec)))


def hand_header(N, w=6, head=14):
    return f"      {'hand':<{head}s}" + " ".join(f"{card(h):>{w}s}" for h in range(N))


def strat_table(cols, labs, pi, mask, N):
    """cols = [(title, E[A,N])...]; one row per hand; per group the probability of each action."""
    out("      " + f"{'hand':5s} {'pi_a':>6s} " + " | ".join(f"{t:^{7 * len(labs) - 1}s}" for t, _ in cols))
    out("      " + f"{'':5s} {'':>6s} " + " | ".join(" ".join(f"{l:>6s}" for l in labs) for _ in cols))
    for h in range(N):
        if not mask[h]:
            out(f"      {card(h):5s} (on the board: impossible)")
            continue
        out(f"      {card(h):5s} {pi[h]:6.3f} " + " | ".join(" ".join(f"{E[j, h]:6.3f}" for j in range(len(labs))) for _, E in cols))
    wgt = pi * mask
    out("      " + f"{'freq':5s} {'':>6s} " + " | ".join(" ".join(f"{(E[j] * wgt).sum() / wgt.sum():6.3f}" for j in range(len(labs))) for _, E in cols)
        + "     <- range-weighted action frequency")


# ----------------------------------------------------------------------------------------------
# the re-solve primitive: the body of resolve.extract_agent, for ONE root
# ----------------------------------------------------------------------------------------------
class Res:
    pass


def resolve_at(full, a, node, pi_a, *, w=None, q=None, K=K, rnr=None, build=None):
    """Re-solve the subgame rooted at node `node` of the full tree (we are seat `a`).

    pi_a   our reach (range) at the root = product of our own past action probabilities  [N]
    w      opponent counterfactual-value ceilings [N]     -> SAFE  (DeepStack gadget on the opponent)
    q      opponent range carried from an earlier solve [N] -> NAIVE (range trusted as is, no gadget)
    w = q = None: the game start, the opponent's range is just "any card not on the board".
    rnr    dict(p=..., kind='station') -> restricted Nash response: the ONLY change is that the opponent
           reach seen by us is  p * reach(model) + (1-p) * reach(free CFR+ opponent)   (cfr.py opp_reach)
    build  optional callable returning the subtree (used by the off-menu case); counted in the wall-clock.
    Returns an object with the subtree F, solver S, average strategies e, values v, reaches r, the
    gadget's follow probabilities gf and the wall-clock seconds."""
    o = 1 - a
    t0 = time.perf_counter()
    F = build() if build else PublicTree(full.cfg, full.B, [full.states[node]], [full.cscale[node]])
    reach = [None, None]
    reach[a] = pi_a[None, :].copy()
    reach[o] = (q if q is not None else full.bmask[node])[None, :].copy()
    gadget = dict(player=o, w=np.asarray(w)[None, :]) if w is not None else None
    rnr_arg = dict(player=o, p=rnr["p"], eff=model_eff(F, o, rnr["kind"])) if rnr else None
    S = Solver(F, reach, gadget=gadget, rnr=rnr_arg)
    S.iterate(K)
    e = [S.avg_eff(0), S.avg_eff(1)]
    gf = S.avg_follow()[0] if gadget is not None else None
    v0, v1, r0, r1 = values(S, e[0], e[1], g_follow=S.avg_follow() if gadget is not None else None)
    r = Res()
    r.F, r.S, r.e, r.v, r.r, r.gf = F, S, e, [v0, v1], [r0, r1], gf
    r.seconds = time.perf_counter() - t0
    return r


def kids(R, i):
    """(labels, child_start, n_child) of node i of the re-solved subtree."""
    cs, cc = int(R.F.child_start[i]), int(R.F.n_child[i])
    return R.F.label[cs: cs + cc], cs, cc


@contextmanager
def p1_offmenu_bet_after_check(fracs):
    """E3 inserts an off-menu size only as P0's OPENING bet (game.legal_actions(extra_open=...)).  Here the
    opponent is P1 and bets right after our check, which that hook cannot express, so inside this block
    tree.legal_actions is wrapped: at P1's node after P0's check the extra pot fractions are added as bets
    (before the all-in).  Nothing on disk is modified."""
    orig = tree_mod.legal_actions

    def patched(cfg, s, extra_open=None):
        acts = orig(cfg, s, extra_open)
        if s.to_act == 1 and s.n_aggr == 0 and s.n_act == 1:        # P1 to act, P0 just checked
            have = {round(x[1], 9) for x in acts if x[2] == "bet" and x[1] is not None}
            extra = []
            for f in fracs:
                target = s.contrib[0] + f * s.pot          # to_call = 0, so target = opponent's chips + f * pot
                if target < cfg.stack - 1e-9 and round(target, 9) not in have:
                    extra.append((f"b{f:g}", target, "bet"))
            acts = acts[:-1] + extra + acts[-1:]
        return acts

    tree_mod.legal_actions = patched
    try:
        yield
    finally:
        tree_mod.legal_actions = orig


# ----------------------------------------------------------------------------------------------
def main():
    cfg, B, full = setup()
    N, a = full.N, AGENT
    o = 1 - a
    wall0 = time.perf_counter()
    flop_key = f"{PRE_PATH}/d{FLOP_CARD}"
    dec_key = f"{flop_key}/k/{OPP_BET}"             # the agent's decision node (it checked, P1 bet)
    n_flop, n_dec, n_chk = (full.key_index[k] for k in (flop_key, dec_key, f"{flop_key}/k"))

    hr(f"WALKTHROUGH: one decision of the safe continual re-solving agent  (oracle leaves, K = {K} CFR+ iterations)")
    out(f"game   : {N} cards = {cfg.n_ranks} ranks x {cfg.n_suits} suits, ante {cfg.ante:g}, stacks {cfg.stack:g} (ante included), 1 private card each")
    out("cards  : " + " ".join(f"{c}={card(c)}" for c in range(N)) + "   (strength = rank + 5 * #board cards of that rank; one board card on the flop)")
    out("menu   : open bet {0.5 pot, 1 pot, all-in}; first raise {1 pot, all-in}; second raise all-in; P0 acts first on every street")
    out(f"agent  : P{a} = out of position (acts first);   opponent: P{o}.   units: antes; mA = milli-antes = 0.001 ante")
    out("reading guide: the re-solve at the decision [3] takes exactly TWO inputs from the past, our range pi_a (b) and the")
    out("  opponent's value ceilings w (c); with ORACLE leaves nothing else (the turn is solved jointly below, d).  It returns our")
    out("  strategy for every hand (e); the opponent's range comes out as a by-product (f).  Nothing is stored between decisions")
    out("  except those two vectors per future node.  Chain: [1] preflop root -> [2] flop root -> [3] the decision.")

    # ---------------------------------------------------------------- [1] preflop root
    hr("CHAIN STEP [1]   re-solve at the preflop root   (game start: nothing to enforce, no gadget)")
    R1 = resolve_at(full, a, 0, np.ones(N))
    out(f"  subtree: {R1.F.M} public nodes (the whole game);  wall-clock {R1.seconds:.1f} s")
    labs0, cs, cc = kids(R1, 0)
    out("  our preflop strategy at the root (this is where the range at the flop root comes from):")
    out(hand_header(N))
    for j, l in enumerate(labs0):
        out(hand_row(l, R1.e[a][cs + j], np.ones(N)))
    eff_a = full.eff_template.copy()                      # the agent's strategy, filled in as re-solves happen
    eff_a[full.E[a]] = np.repeat(1.0 / full.E_cnt[a], full.E_cnt[a])[:, None]
    eff_a[cs: cs + cc] = R1.e[a][cs: cs + cc]
    w_flop = R1.v[o][n_flop].copy()                       # [1] hands over: opponent CFVs at the flop root

    # ---------------------------------------------------------------- [2] flop street root
    hr(f"CHAIN STEP [2]   re-solve at the flop street root '{flop_key}'   (gadget with the CFVs handed over by [1])")
    pi_flop = forward(full, eff_a, full.bmask[:1])[n_flop]
    R2 = resolve_at(full, a, n_flop, pi_flop, w=w_flop)
    out(f"  subtree: {R2.F.M} public nodes (flop + turn);  wall-clock {R2.seconds * 1000:.0f} ms")
    labs2, cs2, cc2 = kids(R2, 0)
    eff_a[full.child_start[n_flop]: full.child_start[n_flop] + cc2] = R2.e[a][cs2: cs2 + cc2]   # [2] owns our flop-root decision
    out("  our action at the flop root (check / bet ...): probability per hand")
    out(hand_header(N))
    for j, l in enumerate(labs2):
        out(hand_row(l, R2.e[a][cs2 + j], full.bmask[n_flop]))
    i_dec2, i_chk2 = R2.F.key_index[dec_key], R2.F.key_index[f"{flop_key}/k"]
    w_dec = R2.v[o][i_dec2].copy()          # [2] hands over: opponent CFVs at the decision node  -> gadget input of [3]
    q_dec_prev = R2.r[o][i_dec2].copy()     # ... and the opponent range there (what a NAIVE agent would carry)
    w_chk = R2.v[o][i_chk2].copy()          # opponent CFVs at P1's node right after our check (off-menu case)

    # ---------------------------------------------------------------- [3] the decision
    pi_dec = forward(full, eff_a, full.bmask[:1])[n_dec]
    R3 = resolve_at(full, a, n_dec, pi_dec, w=w_dec)
    acts, cs3, cc3 = kids(R3, 0)
    E3 = R3.e[a][cs3: cs3 + cc3]
    st = full.states[n_dec]
    mask = full.bmask[n_dec]
    to_call = st.contrib[1] - st.contrib[0]

    hr("(a) THE PUBLIC NODE")
    out(f"  betting so far : {describe_path(full, dec_key)}")
    out(f"  key            : '{dec_key}'      board: flop card {card(FLOP_CARD)}, turn not yet dealt")
    out(f"  contributions  : P0 {st.contrib[0]:g}, P1 {st.contrib[1]:g}  -> pot {st.pot:g};  stacks behind: P0 {cfg.stack - st.contrib[0]:g}, P1 {cfg.stack - st.contrib[1]:g}")
    out(f"  to act         : P{st.to_act} (the agent), facing a bet of {to_call:g} into {st.pot - to_call:g}; calling needs "
        f"{100 * to_call / (st.pot + to_call):.1f}% equity;  legal actions {acts}")

    hr("(b) THE AGENT'S OWN RANGE at this node   (INPUT, exact, carried from its earlier solves)")
    out("  pi_a(h) = product of the agent's own action probabilities along the path:")
    out("            preflop check probability [1]  x  flop check probability [2];   0 for the card on the board.")
    out(hand_header(N))
    first = PRE_PATH.split("/")[0]
    out(hand_row(f"preflop '{first}'", eff_a[cs + labs0.index(first)], np.ones(N)))
    out(hand_row("flop 'k'", eff_a[full.child_start[n_flop] + labs2.index("k")], full.bmask[n_flop]))
    out(hand_row("pi_a (product)", pi_dec, mask))
    out(f"  (sum over the {int(mask.sum())} possible hands = {pi_dec.sum():.3f}; mean {pi_dec.sum() / mask.sum():.3f} = how often the agent's own play brings a random hand here)")

    # the opponent ceilings, in absolute CFV units and as net chips per unit of our range
    M = pi_dec.sum() - pi_dec * mask
    with np.errstate(invalid="ignore", divide="ignore"):
        w_chips = w_dec / (full.cscale[n_dec] * M)
    hr("(c) THE OPPONENT'S COUNTERFACTUAL-VALUE CEILINGS w(h')   = the gadget inputs, from the previous solve [2]")
    out("  w(h') = what the opponent holding h' is worth at THIS node under [2]'s solution.  In the re-solve each hand may")
    out("  'terminate' and take w(h') instead of entering the subgame, so (up to solver error) our new strategy cannot give")
    out("  any opponent hand more than w(h'): the re-solve cannot make us more exploitable than [2] already allowed.")
    out(hand_header(N, 8))
    out(hand_row("w   (mA)", np.array([mA(x) for x in w_dec]), mask, "{:8.2f}", 8))
    out(hand_row("w in net chips", w_chips, mask, "{:8.3f}", 8))
    out("  ('net chips' = w / (chance scale x our range mass excluding h'): the average NET chips (already-invested")
    out("   chips included) opponent hand h' makes against our range reaching this node.)")

    sd = PublicTree(cfg, B, [full.states[n_dec]], [full.cscale[n_dec]], depth_limited=True).summary()
    so = R3.F.summary()
    hr("(d) LOCAL SUBTREE SIZE")
    out(f"  solved here (ORACLE leaves: flop and turn both kept below the root): {so['public_nodes']} public nodes = "
        f"{so['decision']} decision + {so['chance']} chance + {so['fold']} fold terminals + {so['showdown']} showdown "
        f"terminals;  street-boundary leaves: {so['leaf']}")
    out(f"  same root, DEPTH-LIMITED at the end of the flop (what a value net would be asked about): "
        f"{sd['public_nodes']} public nodes = {sd['decision']} decision + {sd['fold']} fold + {sd['showdown']} showdown "
        f"+ {sd['leaf']} street-boundary LEAVES (street closed with chips behind: after our call, and after our raise + P1's call), {sd['levels']} levels")
    out(f"  (full game: 54,957 public nodes.  Solver rows = nodes x {N} hands = {so['public_nodes'] * N}; "
        f"information sets in this subtree: {so['info_sets']})")

    hr(f"(e) THE RE-SOLVED STRATEGY at this node, every hand   (K = {K})")
    out("  action probabilities of the agent; rows with a larger pi_a matter more; the last row is the range-weighted frequency.")
    ref = R1.e[a][full.child_start[n_dec]: full.child_start[n_dec] + full.n_child[n_dec]]
    strat_table([("re-solved [3] (safe)", E3)], acts, pi_dec, mask, N)
    wgt = pi_dec * mask
    out("  for reference, the full-game solve [1] (same K, an equilibrium is not unique, so only roughly comparable):")
    out("      range-weighted frequency  " + "  ".join(f"{l} {(ref[j] * wgt).sum() / wgt.sum():.3f}" for j, l in enumerate(acts)))
    # does the ceiling hold?  exact best response of the opponent (per hand) against the re-solved strategy
    br = values(R3.S, R3.e[0], R3.e[1], best_response_for=o, g_follow=R3.S.avg_follow())[o][0]
    out("  safety check: the opponent's exact best response against this strategy, per hand, vs the ceiling w (mA):")
    out(hand_header(N, 8))
    out(hand_row("ceiling w", mA(w_dec), mask, "{:8.2f}", 8))
    out(hand_row("best response", mA(br), mask, "{:8.2f}", 8))
    out(hand_row("BR - w", mA(br - w_dec), mask, "{:8.3f}", 8))
    gap, scale = np.abs(br - w_dec)[mask > 0].max(), np.abs(w_dec)[mask > 0].max()
    out(f"      largest |BR - w| = {mA(gap):.3f} mA against ceilings up to {mA(scale):.2f} mA (CFR+ error at K={K})."
        + ("  Every hand sits at its ceiling: the re-solve gives the opponent what [2] allowed, and no more." if gap < 0.05 * scale else ""))

    # ---- (f) opponent range
    hr("(f) THE OPPONENT'S RANGE after its bet: a BY-PRODUCT of the safe re-solve, not an input")
    rr = R3.r[o][0]

    def pct(x):
        x = x * mask
        return 100 * x / x.sum()
    out("  safe re-solve: no opponent range is passed in.  The opponent's root reach is (all cards possible) x g_follow(h'),")
    out("  g_follow = the gadget player's own equilibrium probability that h' ENTERS the subgame rather than taking w(h'); it plays")
    out("  the role of P(opponent got here | h'), i.e. how often hand h' bets the pot after our check.")
    out(hand_header(N))
    out(hand_row("g_follow", R3.gf, mask))
    out(hand_row("safe range  %", pct(rr), mask, "{:6.1f}"))
    out("                              ^ OUTPUT of the safe re-solve (this node's reach of the opponent)")
    out(hand_row("[2]'s range %", pct(q_dec_prev), mask, "{:6.1f}"))
    out("                              ^ the reach of this very node in the previous safe solve [2]; NOT used by the safe agent")
    # naive chain: [2n] with q carried from [1]; [3n] with q carried from [2n]
    R2n = resolve_at(full, a, n_flop, pi_flop, q=R1.r[o][n_flop].copy())
    eff_n = eff_a.copy()
    labs2n, cs2n, cc2n = kids(R2n, 0)
    eff_n[full.child_start[n_flop]: full.child_start[n_flop] + cc2n] = R2n.e[a][cs2n: cs2n + cc2n]
    pi_n = forward(full, eff_n, full.bmask[:1])[n_dec]
    q_n = R2n.r[o][R2n.F.key_index[dec_key]].copy()
    R3n = resolve_at(full, a, n_dec, pi_n, q=q_n)
    En = R3n.e[a][cs3: cs3 + cc3]
    out(hand_row("naive INPUT %", pct(q_n), mask, "{:6.1f}"))
    out("                              ^ what the NAIVE agent is handed instead: the range carried from its own previous (gadget-free)")
    out("                                solve [2n]; it trusts it as is.  Its own range here is also slightly different:")
    out(hand_row("naive pi_a", pi_n, mask))
    out(hand_row("safe  pi_a", pi_dec, mask))
    out("  the decision under each:")
    strat_table([("safe [3]", E3), ("naive [3n]", En)], acts, pi_dec, mask, N)
    fs = (E3 * wgt).sum(1) / wgt.sum()
    fn = (En * (pi_n * mask)).sum(1) / (pi_n * mask).sum()
    out("      naive minus safe, range-weighted frequency: " + "  ".join(f"{l} {x - y:+.3f}" for l, x, y in zip(acts, fn, fs)))
    out("  (README E2c, whole extracted agent, exact leaves, K=300: safe 23.7 mA/h exploitable, naive 520.5 mA/h.  Here: one node.)")

    hr("(g) WALL-CLOCK of the re-solve   (single root, 1 core, tree build + average strategy/values included)")
    out(f"  [3] this decision      : {R3.seconds * 1000:8.1f} ms   ({R3.F.M} nodes, {K} CFR+ iterations)")
    out(f"  [2] flop street root   : {R2.seconds * 1000:8.1f} ms   ({R2.F.M} nodes)")
    out(f"  [1] preflop root       : {R1.seconds * 1000:8.1f} ms   ({R1.F.M} nodes, whole game; the oracle agent pays this once, at the start of the hand)")

    # cross-check against E2c's stored strategy for the whole extracted agent
    stored = os.path.join(RES, "agent_oracle_safe_eff0.npy")
    if os.path.exists(stored) and K == 300:
        ag = np.load(stored)
        d1 = np.abs(ag[cs: cs + cc] - R1.e[a][cs: cs + cc]).max()
        d2 = np.abs(ag[full.child_start[n_flop]: full.child_start[n_flop] + cc2] - R2.e[a][cs2: cs2 + cc2]).max()
        d3 = np.abs(ag[full.child_start[n_dec]: full.child_start[n_dec] + cc3] - E3).max()
        out(f"  cross-check vs results/agent_oracle_safe_eff0.npy (E2c's extracted agent, batched forests): max |diff| of the strategy "
            f"at [1] {d1:.1e}, [2] {d2:.1e}, [3] {d3:.1e}")

    # ---------------------------------------------------------------- off-menu
    hr(f"OFF-MENU CASE: after our check the opponent bets {OFF_MENU:g} pot  (menu: 0.5 pot, 1 pot, all-in)")
    off_tag = f"b{OFF_MENU:g}"
    out("  The agent has no stored values for this action (it is not in the tree [2] solved), so [3] cannot be run as above.")
    out("  E3's recipe: add the size to the tree, re-solve with the gadget, use the result only below the new action.")
    out("  E3's own hook (extra_open) can only add P0's OPENING bet, so it cannot add a size to P1's reply to a check; inside")
    out("  this block tree.legal_actions is wrapped to add it (no file is edited).  Root of the re-solve = P1's node right after")
    out(f"  our check ('{flop_key}/k'): our check is already played, so the root range is our range after the check, and the")
    out("  gadget ceilings are the opponent's CFVs at that same node from [2].  Everything else is as in [3].")
    pi_chk = forward(full, eff_a, full.bmask[:1])[n_chk]

    def build_aug():
        with p1_offmenu_bet_after_check([OFF_MENU]):
            return PublicTree(cfg, B, [full.states[n_chk]], [full.cscale[n_chk]])
    Ro = resolve_at(full, a, n_chk, pi_chk, w=w_chk, build=build_aug)
    Fo = Ro.F
    sa = Fo.summary()
    labs_root, cso, cco = kids(Ro, 0)
    i_off = Fo.key_index[f"{flop_key}/k/{off_tag}"]
    acts_o, cs_o, cc_o = kids(Ro, i_off)
    st_o = Fo.states[i_off]
    to_call_o = st_o.contrib[1] - st_o.contrib[0]
    out(f"  augmented subtree: {sa['public_nodes']} public nodes ({sa['decision']} decision, {sa['fold']} fold, {sa['showdown']} showdown);"
        f"  P1's actions at the root: {list(labs_root)}")
    out(f"  node after the off-menu bet: pot {st_o.pot:g}, we face {to_call_o:g}, call needs {100 * to_call_o / (st_o.pot + to_call_o):.1f}% equity; legal {acts_o}")
    out(f"  wall-clock of this re-solve (build the augmented tree + {K} iterations): {Ro.seconds * 1000:.0f} ms")
    ro = Ro.r[o][i_off]
    p_bet = Ro.e[o][i_off]                                  # P1's probability of this action at the root, per hand
    out("  the opponent's range after the off-menu bet (reach of that node in the re-solve; a by-product again):")
    out(hand_header(N))
    out(hand_row("P(bet | h')", p_bet, mask))
    out(hand_row("range %", pct(ro), mask, "{:6.1f}"))
    pi_o = Ro.r[a][i_off]
    Eo = Ro.e[a][cs_o: cs_o + cc_o]
    out("  the agent's strategy facing it:")
    cols_o = [(f"re-solved vs {OFF_MENU:g} pot", Eo)]
    if acts_o == acts and np.allclose(pi_o, pi_dec):        # same legal actions and same range as at the on-menu bet
        cols_o.append((f"snap up to {OPP_BET}: play (e)", E3))     # what translating the size to the menu would play
    strat_table(cols_o, acts_o, pi_o, mask, N)
    br_o = values(Ro.S, Ro.e[0], Ro.e[1], best_response_for=o, g_follow=Ro.S.avg_follow())[o][0]
    out(f"  safety check: opponent best response (any of its actions incl. the new size) vs ceilings w_chk: largest excess "
        f"{mA((br_o - w_chk)[mask > 0].max()):.3f} mA, largest |BR - w| {mA(np.abs(br_o - w_chk)[mask > 0].max()):.3f} mA")

    # ---------------------------------------------------------------- RNR
    hr("RNR EXPLOITATION: the same decision [3] against the 'station' model (calls everything, never folds, never raises)")
    out("  Inputs held fixed: our range pi_a and the gadget ceilings w from the safe chain.  The ONE change is the opponent")
    out("  reach our traversal sees:  Solver(..., rnr=dict(player=1, p=p, eff=model_eff(tree, 1, 'station'))),  i.e.")
    out("      q_seen = p * reach(station) + (1-p) * reach(free CFR+ opponent)      (cfr.py opp_reach; 2 lines; p=0 is Nash).")
    out("  Caveat: the station never bets, so THIS node has probability 0 under the model; the hook applies the model's")
    out("  behaviour below the root (it calls every raise, never folds) starting from the root range.")
    cols = [("safe, p=0", E3)]
    for p_ in RNR_PS:
        Rr = resolve_at(full, a, n_dec, pi_dec, w=w_dec, rnr=dict(p=p_, kind="station"))
        cols.append((f"RNR p={p_:g}", Rr.e[a][cs3: cs3 + cc3]))
    strat_table(cols, acts, pi_dec, mask, N)
    out("  (re-solve wall-clock with RNR: same as without, "
        f"{Rr.seconds * 1000:.0f} ms)")
    out("")
    out(f"total wall-clock of this script: {time.perf_counter() - wall0:.1f} s")

    with open(os.path.join(RES, "walkthrough.txt"), "w") as f:
        f.write("\n".join(_LOG) + "\n")


if __name__ == "__main__":
    main()
