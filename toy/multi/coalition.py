"""Colluding joint strategies for seats 1 and 2 against seat 0 (the hero).

Both coalitions are built from the self-play strategy sigma* (so the colluders still play
"good poker" with their own card) and then overwritten at the nodes where collusion matters.
The coalition does NOT share cards (that would change the information structure; see README).

SOFTPLAY   - when seat 0 has folded, colluders only check/call (chips moving between partners
             are a wash for the coalition, and they avoid bluffing each other);
           - never bet/raise when the last aggressor on the street is the partner.
WHIPSAW    = SOFTPLAY, plus: whenever seat 0 has voluntarily put chips in on the current street
             (bet, raised or called a bet) and a raise is still allowed, the colluder raises
             (pot-size if available, else all-in) with ANY card, squeezing seat 0 between them.
             This rule takes precedence over "never raise the partner".
"""
from __future__ import annotations

import numpy as np

from game import FOLD, CALL, POT, ALLIN
from tree import PublicTree, DECISION
from cfr import normalize

HERO, COALITION = 0, (1, 2)


def hero_entered(tree: PublicTree) -> np.ndarray:
    """entered[n]: seat 0 has put chips in voluntarily on the current street at node n."""
    ent = np.zeros(tree.M, dtype=bool)
    for n in range(1, tree.M):                    # BFS order: parents before children
        u = tree.parent[n]
        if tree.edge_owner[n] == -1:              # board dealt: new street
            continue
        ent[n] = ent[u] or (tree.edge_owner[n] == HERO and tree.contrib[n, HERO] > tree.contrib[u, HERO])
    return ent


def build(tree: PublicTree, sigma_star: np.ndarray, kind: str) -> np.ndarray:
    assert kind in ("softplay", "whipsaw")
    sig = sigma_star.copy()
    w = sigma_star.copy()
    entered = hero_entered(tree)
    for u in np.flatnonzero(tree.kind == DECISION):
        p = tree.player[u]
        if p not in COALITION:
            continue
        kids = np.arange(tree.first_child[u], tree.first_child[u] + tree.n_children[u])
        acts = tree.action[kids]
        partner = COALITION[0] if p == COALITION[1] else COALITION[1]
        aggressive = np.isin(acts, (POT, ALLIN))
        raise_kid = kids[acts == POT] if (acts == POT).any() else kids[acts == ALLIN]
        call_kid = kids[acts == CALL]
        if not tree.alive[u, HERO]:
            target = call_kid                                   # check/call it down
        elif kind == "whipsaw" and entered[u] and len(raise_kid):
            target = raise_kid                                  # squeeze the hero
        elif tree.last_aggr[u] == partner and aggressive.any():
            # keep fold/call proportions of sigma*, move raise mass to call
            mass = w[kids[aggressive]].sum(0)
            w[kids[aggressive]] = 0.0
            w[call_kid] += mass
            continue
        else:
            continue
        w[kids] = 0.0
        w[target] = 1.0
    out = normalize(tree, w)
    mine = tree.dec_edges[np.isin(tree.dec_owner, COALITION)]
    sig[mine] = out[mine]
    return sig
