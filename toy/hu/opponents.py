"""Fixed (non-adaptive) opponent models as eff arrays on the full tree.

station : never folds, never raises: check if possible, else call.
maniac  : always the largest action (all-in) when it may bet/raise; calls
          an all-in it cannot re-raise; never folds.
nit     : continues only with the top of its range: a "strong" hand is a top-2
          rank card preflop and pair-or-better after the flop.  Strong hands bet
          pot (or all-in if pot is not available) when unopened and call when
          facing a bet; weak hands check and fold to any bet.
"""
from __future__ import annotations

import numpy as np

from game import strength
from tree import PublicTree, DEC


def model_eff(T: PublicTree, seat: int, kind: str):
    cfg = T.cfg
    N = T.N
    e = T.eff_template.copy()
    dec = np.nonzero((T.kind == DEC) & (T.player == seat))[0]
    for n in dec:
        cs, cc = T.child_start[n], T.n_child[n]
        labels = T.label[cs: cs + cc]
        P = np.zeros((cc, N))
        if kind == "station":
            j = labels.index("k") if "k" in labels else labels.index("c")
            P[j] = 1
        elif kind == "maniac":
            if "A" in labels:
                P[labels.index("A")] = 1
            else:
                P[labels.index("c") if "c" in labels else labels.index("k")] = 1
        elif kind == "nit":
            board = T.states[n].board
            for h in range(N):
                if len(board) == 0:
                    strong = cfg.rank(h) >= cfg.n_ranks - 2
                else:
                    strong = strength(cfg, h, board) >= cfg.n_ranks
                if "k" in labels:  # unopened
                    if strong:
                        bet = [l for l in labels if l in ("b1",)] or ["A"]
                        P[labels.index(bet[0]), h] = 1
                    else:
                        P[labels.index("k"), h] = 1
                else:
                    P[labels.index("c" if strong else "f"), h] = 1
        else:
            raise ValueError(kind)
        e[cs: cs + cc] = P
    return e
