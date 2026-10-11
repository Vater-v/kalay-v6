"""The PUBLIC tree, flattened into numpy arrays.

A public node is a betting history plus the public board card (if dealt). Private cards are
NEVER branched on: every array that depends on private information carries a trailing
"fiber" axis of length D (one entry per possible private card of a player), e.g.
    reach[p, node, card]   = probability that player p plays to `node` holding `card`
    sigma[node, card]      = probability of the action that leads INTO `node`, for the player
                             that owns the parent of `node`, given that player's card.
So "strategy per action" is stored on the child node (every non-root node has exactly one
incoming edge). This makes all per-action arrays plain [M, D] arrays.

Layout: nodes are numbered in breadth-first order. Consequences we rely on:
  * the children of a node are contiguous and appear in increasing parent order,
  * all nodes of depth L form one contiguous id range [lo_L, hi_L).
A tree pass is therefore a loop over depths (about 15 iterations), each doing a few numpy
gathers / segment reductions (np.add.reduceat) over a contiguous slice.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from game import (GameConfig, BetState, initial_state, legal_actions, apply_action,
                  round_over, start_street2)

DECISION, CHANCE, TERMINAL = 0, 1, 2


class PublicTree:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        n, D = cfg.n_players, cfg.deck
        rec = {k: [] for k in ("kind", "player", "parent", "action", "depth", "board",
                               "contrib", "folded", "last_aggr", "street", "history",
                               "showdown")}
        states: list[BetState] = []

        def add(kind, player, parent, action, depth, board, st: BetState, showdown=False):
            rec["kind"].append(kind); rec["player"].append(player); rec["parent"].append(parent)
            rec["action"].append(action); rec["depth"].append(depth); rec["board"].append(board)
            rec["contrib"].append(st.contrib); rec["folded"].append(st.folded)
            rec["last_aggr"].append(st.last_aggr); rec["street"].append(st.street)
            rec["history"].append(st.history); rec["showdown"].append(showdown)
            states.append(st)
            return len(states) - 1

        def classify(st: BetState):
            """Kind of the node reached in state st (before a board is dealt)."""
            if st.n_alive() <= 1:
                return TERMINAL, -1, False
            if round_over(st):
                return (CHANCE, -1, False) if st.street == 1 else (TERMINAL, -1, True)
            return DECISION, st.to_act, False

        root = initial_state(cfg)
        add(DECISION, root.to_act, -1, -1, 0, -1, root)
        queue = deque([0])
        while queue:  # BFS: children of each node get contiguous ids
            u = queue.popleft()
            st, kind, depth, board = states[u], rec["kind"][u], rec["depth"][u], rec["board"][u]
            if kind == DECISION:
                for a in legal_actions(cfg, st):
                    cst = apply_action(cfg, st, a)
                    ck, cp, sd = classify(cst)
                    queue.append(add(ck, cp, u, a, depth + 1, board, cst, sd))
            elif kind == CHANCE:
                for b in range(D):  # one public child per board card
                    cst = start_street2(cfg, st, b)
                    if len(cst.pending) == 0:   # nobody (or one player) can bet: showdown now
                        v = add(TERMINAL, -1, u, b, depth + 1, b, cst, True)
                    else:
                        v = add(DECISION, cst.to_act, u, b, depth + 1, b, cst)
                    queue.append(v)

        self.M = len(states)
        self.kind = np.array(rec["kind"], dtype=np.int8)
        self.player = np.array(rec["player"], dtype=np.int64)
        self.parent = np.array(rec["parent"], dtype=np.int64)
        self.action = np.array(rec["action"], dtype=np.int64)
        self.depth = np.array(rec["depth"], dtype=np.int64)
        self.board = np.array(rec["board"], dtype=np.int64)
        self.contrib = np.array(rec["contrib"], dtype=np.float64)          # [M, N]
        self.alive = ~np.array(rec["folded"], dtype=bool)                  # [M, N]
        self.last_aggr = np.array(rec["last_aggr"], dtype=np.int64)
        self.street = np.array(rec["street"], dtype=np.int64)
        self.showdown = np.array(rec["showdown"], dtype=bool)
        self.history = rec["history"]
        self._index()

    # ------------------------------------------------------------------ indexing helpers
    def _index(self):
        M, par = self.M, self.parent
        # edge_owner[n]: player choosing the edge into n; -1 chance edge; -2 root
        self.edge_owner = np.full(M, -2, dtype=np.int64)
        self.edge_owner[1:] = np.where(self.kind[par[1:]] == DECISION, self.player[par[1:]], -1)
        self.is_chance_edge = self.edge_owner == -1
        # children ranges
        self.n_children = np.bincount(par[1:], minlength=M)
        has = self.n_children > 0
        first = np.full(M, -1, dtype=np.int64)   # first child id (children are contiguous)
        idx = np.arange(1, M)
        starts = np.r_[True, par[2:] != par[1:-1]]          # first child of each sibling group
        first[par[1:][starts]] = idx[starts]
        self.first_child = first
        assert np.all(has == (self.kind != TERMINAL))
        # depth levels
        self.levels = []
        for L in range(1, self.depth.max() + 1):
            lo = int(np.searchsorted(self.depth, L, "left"))
            hi = int(np.searchsorted(self.depth, L, "right"))
            nodes = np.arange(lo, hi)
            p = par[lo:hi]
            gstart = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])  # sibling-group starts
            gpar = p[gstart]
            lev = dict(lo=lo, hi=hi, parent=p, gstart=gstart, gparent=gpar,
                       gkind=self.kind[gpar], gplayer=self.player[gpar],
                       owner=self.edge_owner[lo:hi],
                       chance_nodes=nodes[self.is_chance_edge[lo:hi]],
                       chance_boards=self.board[lo:hi][self.is_chance_edge[lo:hi]])
            self.levels.append(lev)
        assert np.all(np.diff(self.depth) >= 0), "BFS order violated"
        # decision edges grouped by parent, for regret matching normalisation
        dec = np.flatnonzero(self.edge_owner >= 0)
        self.dec_edges = dec
        gs = np.flatnonzero(np.r_[True, par[dec][1:] != par[dec][:-1]])
        self.dec_gstart = gs
        self.dec_gparent = par[dec][gs]
        self.dec_group_of_edge = np.repeat(np.arange(len(gs)), np.diff(np.r_[gs, len(dec)]))
        self.dec_owner = self.edge_owner[dec]
        self.terminals = np.flatnonzero(self.kind == TERMINAL)

    # ------------------------------------------------------------------ summaries
    def stats(self) -> dict:
        k = self.kind
        return dict(nodes=int(self.M), decision=int((k == DECISION).sum()),
                    chance=int((k == CHANCE).sum()), terminal=int((k == TERMINAL).sum()),
                    showdown=int(self.showdown.sum()),
                    betting_seqs_street1=int(((self.street == 1) & (k != CHANCE)).sum()),
                    max_depth=int(self.depth.max()),
                    infosets=int((k == DECISION).sum()) * self.cfg.deck)
