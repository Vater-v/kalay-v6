"""Flat, level-ordered PUBLIC tree with a private-hand FIBER at every node.

The tree (or a forest of subtrees) is stored as flat numpy arrays in
breadth-first order, so that
  * all nodes of one depth form a contiguous block ("level"),
  * the children of a node are contiguous and the children of the
    non-terminal nodes of level d, taken in order, are exactly level d+1.
Every per-hand quantity (reach, strategy, regret, counterfactual value) is an
array of shape [n_nodes, n_hands].  A CFR pass is then a loop over ~20 levels
with one vectorised numpy operation per level, never a loop over hidden states.

Node kinds
  DEC    decision of player `player[n]`; children = actions
  CHANCE a public card is dealt; children = one per card not on the board
  FOLD   terminal, `player[n]` folded
  SHOW   terminal showdown (if the board is incomplete because a player is
         all-in, the remaining cards are averaged exactly inside W[board])
  LEAF   depth-limit boundary: the street closed and the next card is about to
         be dealt; its value comes from an external value function V
Edge convention: the strategy of action a at node n is stored on the CHILD
node n.a (one row per child).  `chance scale` cscale[n] is the product of the
chance probabilities that are the same for every compatible hand pair; the
card-removal part of chance is a 0/1 mask stored on chance children.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from game import (GameConfig, PState, Boards, legal_actions, apply_action,
                  next_street_root, initial_state)

DEC, CHANCE, FOLD, SHOW, LEAF = 0, 1, 2, 3, 4


class PublicTree:
    def __init__(self, cfg: GameConfig, boards: Boards, roots, root_cscale,
                 depth_limited: bool = False, extra_open=None):
        """roots: list of PState; root_cscale: list of floats.
        depth_limited=True -> stop at the end of the root's street (LEAF)."""
        self.cfg, self.B = cfg, boards
        N = self.N = cfg.n_cards
        kind, player, parent, depth, board_id, c0, c1 = [], [], [], [], [], [], []
        cscale, card, label, root_of, states, street = [], [], [], [], [], []
        child_start, n_child = [], []

        def add(st, k, pl, par, d, cs, cd, lab, r):
            kind.append(k); player.append(pl); parent.append(par); depth.append(d)
            board_id.append(boards.id(st.board)); c0.append(st.contrib[0]); c1.append(st.contrib[1])
            cscale.append(cs); card.append(cd); label.append(lab); root_of.append(r)
            states.append(st); street.append(st.street)
            child_start.append(-1); n_child.append(0)
            return len(kind) - 1

        for r, (s, cs) in enumerate(zip(roots, root_cscale)):
            add(s, DEC, s.to_act, -1, 0, cs, -1, "", r)
        q = deque(range(len(roots)))
        last_street = cfg.n_streets - 1
        root_street = {r: roots[r].street for r in range(len(roots))}
        while q:
            n = q.popleft()
            s = states[n]
            k = kind[n]
            kids = []
            if k == DEC:
                for act in legal_actions(cfg, s, extra_open):
                    res, info = apply_action(cfg, s, act)
                    if res == "fold":
                        ns, folder = info
                        kids.append((ns, FOLD, folder, -1, act[0], cscale[n]))
                    elif res == "dec":
                        kids.append((info, DEC, info.to_act, -1, act[0], cscale[n]))
                    else:  # street closed
                        ns = info
                        allin = max(ns.contrib) >= cfg.stack - 1e-9
                        if allin or ns.street == last_street:
                            kids.append((ns, SHOW, -1, -1, act[0], cscale[n]))
                        elif depth_limited and ns.street == root_street[root_of[n]]:
                            kids.append((ns, LEAF, -1, -1, act[0], cscale[n]))
                        else:
                            kids.append((ns, CHANCE, -1, -1, act[0], cscale[n]))
            elif k == CHANCE:
                nb = len(s.board)
                cs = cscale[n] / (N - 2 - nb)
                for c in range(N):
                    if c in s.board:
                        continue
                    kids.append((next_street_root(s, c), DEC, 0, c, f"d{c}", cs))
            if kids:
                child_start[n] = len(kind)
                n_child[n] = len(kids)
                for (ns, kk, pl, cd, lab, cs) in kids:
                    m = add(ns, kk, pl, n, depth[n] + 1, cs, cd, lab, root_of[n])
                    if kk in (DEC, CHANCE):
                        q.append(m)

        self.M = M = len(kind)
        self.n_roots = len(roots)
        self.kind = np.array(kind, dtype=np.int8)
        self.player = np.array(player, dtype=np.int8)
        self.parent = np.array(parent, dtype=np.int64)
        self.depth = np.array(depth, dtype=np.int64)
        self.board_id = np.array(board_id, dtype=np.int64)
        self.contrib = np.stack([np.array(c0), np.array(c1)], 1)
        self.cscale = np.array(cscale)
        self.card = np.array(card, dtype=np.int64)
        self.label = label
        self.root_of = np.array(root_of, dtype=np.int64)
        self.states = states
        self.street = np.array(street, dtype=np.int64)
        self.child_start = np.array(child_start, dtype=np.int64)
        self.n_child = np.array(n_child, dtype=np.int64)
        self.key_index = {s.key: i for i, s in enumerate(states)}
        assert np.all(np.diff(self.depth) >= 0), "BFS order violated"
        self._build_levels()
        self._build_edges()
        self._build_terminals()

    # ------------------------------------------------------------------
    def _build_levels(self):
        """Per level d: non-terminal parents and segment starts of their
        children inside level d+1 (for np.add.reduceat)."""
        D = int(self.depth.max()) + 1
        bounds = np.searchsorted(self.depth, np.arange(D + 1))
        self.levels = []
        for d in range(D - 1):
            lo, hi = bounds[d], bounds[d + 1]
            nt = lo + np.nonzero(self.n_child[lo:hi] > 0)[0]
            if len(nt) == 0:
                continue
            cnt = self.n_child[nt]
            c_lo, c_hi = bounds[d + 1], bounds[d + 2]
            assert self.child_start[nt[0]] == c_lo and cnt.sum() == c_hi - c_lo
            starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
            is_dec = [(self.kind[nt] == DEC) & (self.player[nt] == p) for p in (0, 1)]
            self.levels.append(dict(nt=nt, cnt=cnt, lo=c_lo, hi=c_hi, starts=starts,
                                    is_dec=is_dec))

    def _build_edges(self):
        """E[p] = child nodes of player p's decisions (= p's action edges),
        sorted, with segment starts per decision node; eff_template = chance
        masks on chance children, 1 elsewhere."""
        N = self.N
        par = self.parent
        self.E, self.E_starts, self.E_cnt, self.E_parent = [], [], [], []
        has_par = par >= 0
        pk = np.where(has_par, self.kind[np.maximum(par, 0)], -1)
        pp = np.where(has_par, self.player[np.maximum(par, 0)], -1)
        for p in (0, 1):
            e = np.nonzero((pk == DEC) & (pp == p))[0]
            ep = par[e]
            # segments: consecutive equal parents
            new = np.concatenate([[True], ep[1:] != ep[:-1]])
            starts = np.nonzero(new)[0]
            cnt = np.diff(np.concatenate([starts, [len(e)]]))
            self.E.append(e); self.E_starts.append(starts); self.E_cnt.append(cnt)
            self.E_parent.append(ep)
        self.eff_template = np.ones((self.M, N))
        ch = np.nonzero(pk == CHANCE)[0]
        self.eff_template[ch] = (np.arange(N)[None, :] != self.card[ch][:, None]).astype(float)
        self.bmask = self.B.mask[self.board_id]  # [M, N] own-card-not-on-board mask

    def _build_terminals(self):
        k = self.kind
        self.F = np.nonzero(k == FOLD)[0]
        folder = self.player[self.F]
        cf = self.contrib[self.F]
        # utility of player i when `folder` folds: folder loses its contribution
        self.F_u = np.zeros((2, len(self.F)))
        for i in (0, 1):
            self.F_u[i] = np.where(folder == i, -cf[:, i], cf[np.arange(len(self.F)), folder])
        self.F_u *= self.cscale[self.F][None, :]
        S = np.nonzero(k == SHOW)[0]
        order = np.argsort(self.board_id[S], kind="stable")
        S = S[order]
        self.S = S
        bids = self.board_id[S]
        self.S_groups = []
        if len(S):
            cuts = np.nonzero(np.diff(bids))[0] + 1
            for g in np.split(np.arange(len(S)), cuts):
                b = bids[g[0]]
                self.S_groups.append((S[g], self.B.W[b].T.copy(),
                                      (self.cscale[S[g]] * self.contrib[S[g], 0])[:, None]))
        self.L = np.nonzero(k == LEAF)[0]

    # ------------------------------------------------------------------
    def info_sets(self):
        """Number of information sets = sum over decision nodes of the
        number of private hands compatible with the board."""
        dec = np.nonzero(self.kind == DEC)[0]
        return int(self.bmask[dec].sum())

    def summary(self):
        k = self.kind
        return dict(public_nodes=self.M, decision=int((k == DEC).sum()),
                    chance=int((k == CHANCE).sum()), fold=int((k == FOLD).sum()),
                    showdown=int((k == SHOW).sum()), leaf=int((k == LEAF).sum()),
                    fiber=self.N, info_sets=self.info_sets(),
                    levels=len(self.levels) + 1)


def full_game_tree(cfg: GameConfig, boards: Boards, extra_open=None) -> PublicTree:
    n = cfg.n_cards
    return PublicTree(cfg, boards, [initial_state(cfg)], [1.0 / (n * (n - 1))],
                      depth_limited=False, extra_open=extra_open)
