"""Sampling-based regret minimisation on the SAME public tree / strategy tables.

ESMCCFR   External-sampling MCCFR (Lanctot et al. 2009). One traversal per player per
          iteration: sample a full deal (private cards + board); at the traverser's nodes
          enumerate all actions, at other players' nodes sample one action from their current
          strategy. Regret update r(I,a) += v(a) - v(I): the sampling probabilities of chance and
          opponents cancel the pi_-i factor of the counterfactual value, so no importance weight.
          Average strategy: sigma(I) added at opponents' visited infosets ("simple averaging").
          Plain regret matching (as in the paper); optional linear weighting of the average.

OutcomeSamplingLearner
          Model-free online learner for ONE seat against an environment (the other seats'
          fixed strategy) that it can only SAMPLE: every iteration is one hand actually played.
          Outcome-sampling estimator: with q_i(z) = prob. that the learner's exploratory
          sampling policy (eps-uniform mix) produced its actions on the played path z,
             W = u_i(z) / q_i(z)
             v~(I, a*) = W * pi_i(a* -> z),  v~(I) = W * pi_i(I -> z)   (own strategy only)
             r(I, a) += [a == a*] v~(I, a*) - v~(I)
          The opponents' and chance probabilities cancel because the environment itself samples
          them; the learner never needs to know or model the opponents' strategy.
"""
from __future__ import annotations

import random

import numpy as np

from tree import PublicTree, DECISION, CHANCE, TERMINAL
from scalar import ScalarTree
from cfr import normalize


def _rm(R, D, children, c):
    """Regret matching at infoset (node, c): returns list of probabilities."""
    pos = [max(R[ch * D + c], 0.0) for ch in children]
    s = sum(pos)
    if s > 0:
        return [x / s for x in pos]
    k = len(children)
    return [1.0 / k] * k


def _sample(probs, rnd):
    x = rnd.random()
    acc = 0.0
    for k, p in enumerate(probs):
        acc += p
        if x < acc:
            return k
    return len(probs) - 1


class ESMCCFR:
    def __init__(self, tree: PublicTree, seed: int = 0, linear_avg: bool = True):
        self.tree, self.st = tree, ScalarTree(tree)
        self.N, self.D = tree.cfg.n_players, tree.cfg.deck
        self.R = [0.0] * (tree.M * self.D)
        self.S = [0.0] * (tree.M * self.D)
        self.rnd = random.Random(seed)
        self.t = 0
        self.visits = 0          # decision/terminal nodes touched (work counter)
        self.linear_avg = linear_avg

    def _walk(self, n, i, cards, board):
        st, D = self.st, self.D
        self.visits += 1
        k = st.kind[n]
        if k == TERMINAL:
            return st.utility(n, cards, board)[i]
        if k == CHANCE:
            return self._walk(st.children[n][board], i, cards, board)
        p, ch = st.player[n], st.children[n]
        c = cards[p]
        sig = _rm(self.R, D, ch, c)
        if p == i:
            vals = [self._walk(x, i, cards, board) for x in ch]
            v = sum(s * x for s, x in zip(sig, vals))
            for x, va in zip(ch, vals):
                self.R[x * D + c] += va - v
            return v
        w = self.t if self.linear_avg else 1.0
        for x, s in zip(ch, sig):
            self.S[x * D + c] += w * s
        return self._walk(ch[_sample(sig, self.rnd)], i, cards, board)

    def iterate(self, n_iter: int = 1):
        for _ in range(n_iter):
            self.t += 1
            for i in range(self.N):
                deal = self.rnd.sample(range(self.D), self.N + 1)
                self._walk(0, i, deal[:self.N], deal[self.N])

    def average_strategy(self) -> np.ndarray:
        return normalize(self.tree, np.array(self.S).reshape(self.tree.M, self.D))


class OutcomeSamplingLearner:
    """Seat `me` learns online by playing hands against fixed `env_sigma` (other seats)."""

    def __init__(self, tree: PublicTree, env_sigma: np.ndarray, me: int = 0, eps: float = 0.6,
                 seed: int = 0, rm_plus: bool = True, R0: np.ndarray | None = None):
        self.tree, self.st = tree, ScalarTree(tree)
        self.N, self.D, self.me, self.eps = tree.cfg.n_players, tree.cfg.deck, me, eps
        self.env = env_sigma.tolist()           # env[node][card] = edge probability
        # R0: optional prior regrets (e.g. self-play CFR+ regrets / n_deals, which puts them in
        # the same units as one sampled hand) -> the learner starts from the self-play policy
        self.R = [0.0] * (tree.M * self.D) if R0 is None else np.asarray(R0, float).ravel().tolist()
        self.S = [0.0] * (tree.M * self.D)
        self.rnd = random.Random(seed)
        self.rm_plus = rm_plus
        self.hands = 0
        self.realised = 0.0                       # sum of actual payoffs of the played hands

    def play_hand(self):
        st, D, me, rnd = self.st, self.D, self.me, self.rnd
        deal = rnd.sample(range(D), self.N + 1)
        cards, board = deal[:self.N], deal[self.N]
        path = []                                 # learner decisions: (children, c, sig, k)
        n, pi_me, q_me = 0, 1.0, 1.0
        while st.kind[n] != TERMINAL:
            if st.kind[n] == CHANCE:
                n = st.children[n][board]
                continue
            p, ch = st.player[n], st.children[n]
            c = cards[p]
            if p == me:
                sig = _rm(self.R, D, ch, c)
                # stochastically-weighted averaging: weight pi_me / q_me (opponents stationary)
                w = pi_me / q_me
                for x, s in zip(ch, sig):
                    self.S[x * D + c] += w * s
                explore = [self.eps / len(ch) + (1 - self.eps) * s for s in sig]
                k = _sample(explore, rnd)
                path.append((ch, c, sig, k))
                pi_me *= sig[k]
                q_me *= explore[k]
            else:
                k = _sample([self.env[x][c] for x in ch], rnd)
            n = ch[k]
        u = st.utility(n, cards, board)[me]
        self.realised += u
        self.hands += 1
        W = u / q_me
        tail = 1.0                                # pi_me from (I, a*) down to z
        for ch, c, sig, k in reversed(path):
            v_a = W * tail
            v_I = v_a * sig[k]
            for j, x in enumerate(ch):
                r = (v_a if j == k else 0.0) - v_I
                idx = x * D + c
                self.R[idx] = max(self.R[idx] + r, 0.0) if self.rm_plus else self.R[idx] + r
            tail *= sig[k]

    def current_strategy(self) -> np.ndarray:
        out = np.array(self.env)
        mine = self.tree.dec_edges[self.tree.dec_owner == self.me]
        cur = normalize(self.tree, np.array(self.R).reshape(self.tree.M, self.D))
        out[mine] = cur[mine]
        return out

    def average_strategy(self) -> np.ndarray:
        out = np.array(self.env)
        mine = self.tree.dec_edges[self.tree.dec_owner == self.me]
        avg = normalize(self.tree, np.array(self.S).reshape(self.tree.M, self.D))
        out[mine] = avg[mine]
        return out
