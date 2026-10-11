"""Rules of "Tri-Leduc": an N-player (default 3) two-street toy poker game.

Deck
    D = n_ranks * n_suits cards; card c has rank c // n_suits. Default 6 ranks x 2 suits = 12.
Deal
    every player gets ONE private card; after street 1 ONE public card (the board) is dealt.
    The joint hidden state is the ordered tuple of distinct private cards (c_1, ..., c_N).
Money (unit = 1 ante)
    every player antes 1; every player starts with the same total stack S (incl. the ante),
    default S = 10. Equal stacks => there are never side pots (anybody can always call).
Betting (both streets, same menu)
    not facing a bet : CHECK, BET-POT (bet the size of the pot), ALL-IN
    facing a bet     : FOLD, CALL, RAISE-POT (call, then raise by the pot after calling), ALL-IN
    at most `raise_cap` aggressive actions (bet or raise) per street. A pot-size action that would
    put a player at or above the stack is not offered (ALL-IN covers it).
    Street 1 and street 2 are both opened by the lowest seat still able to act.
Showdown
    pair with the board beats any unpaired card; otherwise the higher rank wins; equal strength
    splits the pot among the tied players. Payoff of player i = share_i * pot - contribution_i.

This module only knows rules. `PublicTree` (tree.py) turns the rules into a flat public tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

FOLD, CALL, POT, ALLIN = 0, 1, 2, 3
ACTION_CHARS = "fcpa"  # fold, check/call, pot bet/raise, all-in


@dataclass(frozen=True)
class GameConfig:
    n_players: int = 3
    n_ranks: int = 6
    n_suits: int = 2
    stack: int = 10       # total chips per player in antes, ante included
    raise_cap: int = 2    # aggressive actions allowed per street

    @property
    def deck(self) -> int:
        return self.n_ranks * self.n_suits

    @property
    def n_deals(self) -> int:
        """Number of ordered private deals (c_1..c_N distinct) = D!/(D-N)!."""
        return factorial(self.deck) // factorial(self.deck - self.n_players)

    def describe(self) -> str:
        return (f"N={self.n_players} D={self.deck} ({self.n_ranks}r x {self.n_suits}s) "
                f"stack={self.stack} raise_cap={self.raise_cap}")


def strength_table(cfg: GameConfig) -> np.ndarray:
    """s[b, c] = showdown strength of private card c on board b (higher wins).

    Unpaired: rank(c) in [0, R). Paired with board: R + rank(c). s[b, b] is never used
    (card b is on the board and cannot be held), we leave it at its formula value.
    """
    r = np.arange(cfg.deck) // cfg.n_suits
    return r[None, :] + cfg.n_ranks * (r[None, :] == r[:, None])


@dataclass(frozen=True)
class BetState:
    """Public betting state. Everything here is common knowledge."""
    street: int                 # 1 or 2
    contrib: tuple              # chips put in by each player (ante included)
    folded: tuple               # bool per player
    pending: tuple              # players that still have to act this round, in order
    raises: int                 # aggressive actions so far this street
    last_aggr: int              # seat of last bettor/raiser this street, -1 if none
    history: str                # e.g. "0c1p2c|7|0c1c2c"

    @property
    def to_act(self) -> int:
        return self.pending[0]

    def n_alive(self) -> int:
        return sum(not f for f in self.folded)


def initial_state(cfg: GameConfig) -> BetState:
    n = cfg.n_players
    return BetState(1, (1,) * n, (False,) * n, tuple(range(n)), 0, -1, "")


def legal_actions(cfg: GameConfig, st: BetState) -> list[int]:
    p = st.to_act
    mx = max(st.contrib)
    to_call = mx - st.contrib[p]
    acts = [FOLD] if to_call > 0 else []
    acts.append(CALL)
    if st.raises < cfg.raise_cap and mx < cfg.stack:
        pot_after_call = sum(st.contrib) + to_call
        if mx + pot_after_call < cfg.stack:
            acts.append(POT)
        acts.append(ALLIN)
    return acts


def apply_action(cfg: GameConfig, st: BetState, a: int) -> BetState:
    n = cfg.n_players
    p = st.to_act
    contrib = list(st.contrib)
    folded = list(st.folded)
    mx = max(contrib)
    raises, last_aggr = st.raises, st.last_aggr
    if a == FOLD:
        folded[p] = True
        pending = st.pending[1:]
    elif a == CALL:
        contrib[p] = mx
        pending = st.pending[1:]
    else:
        to_call = mx - contrib[p]
        target = mx + sum(contrib) + to_call if a == POT else cfg.stack
        contrib[p] = target
        raises += 1
        last_aggr = p
        # everybody else who is alive and not all-in must act again, in seat order after p
        order = [(p + k) % n for k in range(1, n)]
        pending = tuple(q for q in order if not folded[q] and contrib[q] < cfg.stack)
    return BetState(st.street, tuple(contrib), tuple(folded), tuple(pending), raises,
                    last_aggr, st.history + f"{p}{ACTION_CHARS[a]}")


def round_over(st: BetState) -> bool:
    return st.n_alive() <= 1 or len(st.pending) == 0


def start_street2(cfg: GameConfig, st: BetState, board: int) -> BetState:
    active = tuple(q for q in range(cfg.n_players)
                   if not st.folded[q] and st.contrib[q] < cfg.stack)
    # betting only happens if at least two players can still act
    pending = active if len(active) >= 2 else ()
    return BetState(2, st.contrib, st.folded, pending, 0, -1, st.history + f"|{board}|")


def showdown_share(cfg: GameConfig, strengths: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Pot share of each player (vectorized over leading axes).

    strengths: [..., N] strength of each player's card; alive: [N] bool.
    Returns [..., N]: 1/k for each of the k alive players with maximal strength, else 0.
    """
    s = np.where(alive, strengths, -10**6)
    best = s.max(axis=-1, keepdims=True)
    win = (s == best) & alive
    return win / win.sum(axis=-1, keepdims=True)
