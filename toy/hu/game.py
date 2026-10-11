"""Mini hold'em: rules, cards, hand strength, bet menu, public states.

Game ("HU mini hold'em")
------------------------
* Deck: R ranks x S suits, card id k -> rank k // S.  Shared deck: the two
  private cards and the board are dealt without replacement, so card removal
  matters (if I hold a K, you are less likely to).
* Each player antes `ante` (=1) and holds ONE private card.  Stacks are
  `stack` chips each (ante included).
* Streets: 0 = preflop (no board), 1 = flop (1 public card), 2 = turn
  (1 more public card).  Player 0 acts first on every street.
* Actions: fold 'f', check 'k', call 'c', bet/raise from a pot-fraction menu,
  all-in 'A'.  The menu depends on how many aggressive actions were already
  made on this street (fewer sizes deeper in the sequence).  With the default
  menu: open bet {0.5 pot, 1 pot, all-in}; first raise {1 pot, all-in};
  second raise {all-in}; no third raise (raise cap = 2).
  A size whose amount reaches the stack becomes the all-in (de-duplicated).
* Showdown: strength = rank + R * (number of board cards with the same rank)
  -> trips > pair > high card, higher rank wins inside a class, kicker is
  irrelevant (one private card), equal strength splits the pot.

Public states never contain private cards: everything that depends on the
private cards lives in the "fiber" arrays indexed by hand (= card id).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import permutations
import numpy as np


@dataclass(frozen=True)
class GameConfig:
    n_ranks: int = 4
    n_suits: int = 2
    ante: float = 1.0
    stack: float = 12.0
    # bet_menu[level] = pot fractions allowed when `level` aggressive actions were
    # already made on this street (level 0 = opening bet). All-in is always added
    # while level < len(bet_menu). len(bet_menu) - 1 = raise cap.
    bet_menu: tuple = ((0.5, 1.0), (1.0,), ())
    n_streets: int = 3

    @property
    def n_cards(self) -> int:
        return self.n_ranks * self.n_suits

    def rank(self, card: int) -> int:
        return card // self.n_suits


@dataclass(frozen=True)
class PState:
    """A public state (betting history + board). No private information."""
    street: int
    board: tuple            # public cards, in dealing order
    contrib: tuple          # chips put in the pot by (P0, P1), ante included
    to_act: int
    n_aggr: int             # bets/raises made on this street so far
    n_act: int              # actions taken on this street so far
    key: str                # full public history, unique id of the node

    @property
    def pot(self) -> float:
        return self.contrib[0] + self.contrib[1]


def initial_state(cfg: GameConfig) -> PState:
    return PState(0, (), (cfg.ante, cfg.ante), 0, 0, 0, "")


def strength(cfg: GameConfig, card: int, board) -> int:
    r = cfg.rank(card)
    m = sum(1 for b in board if cfg.rank(b) == r)
    return r + cfg.n_ranks * m


def legal_actions(cfg: GameConfig, s: PState, extra_open=None):
    """Return a list of (label, new_contrib_of_actor or None, kind).

    kind in {'fold','check','call','bet'}.  `extra_open` is an optional dict
    {street: [fractions]} of OFF-MENU opening bet sizes for player 0 at the
    street root (used by experiment E3)."""
    p, o = s.to_act, 1 - s.to_act
    cp, co = s.contrib[p], s.contrib[o]
    to_call = co - cp
    acts = []
    if to_call > 0:
        acts.append(("f", None, "fold"))
        acts.append(("c", co, "call"))
    else:
        acts.append(("k", None, "check"))
    level = s.n_aggr
    if level < len(cfg.bet_menu) and co < cfg.stack and cp < cfg.stack:
        fracs = list(cfg.bet_menu[level])
        if extra_open and level == 0 and s.n_act == 0 and p == 0:
            fracs = fracs + list(extra_open.get(s.street, []))
        seen = set()
        for f in fracs:
            # bet: f * pot ; raise: call first, then f * (pot after the call)
            target = co + f * (s.pot + to_call)
            if target >= cfg.stack - 1e-9:
                continue  # becomes the all-in below
            tag = ("b" if level == 0 else "r") + f"{f:g}"
            if round(target, 9) in seen:
                continue
            seen.add(round(target, 9))
            acts.append((tag, target, "bet"))
        acts.append(("A", cfg.stack, "bet"))
    return acts


def apply_action(cfg: GameConfig, s: PState, act):
    """Return (kind, new_state_or_info).

    kind: 'dec'      -> next decision state
          'fold'     -> terminal, info = (state, folder)
          'close'    -> street closed, info = state with contributions updated
    """
    label, amount, k = act
    p = s.to_act
    key = s.key + ("/" if s.key else "") + label
    if k == "fold":
        return "fold", (replace(s, key=key), p)
    contrib = list(s.contrib)
    if k == "call":
        contrib[p] = amount
        return "close", replace(s, contrib=tuple(contrib), key=key, n_act=s.n_act + 1)
    if k == "check":
        ns = replace(s, key=key, n_act=s.n_act + 1, to_act=1 - p)
        if p == 1:  # second check closes the street (P0 always acts first)
            return "close", ns
        return "dec", ns
    contrib[p] = amount
    ns = replace(s, contrib=tuple(contrib), key=key, to_act=1 - p,
                 n_aggr=s.n_aggr + 1, n_act=s.n_act + 1)
    return "dec", ns


def next_street_root(s: PState, card: int) -> PState:
    return PState(s.street + 1, s.board + (card,), s.contrib, 0, 0, 0,
                  s.key + f"/d{card}")


# ----------------------------------------------------------------------------
# Board equity matrices (all card removal is handled HERE, explicitly)
# ----------------------------------------------------------------------------
class Boards:
    """Enumerates every public board (ordered, length 0..2) and precomputes

    W[b][h, h'] = E_runout[ sign(strength(h) - strength(h')) ]
    for the private cards h (player holding h) and h' (opponent), where the
    expectation is over the remaining board cards dealt uniformly from the deck
    minus {board, h, h'}.  W[b][h,h'] = 0 whenever h == h' or either card is on
    the board (impossible combinations), and W is antisymmetric.
    `mask[b][h]` = 1 if card h is not on the board.
    """

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        n = cfg.n_cards
        boards = [()]
        boards += [(c,) for c in range(n)]
        boards += list(permutations(range(n), 2))
        self.boards = boards
        self.index = {b: i for i, b in enumerate(boards)}
        nb = len(boards)
        self.mask = np.zeros((nb, n))
        self.W = np.zeros((nb, n, n))
        # strength table for complete boards
        full_str = {}
        for b in boards:
            if len(b) == 2:
                full_str[b] = np.array([strength(cfg, h, b) for h in range(n)])
        for bi, b in enumerate(boards):
            self.mask[bi] = [0.0 if h in b else 1.0 for h in range(n)]
            if len(b) == 2:
                st = full_str[b]
                sg = np.sign(st[:, None] - st[None, :]).astype(float)
                m = self.mask[bi]
                sg *= m[:, None] * m[None, :]
                np.fill_diagonal(sg, 0.0)
                self.W[bi] = sg
        # incomplete boards: average the complete-board matrices over runouts,
        # excluding runout cards that collide with h or h'
        for bi, b in enumerate(boards):
            if len(b) == 2:
                continue
            k = len(b)
            acc = np.zeros((n, n))
            cnt = np.zeros((n, n))
            for rest in permutations([c for c in range(n) if c not in b], 2 - k):
                fb = b + rest
                Wf = self.W[self.index[fb]]
                ok = np.ones(n)
                ok[list(rest)] = 0.0  # h or h' equal to a runout card: impossible
                okm = ok[:, None] * ok[None, :]
                acc += Wf * okm
                cnt += okm
            with np.errstate(invalid="ignore", divide="ignore"):
                Wb = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
            m = self.mask[bi]
            Wb *= m[:, None] * m[None, :]
            np.fill_diagonal(Wb, 0.0)
            self.W[bi] = Wb

    def id(self, board) -> int:
        return self.index[tuple(board)]
