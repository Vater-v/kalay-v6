"""Offline Slumbot protocol normalization for the official 50/100, 200-BB game.

Reference, checked 2026-09-21: https://slumbot.com/sample_api.py
Current endpoints: /slumbot/api/{new_hand,act,login}. This module performs no
network requests. bN is the player's total contribution on the current street,
not an increment or a whole-hand total. Positions are 0=BB and 1=SB/button.

The public event ledger is a protocol boundary, not a training environment.
Slumbot clients replay its exact events through poker.envs.transition.step_target.
Unmatched opponent bets are never rounded to buckets.
"""
from dataclasses import dataclass, field
from numbers import Integral
from typing import Mapping, Sequence

SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20_000
SEAT = (1, 0)                      # protocol position (0 = BB, 1 = SB/button) -> our seat
SLUMBOT_REGIME = 1                 # heads-up, 200 BB, no rake (poker/envs/config.py REGIMES)


def our_card(protocol_id):
    """protocol.card_id (rank * 4 + suit) -> engine card (suit * 13 + rank)."""
    rank, suit = divmod(int(protocol_id), 4)
    return suit * 13 + rank


@dataclass(frozen=True)
class Decision:                    # the fields poker/arena/slumbot/parallel.py records
    increment: str
    bucket: int
    probabilities: tuple
    inference_ms: float


class ProtocolError(ValueError):
    """Malformed response, illegal action, or inconsistent public history."""


class OffGridActionError(ProtocolError):
    """The exact action cannot be represented by the configured buckets."""


@dataclass(frozen=True)
class PublicAction:
    actor: int
    street: int
    kind: str
    contribution: int
    street_total: int
    pot_before: int
    to_call_before: int
    stacks_before: tuple[int, int]

    @property
    def pot_after(self):
        return self.pot_before + self.contribution


@dataclass(frozen=True)
class PublicHistory:
    action: str
    events: tuple[PublicAction, ...]
    street: int
    next_player: int | None
    terminal_reason: str | None
    stacks: tuple[int, int]
    invested: tuple[int, int]
    street_bets: tuple[int, int]
    last_full_raise: int
    separator_pending: bool

    @property
    def terminal(self):
        return self.terminal_reason is not None

    @property
    def pot(self):
        """Committed pot before terminal settlement/refunds; not final stacks."""
        return sum(self.invested)

    @property
    def to_call(self):
        if self.next_player is None:
            return 0
        return max(self.street_bets) - self.street_bets[self.next_player]

    @property
    def raise_bounds(self):
        if self.next_player is None or min(self.stacks) == 0:
            return None
        high = self.street_bets[self.next_player] + self.stacks[self.next_player]
        low = min(high, max(self.street_bets) + self.last_full_raise)
        return (low, high) if high > max(self.street_bets) else None


def parse_history(action: str) -> PublicHistory:
    """Parse all public actions, preserving every exact chip amount.

    Completed non-river streets can end with or without their final '/'.
    Called all-ins can omit all runout separators or include all of them;
    partial runout suffixes and actions after hand completion are rejected.
    """
    if not isinstance(action, str):
        raise ProtocolError('action must be a string')
    bets = [BIG_BLIND, SMALL_BLIND]
    invested = bets.copy()
    stacks = [STACK_SIZE - x for x in invested]
    acted = [False, False]
    actor, street, last_full_raise = 1, 0, BIG_BLIND
    pending, terminal = False, None
    events = []
    offset = 0
    while offset < len(action):
        if terminal:
            raise ProtocolError('Action after terminal hand')
        code = action[offset]
        offset += 1
        if pending:
            if code != '/':
                raise ProtocolError('Missing street separator')
            pending = False
            continue
        if code not in ('k', 'c', 'f', 'b'):
            raise ProtocolError(f'Unexpected action character {code!r}')
        to_call = max(bets) - bets[actor]
        amount = 0
        target = bets[actor]
        if code == 'b':
            start = offset
            while offset < len(action) and '0' <= action[offset] <= '9':
                offset += 1
            digits = action[start:offset]
            if not digits:
                raise ProtocolError('Missing integer bet size')
            significant = digits.lstrip('0') or '0'
            if len(significant) > 5:
                raise ProtocolError('Bet exceeds stack')
            target = int(significant)
            high = bets[actor] + stacks[actor]
            current = max(bets)
            if stacks[1 - actor] == 0 or target <= current:
                raise ProtocolError('Raise requires an opponent with chips and a larger bet')
            if target > high:
                raise ProtocolError('Bet exceeds remaining stack on this street')
            increment = target - current
            if increment < last_full_raise and target != high:
                raise ProtocolError('Under-minimum raise is only allowed when all-in')
            if increment >= last_full_raise:
                last_full_raise = increment
            amount = target - bets[actor]
            acted[1 - actor] = False
        elif code == 'c':
            if to_call == 0:
                raise ProtocolError('Cannot call when check is free')
            amount = to_call
            target += amount
        elif code == 'k':
            if to_call:
                raise ProtocolError('Cannot check facing a bet')
        elif not to_call:
            raise ProtocolError('Cannot fold when check is free')

        kind = {'k': 'check', 'c': 'call', 'f': 'fold', 'b': 'raise'}[code]
        events.append(PublicAction(actor, street, kind, amount, target,
                                   sum(invested), to_call, tuple(stacks)))
        stacks[actor] -= amount
        invested[actor] += amount
        bets[actor] = target
        acted[actor] = True
        if code == 'f':
            terminal, actor = 'fold', None
        elif all(acted) and bets[0] == bets[1]:
            if min(stacks) == 0:
                # Equal initial stacks and matched contributions imply both all-in.
                suffix = action[offset:]
                if suffix not in ('', '/' * (3 - street)):
                    raise ProtocolError('Invalid all-in runout suffix')
                offset = len(action)
                street, terminal, actor = 3, 'showdown', None
            elif street == 3:
                terminal, actor = 'showdown', None
            else:
                street += 1
                bets, acted = [0, 0], [False, False]
                actor, last_full_raise, pending = 0, BIG_BLIND, True
        else:
            actor = 1 - actor
    return PublicHistory(action, tuple(events), street, actor, terminal,
                         tuple(stacks), tuple(invested), tuple(bets),
                         last_full_raise, pending)


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ProtocolError(f'{label} must be an integer')
    return int(value)


def serialize_action(state: PublicHistory, kind: str, *, raise_to=None) -> str:
    """Validate and serialize one action; raise_to is exact street-total chips."""
    if state.terminal:
        raise ProtocolError('Cannot act after hand completion')
    if kind == 'check_call':
        kind = 'call' if state.to_call else 'check'
    if kind == 'raise':
        if raise_to is None:
            raise ProtocolError('raise_to is required for a raise')
        text = f'b{_integer(raise_to, "raise_to")}'
    else:
        if raise_to is not None:
            raise ProtocolError('Only a raise accepts raise_to')
        try:
            text = {'fold': 'f', 'check': 'k', 'call': 'c'}[kind]
        except KeyError as error:
            raise ProtocolError(f'Unknown action kind {kind!r}') from error
    separator = '/' if state.separator_pending else ''
    parse_history(state.action + separator + text)
    return text


def serialize_bucket_action(state: PublicHistory, index, legal_masks: Sequence,
                            action_amounts: Sequence) -> str:
    """Bridge NLH output buckets to protocol using exact incremental amounts."""
    index = _integer(index, 'action index')
    if len(legal_masks) != len(action_amounts) or len(legal_masks) < 3:
        raise ProtocolError('Expected the NLH fold/check-call/raise bucket schema')
    if index < 0 or index >= len(legal_masks) or not legal_masks[index]:
        raise ProtocolError('Selected bucket is illegal')
    amount = _integer(action_amounts[index], 'action amount')
    if index == 0:
        if amount != 0:
            raise ProtocolError('Fold contribution must be zero')
        return serialize_action(state, 'fold')
    if index == 1:
        if amount != state.to_call:
            raise ProtocolError('Bucket call amount disagrees with protocol state')
        return serialize_action(state, 'check_call')
    if amount < 0 or state.next_player is None:
        raise ProtocolError('Invalid raise contribution or terminal state')
    return serialize_action(state, 'raise', raise_to=state.street_bets[state.next_player] + amount)


def match_exact_bucket(event: PublicAction, legal_masks: Sequence,
                       action_amounts: Sequence) -> int:
    """Match only exact amounts; fail explicitly on an off-grid opponent action.

    This does not import cards, board or actor into VecCardEnv. It only checks
    whether one normalized event fits an already synchronized NLH observation.
    """
    if len(legal_masks) != len(action_amounts) or len(legal_masks) < 3:
        raise ProtocolError('Expected the NLH fold/check-call/raise bucket schema')
    candidates = [0] if event.kind == 'fold' else ([1] if event.kind in ('check', 'call')
                                                    else range(2, len(legal_masks)))
    matches = [i for i in candidates if legal_masks[i]
               and _integer(action_amounts[i], 'action amount') == event.contribution]
    if len(matches) != 1:
        raise OffGridActionError('Exact public action has no unique legal bucket; '
                                 'VecCardEnv bucket-only step cannot replay it. '
                                 'An exact Rust replay API is required; rounding is forbidden.')
    return matches[0]


def card_id(card: str) -> int:
    """Official card strings to Rust rank-major card encoding."""
    if not isinstance(card, str) or len(card) != 2:
        raise ProtocolError('Card must have one rank and one suit')
    ranks, suits = '23456789TJQKA', 'shdc'
    if card[0] not in ranks or card[1] not in suits:
        raise ProtocolError(f'Invalid card {card!r}')
    return ranks.index(card[0]) * 4 + suits.index(card[1])


@dataclass(frozen=True)
class HandResponse:
    state: PublicHistory
    client_pos: int
    hole_cards: tuple[int, int]
    board: tuple[int, ...]
    token: str = field(repr=False)
    winnings: int | None = None
    baseline_winnings: int | None = None


def parse_response(payload: Mapping, *, previous_token: str | None = None) -> HandResponse:
    """Validate one completed /new_hand or /act response, without network access."""
    if not isinstance(payload, Mapping):
        raise ProtocolError('Response must be a JSON object')
    if 'error_msg' in payload:
        raise ProtocolError(f'Slumbot error: {payload["error_msg"]}')
    state = parse_history(payload.get('action'))
    client = _integer(payload.get('client_pos'), 'client_pos')
    if client not in (0, 1):
        raise ProtocolError('client_pos must be 0=BB or 1=SB')
    if not state.terminal and state.next_player != client:
        raise ProtocolError('Response is not the client turn')
    token = payload.get('token', previous_token)
    if not isinstance(token, str) or not token:
        raise ProtocolError('Missing valid session token')
    hole, board = payload.get('hole_cards'), payload.get('board')
    if not isinstance(hole, (list, tuple)) or len(hole) != 2:
        raise ProtocolError('Expected exactly two client hole cards')
    if not isinstance(board, (list, tuple)) or len(board) != (0, 3, 4, 5)[state.street]:
        raise ProtocolError('Board length disagrees with public history')
    holes = tuple(card_id(x) for x in hole)
    boards = tuple(card_id(x) for x in board)
    if len(set(holes + boards)) != len(holes + boards):
        raise ProtocolError('Duplicate known card')
    if 'old_action' in payload:
        old = parse_history(payload['old_action'])
        if state.events[:len(old.events)] != old.events:
            raise ProtocolError('Response rewrites previous public actions')
    winnings = payload.get('winnings')
    if state.terminal != (winnings is not None):
        raise ProtocolError('Terminal history and winnings disagree')
    if winnings is not None:
        winnings = _integer(winnings, 'winnings')
        if state.terminal_reason == 'fold':
            folded = state.events[-1].actor
            possible = {-state.invested[client] if folded == client else state.invested[folded]}
        else:
            possible = {-state.invested[client], 0, state.invested[1 - client]}
        if winnings not in possible:
            raise ProtocolError('Winnings disagree with committed chips')
    baseline = payload.get('baseline_winnings')
    if baseline is not None:
        if not state.terminal:
            raise ProtocolError('Baseline winnings require a terminal hand')
        baseline = _integer(baseline, 'baseline_winnings')
    return HandResponse(state, client, holes, boards, token, winnings, baseline)
