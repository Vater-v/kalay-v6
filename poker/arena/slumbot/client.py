"""Bounded Slumbot match runner with injectable policy and transport.

No match, request, login or registration runs on import. The production model
adapter is in slumbot_policy; the durable concurrent runner is slumbot_parallel.
API source: https://slumbot.com/sample_api.py

Every mutating request is sent once. A timeout, malformed response, or protocol
failure stops the match with completed and unresolved-hand evidence. Such a
partial match must not be reported as an unbiased estimate over requested hands.
"""
from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral
import re
from typing import Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from http.client import HTTPSConnection

from poker.arena.slumbot.protocol import BIG_BLIND, HandResponse, ProtocolError, parse_history, parse_response

ORIGIN = 'https://slumbot.com'
NEW_HAND = '/slumbot/api/new_hand'
ACT = '/slumbot/api/act'
_ACTION = re.compile(r'(?:[kcf]|b[0-9]+)\Z')
_REDACTED = '[REDACTED]'


class Transport(Protocol):
    def post(self, endpoint: str, body: dict) -> Mapping:
        """Send exactly once; return decoded JSON or raise without retrying."""


class TransportError(RuntimeError):
    def __init__(self, message, *, response=None):
        super().__init__(message)
        self.response = response


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not turn a mutating POST into another HTTP request.


class HTTPTransport:
    """HTTPS JSON transport, with no authentication workflow or mutation retry."""

    def __init__(self, *, timeout=15.0, max_response_bytes=1_048_576, opener=None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive')
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, Integral) or max_response_bytes < 1:
            raise ValueError('max_response_bytes must be a positive integer')
        self.timeout = timeout
        self.max_response_bytes = int(max_response_bytes)
        self._opener = opener if opener is not None else build_opener(_NoRedirect())

    def post(self, endpoint, body):
        if endpoint not in (NEW_HAND, ACT):
            raise ValueError('Only explicit new_hand and act endpoints are supported')
        if not isinstance(body, dict):
            raise ValueError('Request body must be an object')
        keys = set(body)
        if endpoint == NEW_HAND and keys not in (set(), {'token'}):
            raise ValueError('new_hand accepts only an optional token')
        if endpoint == ACT and keys != {'token', 'incr'}:
            raise ValueError('act requires token and incr')
        if 'token' in body and (not isinstance(body['token'], str) or not body['token']):
            raise ValueError('token must be a nonempty string')
        if endpoint == ACT and (not isinstance(body['incr'], str) or not _ACTION.fullmatch(body['incr'])):
            raise ValueError('incr must be one protocol action')
        request = Request(ORIGIN + endpoint, method='POST',
                          data=json.dumps(body, allow_nan=False).encode('utf-8'),
                          headers={'Content-Type': 'application/json'})
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
                if response.status != 200:
                    raise TransportError(f'HTTP status {response.status}; request outcome unknown')
                if len(raw) > self.max_response_bytes:
                    raise TransportError('Response size limit exceeded; request outcome unknown')
                try:
                    decoded = json.loads(raw.decode('utf-8'))
                except (UnicodeError, ValueError) as error:
                    raise TransportError('Malformed JSON response; request outcome unknown') from error
                if not isinstance(decoded, dict):
                    raise TransportError('Response is not a JSON object; request outcome unknown')
                return decoded
        except HTTPError as error:
            try:
                raw = error.read(self.max_response_bytes + 1)
                decoded = json.loads(raw.decode('utf-8')) if len(raw) <= self.max_response_bytes else None
            except (UnicodeError, ValueError, OSError):
                decoded = None
            raise TransportError(f'HTTP status {error.code}; request outcome unknown', response=decoded) from None
        except (URLError, TimeoutError, OSError):
            raise TransportError('HTTP request failed; request outcome unknown') from None


class _PersistentHTTPSOpener:
    """One keep-alive connection per serial worker, never an automatic retry."""
    def __init__(self):
        self.connection = None

    def open(self, request, timeout):
        if not request.full_url.startswith(ORIGIN + '/slumbot/api/'):
            raise ValueError('Unexpected Slumbot endpoint')
        if self.connection is None:
            self.connection = HTTPSConnection('slumbot.com',timeout=timeout)
        self.connection.request('POST',request.full_url[len(ORIGIN):],
            body=request.data,headers=dict(request.header_items()))
        return self.connection.getresponse()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class PooledHTTPTransport(HTTPTransport):
    """Reusable TLS connection; ambiguous outcomes remain terminal to the match."""
    def __init__(self, **kwargs):
        super().__init__(opener=_PersistentHTTPSOpener(),**kwargs)

    def close(self):
        self._opener.close()


def _secret_key(key):
    normalized = str(key).lower().replace('_', '').replace('-', '')
    return 'token' in normalized or normalized in ('authorization', 'password', 'secret', 'apikey')


def _collect_secrets(value, secrets):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _secret_key(key) and isinstance(child, str) and child:
                secrets.add(child)
            _collect_secrets(child, secrets)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_secrets(child, secrets)


def _redact(value, secrets):
    if isinstance(value, Mapping):
        return {_redact(str(key), secrets): _REDACTED if _secret_key(key) else _redact(child, secrets)
                for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(child, secrets) for child in value]
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, _REDACTED)
    return value


@dataclass(frozen=True)
class RequestEvidence:
    hand: int
    endpoint: str
    request: dict
    response: object | None
    error: str | None


@dataclass(frozen=True)
class CompletedHand:
    number: int
    client_pos: int
    action: str
    hole_cards: tuple[int, int]
    board: tuple[int, ...]
    winnings_chips: int
    baseline_winnings_chips: int | None


@dataclass(frozen=True)
class UnresolvedHand:
    number: int
    last_confirmed_action: str | None
    reason: str
    mutation_outcome_unknown: bool


@dataclass(frozen=True)
class MatchResult:
    requested_hands: int
    policy_id: str
    completed: tuple[CompletedHand, ...]
    requests: tuple[RequestEvidence, ...]
    unresolved: UnresolvedHand | None

    def to_dict(self):
        """JSON-ready evidence and explicit completed-hand-only reporting units."""
        result = asdict(self)
        n = len(self.completed)
        profit = sum(hand.winnings_chips for hand in self.completed)
        baseline = (sum(hand.baseline_winnings_chips for hand in self.completed)
                    if n and all(hand.baseline_winnings_chips is not None for hand in self.completed) else None)
        result.update(
            status='completed' if self.unresolved is None else 'unresolved',
            completed_hands=n, big_blind_chips=BIG_BLIND, profit_chips=profit,
            profit_bb=profit / BIG_BLIND,
            chips_per_100=100 * profit / n if n else None,
            bb_per_100=100 * profit / BIG_BLIND / n if n else None,
            baseline_profit_chips=baseline,
            baseline_bb_per_100=(100 * baseline / BIG_BLIND / n if baseline is not None else None),
            seat_counts={'bb': sum(h.client_pos == 0 for h in self.completed),
                         'sb': sum(h.client_pos == 1 for h in self.completed)},
            rates_scope=('all_requested_hands' if self.unresolved is None else
                         'completed_hands_only; unresolved hand prevents an unbiased match conclusion'),
            quality_verdict=None,
        )
        return result


class MatchInterrupted(RuntimeError):
    """The partial result is evidence, not permission to silently skip a hand."""

    def __init__(self, result):
        super().__init__(result.unresolved.reason)
        self.result = result


def run_match(policy: Callable[[HandResponse], str], transport: Transport, *, hands: int,
              policy_id: str, hand_cap: int = 100, action_cap_per_hand: int = 512,
              initial_token: str | None = None) -> MatchResult:
    """Run a bounded explicitly requested match using an injected policy adapter.

    policy_id should identify a frozen artifact; the runner does not verify or
    create a neural adapter. Auth/session tokens exist only in working memory
    and are recursively removed from request/response evidence and error text.
    There is no automatic retry or resume after an unresolved mutation.
    """
    for name, value in (('hands', hands), ('hand_cap', hand_cap), ('action_cap_per_hand', action_cap_per_hand)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if hands > hand_cap:
        raise ValueError('Requested hands exceed explicit hand_cap')
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError('A frozen policy identifier is required')
    if initial_token is not None and (not isinstance(initial_token, str) or not initial_token):
        raise ValueError('initial_token must be a nonempty string or None')
    token = initial_token
    secrets = {token} if token else set()
    completed, evidence = [], []

    def interrupted(number, current, reason, ambiguous):
        # Sanitize the entire accumulated record again in case a later response
        # disclosed a token that also appeared in earlier server text.
        safe = tuple(RequestEvidence(e.hand, e.endpoint, _redact(e.request, secrets),
                                     _redact(e.response, secrets), _redact(e.error, secrets)) for e in evidence)
        unresolved = UnresolvedHand(number, current.state.action if current else None,
                                    _redact(reason, secrets), ambiguous)
        result = MatchResult(int(hands), _redact(policy_id, secrets), tuple(completed), safe, unresolved)
        raise MatchInterrupted(result) from None

    def request(number, endpoint, body, current=None, expected=None):
        nonlocal token
        _collect_secrets(body, secrets)
        try:
            payload = transport.post(endpoint, body)
        except Exception as error:
            raw = error.response if isinstance(error, TransportError) else None
            _collect_secrets(raw, secrets)
            # Transport exceptions may contain arbitrary URLs/body details.
            # Only our explicit TransportError text is retained, after redaction.
            reason = str(error) if isinstance(error, TransportError) else f'Transport failed ({type(error).__name__}); request outcome unknown'
            evidence.append(RequestEvidence(number, endpoint, _redact(body, secrets),
                                             _redact(raw, secrets), _redact(reason, secrets)))
            interrupted(number, current, reason, True)
        _collect_secrets(payload, secrets)
        try:
            response = parse_response(payload, previous_token=token)
            if current is not None:
                if response.client_pos != current.client_pos or response.hole_cards != current.hole_cards:
                    raise ProtocolError('Client seat or hole cards changed within the hand')
                if response.board[:len(current.board)] != current.board:
                    raise ProtocolError('Previously revealed board cards changed')
                if response.state.events[:len(expected.events)] != expected.events:
                    raise ProtocolError('Response does not confirm the exact submitted action')
        except (ProtocolError, TypeError, ValueError) as error:
            reason = f'Protocol failure: {error}'
            evidence.append(RequestEvidence(number, endpoint, _redact(body, secrets),
                                             _redact(payload, secrets), _redact(reason, secrets)))
            interrupted(number, current, reason, True)
        evidence.append(RequestEvidence(number, endpoint, _redact(body, secrets),
                                         _redact(payload, secrets), None))
        token = response.token
        return response

    for number in range(1, int(hands) + 1):
        current = request(number, NEW_HAND, {'token': token} if token else {})
        actions = 0
        while not current.state.terminal:
            if actions >= action_cap_per_hand:
                interrupted(number, current, 'Action cap reached with hand still unresolved', False)
            try:
                increment = policy(current)
            except Exception as error:
                interrupted(number, current, f'Policy adapter failed ({type(error).__name__}): {str(error)[:300]}', False)
            try:
                if not isinstance(increment, str) or not _ACTION.fullmatch(increment):
                    raise ProtocolError('Policy must return exactly one protocol action')
                separator = '/' if current.state.separator_pending else ''
                expected = parse_history(current.state.action + separator + increment)
            except ProtocolError as error:
                interrupted(number, current, f'Invalid policy action: {error}', False)
            actions += 1
            current = request(number, ACT, {'token': token, 'incr': increment}, current, expected)
        completed.append(CompletedHand(number, current.client_pos, current.state.action,
                                       current.hole_cards, current.board, current.winnings,
                                       current.baseline_winnings))
    safe = tuple(RequestEvidence(e.hand, e.endpoint, _redact(e.request, secrets),
                                 _redact(e.response, secrets), _redact(e.error, secrets)) for e in evidence)
    return MatchResult(int(hands), _redact(policy_id, secrets), tuple(completed), safe, None)
