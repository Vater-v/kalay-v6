"""The live play path as a Slumbot client (poker/arena/slumbot/run.py).

Every request of Slumbot carries the whole public history of the hand; it is replayed exactly on our engine from the
hand's start with the known board (every action at its exact chip amount, poker.envs.transition.step_target), and the
events not yet seen by this worker's live player are fed to it one by one (poker.player.live.LivePlayer.observe: the
opponent's actions update his range through the likelihoods of the solves, our own actions are the ones act() chose),
then the player decides at the replayed state (act(): a street-start PCFR+ solve with the history on the slot
topology, through the solve server shared by all workers). Exact chip amounts and ranges are
handled by the same LivePlayer as local matches.

A worker thread handles one hand at a time (parallel.run_parallel: run_match(hands=1) per hand), so each thread keeps
its own LivePlayer and the count of events it has observed; a new hand is recognised by a changed seat / hole cards or
a history that is not a continuation of the observed one. A failure of the decision (BudgetExceeded, NonFinite,
IllegalAction, TableUnreachable, InvalidHand, ServerUnavailable, ...) is counted and recorded with its type, message,
hand key, street and the player's counters (self.failures), then raised: the hand stays unresolved in the runner's
record (never a default action; the runner stops the match on it, its contract).

Ready gate: warm_up() compiles every topology x ladder step of config.regime_field(), verifies the flop class cache and
turns the server ready; run.py starts the workers only after it. The summary carries config.game_meta() and the
code fingerprint from poker.envs.fingerprint.code_fingerprint().

Seats and cards as in poker/arena/slumbot/protocol.py (client_pos 0 = BB = our seat 1; protocol card ids -> ours).
"""
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.arena.slumbot.protocol import SEAT, SLUMBOT_REGIME, STACK_SIZE, Decision, ProtocolError, our_card, serialize_action
from poker.arena.slumbot.util import file_sha256
from poker.envs import config, guards
from poker.envs.betting import init_betting
from poker.envs.fingerprint import code_fingerprint
from poker.envs.transition import step_target
from poker.player import live as LV
from poker.solver import critics as CR
from poker.solver import runout
from poker.solver import subgame as SG

ROOT = Path(__file__).resolve().parents[3]


class LiveDecisionFailure(RuntimeError):
    """A decision of the live player failed (the type and message of the failure in the text); the hand is not
    answered: the runner records it unresolved and stops (never a default action)."""


@dataclass(frozen=True)
class LiveDecision(Decision):      # Decision's fields (parallel.py records asdict) plus the live player's candidates
    street: int = -1
    candidates: tuple = ()         # ((kind, street total), ...) the row's columns
    solve_seconds: float = 0.0
    engine_hole_cards: tuple = ()
    engine_board: tuple = ()


def check_live_regime():
    """The live bot plays Slumbot's table: this process must be regime 1 (or POKER_REGIME_FORCE=1, logged)."""
    slumbot = config.REGIMES[SLUMBOT_REGIME]
    if (slumbot["players"] != (2,) or slumbot["rake_mix"] != ((0.0, 0.0, 1.0),)
            or [tuple(r) for r in slumbot["depth_mix"]] != [(STACK_SIZE / config.BB, STACK_SIZE / config.BB, 1.0)]):
        raise RuntimeError(f"regime {SLUMBOT_REGIME} {slumbot} is no longer Slumbot's table (HU, {STACK_SIZE} chips, no rake)")
    if config.REGIME == SLUMBOT_REGIME:
        return config.regime_record()
    problem = (f"Slumbot plays {config.describe_regime(config.regime_record(SLUMBOT_REGIME))}; this process is "
               f"{config.describe_regime(config.regime_record())}")
    if not config.REGIME_FORCE:
        raise ValueError(f"{problem}; POKER_REGIME=1, or POKER_REGIME_FORCE=1 for a deliberate cross-regime evaluation")
    print(time.strftime("%H:%M:%S"), f"WARNING: {problem}; played anyway (POKER_REGIME_FORCE=1)", flush=True)
    return config.regime_record()


class LiveSlumbotBot:
    """policy.choose(response, rng) -> LiveDecision for parallel.run_parallel; policy_id / metadata / stats /
    decisions / batches as the runner reads them."""

    def __init__(self, *, critics_dir, preflop_table, flop_cache, rank7, preflop_equity, iterations=None, chunk=50,
                 window_seconds=0.3, dev_size=0, budget_seconds=4.0, eps=LV.DEFAULT_EPS, log=None, jax_cache=None, gadget=True, gadget_deal="mix",
                 certificate=True):
        self.regime = check_live_regime()
        self.log = log
        self.n = len(SEAT)
        self.gadget = bool(gadget)
        self.gadget_deal = gadget_deal
        critics_dir, preflop_table, flop_cache, rank7, preflop_equity = (Path(x) for x in (critics_dir, preflop_table, flop_cache, rank7,
                                                                                      preflop_equity))
        post_names = tuple(f"{name}.npz" for name in CR.POST_CRITICS.values()) if self.gadget else ()
        for p in (critics_dir / "pre_end.npz", critics_dir / "flop_end.npz", critics_dir / "turn_end.npz", preflop_table, rank7, preflop_equity,
                  *(critics_dir / name for name in post_names)):
            if not p.exists():
                raise FileNotFoundError(f"live bot: {p} is missing" + (" (the re-solving gadget needs the post-card street critics "
                                                                        "flop_critic / turn_critic / river_critic of the same run; "
                                                                        "--no-gadget plays with Bayesian ranges instead)"
                                                                        if p.name in post_names else ""))
        if not flop_cache.is_dir():
            raise FileNotFoundError(f"live bot: the flop class cache {flop_cache} is missing")
        self.jax_cache = None if jax_cache is None else Path(jax_cache)     # the base; checked and installed at warm_up()
        self.critics = CR.load_live_critics(critics_dir)
        self.post_critics = CR.load_post_critics(critics_dir) if self.gadget else {}
        self.preflop_equity = np.load(preflop_equity)
        self.table = LV.PreflopTable(preflop_table, critic_path=critics_dir / "pre_end.npz")
        self.flop_cache = flop_cache
        self.flop_equity = lambda f: SG.flop_equity(f, read_only=flop_cache)
        self.turn_equity = SG.default_turn_equity(rank7) if hasattr(SG, "default_turn_equity") else (
            lambda b, _t=np.load(rank7, mmap_mode="r"): SG.turn_equity(b, _t))
        self.iterations = dict(config.PLAY_ITERATIONS if iterations is None else iterations)
        self.chunk = int(chunk)
        self.budget_seconds = budget_seconds
        self.eps = eps
        self.solver = LV.Solver(iterations=self.iterations, chunk=self.chunk, critics=self.critics, certificate=certificate)
        self.server = LV.SolveServer(self.solver, window_seconds=window_seconds, dev_size=dev_size, log=None)   # quiet: one line per trap otherwise
        base = init_betting(self.n, jnp.full(self.n, STACK_SIZE, jnp.int32), history=True, max_stack=STACK_SIZE,
                            rake_level=0.0)
        self.base = jax.tree.map(lambda x: np.array(x, copy=True), base)
        if np.asarray(self.base.bet_street).tolist() != [config.SB, config.BB] or int(self.base.to_act) != 0:
            raise RuntimeError("engine HU layout changed: seat 0 must post SB and act first preflop")
        self._step = jax.jit(step_target)
        self.local = threading.local()
        self.lock = threading.Lock()
        self.stats = {"hands": 0, "decisions": 0, "events": 0, "failures": 0, "replays": 0, "new_hand_by_signature": 0,
                      "new_hand_by_history": 0, "forced_own": 0}
        self.failures = []
        self.decision_seconds = []
        self.decisions = self.batches = 0
        self.max_observed_batch = 1
        self.ready = False
        self.ready_line = None
        fingerprint = code_fingerprint()
        artifacts = {"preflop_table": file_sha256(preflop_table), "pre_end": file_sha256(critics_dir / "pre_end.npz"),
                     "flop_end": file_sha256(critics_dir / "flop_end.npz"), "turn_end": file_sha256(critics_dir / "turn_end.npz"),
                     "preflop_equity": file_sha256(preflop_equity),
                     **{name: file_sha256(critics_dir / f"{name}.npz") for name in (CR.POST_CRITICS.values() if self.gadget else ())}}
        self.metadata = {"path": "live (poker/player/live.py, poker/solver/subgame.py)", "artifacts": artifacts, **config.game_meta(),
                         "regime": self.regime, "iterations": {str(k): v for k, v in self.iterations.items()}, "chunk": self.chunk,
                         "eps": self.eps, "window_seconds": window_seconds, "dev_size": dev_size, "gadget": self.gadget, "gadget_deal": self.gadget_deal,
                         "certificate": bool(certificate),
                         "budget_seconds": budget_seconds, "ladder": list(SG.LADDER), "device": jax.default_backend(),
                         "field": {k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in SG.FIELD.items()},
                         "fingerprint": fingerprint, "fingerprint_note": "poker.envs.fingerprint.code_fingerprint()",
                         "translation": "none (exact chip totals, slot topology)",
                         "legacy_tracker": "retired from this path"}
        self.policy_id = hashlib.sha256(json.dumps(self.metadata, sort_keys=True, default=str).encode()).hexdigest()

    # ------------------------------------------------------------------ ready gate

    def warm_up(self):
        """Compile the whole topology x ladder set, verify the flop cache, turn the server ready; the ready line."""
        t0 = time.perf_counter()
        cache_path, _ = SG.compilation_cache_dir(self.jax_cache)             # CacheUnusable (tmpfs, full disk): not ready
        from_cache = any(cache_path.iterdir())
        flop = self.flop_equity(np.asarray([0, 14, 28]))
        turn = np.asarray(self.turn_equity(np.asarray([0, 14, 28, 3])))
        self.server.warm_up(flop, turn, self.preflop_equity, from_cache=from_cache, flop_cache=self.flop_cache,
                            turn_builder=self.turn_equity, cache_dir=self.jax_cache)
        if self.gadget:                                                       # the ceiling computation of every street compiles now
            self._player().player.warm_gadget(flop, turn)
        # The engine jits of the replay and of the player compile now, not in a hand: a step, the legality check, a reward
        dummy = jax.tree.map(np.asarray, self.base)
        np.asarray(self._step(dummy, config.CHECK_CALL, config.BB).stack)
        LV.legal(dummy, LV.CALL, config.BB)
        jax.tree.map(np.asarray, LV.engine_step(dummy, LV.CALL, config.BB))
        self.ready = bool(self.server.ready)
        self.ready_line = self.server.ready_line
        self.compile_events_at_ready = SG.compile_events()
        self.warm_up_seconds = time.perf_counter() - t0
        if self.log is not None:
            self.log(self.ready_line)
        return self.ready_line

    def close(self):
        self.server.close()

    # ------------------------------------------------------------------ per-worker players

    def _player(self):
        th = self.local
        if getattr(th, "player", None) is None:
            th.player = LV.LivePlayer(0, solver=LV.ServerClient(self.server, self.budget_seconds), preflop=self.table,
                                      eps=self.eps, equity=self.flop_equity, turn_equity=self.turn_equity,
                                      preflop_equity=self.preflop_equity, budget_seconds=self.budget_seconds, log=self.log,
                                      gadget=self.gadget, post_critics=self.post_critics, gadget_deal=self.gadget_deal)
            th.observed = 0
            th.events = ()
            th.signature = None
            th.hand_key = None
            th.hands = 0
        return th

    def _hand_start(self, response):
        """The engine state at the hand's start with our cards and the known board (the opponent's cards: two unused
        ones, never read - the live player tracks his range, not his cards)."""
        hero = SEAT[response.client_pos]
        hole = [our_card(c) for c in response.hole_cards]
        board = [our_card(c) for c in response.board]
        used = set(hole + board)
        dummy = [c for c in range(52) if c not in used][:2]
        holes = np.zeros((self.n, 2), np.int32)
        holes[hero], holes[1 - hero] = hole, dummy
        # The response and replay state are host data. Each hand owns its arrays;
        # cards need no device transfer before the betting replay or hand lookup.
        state = jax.tree.map(lambda x: np.array(x, copy=True), self.base)
        return state.replace(hole=holes, board=np.array(board + [-1] * (5 - len(board)), np.int32)), hero

    @staticmethod
    def _event_action(state, event):
        """(kind, street total) of a protocol event in the live player's convention: a fold at the actor's current
        street bet, a check/call at the level to call, a bet/raise at its total (ALLIN when it puts the actor all-in)."""
        seat = SEAT[event.actor]
        if event.kind == "fold":
            return LV.FOLD, int(state.bet_street[seat])
        if event.kind in ("check", "call"):
            return LV.CALL, int(max(state.bet_street))
        all_in = int(event.street_total) == int(state.bet_street[seat]) + int(state.stack[seat])
        return (LV.ALLIN if all_in else LV.RAISE), int(event.street_total)

    def _replay(self, response):
        """[(state_before, kind, total, state_after)] of every event, and the state to decide at."""
        state, hero = self._hand_start(response)
        expected_holes, expected_board = state.hole.copy(), state.board.copy()
        steps = []
        for e in response.state.events:
            seat = SEAT[e.actor]
            if int(state.to_act) != seat or bool(state.terminal):
                raise ProtocolError(f"replay: engine actor {int(state.to_act)} for protocol seat {seat}")
            kind, total = self._event_action(state, e)
            index = {LV.FOLD: config.FOLD, LV.CALL: config.CHECK_CALL}.get(kind, config.ALLIN if kind == LV.ALLIN else 2)
            before = int(state.bet_total[seat])
            try:                                     # the engine's own legality, on the host, before the jitted step: an action
                guards.target(state, index, total, "slumbot replay")   # the jitted engine would apply silently (audit envs F3)
            except ValueError as err:
                raise ProtocolError(f"replay: {err}") from None
            nxt = jax.tree.map(lambda x: np.array(x, copy=True), self._step(state, index, total))
            if not np.array_equal(nxt.hole, expected_holes) or not np.array_equal(nxt.board, expected_board):
                raise ProtocolError(f"replay changed cards after event {len(steps)}: "
                                    f"hole {np.asarray(nxt.hole).tolist()} expected {expected_holes.tolist()}, "
                                    f"board {np.asarray(nxt.board).tolist()} expected {expected_board.tolist()}")
            if int(nxt.bet_total[seat]) - before != e.contribution:
                raise ProtocolError(f"replay: engine added {int(nxt.bet_total[seat]) - before}, protocol {e.contribution}")
            steps.append((state, kind, total, nxt))
            state = nxt
        h = response.state
        if not h.terminal:
            ok = (int(state.street) == h.street and int(state.to_act) == SEAT[h.next_player]
                  and all(int(state.stack[SEAT[p]]) == h.stacks[p] for p in (0, 1))
                  and all(int(state.bet_street[SEAT[p]]) == h.street_bets[p] for p in (0, 1)))
            if not ok:
                raise ProtocolError(f"replay disagrees with the protocol at {h.action!r}")
        with self.lock:
            self.stats["replays"] += 1
        return steps, state, hero

    # ------------------------------------------------------------------ the decision

    def choose(self, response, rng):
        t0 = time.perf_counter()
        th = self._player()
        player = th.player
        hero = SEAT[response.client_pos]
        events = response.state.events
        signature = (response.client_pos, tuple(response.hole_cards))
        new_hand = (th.signature != signature or len(events) < th.observed or events[:th.observed] != th.events[:th.observed])
        if new_hand:
            with self.lock:
                self.stats["hands"] += 1
                self.stats["new_hand_by_signature" if th.signature != signature else "new_hand_by_history"] += 1
                th.hands += 1
                th.hand_key = f"w{threading.get_ident() % 1000}-h{th.hands}-pos{response.client_pos}"
            if player.hero != hero:                                             # the seat changes hand by hand
                th.player = player = LV.LivePlayer(hero, solver=LV.ServerClient(self.server, self.budget_seconds), preflop=self.table,
                                                   eps=self.eps, equity=self.flop_equity, turn_equity=self.turn_equity,
                                                   preflop_equity=self.preflop_equity, budget_seconds=self.budget_seconds, log=self.log,
                                                   gadget=self.gadget, post_critics=self.post_critics, gadget_deal=self.gadget_deal)
            th.observed = 0
        try:
            steps, state, _ = self._replay(response)
            if new_hand:
                player.new_hand(steps[0][0] if steps else state, hand_id=th.hand_key)
        except Exception as e:
            self._failed(e, th, player, response)
        try:
            for i in range(th.observed, len(steps)):
                sb, kind, total, sa = steps[i]
                if int(sb.to_act) == player.hero and (player.pending is None or player.pending[:2] != (kind, total)):
                    player.choose(sb, kind, total)           # a replayed history (an offline harness): our action fixed from
                    with self.lock:                          # outside; in a live match our events are the ones act() chose
                        self.stats["forced_own"] += 1
                player.observe(sb, kind, total, sa)
                with self.lock:
                    self.stats["events"] += 1
            th.observed = len(steps)
            th.events = events
            th.signature = signature
            kind, total = player.act(state, rng)
        except Exception as e:
            self._failed(e, th, player, response)
        c = player.pending[2]
        row = c.probs[player._hand(state)]
        if kind == LV.FOLD:
            increment = serialize_action(response.state, "fold")
        elif kind == LV.CALL:
            increment = serialize_action(response.state, "check_call")
        else:
            increment = serialize_action(response.state, "raise", raise_to=int(total))
        dt = time.perf_counter() - t0
        with self.lock:
            self.decisions += 1
            self.batches += 1
            self.stats["decisions"] += 1
            self.decision_seconds.append(dt)
        return LiveDecision(increment, int(kind), tuple(float(x) for x in row), 1000.0 * dt, int(state.street),
                            tuple((int(k), int(t)) for k, t in zip(c.kinds, c.targets)),
                            float(player.stats["solve_seconds"][-1]) if player.stats["solve_seconds"] else 0.0,
                            tuple(int(c) for c in state.hole[hero]), tuple(int(c) for c in state.board))

    def _failed(self, exc, th, player, response):
        record = {"hand": th.hand_key, "seat": player.hero, "street": response.state.street,
                  "player_street": getattr(player, "street", None), "error": type(exc).__name__,
                  "message": str(exc)[:500], "history": response.state.action, "time": time.strftime("%H:%M:%S"),
                  "client_pos": response.client_pos, "protocol_hole_cards": list(response.hole_cards),
                  "protocol_board": list(response.board),
                  "player_failure": player.failures[-1] if player.failures else None,
                  "counters": {k: (dict(v) if isinstance(v, dict) else v) for k, v in player.stats.items()
                               if k in ("decisions", "solves", "off_tree", "translated", "slot_overflow", "solve_errors",
                                        "budget_exceeded", "illegal_actions", "table_unreachable", "invalid_hand", "floor_hits",
                                        "gadget_streets")}}
        with self.lock:
            self.stats["failures"] += 1
            self.failures.append(record)
        if self.log is not None:
            self.log(f"live bot: decision FAILED ({record['error']}: {record['message'][:120]}) in hand {th.hand_key}")
        th.signature = None                                                     # the next request starts a fresh hand
        raise LiveDecisionFailure(f"{record['error']}: {record['message'][:200]}") from exc

    def summary(self):
        """What run.py writes next to the runner's summary: the counters, the failures, the decision times, the
        server's batches and errors, the merged live-player counters of every worker."""
        dec = np.asarray(self.decision_seconds, np.float64)
        return {"stats": dict(self.stats), "failures": list(self.failures), "ready_line": self.ready_line,
                "warm_up_seconds": getattr(self, "warm_up_seconds", None),
                "compiles_in_play": SG.compile_events() - getattr(self, "compile_events_at_ready", SG.compile_events()),
                "decision_seconds": ({"count": int(dec.size), "mean": float(dec.mean()), "p50": float(np.percentile(dec, 50)),
                                      "p90": float(np.percentile(dec, 90)), "max": float(dec.max())} if dec.size else None),
                "server": {"batches": list(self.server.batches), "errors": dict(self.server.errors),
                           "handovers": len(self.server.handovers)},
                "metadata": self.metadata}
