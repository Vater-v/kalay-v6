"""Duplicate self-play of the live play path (poker/arena/match.py): every seat a poker.player.live.LivePlayer,
every decision a street-start PCFR+ solve through the shared solve server (the trap collector), every action applied
on the engine at its exact chip total (poker.player.live.engine_step: legality asserted), no policy network, no legacy range
tracker. Each deal is played twice with the seats swapped (the same cards, depth and rake level); the pair sum nets the
seat advantage out, so a player against itself scores 0 up to the sampling noise of its own decisions.

A failed decision (BudgetExceeded, NonFinite, IllegalAction, TableUnreachable, InvalidHand, ServerUnavailable, ...) is
counted and the hand recorded with its failure type and the player's counters (the hand is void: no default action, no
reward); hands run on `threads` worker threads so the server batches their solves.
"""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np

from poker.arena.match import deal, depth_record, table_depths
from poker.arena.slumbot.util import file_sha256
from poker.envs import config
from poker.envs.evaluator import get_rewards
from poker.envs.fingerprint import code_fingerprint
from poker.player import live as LV
from poker.solver import critics as CR
from poker.solver import subgame as SG


class LiveArena:
    """The artifacts, the solver and the solve server shared by every hand of a match; warm_up() = the ready gate."""

    def __init__(self, *, critics_dir, preflop_table, flop_cache, rank7, preflop_equity, iterations=None, chunk=50,
                 window_seconds=0.3, dev_size=0, budget_seconds=4.0, eps=LV.DEFAULT_EPS, log=None, jax_cache=None,
                 engine_max_stack=None):
        self.log = log
        critics_dir, preflop_table, flop_cache, rank7, preflop_equity = (Path(x) for x in (critics_dir, preflop_table, flop_cache, rank7,
                                                                                      preflop_equity))
        for p in (critics_dir / "pre_end.npz", critics_dir / "flop_end.npz", critics_dir / "turn_end.npz", preflop_table, rank7, preflop_equity):
            if not p.exists():
                raise FileNotFoundError(f"live arena: {p} is missing")
        if not flop_cache.is_dir():
            raise FileNotFoundError(f"live arena: the flop class cache {flop_cache} is missing")
        self.critics = CR.load_live_critics(critics_dir)
        self.preflop_equity = np.load(preflop_equity)
        self.table = LV.PreflopTable(preflop_table, critic_path=critics_dir / "pre_end.npz")
        self.flop_cache = flop_cache
        self.flop_equity = lambda f: SG.flop_equity(f, read_only=flop_cache)
        self.turn_equity = SG.default_turn_equity(rank7)
        self.iterations = dict(config.PLAY_ITERATIONS if iterations is None else iterations)
        self.chunk = int(chunk)
        self.budget_seconds = budget_seconds
        self.eps = eps
        self.jax_cache = jax_cache
        self.engine_max_stack = int(table_depths(None)[1] if engine_max_stack is None else engine_max_stack)
        config.history_bound(2, start=self.engine_max_stack)
        self.solver = LV.Solver(iterations=self.iterations, chunk=self.chunk, critics=self.critics)
        self.server = LV.SolveServer(self.solver, window_seconds=window_seconds, dev_size=dev_size, log=None)   # quiet: one line per trap otherwise
        self.ready = False
        fingerprint = code_fingerprint()
        self.metadata = {"path": "live (poker/player/live.py, poker/solver/subgame.py)", **config.game_meta(), "regime": config.regime_record(),
                         "artifacts": {"preflop_table": file_sha256(preflop_table), "pre_end": file_sha256(critics_dir / "pre_end.npz"),
                                       "flop_end": file_sha256(critics_dir / "flop_end.npz"), "turn_end": file_sha256(critics_dir / "turn_end.npz"),
                                       "preflop_equity": file_sha256(preflop_equity)},
                         "iterations": {str(k): v for k, v in self.iterations.items()}, "chunk": self.chunk,
                         "eps": eps, "window_seconds": window_seconds, "dev_size": dev_size,
                         "budget_seconds": budget_seconds, "engine_warmup_max_stack_chips": self.engine_max_stack, "ladder": list(SG.LADDER), "device": jax.default_backend(),
                         "turn_matrix": getattr(self.turn_equity, "where", "?"), "fingerprint": fingerprint,
                         "fingerprint_note": "poker.envs.fingerprint.code_fingerprint()"}
        self.policy_id = hashlib.sha256(json.dumps(self.metadata, sort_keys=True, default=str).encode()).hexdigest()

    def warm_up(self):
        t0 = time.perf_counter()
        cache = Path(self.jax_cache if self.jax_cache is not None else config.JAX_CACHE_DIR).expanduser()
        from_cache = cache.exists() and any(cache.rglob("*"))
        flop = self.flop_equity(np.asarray([0, 14, 28]))
        turn = np.asarray(self.turn_equity(np.asarray([0, 14, 28, 3])))
        self.server.warm_up(flop, turn, self.preflop_equity, from_cache=from_cache, flop_cache=self.flop_cache,
                            turn_builder=self.turn_equity, cache_dir=self.jax_cache)
        # The engine jits of a hand compile now, not in a hand: a dealt state's step, the legality check, the rewards
        from poker.envs.betting import init
        dummy = jax.tree.map(np.asarray, init(jax.random.key(0), 2, np.full(2, self.engine_max_stack, np.int32), history=True,
                                              max_stack=self.engine_max_stack))
        LV.legal(dummy, LV.CALL, config.BB)
        jax.tree.map(np.asarray, LV.engine_step(dummy, LV.CALL, config.BB))
        np.asarray(get_rewards(dummy))
        self.ready = bool(self.server.ready)
        self.compile_events_at_ready = SG.compile_events()
        self.warm_up_seconds = time.perf_counter() - t0
        return self.server.ready_line

    def player(self, seat):
        return LV.LivePlayer(seat, solver=LV.ServerClient(self.server, self.budget_seconds), preflop=self.table, eps=self.eps,
                             equity=self.flop_equity, turn_equity=self.turn_equity, preflop_equity=self.preflop_equity,
                             budget_seconds=self.budget_seconds, log=self.log)

    def close(self):
        self.server.close()


def play_hand(arena, state0, rng, hand_id):
    """One hand from `state0` (a dealt host State), both seats live players: (rewards [n] in chips, decisions, seconds)
    or a failure record (the hand void)."""
    n = int(state0.stack.shape[0])
    players = [arena.player(seat) for seat in range(n)]
    for seat, p in enumerate(players):
        p.new_hand(state0, hand_id=f"{hand_id}-seat{seat}")
    state = state0
    decisions, t0 = [], time.perf_counter()
    while not bool(state.terminal):
        seat = int(state.to_act)
        player = players[seat]
        t1 = time.perf_counter()
        try:
            kind, total = LV.decide(player, state, rng, arena.budget_seconds)
        except Exception as e:
            return {"failed": True, "hand": hand_id, "seat": seat, "street": int(state.street), "error": type(e).__name__,
                    "message": str(e)[:300], "decisions": decisions, "depth_coverage": player.coverage, "player_failure": player.failures[-1] if player.failures else None,
                    "counters": {k: v for k, v in player.stats.items() if k in ("decisions", "solves", "off_tree", "solve_errors",
                                                                                 "budget_exceeded", "illegal_actions", "table_unreachable",
                                                                                 "invalid_hand", "floor_hits")}}
        nxt = jax.tree.map(np.asarray, LV.engine_step(state, kind, total))
        decisions.append({"seat": seat, "street": int(state.street), "kind": int(kind), "total": int(total),
                          "seconds": time.perf_counter() - t1, "solve_seconds": player.stats["solve_seconds"][-1] if player.stats["solve_seconds"] else 0.0})
        for p in players:
            p.observe(state, kind, total, nxt)
        state = nxt
    rewards = np.asarray(get_rewards(state), np.float64)
    stats = {k: sum(int(p.stats[k]) for p in players) for k in ("decisions", "table_decisions", "translated", "slot_overflow",
                                                               "floor_hits", "opponent_updates", "solve_errors")}
    stats["outside_training_depth_hands"] = int(players[0].coverage["status"] == "extrapolation")
    return {"failed": False, "hand": hand_id, "rewards": rewards.tolist(), "decisions": decisions, "seconds": time.perf_counter() - t0,
            "stats": stats, "depth_coverage": players[0].coverage, "min_ratio": float(min(p.stats["min_ratio"] for p in players)),
            "reach_bit_equal": all(np.array_equal(p.hero_reach(), p.ranges[p.hero]) for p in players)}


def live_duplicate(arena, deals, seed=0, stack_bb=None, rake_level=None, threads=4, log=None):
    """`deals` duplicate pairs through the production path: bb/100 of the pair sums (0 up to sampling noise: the
    player against itself), the failures, the decision times. Depths and rake levels follow the configured training regime."""
    config.heads_up_part("poker.arena.live_match")
    rng = np.random.default_rng(seed)
    depth_rng = np.random.default_rng([seed, 29])
    sample_depth, max_stack = table_depths(stack_bb)
    keys = jax.random.split(jax.random.key(seed * 100003), deals)
    depth = sample_depth(depth_rng, deals)
    stacks = (depth.astype(np.int64) * config.BB).astype(np.int32)
    levels = (config.sample_rake_levels(rng, deals) if rake_level is None else np.full(deals, rake_level)).astype(np.float32)
    states = deal(keys, stacks, levels, max_stack)
    first = jax.tree.map(lambda x: np.asarray(x)[0], states)              # the engine jits on the dealt shape, before any hand:
    LV.legal(first, LV.CALL, config.BB)                                    # a legality check, a step, the rewards
    jax.tree.map(np.asarray, LV.engine_step(first, LV.CALL, config.BB))
    np.asarray(get_rewards(first))
    events0 = SG.compile_events()                                          # compiles_in_play counts from the first hand
    hands = []
    for i in range(deals):
        s = jax.tree.map(lambda x: np.asarray(x)[i], states)
        for game in (0, 1):
            swapped = s if game == 0 else _swap_seats(s)
            hands.append((i, game, swapped))
    results = {}
    lock = threading.Lock()
    t0 = time.perf_counter()

    def run(item):
        i, game, s = item
        r = play_hand(arena, s, np.random.default_rng([seed, i, game]), f"deal{i}-game{game}")
        with lock:
            results[(i, game)] = r
            done = len(results)
            if log is not None and (done % max(1, 2 * deals // 10) == 0 or r["failed"]):
                fails = sum(x["failed"] for x in results.values())
                log(f"  {done}/{2 * deals} hands, failures {fails}, {done / (time.perf_counter() - t0):.2f} hands/s")

    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="live-hand") as pool:
        list(pool.map(run, hands))
    pair_sums, failures, dec, solve = [], [], [], []
    for i in range(deals):
        a, b = results[(i, 0)], results[(i, 1)]
        for r in (a, b):
            if r["failed"]:
                failures.append(r)
            for d in r["decisions"]:
                dec.append(d["seconds"])
                solve.append(d["solve_seconds"])
        if not a["failed"] and not b["failed"]:
            pair_sums.append(a["rewards"][0] + b["rewards"][1])        # seat 0 of game 0 and seat 1 of game 1 hold the same cards
    s = np.asarray(pair_sums, np.float64) / config.BB / 2
    dec, solve = np.asarray(dec), np.asarray(solve)
    stats = {}
    for r in results.values():
        for k, v in r.get("stats", {}).items():
            stats[k] = stats.get(k, 0) + v
    return {"deals": int(deals), "hands": int(2 * deals), "pairs_scored": int(len(s)), "failures": failures, "failed_hands": len(failures),
            "bb_per_100_pair_sum": float(100 * s.mean()) if len(s) else None, "ci95": float(196 * s.std() / np.sqrt(len(s))) if len(s) else None,
            "decision_seconds": {"count": int(dec.size), "mean": float(dec.mean()), "p50": float(np.percentile(dec, 50)),
                                 "p90": float(np.percentile(dec, 90)), "max": float(dec.max())} if dec.size else None,
            "solve_seconds": {"mean": float(solve.mean()), "p90": float(np.percentile(solve, 90)), "max": float(solve.max())} if solve.size else None,
            "within_budget": bool(dec.size and dec.max() <= arena.budget_seconds), "budget_seconds": arena.budget_seconds,
            "counters": stats, "min_ratio": float(min([r["min_ratio"] for r in results.values() if not r["failed"]] or [float("nan")])),
            "reach_bit_equal": all(r.get("reach_bit_equal", False) for r in results.values() if not r["failed"]),
            "seconds": time.perf_counter() - t0, "hands_per_second": 2 * deals / (time.perf_counter() - t0),
            "server": {"batches": list(arena.server.batches), "errors": dict(arena.server.errors), "handovers": len(arena.server.handovers)},
            "players": 2, "engine_history_capacity": int(first.history_amount.shape[0]), **depth_record(stack_bb, depth), "rake": [list(r) for r in config.RAKE_MIX] if rake_level is None else rake_level,
            "regime": config.regime_record(), "game_hash": config.game_hash(), "metadata": arena.metadata, "policy_id": arena.policy_id,
            "ready_line": arena.server.ready_line, "warm_up_seconds": arena.warm_up_seconds, "threads": threads,
            "compiles_in_play": SG.compile_events() - events0, "compiles_setup": events0 - arena.compile_events_at_ready,
            "loadavg": open("/proc/loadavg").read().split()[:3]}


def _swap_seats(s):
    """The same deal with the two seats' cards exchanged (the duplicate game): the hole cards swap, the board stays."""
    hole = np.asarray(s.hole).copy()
    return s.replace(hole=hole[::-1].copy())
