"""The live player: every decision from a PCFR+ solve of the street from its START on one fixed
topology per (street, bucket, players, hero seat), the played history on the tree (our actions frozen
to what we played, the opponent's actual sizes in the topology's slots), the preflop table as the warm
cache of the preflop topology, exact own range, Bayesian opponent ranges with a likelihood floor, no
translation of any opponent action.

Design in ten lines:
 1. Root of every solve = the street-start State as the engine had it (pot, stacks, blinds preflop), the
    ranges at the street start (ours exact, theirs Bayesian), the played history of the street on the
    tree; the policy at the node the history reaches is what we play. No translation: every action is
    applied to the engine at its exact amount (poker.envs.transition.step_target), an opponent's size the size
    set does not have goes into the node's slot of the topology (poker/solver/subgame.py).
 2. Size sets = config.street_sizes(street, bucket), the bucket decided at the street start
    (config.size_bucket), cap = config.tree_rule's cap of the menu; the all-in always.
 3. Our played actions are frozen in every later solve of the street (pcfr fixed_mask / fixed_pi): the
    solve's reach at the current node is our exact range. The opponent's likelihoods come from the first
    solve whose tree holds his exact action in its slot ("insert" likelihoods, checked by checks/s8_0.py).
 4. Our range = the exact reach of the strategies we played (times board restrictions; float64 on the
    host; recomputed from the records and asserted bit-equal). Every other seat: r'(h) = r(h) max(L(h),
    eps) / Z, eps = 1e-3 per hand on the LIKELIHOOD; Z <= 1, so r'(h) >= eps r(h) at every update - no
    collapse by construction (worst ratio and floor hits counted).
 5. Fixed T per street and bucket (iterations), chunk per device call, no certificate unless asked; no
    partial solve, no default action: a solve that cannot run (or a decision over the budget when one is
    set) raises and is counted (solve_errors, budget_exceeded).
 6. Players from the State (ranges [n, H], every non-hero seat an opponent; the field the bot covers
    comes from config.regime_field: table sizes, seats, streets, menus, the batch ladder).
 7. Preflop: the precomputed table (PreflopTable, results/preflop_table/) answers when the hand is the
    table's spot (200 BB, the blinds, no rake) and the whole history is in its tree; anything else (an
    off-table size, other stacks) is a root solve of the preflop topology with the slots (the checkpoint's pre_end
    critic at the flop starts, the exact preflop matrix).
 8. Flop and turn solves need the next street's critic (critics={1: flop_end, 2: turn_end}) and the
    all-in matrix of the board (flop: the cache of the 1755 classes, turn: the rank7 table on the host).
 9. Batching: LivePlayer asks a Solver (solve_batch of poker/solver/subgame.py) - one request per decision;
    a shared Solver collects concurrent requests of one topology into one vmapped call (many tables).
10. The re-solving gadget (CFR-D / DeepStack; pcfr.gadget_*): from the second street of a hand on, the solve
    carries OUR exact range and, for every other seat, per-hand counterfactual-value CEILINGS at the street
    start: the post-card critic of the new street (critics.load_post_critics) evaluated at the leaf the previous
    street ended in (both seats' reach there under the previous solve's average strategy), for the card that was
    dealt. The seat's DEAL range in the solve (gadget_deal): "mix" (default) = gadget_mix x his tracked range (item
    4) + (1 - gadget_mix) x uniform over the hands the board allows; "range" = his tracked range alone (Burch,
    Johanson, Bowling 2014's form: Terminate can only remove hands); "uniform" = no range at all. Measured on the
    real HU river tree (tests/gadget_river_resolve.py: exact leaves, 3 spots x 3 trunk bets, T 800, the re-solved
    subgame spliced into the full game), in mbb/hand over the full solve: range only +16..+942; with a WRONG carried
    range (his bluffs crushed to 1e-3 of their weight, what the likelihood floor does to hands the solves called
    unlikely) range only +1100..+7700 and the range deal +1..+4200; the uniform deal +2..+137 with exact ceilings;
    the half-and-half mix +0..+161 with exact ceilings and +1..+147 with 0.01 pot of per-hand ceiling noise, in
    8 of 9 cases below +40. A ceiling BIAS of 0.05 pot costs up to +324 (too high: he Terminates everything) or
    +506 (too low): the critic's per-hand error is what the gadget needs small. gadget=False is the old path.

Protocol (one hand): new_hand(state0); at our turn act(state, rng) -> (kind, total) (or choose(state,
kind, total) for a fixed action); after EVERY action of every seat observe(state_before, kind, total,
state_after), ours included; step(state, kind, total) applies an action on the engine.
"""

import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.envs.actions import action_table
from poker.envs.transition import step_target
from poker.solver import cards as C
from poker.solver import checkdown as CD
from poker.solver import critics as CR
from poker.solver import situations as SI
from poker.solver import subgame as SG
from poker.solver.cards import NUM_HANDS as H

FOLD, CALL, RAISE, ALLIN = SG.FOLD, SG.CALL, SG.RAISE, SG.ALLIN
DEFAULT_ITERATIONS = dict(config.PLAY_ITERATIONS)        # per street (a (street, menu) key overrides): poker/envs/config.py
DEFAULT_EPS = 1e-3                                       # likelihood floor (module doc, item 4)
GRID13_KINDS = {0: FOLD, 1: CALL, 12: ALLIN}             # the preflop table's action index -> kind (2..11: sizes)
street_cap = SG.street_cap


@dataclass
class Candidates:
    """The actions a model offers at one decision and their per-hand probabilities."""
    kinds: np.ndarray        # [K] FOLD / CALL / RAISE / ALLIN
    targets: np.ndarray      # [K] street totals in chips
    probs: np.ndarray        # [H, K] float64, rows sum to 1 over the candidates
    level: int               # bet level of the state (chips)
    base: int                # pot + to_call
    maximum: int             # the actor's all-in total

    def column(self, kind, target):
        for k in range(len(self.kinds)):
            if int(self.kinds[k]) == kind and (kind in (FOLD, CALL) or int(self.targets[k]) == int(target)):
                return k
        return None


def inference_coverage(state):
    """Observable input coverage, never a restriction on legal engine states.

    In HU only the smaller starting stack is contestable. The larger player's
    unmatched reserve stays in the real state but does not create a deeper game.
    This report describes depth only; it is not a certificate of strategy quality.
    """
    depths = (np.asarray(state.stack, np.float64) + np.asarray(state.bet_total, np.float64)) / config.BB
    effective = float(depths.min()) if len(depths) == 2 else None
    report = config.depth_coverage([effective] if effective is not None else depths)
    return {**report, "starting_stacks_bb": depths.tolist(), "effective_hu_depth_bb": effective,
            "uncontestable_reserve_bb": (depths - effective).tolist() if effective is not None else None,
            "scope": "declared starting-depth coverage only; no quality guarantee"}


class BudgetExceeded(RuntimeError):
    """A decision took longer than the configured budget (counted; never a default action)."""


class IllegalAction(ValueError):
    """An action the engine's rules do not allow at this state (poker/envs/actions.py action_table): never
    applied - poker/envs/transition.py clips an out-of-range index silently and its checkify calls are no-ops
    unless wrapped (spy hunt 2026-10-11), so the play path checks before every step."""


class InvalidHand(ValueError):
    """Our hole cards are not a hand: the absent sentinel (-1, hand_index 1326) where a decision needs our hand, a card
    out of 0..51, a duplicate, or a card on the visible board. Refused before anything is solved or played (counted
    `invalid_hand`, the hand failed with its record)."""


class TableUnreachable(RuntimeError):
    """The preflop table (or a solve) holds no strategy for our hand at this node: our own reach of
    the node with this hand is below TABLE_REACH_MIN (pcfr's never-reached rows are uniform; the
    preflop table v1 has 1,045 such rows of 37,128). Unreachable for us by construction: counted,
    never played, and no root re-solve is triggered by it."""


_STEP = jax.jit(step_target)
_INFO = jax.jit(SG._node_info)
_TABLE = jax.jit(action_table)
ENGINE_INDEX = {FOLD: config.FOLD, CALL: config.CHECK_CALL, RAISE: 2, ALLIN: config.ALLIN}
TABLE_REACH_MIN = 1e-7                                   # pcfr's fallback appears below own reach ~8.7e-8 (spy hunt)


def legal(state, kind, target):
    """The engine's rules for (kind, total) at `state`: fold only facing a bet (total = our street bet),
    check/call at the call total, the all-in at the actor's maximum, a raise at least the minimum raise
    and below the all-in when the rules allow a raise. False for any other kind."""
    if kind not in (FOLD, CALL, RAISE, ALLIN) or bool(np.asarray(state.terminal)):
        return False
    mask, _ = (np.asarray(x) for x in _TABLE(state))
    i = _info(state)
    target = int(target)
    if kind == FOLD:
        return bool(mask[config.FOLD]) and target == i["current"]
    if kind == CALL:
        return bool(mask[config.CHECK_CALL]) and target == i["call_target"]
    if kind == ALLIN:
        return bool(mask[config.ALLIN]) and target == i["maximum"]
    return bool(mask[config.ALLIN]) and i["minimum"] <= target < i["maximum"]


def check_legal(state, kind, target):
    if not legal(state, kind, target):
        raise IllegalAction(f"{SG.KIND_NAMES.get(kind, kind)} to {target} is not a legal action of this state")
    capacity = int(np.asarray(state.history_player).shape[0])
    if capacity and int(np.asarray(state.history_len)) >= capacity:
        raise IllegalAction(f"the hand's history is full ({capacity} decisions): the engine would drop this step silently "
                            "(config.history_bound: exact min-raise wars beyond its bound)")


def engine_step(state, kind, target):
    """Apply (kind, total) on the engine at its exact amount after the legality check (history index:
    FOLD, CHECK_CALL, 2 for a raise to `total`, ALLIN for the all-in - raises_on_street counts the last
    two). Raises IllegalAction instead of letting the engine clip or ignore the action."""
    check_legal(state, int(kind), int(target))
    return _STEP(state, jnp.int32(ENGINE_INDEX[int(kind)]), jnp.int32(int(target)))


def decide(player, state, rng, budget_seconds, executor=None):
    """act() under a wall-clock budget: the decision runs in a worker thread and must return within
    budget_seconds, else BudgetExceeded is raised and counted (poker.player.stats budget_exceeded) - the
    runner reports the hand as failed, nothing is played by default (the stalled computation finishes
    in its thread; a new executor per call unless one is given)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    own = executor is None
    executor = executor or ThreadPoolExecutor(max_workers=1)
    future = executor.submit(player.act, state, rng)
    try:
        return future.result(timeout=budget_seconds)
    except FutureTimeout:
        player.stats["budget_exceeded"] += 1
        player.fail(BudgetExceeded(f"decision did not return within {budget_seconds} s"))
    finally:
        if own:
            executor.shutdown(wait=False)


def hand_index(cards):
    a, b = sorted(int(x) for x in cards)
    return int(C.PAIR_INDEX[a, b])


def _info(state):
    return jax.tree.map(lambda x: int(np.asarray(x)), _INFO(state))


class PreflopTable:
    """The precomputed preflop table (results/preflop_table/preflop_<set>_regime<r>.npz)
    as the warm cache of the preflop topology. The NPZ stores the average policy, topology, legal
    actions and chip targets; the adjacent JSON records the spot and metadata. It returns the average strategy
    at the node a played history reaches. fits(state): the hand root is the table's spot (two players,
    its stacks, the blinds, no rake). node(history): the table's node, or None when the history leaves
    its tree (an action it does not have: the player solves the preflop from the root instead)."""

    def __init__(self, path, *, critic_path):
        if config.GRID != config.STREET_SIZES_GRID:
            raise ValueError("live preflop tables require current grid 13")
        path = Path(path)
        self.meta = json.loads(path.with_suffix(".json").read_text())
        config.check_game(self.meta, str(path))
        if self.meta.get("format") != "preflop-canonical-v1" or self.meta.get("game_hash") != config.game_hash():
            raise ValueError(f"{path}: canonical preflop table of the current game required")
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if self.meta.get("critic", {}).get("sha256") != digest(critic_path):
            raise ValueError(f"{path}: pre_end checkpoint differs; regenerate the preflop table")
        if self.meta.get("npz_sha256") != digest(path):
            raise ValueError(f"{path}: table checksum differs from its metadata")
        z = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.policy = np.asarray(z["policy"], np.float64)                 # [decisions, H, 13]
        self.internal = np.asarray(z["internal"])
        self.to_internal = np.full(len(z["parent"]), -1, np.int32)
        self.to_internal[self.internal] = np.arange(len(self.internal))
        self.parent, self.label, self.actor = (np.asarray(z[k]) for k in ("parent", "label", "actor"))
        self.children, self.legal, self.targets = (np.asarray(z[k]) for k in ("children", "legal", "targets"))
        self.terminal, self.street_bet, self.pot = (np.asarray(z[k]) for k in ("terminal", "street_bet", "pot"))
        self.stack = int(self.meta["spot"]["stack_chips"])
        if self.policy.shape != (len(self.internal), H, config.NUM_ACTIONS):
            raise ValueError(f"{path}: unexpected policy shape {self.policy.shape}")
        if not np.isfinite(self.policy).all() or np.any(self.policy < 0):
            raise ValueError(f"{path}: invalid action probabilities")
        self.n = int(self.street_bet.shape[1])

    def fits(self, state):
        s = jax.tree.map(np.asarray, state)
        if int(s.stack.shape[0]) != self.n or int(s.street) != 0 or int(s.history_len) != 0:
            return False
        if not np.array_equal(s.bet_street, np.asarray([config.SB, config.BB])):
            return False
        return bool(np.all(s.stack + s.bet_street == self.stack)) and int(s.rake_bps) == 0

    def columns(self, v):
        """[(column, kind, total)] of the table's legal actions at node v."""
        return [(a, GRID13_KINDS.get(a, RAISE), int(self.targets[v, a])) for a in range(self.legal.shape[1])
                if self.legal[v, a] and self.children[v, a] >= 0]

    def node(self, history):
        v = 0
        for kind, target in history:
            if self.terminal[v]:
                return None
            hit = [a for a, k, t in self.columns(v) if k == kind and (kind in (FOLD, CALL) or t == int(target))]
            if not hit:
                return None
            v = int(self.children[v, hit[0]])
        return v

    def candidates(self, v):
        """The table's strategy at node v as Candidates."""
        cols = self.columns(v)
        probs = self.policy[self.to_internal[v]][:, [a for a, _, _ in cols]]
        probs = probs / np.maximum(probs.sum(1, keepdims=True), 1e-300)
        actor = int(self.actor[v])
        level = max(int(self.street_bet[v].max()), config.BB)
        to_call = level - int(self.street_bet[v, actor])
        return Candidates(np.asarray([k for _, k, _ in cols]), np.asarray([t for _, _, t in cols]), probs,
                          level, int(self.pot[v]) + to_call, int(self.targets[v, 12]))


def _restrict(r, valid):
    r = r * valid
    return r / r.sum()


def _bayes(r, like):
    r = r * like
    return r / r.sum()


def hero_reach(records):
    """The reach of the played strategy from the recorded events: ("like", L) multiplies and
    renormalises, ("board", valid) restricts and renormalises - the same float64 operations in the
    same order as the player's own bookkeeping, so the result is bit-equal to the tracked range."""
    r = np.full(H, 1.0 / H)
    for kind, arr in records:
        if kind == "start":
            r = np.array(arr, np.float64)
        else:
            r = _bayes(r, arr) if kind == "like" else _restrict(r, arr)
    return r


def candidates_at(sol, v):
    """The strategy of a Solution at its node v as Candidates (legal columns, rows renormalised)."""
    tree = sol.tree
    cols = tree.columns(v)
    sigma = sol.strategy(v).astype(np.float64)
    probs = sigma[:, [a for a, _, _ in cols]]
    probs = probs / np.maximum(probs.sum(1, keepdims=True), 1e-300)
    info = tree.info
    return Candidates(np.asarray([k for _, k, _ in cols]), np.asarray([t for _, _, t in cols]), probs,
                      int(info["level"][v]), int(info["pot"][v]) + int(info["to_call"][v]), int(info["maximum"][v]))


class ServerUnavailable(RuntimeError):
    """The solve server is not running: the request fails at once (counted; never a default action)."""


class Solver:
    """The solving service: solve(request) for one decision, solve_batch(requests) for concurrent
    decisions of ONE topology in one vmapped call (many tables). A request: (tree, ranges, fixed,
    equity). iterations / chunk / critics / certificate are its configuration; the device is JAX's
    (JAX_PLATFORMS), T may differ per device through `iterations`. SolveServer puts it behind a
    queue for many tables."""

    def __init__(self, iterations=None, chunk=50, critics=None, rake_level=0.0, certificate=False, log=None):
        self.iterations = dict(DEFAULT_ITERATIONS if iterations is None else iterations)
        self.chunk = int(chunk)
        self.critics = critics or {}
        self.rake_level = float(rake_level)
        self.certificate = bool(certificate)
        self.log = log
        self.calls, self.spots, self.seconds = 0, 0, 0.0

    def iterations_for(self, street, bucket):
        T = self.iterations.get((street, bucket), self.iterations.get(street))
        if T is None:
            raise ValueError(f"no iteration count for street {street} (bucket {bucket})")
        return int(T)

    def solve_batch(self, requests):
        """requests: [(tree, ranges, fixed, equity[, gadget])] of one topology -> [Solution]."""
        if not requests:
            return []
        trees = [r[0] for r in requests]
        street, bucket = trees[0].street, trees[0].shape.bucket
        T = self.iterations_for(street, bucket)
        sols = SG.solve_batch(trees, [r[1] for r in requests], T, self.chunk, aux=self.critics.get(street),
                              rake_level=self.rake_level, equity=[r[3] for r in requests], fixed=[r[2] for r in requests],
                              certificate=self.certificate, gadget=[r[4] if len(r) > 4 else None for r in requests])
        self.calls += 1
        self.spots += len(requests)
        self.seconds += sols[0].seconds
        return sols

    def solve(self, tree, ranges, fixed=None, equity=None, gadget=None):
        return self.solve_batch([(tree, ranges, fixed, equity, gadget)])[0]


class SolveServer:
    """The device's owner for many tables - the trap collector (user's rules, 2026-10-11): a trap is a
    buffer of requests that is always standing. The first request landing in the active trap starts its
    timer (window_seconds, default 0.3 s; 0 = hand over at once); everything arriving within the window
    joins; at the trigger (the timer, or size_cap requests) the trap is handed to the solver and
    processed at once, the spare trap becomes the active one and the cleaner creates a new spare, so
    one active + one spare always stand (a request never finds no trap, never waits behind a lock, is
    never refused); requests arriving while a batch is processed land in the active trap and go next.
    When idle the cleaner trims any surplus back to exactly one active + one spare (a trap holding a
    request is never destroyed). A handed-over trap is processed by topology: every request of one
    topology in one vmapped solve_batch. The developer toggle dev_size = N (the 16-worker checkpoint test)
    makes a single trap wait for N requests with no timer. request() returns the Solution within
    `timeout` seconds or raises BudgetExceeded (ServerUnavailable once closed), counted in `errors`;
    nothing is ever played by default. One worker thread; JAX releases the GIL while the device computes."""

    MAX_WINDOW = 0.3

    def __init__(self, solver, window_seconds=0.3, size_cap=None, dev_size=0, log=None):
        import threading
        if not 0.0 <= window_seconds <= self.MAX_WINDOW:
            raise ValueError(f"the collecting window is at most {self.MAX_WINDOW} s, got {window_seconds}")
        self.solver = solver
        self.window = float(window_seconds)
        self.size_cap = int(SG.LADDER[-1] if size_cap is None else size_cap)     # a trap never exceeds the ladder's top
        if self.size_cap > SG.LADDER[-1]:
            raise ValueError(f"size_cap {self.size_cap} above the ladder's top {SG.LADDER[-1]}: that batch shape is never compiled")
        self.ready = False                      # true after warm_up(): every topology x ladder step compiled, artifacts loaded
        self.ready_line = None
        self.field, self.warmed = None, set()
        self.dev_size = int(dev_size)
        self.log = log
        self.cond = threading.Condition()
        self.traps = [self._new_trap(), self._new_trap()]      # [active, spare, ...]
        self.processing = None                                  # the trap handed to the solver (not standing)
        self.running = True
        self.batches = []                                       # sizes of the solve batches, in order
        self.handovers = []                                     # (requests in the trap, topologies) per hand-over
        self.errors = {"timeout": 0, "unavailable": 0, "failed": 0, "outside_field": 0}
        self.thread = threading.Thread(target=self._loop, name="solve-server", daemon=True)
        self.thread.start()

    @staticmethod
    def _new_trap():
        return {"requests": [], "started": None}

    @property
    def standing(self):
        """Standing traps (active + spares); the invariant when idle: exactly 2."""
        with self.cond:
            return len(self.traps)

    def warm_up(self, flop, turn, preflop, field=None, from_cache=None, flop_cache=None, turn_builder=None, cache_dir=None):
        """Compile every topology of the field (config.regime_field, or a sub-field for a test) at every
        ladder step with the solver's T / chunk / critics (subgame.warm_up), then the server is ready and
        refuses any request outside the warmed set. flop / turn / preflop: the all-in matrices of a dummy
        board (the loaders the bot uses: the flop cache, the rank7 table, the preflop matrix - loaded
        before this call). Returns the warm-up rows; prints the ready line through `log`."""
        field = field or SG.FIELD
        streets = set(field["streets"])
        needed = {st for st in streets if st < 3}
        if not needed <= set(self.solver.critics):
            raise ValueError(f"warm_up: critics for the streets {sorted(needed - set(self.solver.critics))} are missing")
        if flop_cache is not None and 1 in streets:
            n_classes, _, _ = SG.flop_cache_status(flop_cache)
            if n_classes < len(SG.FL.FLOPS):
                raise SG.MissingArtifact(f"the flop cache {flop_cache} has {n_classes} of {len(SG.FL.FLOPS)} classes: not ready")
        t0 = time.perf_counter()
        cache = None
        if cache_dir is not False:                                # the persistent cache on disk, checked (CacheUnusable: not ready)
            cache = SG.compilation_cache_dir(cache_dir)
        rows = SG.warm_up(self.solver.iterations, self.solver.chunk, self.solver.critics, flop, turn, preflop, field, log=self.log,
                          certificate=self.solver.certificate)
        if turn_builder is not None and 2 in streets:             # the device's turn matrix builder compiles now, not in a hand
            np.asarray(turn_builder(np.asarray([0, 14, 28, 3])))
        self.field = field
        self.warmed = {r[0] for r in rows}
        self.ready_line = SG.ready_line(rows, time.perf_counter() - t0, from_cache, field, flop_cache)
        if cache is not None:
            self.ready_line += f"; jax cache {cache[0]} ({cache[1]:.0f} MB free)"
        if turn_builder is not None:
            self.ready_line += f"; turn matrix {getattr(turn_builder, 'where', 'builder')}"
        self.ready = True
        if self.log is not None:
            self.log(self.ready_line)
        return rows

    def submit(self, tree, ranges, fixed=None, equity=None, gadget=None):
        """Drop a request into the active trap; returns a concurrent.futures.Future of its Solution. Before
        warm_up() the request is refused (ServerUnavailable, counted): the scheme is ready after the warm-up,
        it never compiles at play."""
        from concurrent.futures import Future
        future = Future()
        with self.cond:
            if not self.running or not self.ready:
                self.errors["unavailable"] += 1
                future.set_exception(ServerUnavailable("the solve server is closed" if not self.running
                                                       else "the solve server is not ready (warm_up first)"))
                return future
            if tree.key not in self.warmed:                       # another table size or menu: never a compile mid-session
                self.errors["outside_field"] += 1
                field = self.field or {"players": (), "menus": ()}
                future.set_exception(ServerUnavailable(f"topology {tree.key} is outside the warmed field "
                                                       f"(n={list(field['players'])}, menus={list(field['menus'])})"))
                return future
            trap = self.traps[0]
            if not trap["requests"]:
                trap["started"] = time.perf_counter()           # the timer starts with the first request
            trap["requests"].append((tree.key, (tree, ranges, fixed, equity, gadget), future, time.perf_counter()))
            self.cond.notify()
        return future

    def request(self, tree, ranges, fixed=None, equity=None, timeout=None, gadget=None):
        from concurrent.futures import TimeoutError as FutureTimeout
        future = self.submit(tree, ranges, fixed, equity, gadget)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            self.errors["timeout"] += 1
            raise BudgetExceeded(f"the solve did not return within {timeout} s")

    def close(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()
        self.thread.join(timeout=5.0)

    def _ready(self, trap):
        """Under the lock: the active trap's trigger - the window elapsed, or the size cap, or (developer
        mode) dev_size requests; returns the seconds to wait when not ready yet."""
        n = len(trap["requests"])
        if n == 0:
            return None
        if self.dev_size > 0:
            return 0.0 if n >= self.dev_size else None
        if n >= self.size_cap:
            return 0.0
        remaining = trap["started"] + self.window - time.perf_counter()
        return 0.0 if remaining <= 0 else remaining

    def _clean(self):
        """Under the lock: exactly one active + one spare standing (a spare is created at a hand-over,
        surplus empty traps are trimmed when idle; a trap holding a request is never destroyed)."""
        while len(self.traps) < 2:
            self.traps.append(self._new_trap())
        while len(self.traps) > 2 and not self.traps[-1]["requests"]:
            self.traps.pop()

    def _loop(self):
        while True:
            with self.cond:
                while self.running:
                    wait = self._ready(self.traps[0])
                    if wait == 0.0:
                        break
                    self.cond.wait(wait)                      # None: until a request lands; else until the trigger
                    self._clean()
                if not self.running:
                    for trap in self.traps:
                        for _, _, future, _ in trap["requests"]:
                            future.set_exception(ServerUnavailable("the solve server is closed"))
                        trap["requests"].clear()
                    return
                trap = self.traps[0]
                if len(trap["requests"]) > self.size_cap:     # a trap that grew past the cap while the device was busy:
                    rest = self._new_trap()                   # at most size_cap requests go (the ladder's top), the rest
                    rest["requests"] = trap["requests"][self.size_cap:]   # stay active and go next
                    rest["started"] = time.perf_counter()
                    trap["requests"] = trap["requests"][:self.size_cap]
                    self.traps[0] = rest
                else:
                    self.traps.pop(0)                         # hand-over: the spare becomes active,
                self._clean()                                 # the cleaner creates the new spare
                self.processing = trap
            groups = {}
            for item in trap["requests"]:
                groups.setdefault(item[0], []).append(item)
            self.handovers.append((len(trap["requests"]), len(groups)))
            n_requests, n_groups = len(trap["requests"]), len(groups)
            for key, items in groups.items():
                try:
                    sols = self.solver.solve_batch([req for _, req, _, _ in items])
                    self.batches.append(len(items))
                    for (_, _, future, _), sol in zip(items, sols):
                        future.set_result(sol)
                    del sols
                except Exception as e:                        # the group fails: every requester sees the error
                    self.errors["failed"] += len(items)
                    for _, _, future, _ in items:
                        future.set_exception(e)
            groups.clear()                                    # the payloads and solutions belong to the requesters now
            trap["requests"] = []
            with self.cond:
                self.processing = None
                self._clean()
            if self.log is not None:
                self.log(f"solve server: trap of {n_requests} requests, {n_groups} topologies")


class ServerClient:
    """A table's handle on a SolveServer with the Solver interface LivePlayer uses: solve() requests
    with the table's budget (seconds; None = no budget)."""

    def __init__(self, server, budget_seconds=None):
        self.server = server
        self.budget_seconds = budget_seconds
        self.chunk = server.solver.chunk
        self.critics = server.solver.critics
        self.iterations = server.solver.iterations

    def solve(self, tree, ranges, fixed=None, equity=None, gadget=None):
        return self.server.request(tree, ranges, fixed, equity, timeout=self.budget_seconds, gadget=gadget)


class LivePlayer:
    def __init__(self, hero, *, solver=None, preflop=None, sizes=None, bucket=None, eps=DEFAULT_EPS, cap=None, equity=None,
                 turn_equity=None, preflop_equity=None, budget_seconds=None, log=None, prefetch_equity=True, gadget=True,
                 post_critics=None, gadget_deal="mix", gadget_mix=0.5):
        """hero: our seat. solver: the Solver (default: a private one with default iterations). preflop:
        the PreflopTable (warm cache) or None. sizes: (street, bucket) -> fractions per raise level for
        every seat (default config.street_sizes). bucket: street-start State -> bucket (default
        subgame.bucket_of). cap: {street: aggressive actions} (default street_cap of the bucket). equity:
        flop board -> all-in matrix (default subgame.flop_equity); turn_equity: turn board -> matrix
        (default subgame.turn_equity with the rank7 table when results/cache has it); preflop_equity:
        the exact preflop matrix [1326, 1326] (needed, with poker.solver.critics[0], for preflop solves).
        budget_seconds: a decision (range update + tree + solve + policy, re-solves included) that
        takes longer raises BudgetExceeded and is counted; None = no budget. gadget: re-solve every street after
        the hand's first with the opponents' per-hand ceilings (module doc, item 10) from post_critics = {street:
        critic} (critics.load_post_critics: 1 flop, 2 turn, 3 river); gadget_deal: "mix" (gadget_mix x their tracked
        ranges + (1 - gadget_mix) x uniform as the deal, the ceilings on top; module doc item 10), "range" or "uniform"."""
        self.hero = int(hero)
        self.gadget = bool(gadget)
        if gadget_deal not in ("mix", "range", "uniform"):
            raise ValueError(f"gadget_deal must be 'mix', 'range' or 'uniform', got {gadget_deal!r}")
        if not 0.0 <= float(gadget_mix) <= 1.0:
            raise ValueError(f"gadget_mix must be in [0, 1], got {gadget_mix!r}")
        self.gadget_deal = gadget_deal
        self.gadget_mix = float(gadget_mix)
        self.post_critics = dict(post_critics or {})
        if self.gadget and set(self.post_critics) != {1, 2, 3}:
            raise ValueError("gadget=True needs the post-card critics of the flop, turn and river "
                             "(critics.load_post_critics, keys 1, 2, 3); gadget=False carries Bayesian ranges instead")
        self.solver = solver or Solver()
        self.preflop = preflop
        self.sizes = sizes
        self.bucket_of = bucket or SG.bucket_of
        self.eps = float(eps)
        self.cap = cap or {}
        self.equity = equity or SG.flop_equity
        self.turn_equity = turn_equity or SG.default_turn_equity()      # the device's builder
        self.prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="equity") if prefetch_equity else None
        self.equity_future = None
        self.preflop_equity = None if preflop_equity is None else np.asarray(preflop_equity, np.float32)
        self.budget_seconds = budget_seconds
        self.reach_min = TABLE_REACH_MIN         # a scenario that forces our line sets 0.0 (checks only)
        self.log = log
        self.stats = self.empty_stats()
        self.failures = []
        self.hand_id = None
        self.n = None

    @staticmethod
    def empty_stats():
        return {"hands": 0, "outside_training_depth_hands": 0, "decisions": 0, "table_decisions": 0, "translated": 0, "solves": {}, "off_tree": {},
                "slot_overflow": 0, "opponent_updates": 0, "floor_hits": 0, "min_ratio": np.inf, "solve_seconds": [],
                "solve_cpu_seconds": [], "build_seconds": [], "decision_seconds": [], "compile_seconds": 0.0,
                "certificates": [], "structures": set(), "solve_errors": 0, "budget_exceeded": 0, "illegal_actions": 0,
                "table_unreachable": 0, "invalid_hand": 0, "equity_seconds": [], "gadget_streets": 0, "ceiling_seconds": [],
                "follow_mean": []}

    # ------------------------------------------------------------------ hand bookkeeping

    def new_hand(self, state, ranges=None, hand_id=None):
        """Start of a hand (or of a mid-hand scenario at a street start): uniform ranges over the hands
        the visible board allows - or the given ranges [n, H] (a scenario: recorded as our range's
        starting point) - and the street-start values from this state. hand_id: for failure records."""
        self.n = int(state.stack.shape[0])
        self.hand_id = hand_id if hand_id is not None else self.stats["hands"]
        if not 0 <= self.hero < self.n:
            raise ValueError(f"hero seat {self.hero} outside the table of {self.n} players")
        self.coverage = inference_coverage(state)
        if self.coverage["status"] == "extrapolation":
            self.stats["outside_training_depth_hands"] += 1
            if self.preflop is not None or getattr(self.solver, "critics", None):
                warnings.warn(f"effective starting depth {self.coverage['actual_depths_bb']} BB is outside "
                              f"the declared training mix {self.coverage['training_depth_mix_bb']}; "
                              "continuing with unvalidated critic extrapolation", RuntimeWarning, stacklevel=2)
        self._check_hand(state, "new_hand", required=False)       # malformed cards refused at the start; absent = range-only
        valid = self._valid(state)
        uniform = np.tile(_restrict(np.full(H, 1.0 / H), valid)[None], (self.n, 1))
        if ranges is None:
            # The same operations hero_reach replays: bit-equal by construction.
            self.ranges = uniform
            self.records = [("board", valid.astype(np.float64))]
        else:
            r = np.asarray(ranges, np.float64)
            if r.shape != (self.n, H) or not np.isfinite(r).all() or np.any(r < 0):
                raise ValueError(f"ranges must be finite nonnegative [{self.n}, {H}] weights")
            r = r * valid[None]
            if np.any(r.sum(1) <= 0):
                raise ValueError("ranges must retain positive mass after board removal")
            self.ranges = r / r.sum(1, keepdims=True)
            self.records = [("start", self.ranges[self.hero].copy())]
        self.pending = None
        self.leaf_reach = None                   # [n, H] reach at the leaf the previous street ended in (gadget source)
        # The cache was solved for uniform starting ranges. Explicit scenario ranges
        # change the game and must be solved, even when stacks and blinds match.
        self.table_fits = bool(self.preflop is not None and self.preflop.fits(state)
                               and np.array_equal(self.ranges, uniform))
        self._start_street(state)
        self.stats["hands"] += 1

    def _start_street(self, state):
        self.street = int(np.asarray(state.street))
        self.street_start = jax.tree.map(np.asarray, state)
        live = ~np.asarray(state.folded, bool)
        pot = float(np.asarray(state.bet_total, np.float64).sum())
        effective = float(np.asarray(state.stack, np.float64)[live].min()) if np.any(live) else 0.0
        self.coverage.update(street=self.street, pot_bb=pot / config.BB,
                             effective_behind_bb=effective / config.BB,
                             effective_spr=(effective / pot if self.street > 0 and pot > 0 else None),
                             declared_training_max_spr=(max(high for _, high, probability in config.DEPTH_MIX
                                                            if probability > 0) - 1) / 2)
        self.street_ranges = self.ranges.copy()
        self.bucket = self.bucket_of(self.street_start)
        self.path = []                          # (kind, total) of every action on this street
        self.frozen = {}                        # position in path -> probs [H, K] over the columns we had (ours)
        self.unapplied = []                     # positions of opponent actions whose likelihood is not applied yet
        self.solution = None                    # the last solve of this street (Solution) or None
        self.ceilings = None                    # the gadget of this street's solves ({"ev", "mask"}) once computed
        self.equity_cache = None
        self.equity_future = None
        if self.prefetch is not None and self.street in (1, 2):   # the board is known now: the all-in matrix is built
            board = SG.visible_board(self.street_start)           # in the background while the opponent thinks
            self.equity_future = self.prefetch.submit(self._build_equity, self.street, board)

    @staticmethod
    def _valid(state):
        board = SG.visible_board(state)
        return (C.board_valid(board[board >= 0]) if (board >= 0).any() else np.ones(H, bool)).astype(np.float64)

    def _count(self, key, street):
        d = self.stats[key]
        d[street] = d.get(street, 0) + 1

    def fail(self, exc):
        """Record a failure of this hand (type, message, traceback, hand id, street, the counters at the
        time) in self.failures and re-raise: a failed hand is reported with its record, never played."""
        import traceback
        self.failures.append({"hand": getattr(self, "hand_id", None), "street": getattr(self, "street", None),
                              "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
                              "counters": {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.stats.items()
                                           if k in ("decisions", "solves", "off_tree", "translated", "slot_overflow", "solve_errors",
                                                    "budget_exceeded", "illegal_actions", "table_unreachable", "invalid_hand", "floor_hits")}})
        raise exc

    def _hand(self, state):
        return hand_index(np.asarray(state.hole)[self.hero])

    def _check_hand(self, state, where, required):
        """Our hole cards at `state`: the hand index when they are a hand (two distinct cards 0..51, none on the
        visible board); None when both are absent (-1: a range-only scenario) and `required` is False; otherwise
        InvalidHand (counted, the hand failed) - the sentinel index 1326 never reaches a strategy row."""
        hole = np.asarray(state.hole)[self.hero].reshape(-1)
        board = SG.visible_board(state)
        board = board[board >= 0]
        absent = bool((hole < 0).all())
        a_hand = (hole.shape == (2,) and bool(((hole >= 0) & (hole < 52)).all()) and int(hole[0]) != int(hole[1])
                  and not bool(np.isin(hole, board).any()))
        if a_hand:
            return hand_index(hole)
        if absent and not required:
            return None
        self.stats["invalid_hand"] += 1
        self.fail(InvalidHand(f"{where}: our hole cards {hole.tolist()} are not a hand on board {board.tolist()} "
                              f"({'absent' if absent else 'out of range, duplicate or on the board'}): refused, nothing played"))

    def _build_equity(self, street, board):
        """The all-in matrix of a flop (the class cache) or a turn (the device's builder); in the prefetch thread or
        synchronously at the first decision of the street."""
        return self.equity(board[:3]) if street == 1 else self.turn_equity(board[:4])

    def _street_equity(self):
        if self.equity_cache is None:
            t0 = time.perf_counter()
            board = SG.visible_board(self.street_start)
            if self.street == 0:
                if self.preflop_equity is None:
                    raise ValueError("a preflop solve needs preflop_equity (runout.preflop_equity_cached)")
                self.equity_cache = self.preflop_equity
            elif self.street in (1, 2):
                future, self.equity_future = self.equity_future, None
                try:
                    self.equity_cache = future.result() if future is not None else self._build_equity(self.street, board)
                except SG.MissingArtifact as e:           # never computed mid-hand: counted, the hand failed
                    self.stats["solve_errors"] += 1
                    self.fail(e)
            else:
                self.equity_cache = False
            self.stats["equity_seconds"].append(time.perf_counter() - t0)   # the wait for the matrix at the decision
        return None if self.equity_cache is False else self.equity_cache

    # ------------------------------------------------------------------ the solve of the street

    def _tree(self):
        """The street's topology replayed at the street start with the played history."""
        sizes = None if self.sizes is None else {s: self.sizes(self.street, self.bucket) for s in range(self.n)}
        try:
            return SG.build_tree(self.street_start, self.hero, self.path, self.bucket, sizes, self.cap.get(self.street))
        except SG.OffTree:
            self.stats["slot_overflow"] += 1
            raise

    def _solve_street(self, why):
        """One solve of the street from its start with the history on the tree: our played nodes frozen."""
        try:
            tree = self._tree()
            fixed = {}
            for pos, (probs, _) in self.frozen.items():
                v = int(tree.path[pos])
                cols = tree.columns(v)
                if len(cols) != probs.shape[1]:
                    raise RuntimeError("the frozen strategy's columns do not match the replayed node")
                row = np.zeros((H, tree.kinds.shape[1]), np.float32)
                for k, (a, _, _) in enumerate(cols):
                    row[:, a] = probs[:, k]
                fixed[v] = row
            equity = self._street_equity()
            ranges = self.street_ranges
            gadget = None
            if self.gadget and self.street > 0 and self.leaf_reach is not None:
                gadget = self._gadget(equity)
                ranges = self.street_ranges.copy()
                for seat in range(self.n):
                    if gadget["mask"][seat]:
                        ranges[seat] = self._deal(seat)
            sol = self.solver.solve(tree, ranges, fixed, equity, gadget)
        except Exception as e:
            self.stats["solve_errors"] += 1
            self.fail(e)
        st = self.stats
        self._count("solves", self.street)
        if gadget is not None and sol.follow is not None:
            st["follow_mean"].append([float(np.mean(sol.follow[seat][ranges[seat] > 0])) for seat in range(self.n)
                                      if gadget["mask"][seat]])
        st["solve_seconds"].append(sol.seconds)
        st["solve_cpu_seconds"].append(sol.cpu_seconds)
        st["build_seconds"].append(tree.seconds)
        st["compile_seconds"] += sol.compile_seconds
        st["structures"].add(tree.key)
        if sol.e is not None:
            st["certificates"].append(sol.e)
        if self.log is not None:
            self.log(f"solve ({why}) street {self.street} bucket {self.bucket} history {len(self.path)} nodes {tree.size} "
                     f"T {sol.iterations}: {sol.seconds:.2f} s (compile {sol.compile_seconds:.1f} s, build {tree.seconds:.2f} s)"
                     + (f" E {sol.e:.4f}" if sol.e is not None else ""))
        self.solution = sol
        return sol

    # ------------------------------------------------------------------ the re-solving gadget (module doc, item 10)

    def _gadget(self, equity):
        """The ceilings of this street's solves: the post-card critic of this street at the leaf the previous street
        ended in (self.leaf_reach: both seats' reach there, the hands holding the new card removed), for the board
        as it is now; whole-hand EV per hand of every other seat in units of this street's start pot (pcfr's
        gadget_ev; the solver multiplies by our range's compatible mass). Computed once per street."""
        if self.ceilings is not None:
            return self.ceilings
        t0 = time.perf_counter()
        critic = self.post_critics[self.street]
        s = self.street_start
        n = self.n
        board = SG.visible_board(s)
        valid = self._valid(s)
        reach = np.asarray(self.leaf_reach, np.float64) * valid[None, :]
        total = reach.sum(1, keepdims=True)
        if not np.all(total > 0):
            raise ValueError("gadget: a seat's reach at the previous leaf has no mass on the hands this board allows")
        ranges = (reach / total).astype(np.float32)                                # [n, H], seat order
        slot = (np.arange(n) - 1) % n                                               # seat -> critic slot (pcfr.Static.slot)
        live = ~np.asarray(s.folded, bool)
        contrib = np.asarray(s.bet_total, np.float64)
        pot = float(contrib.sum())
        behind = np.asarray(s.stack, np.float64)
        slots = np.zeros((CR.P_MAX, H), np.float32)
        present, behind_pot, contrib_pot = (np.zeros(CR.P_MAX, np.float32) for _ in range(3))
        for seat in range(n):
            slots[slot[seat]] = ranges[seat] * live[seat]
            present[slot[seat]] = float(live[seat])
            behind_pot[slot[seat]] = behind[seat] / pot
            contrib_pot[slot[seat]] = contrib[seat] / pot
        bps, cap = int(np.asarray(s.rake_bps)), int(np.asarray(s.rake_cap))
        pub = CR.public_row(jnp.float32(pot / config.BB), jnp.float32(self.solver_rake_level()), jnp.float32(bps / 500.0),
                            jnp.float32(cap / pot), jnp.asarray(present), jnp.asarray(behind_pot), jnp.asarray(contrib_pot))
        pair = jnp.asarray(slots[:n])                                               # the live slots
        if self.street == 3:
            tables = jax.tree.map(lambda x: x[0], SI._tables(jnp.asarray(board[None], jnp.int32)))
            share = CD.two(lambda w: CD.share_river(w, tables), pair)
        else:
            w = jnp.asarray(np.asarray(equity, np.float32))
            share = CD.two(lambda x: CD.share_matrix(x, w), pair)
        net = CD.net_fraction(jnp.int32(round(pot)), jnp.int32(bps), jnp.int32(cap))
        f = share * net * jnp.asarray(valid, jnp.float32)[None, :]
        board1h = jnp.asarray(np.sum(np.eye(52, dtype=np.float32)[board[board >= 0]], axis=0))
        ev_rel = np.asarray(critic(board1h[None], pub[None], pair[None], f[None], None))[0]       # [n slots, H]
        if not np.isfinite(ev_rel).all():
            raise SG.NonFinite("gadget: the post-card critic returned non-finite values")
        ev = np.zeros((n, H), np.float64)
        mask = np.zeros(n, bool)
        for seat in range(n):
            if seat != self.hero and live[seat]:
                ev[seat] = (ev_rel[slot[seat]] - contrib[seat] / pot) * valid       # relative -> whole-hand accounting
                mask[seat] = True
        self.ceilings = {"ev": ev, "mask": mask}
        self.stats["ceiling_seconds"].append(time.perf_counter() - t0)
        self.stats["gadget_streets"] += 1
        return self.ceilings

    def solver_rake_level(self):
        return float(getattr(self.solver, "rake_level", 0.0) or 0.0)

    def _deal(self, seat):
        """A gadget seat's deal range in this street's solves (gadget_deal, module doc item 10)."""
        valid = self._valid(self.street_start)
        uniform = valid / valid.sum()
        tracked = np.asarray(self.street_ranges[seat], np.float64)
        if self.gadget_deal == "uniform":
            return uniform
        if self.gadget_deal == "range":
            return tracked
        return self.gadget_mix * tracked + (1.0 - self.gadget_mix) * uniform

    def warm_gadget(self, flop, turn, n=2):
        """Compile the gadget's ceiling computation of every postflop street now (a dummy board of an `n`-seat table,
        uniform reach; the all-in matrices `flop` / `turn` [H, H] of the warm-up), so that no hand compiles it
        (checks/s8_1 contract: compile at the warm-up only). Leaves the player as new."""
        if not self.gadget:
            return
        self.n = int(n)
        board = np.asarray([0, 14, 28, 3, 17], np.int32)
        for street, equity in ((1, flop), (2, turn), (3, None)):
            root = jax.tree.map(np.asarray, SG.TR.street_root(self.n, street, 6 * config.BB, 100 * config.BB))
            shown = {1: 3, 2: 4, 3: 5}[street]
            self.street_start = root.replace(board=np.concatenate([board[:shown], np.full(5 - shown, -1, np.int32)]))
            self.street = street
            self.leaf_reach = np.tile(self._valid(self.street_start)[None], (self.n, 1))
            self.ceilings = None
            self._gadget(equity)
        self.leaf_reach, self.ceilings, self.street_start, self.n = None, None, None, None
        self.stats["gadget_streets"], self.stats["ceiling_seconds"] = 0, []

    def _leaf_reach_from_solution(self, sol):
        """[n, H] both seats' reach at the node the street's history ends in, under the solve's average strategies:
        ours exact (the tracked range), every other seat's = his deal range in that solve (uniform x his average
        Follow when the gadget was on, his street-start range otherwise) times his average action probabilities
        along the path."""
        tree = sol.tree
        valid = self._valid(self.street_start)
        reach = np.zeros((self.n, H), np.float64)
        reach[self.hero] = self.ranges[self.hero]
        for seat in range(self.n):
            if seat == self.hero:
                continue
            r = np.asarray(self.street_ranges[seat], np.float64)
            if self.ceilings is not None and self.ceilings["mask"][seat]:
                r = self._deal(seat)
                if sol.follow is not None:
                    r = r * np.asarray(sol.follow[seat], np.float64)
            for v, a in zip(tree.path[:-1], tree.path_cols):
                if int(tree.topo.actor[v]) == seat:
                    sigma = sol.strategy(int(v)).astype(np.float64)
                    cols = [col for col, _, _ in tree.columns(int(v))]
                    probs = sigma[:, cols] / np.maximum(sigma[:, cols].sum(1, keepdims=True), 1e-300)
                    r = r * probs[:, cols.index(int(a))]
            reach[seat] = r
        return reach

    def _leaf_reach_from_table(self):
        """As _leaf_reach_from_solution for a preflop played on the table (uniform deal, the table's strategies)."""
        reach = np.zeros((self.n, H), np.float64)
        reach[self.hero] = self.ranges[self.hero]
        valid = self._valid(self.street_start)
        for seat in range(self.n):
            if seat == self.hero:
                continue
            r = valid / valid.sum()
            for pos in range(len(self.path)):
                u = self.preflop.node(self.path[:pos])
                if int(self.preflop.actor[u]) == seat:
                    c = self.preflop.candidates(u)
                    r = r * c.probs[:, c.column(*self.path[pos])]
            reach[seat] = r
        return reach

    def _apply_unapplied(self, sol):
        """The likelihoods of the opponent actions not yet applied, from a solve whose tree holds them."""
        tree = sol.tree
        for pos in list(self.unapplied):
            v, a = int(tree.path[pos]), int(tree.path_cols[pos])
            c = candidates_at(sol, v)
            like = c.probs[:, [col for col, _, _ in tree.columns(v)].index(a)]
            self._update_opponent(int(tree.topo.actor[v]), like, bool(tree.in_slot[pos]))
            self.unapplied.remove(pos)

    def _apply_from_table(self):
        for pos in list(self.unapplied):
            u = self.preflop.node(self.path[:pos])
            c = self.preflop.candidates(u)
            like = c.probs[:, c.column(*self.path[pos])]
            self._update_opponent(int(self.preflop.actor[u]), like, False)
            self.unapplied.remove(pos)

    def _update_opponent(self, seat, like, off_tree):
        before = self.ranges[seat]
        floored = np.maximum(like, self.eps)
        after = _bayes(before, floored)
        ok = before > 0
        ratio = after[ok] / before[ok]
        st = self.stats
        st["min_ratio"] = min(st["min_ratio"], float(ratio.min()))
        st["floor_hits"] += int(np.any(like[ok] < self.eps))
        st["opponent_updates"] += 1
        if off_tree:
            self._count("off_tree", self.street)
        self.last_update = {"seat": seat, "min_ratio": float(ratio.min()), "like_min": float(like[ok].min()),
                            "like_max": float(like[ok].max()), "off_tree": off_tree,
                            "min_weight_over_uniform": float(after[ok].min() * ok.sum())}
        self.ranges[seat] = after

    def _table_node(self):
        """The preflop table's node of the current history when the table serves this hand, else None."""
        if self.street == 0 and self.table_fits and self.solution is None:
            return self.preflop.node(self.path)
        return None

    def _own_reach(self, state):
        """Our own reach of the current node with our hand: our range weight at the street start times
        the product of the played probabilities of our actions on this street (what pcfr's rows were
        accumulated with), or None when the state has no hole cards (a scenario)."""
        hole = np.asarray(state.hole)[self.hero]
        if (hole < 0).any():
            return None
        h = hand_index(hole)
        reach = float(self.street_ranges[self.hero][h])
        for probs, k in self.frozen.values():
            reach *= float(probs[h, k])
        return reach

    def _guard_reach(self, state, where):
        reach = self._own_reach(state)
        if reach is not None and reach < self.reach_min:
            self.stats["table_unreachable"] += 1
            self.fail(TableUnreachable(f"our reach of this node with our hand is {reach:.1e} < {self.reach_min:g} ({where}): "
                                       "the strategy row there is pcfr's never-reached fallback, not a decision of ours"))

    def _candidates(self, state=None):
        """Our strategy at the current history (hero to act): the table when it fits and holds the
        history, else the solve of the street with the history; the opponent's pending likelihoods are
        applied from the same source. Returns (Candidates, "table" or the Solution). With `state`, our
        own reach of the node with our hand is checked first (TableUnreachable below TABLE_REACH_MIN)."""
        v = self._table_node()
        if v is not None:
            if state is not None:
                self._guard_reach(state, "preflop table")
            self._apply_from_table()
            self.stats["table_decisions"] += 1
            return self.preflop.candidates(v), "table"
        if state is not None:
            self._guard_reach(state, "street solve")
        sol = self.solution
        if sol is None or len(sol.tree.history) != len(self.path):
            sol = self._solve_street("decision")
        self._apply_unapplied(sol)
        v = sol.tree.node
        if int(sol.tree.topo.actor[v]) != self.hero:
            raise RuntimeError("the history does not reach a decision of ours")
        return candidates_at(sol, v), sol

    # ------------------------------------------------------------------ play

    def act(self, state, rng):
        """Our decision at `state` (hero to act): (kind, total) sampled with `rng` from the strategy of
        our hand; the probabilities of every hand are kept for observe()."""
        t0 = time.perf_counter()
        if int(np.asarray(state.to_act)) != self.hero:
            raise ValueError("act: it is not the hero's turn")
        h = self._check_hand(state, "act", required=True)         # before the solve: no hand, no decision (InvalidHand)
        c = self._candidates(state)[0]
        p = c.probs[h]
        if not p.sum() > 0:
            raise RuntimeError("the strategy gives our hand no action")
        k = int(rng.choice(len(p), p=p / p.sum()))
        self.pending = (int(c.kinds[k]), int(c.targets[k]), c, k)
        self.stats["decisions"] += 1
        self._budget(t0)
        return int(c.kinds[k]), int(c.targets[k])

    def choose(self, state, kind, target):
        """Our action fixed from outside (a replayed history, a check): the bookkeeping act() would
        do for (kind, total), which must be one of the strategy's actions here."""
        t0 = time.perf_counter()
        if int(np.asarray(state.to_act)) != self.hero:
            raise ValueError("choose: it is not the hero's turn")
        c = self._candidates(state)[0]
        k = c.column(int(kind), int(target))
        if k is None:
            raise ValueError(f"choose: ({SG.KIND_NAMES[int(kind)]}, {target}) is not an action of our strategy here")
        self.pending = (int(c.kinds[k]), int(c.targets[k]), c, k)
        self.stats["decisions"] += 1
        self._budget(t0)
        return int(c.kinds[k]), int(c.targets[k])

    def candidates(self, state):
        """Our normalised strategy at `state`, including every nonzero solved probability."""
        if int(np.asarray(state.to_act)) != self.hero:
            raise ValueError("candidates: it is not the hero's turn")
        return self._candidates(state)[0]

    def _budget(self, t0):
        dt = time.perf_counter() - t0
        self.stats["decision_seconds"].append(dt)
        if self.budget_seconds is not None and dt > self.budget_seconds:
            self.stats["budget_exceeded"] += 1
            self.fail(BudgetExceeded(f"decision took {dt:.2f} s > {self.budget_seconds} s"))

    def observe(self, state_before, kind, target, state_after):
        """Bookkeeping for one action of any seat at state_before (ours right after act())."""
        seat = int(np.asarray(state_before.to_act))
        street = int(np.asarray(state_before.street))
        kind, target = int(kind), int(target)
        if street != self.street:
            raise ValueError("observe: the state is not on the current street")
        if not legal(state_before, kind, target):
            self.stats["illegal_actions"] += 1
            self.fail(IllegalAction(f"seat {seat}: {SG.KIND_NAMES.get(kind, kind)} to {target} is not a legal action here"))
        self._check_hand(state_after, "observe", required=False)  # a new board card equal to one of ours: refused here
        if seat == self.hero:
            if self.pending is None or self.pending[:2] != (kind, target):
                raise ValueError("observe: the hero's action is not the one act() chose")
            _, _, c, k = self.pending
            self.pending = None
            like = c.probs[:, k].astype(np.float64)
            self.frozen[len(self.path)] = (c.probs, k)
            self.ranges[self.hero] = _bayes(self.ranges[self.hero], like)
            self.records.append(("like", like))
        else:
            self.unapplied.append(len(self.path))
        self.path.append((kind, target))
        added = int(np.asarray(state_after.bet_total)[seat]) - int(np.asarray(state_before.bet_total)[seat])
        expected = 0 if kind == FOLD else target - int(np.asarray(state_before.bet_street)[seat])
        if added != expected:
            self.stats["translated"] += 1
        moved = int(np.asarray(state_after.street)) != street
        if moved or bool(np.asarray(state_after.terminal)):
            self._close_street(continuing=moved and not bool(np.asarray(state_after.terminal)))
        if moved and not bool(np.asarray(state_after.terminal)):
            valid = self._valid(state_after)
            for s in range(self.n):
                self.ranges[s] = _restrict(self.ranges[s], valid)
            self.records.append(("board", valid))
            self._start_street(state_after)

    def _close_street(self, continuing=False):
        """The street (or the hand) ended: the opponent actions not applied yet get their likelihoods
        from the table, from the last solve of the street when its tree holds them, else from one more
        solve (a closing solve: his off-tree raise ended the street with nobody acting after it). When the
        hand continues on the next street (`continuing`) and the gadget is on, the reach of every seat at
        the leaf is recorded from the same source (self.leaf_reach: the next street's ceilings come from it)."""
        need_reach = continuing and self.gadget
        if not self.unapplied and not need_reach:
            return
        if self._table_node() is not None:
            self._apply_from_table()
            if need_reach:
                self.leaf_reach = self._leaf_reach_from_table()
            return
        sol = self.solution
        if sol is None or sol.tree.find(self.path) is None:
            if not continuing:                 # the hand is over: his last likelihoods are a diagnostic, not worth a solve
                self.unapplied.clear()
                return
            sol = self._solve_street("closing")
        else:
            tree = SG.replay(sol.tree.shape, self.street_start, self.path)
            sol = SG.Solution(tree, sol.sigma, sol.iterations, sol.e, sol.seconds, sol.cpu_seconds, 0.0, sol.batch_size,
                              None, sol.follow)
        self._apply_unapplied(sol)
        if need_reach:
            self.leaf_reach = self._leaf_reach_from_solution(sol)

    @staticmethod
    def step(state, kind, target):
        return engine_step(state, kind, target)

    def hero_reach(self):
        """The reach of the strategies we played this hand, recomputed from the records (hero_reach)."""
        return hero_reach(self.records)
