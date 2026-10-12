"""Live subgames: ONE fixed topology per (street, bucket, players, hero seat) with an actual-size slot at
every opponent decision node, replayed at the actual street-start state with the played history on the
tree, solved by the teacher's PCFR+ (poker/solver/pcfr.py) for a fixed number of iterations, batched over
concurrent decisions of the same topology.

Topology (Shape): built once from a DEEP root of the street (every player 10^8 chips, poker/solver/tree.py's
model) so every size is a legal raise. Columns of a decision node: fold (facing a bet), check/call, the
actor's size fractions at the node's raise level (config.street_sizes of the bucket, every fraction a
column, duplicates resolved only at replay), one SLOT when the actor is not the hero (the opponent's
actual size, filled at replay from the played history, dead otherwise; it exists at every opponent node
whatever the raise level, so a raise beyond the cap is on the tree too), and the all-in. Fraction
raises stop after `cap` aggressive actions (the all-in stays), slots do not; a path visits each raise
level once, so one slot per node is never short: "slot overflow" cannot happen on a played history
(the counter in poker/player/live.py measures it anyway).

Replay (LiveTree): the real street-start State stepped through the topology (poker.envs.transition.step_target
at exact chip amounts): per column the real total and its legality (a size at or above the all-in, a
clamped duplicate of a smaller size, a raise the rules do not allow - dead children with zero reach,
masked by the solver like the dead branches of the teacher's trees), the played history matched
column by column (an opponent raise that is no size of the node fills the node's slot), the node of
every prefix of the history, boundary kinds from the real states; a street end where a live player is
all-in is valued exactly (pcfr.leaf_values) so unequal stacks replay onto the same topology.

Solve: spot_batch builds the solver's inputs in units of the street-start pot (ranges at the street
start, our played nodes frozen to the strategies we played: fixed_mask / fixed_pi, pcfr), solve_batch
stacks several spots of one topology into one vmapped call (many tables, one compile), solve() is the
single-spot case. One compiled solver per (topology, T, chunk, certificate); warm_up() compiles every
topology of a table size before the first hand; a persistent compilation cache (jax_compilation_cache_dir)
is measured in checks/s8_2.py.
"""

import contextlib
import hashlib
import os
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config, derived
from poker.envs.actions import action_candidates, action_table
from poker.envs.evaluator import rake_paid
from poker.envs.state import State
from poker.envs.transition import step_target
from poker.solver import cards as C
from poker.solver import flops as FL
from poker.solver import pcfr as SV
from poker.solver import runout
from poker.solver import situations as SI
from poker.solver import tree as TR
from poker.solver.cards import NUM_HANDS as H

FOLD, CALL, RAISE, ALLIN, SLOT = 0, 1, 2, 3, 4   # column kinds (SLOT = an opponent's actual size at replay)
KIND_NAMES = {FOLD: "fold", CALL: "check/call", RAISE: "raise", ALLIN: "all-in", SLOT: "slot"}
ENGINE_INDEX = {FOLD: config.FOLD, CALL: config.CHECK_CALL, RAISE: 2, SLOT: 2, ALLIN: config.ALLIN}
DEEP = 10 ** 8                                   # chips of every player in the deep root of a topology
FIELD = config.regime_field()                    # the regime's table sizes, seats, streets, menus, batch ladder (config only)
DEVICE = "gpu" if jax.default_backend() == "gpu" else "cpu"
LADDER = FIELD["ladder"][DEVICE]                 # batch sizes a solve is padded to: the closed set of compiled shapes
DEEP_POT = 1200                                  # the deep root's street-start pot (poker/solver/tree.py)
_CHUNK = 256                                     # engine calls of the builder: one compiled shape per n


def _strip(state):
    """The State without history arrays (zero length: the engine counts decisions, records nothing)."""
    z = jnp.zeros(0, jnp.int32)
    return state.replace(history_player=z, history_action=z, history_amount=z, history_street=z)


def _node_info(s, offered=None):
    # One canonical calculation for teacher and live fraction raises. Retaining masked targets lets
    # replay step dead branches safely; the exact opponent SLOT is handled separately below.
    mask, targets = action_table(s) if offered is None else action_candidates(s, offered)
    me = jnp.clip(s.to_act, 0, s.stack.shape[0] - 1)
    out = {"level": derived.bet_level(s), "to_call": derived.to_call(s), "pot": derived.pot(s),
           "minimum": derived.min_raise_to(s), "maximum": derived.max_raise_to(s),
           "fold_ok": mask[config.FOLD], "call_ok": mask[config.CHECK_CALL], "raise_ok": mask[config.ALLIN],
           "current": s.bet_street[me], "call_target": targets[config.CHECK_CALL]}
    if offered is not None:
        out.update(size_legal=mask, size_targets=targets)
    return out


_INFO = jax.jit(jax.vmap(_node_info))
_STEP = jax.jit(jax.vmap(step_target))
_RAKE = jax.jit(jax.vmap(rake_paid))
try:                                   # the replay of a tree = many small engine batches: on the host device, not the GPU
    _HOST_DEVICE = jax.devices("cpu")[0]        # (0.25-0.33 s per tree on the 1070 against 0.04 s on the host; task #41)
except RuntimeError:                   # JAX_PLATFORMS without a cpu entry: the default device then
    _HOST_DEVICE = None


def _chunked(fn, states, *args):
    """fn over a batched host State (numpy leaves) in padded chunks of _CHUNK rows -> numpy leaves."""
    count = states.terminal.shape[0]
    outs = []
    with (jax.default_device(_HOST_DEVICE) if _HOST_DEVICE is not None else contextlib.nullcontext()):
        for start in range(0, count, _CHUNK):
            idx = np.arange(start, min(count, start + _CHUNK))
            pad = np.concatenate([idx, np.full(_CHUNK - len(idx), idx[0])])
            out = fn(jax.tree.map(lambda x: x[pad], states), *(a[pad] for a in args))
            outs.append(jax.tree.map(lambda x: np.asarray(x)[:len(idx)], out))
    return jax.tree.map(lambda *x: np.concatenate(x), *outs)


def raises_on_street(state) -> int:
    """Aggressive actions recorded on the current street of a host State (history index >= 2:
    the grid's raises, the live path's raise index 2 and ALLIN). 0 when the history is off."""
    n = int(np.asarray(state.history_len))
    if state.history_action.shape[0] == 0 or n == 0:
        return 0
    act = np.asarray(state.history_action)[:n]
    street = np.asarray(state.history_street)[:n]
    return int(np.sum((street == int(np.asarray(state.street))) & (act >= 2)))


def level_sizes(levels, level):
    """The fractions of raise level `level` (the last entry repeats)."""
    return tuple(tuple(f) for f in levels[min(int(level), len(levels) - 1)])


def street_cap(street, bucket, n):
    """Aggressive actions per street of the deployed trees for `n` players in the menu `bucket`:
    config.tree_rule (the caps by menu live under the decided grid, STREET_SIZES_GRID)."""
    return int(config.tree_rule(n, street, bucket)["cap"])


@dataclass(frozen=True, eq=False)
class Shape:
    """The topology of one class (street, bucket, players, hero seat, size sets, cap) and its columns."""
    topo: TR.Topology
    st: SV.Static
    key: str
    kinds: np.ndarray             # [V, A] FOLD / CALL / RAISE / SLOT / ALLIN, -1 no child
    frac: np.ndarray              # [V, A] index of a RAISE column's fraction in the actor's level set (-1 otherwise)
    level: np.ndarray             # [V] aggressive actions on the street before the node
    slot: np.ndarray              # [V] the SLOT column of an opponent node, -1 otherwise
    street: int
    n: int
    hero: int
    bucket: str
    sizes: dict                   # seat -> fractions per raise level
    offered: np.ndarray          # [V, config.NUM_ACTIONS] offered fractions in the canonical grid
    cap: int
    seconds: float

    @property
    def size(self):
        return self.topo.size

    @property
    def internal(self):
        return self.topo.internal


_SHAPES = {}
_FRACTION_INDEX = {tuple(f): 2 + i for i, f in enumerate(config.RAISE_FRACS)}


def _offered(sizes, actors, levels, cap):
    """Map topology columns to the canonical grid; no sizing or legality rule is duplicated here."""
    out = np.zeros((len(actors), config.NUM_ACTIONS), bool)
    for v, (seat, level) in enumerate(zip(actors, levels)):
        if seat >= 0 and level < cap:
            for fraction in level_sizes(sizes[int(seat)], level):
                out[v, _FRACTION_INDEX[fraction]] = True
    return out


def shape_key(street, bucket, n, hero, sizes, cap):
    sz = repr(sorted((s, tuple(map(tuple, lv))) for s, lv in sizes.items()))
    return f"s{street}|{bucket}|n{n}|h{hero}|cap{cap}|{hashlib.sha1(sz.encode()).hexdigest()[:12]}"


def shape_for(street, bucket, n, hero, sizes=None, cap=None) -> Shape:
    """The topology of (street, menu `bucket`, n players, hero seat): sizes seat -> levels (default
    config.street_sizes of the menu for every seat), cap (default street_cap). Built once, cached."""
    sizes = sizes or {seat: config.street_sizes(street, bucket) for seat in range(n)}
    sizes = {int(s): tuple(tuple(tuple(f) for f in lv) for lv in levels) for s, levels in sizes.items()}
    for levels in sizes.values():
        config._size_indices(levels)       # own fractions belong to the decided grid; arbitrary opponent amounts use SLOT
    cap = street_cap(street, bucket, n) if cap is None else int(cap)
    key = shape_key(street, bucket, n, hero, sizes, cap)
    if key in _SHAPES:
        return _SHAPES[key]
    t0 = time.perf_counter()
    deep = TR.deep_stack(n) if hasattr(TR, "deep_stack") else DEEP        # the int32-proven builder stack of n seats
    root = TR.preflop_root(n, deep) if street == 0 else TR.street_root(n, street, DEEP_POT, deep)
    base = jax.tree.map(np.asarray, _strip(root))
    frontier = jax.tree.map(lambda x: x[None], base)
    parent, label, actor, live, terminal, leaf, depth = [-1], [-1], [], [], [], [], [0]
    levels, kinds, fracs = [0], [], []
    first = 0
    while True:
        count = int(frontier.terminal.shape[0])
        term = np.asarray(frontier.terminal)
        at_leaf = (np.asarray(frontier.street) > street) & ~term
        done = term | at_leaf
        to_act = np.asarray(frontier.to_act)
        folded = np.asarray(frontier.folded)
        offered = _offered(sizes, np.where(done, -1, to_act), levels[first:first + count], cap)
        info = _chunked(_INFO, frontier, offered)
        src, acts, tgts = [], [], []
        for j in range(count):
            i = first + j
            terminal.append(bool(done[j]))
            leaf.append(bool(at_leaf[j]))
            live.append(~folded[j])
            actor.append(-1 if done[j] else int(to_act[j]))
            row = []                                        # (kind, fraction index, deep total)
            if not done[j]:
                seat = int(to_act[j])
                lvl = levels[i]
                row.append((FOLD if bool(info["fold_ok"][j]) else -1, -1, int(info["current"][j])))
                row.append((CALL, -1, int(info["call_target"][j])))
                if bool(info["raise_ok"][j]):
                    minimum, maximum = int(info["minimum"][j]), int(info["maximum"][j])
                    totals = []
                    if lvl < cap:
                        for f, fraction in enumerate(level_sizes(sizes[seat], lvl)):
                            t = int(info["size_targets"][j, _FRACTION_INDEX[fraction]])
                            row.append((RAISE, f, t))
                            totals.append(t)
                    if seat != hero:
                        row.append((SLOT, -1, max(totals + [minimum]) + 1))
                    row.append((ALLIN, -1, maximum))
                for a, (k, f, t) in enumerate(row):
                    if k < 0:
                        continue
                    parent.append(i)
                    label.append(a)
                    levels.append(lvl + int(k in (RAISE, SLOT, ALLIN)))
                    depth.append(depth[i] + 1)
                    src.append(j)
                    acts.append(ENGINE_INDEX[k])
                    tgts.append(t)
            kinds.append([k for k, _, _ in row])
            fracs.append([f for _, f, _ in row])
        first += count
        if not src:
            break
        src = np.asarray(src)
        frontier = _chunked(_STEP, jax.tree.map(lambda x: x[src], frontier), np.asarray(acts, np.int32),
                            np.asarray(tgts, np.int32))
    V = len(parent)
    A = max(1, max(len(r) for r in kinds))
    kind_arr, frac_arr = (np.full((V, A), -1, np.int32) for _ in range(2))
    children = np.full((V, A), -1, np.int32)
    for v in range(V):
        for a in range(len(kinds[v])):
            kind_arr[v, a], frac_arr[v, a] = kinds[v][a], fracs[v][a]
    for v in range(1, V):
        children[parent[v], label[v]] = v
    depth = np.asarray(depth)
    level_groups = tuple(np.flatnonzero(depth == d) for d in range(depth.max() + 1))
    parent, label, actor = (np.asarray(x, np.int32) for x in (parent, label, actor))
    terminal, live, leaf = (np.asarray(x, bool) for x in (terminal, live, leaf))
    slot = np.full(V, -1, np.int32)
    for v in range(V):
        hits = np.flatnonzero(kind_arr[v] == SLOT)
        if len(hits):
            slot[v] = hits[0]
    topo = TR.Topology(n, parent, label, actor, children, children >= 0, live, terminal, level_groups, street, leaf)
    _SHAPES[key] = Shape(topo, SV.Static(topo), key, kind_arr, frac_arr, np.asarray(levels, np.int32), slot, int(street),
                         int(n), int(hero), str(bucket), sizes, _offered(sizes, actor, levels, cap), int(cap),
                         time.perf_counter() - t0)
    return _SHAPES[key]


def shapes():
    """The topologies built so far: {key: (street, nodes, decisions, build seconds)}."""
    return {k: (s.street, int(s.size), int(len(s.internal)), s.seconds) for k, s in _SHAPES.items()}


@dataclass(frozen=True, eq=False)
class LiveTree:
    """A topology replayed at a real street-start state with a played history on it."""
    shape: Shape
    kinds: np.ndarray             # [V, A] FOLD / CALL / RAISE / ALLIN of every column (a filled slot is a RAISE)
    targets: np.ndarray           # [V, A] street totals at the real root (chips; dead columns: what was stepped)
    legal: np.ndarray             # [V, A] the column is a legal action of the real state
    alive: np.ndarray             # [V] every action on the path is legal
    states: State                 # the real State of every node (host, history off; garbage below dead columns)
    info: dict                    # per node: level (bet level), to_call, pot, maximum (chips; meaningful at decisions)
    kind: np.ndarray              # [V] boundary kind of the real state (poker/solver/tree.py FOLD, SHOWDOWN, ALLIN_RUNOUT, LEAF)
    history: tuple                # the (kind, total) actions replayed on the tree
    path: np.ndarray              # [len(history) + 1] node of every prefix of the history
    path_cols: np.ndarray         # [len(history)] the column each action took
    in_slot: np.ndarray           # [len(history)] the action filled the node's slot (an off-tree size)
    seconds: float

    @property
    def topo(self):
        return self.shape.topo

    @property
    def st(self):
        return self.shape.st

    @property
    def key(self):
        return self.shape.key

    @property
    def street(self):
        return self.shape.street

    @property
    def n(self):
        return self.shape.n

    @property
    def hero(self):
        return self.shape.hero

    @property
    def size(self):
        return self.topo.size

    @property
    def internal(self):
        return self.topo.internal

    @property
    def node(self):
        """The node the whole history reaches."""
        return int(self.path[-1])

    def columns(self, v):
        """[(column, kind, target)] of node v's legal actions."""
        return [(a, int(self.kinds[v, a]), int(self.targets[v, a])) for a in range(self.kinds.shape[1]) if self.legal[v, a]]

    def child(self, v, kind, target):
        """(column, child index) of the action (kind, target) at node v, or None when the tree has no
        such action."""
        for a, k, t in self.columns(v):
            if k == kind and (kind in (FOLD, CALL) or t == int(target)):
                return a, int(self.topo.children[v, a])
        return None

    def node_path(self, v):
        """Actions (kind, target) from the root to node v."""
        out = []
        while self.topo.parent[v] >= 0:
            p = int(self.topo.parent[v])
            a = int(self.topo.label[v])
            out.append((int(self.kinds[p, a]), int(self.targets[p, a])))
            v = p
        return out[::-1]

    def find(self, actions):
        """Node reached by a list of (kind, target) actions from the root, or None."""
        v = 0
        for kind, target in actions:
            hit = self.child(v, kind, target)
            if hit is None:
                return None
            v = hit[1]
        return v

    def node_state(self, v):
        return jax.tree.map(lambda x: x[v], self.states)


class OffTree(ValueError):
    """A played action the topology cannot hold (a raise where the actor has no slot and no size)."""


class NonFinite(RuntimeError):
    """A live solve produced a non-finite policy, range or root value (pcfr's regret_policy and
    average_policy would silently turn NaN into a uniform policy: never played, raised and counted)."""


def replay(shape: Shape, root: State, history=()) -> LiveTree:
    """`shape` replayed at the real street-start State `root` (module doc) with the played `history`
    [(kind, total)] of this street on it: an opponent raise that is no size column of its node fills
    the node's slot. Raises OffTree when an action has no column (our own off-tree size: we only play
    tree actions; or a slot already taken, which a single history cannot do)."""
    t0 = time.perf_counter()
    topo = shape.topo
    V, A = shape.kinds.shape
    n = int(root.stack.shape[0])
    street = int(np.asarray(root.street))
    if street != shape.street or n != shape.n:
        raise ValueError(f"the state is street {street} with {n} players, the shape {shape.street}/{shape.n}")
    history = [(int(k), int(t)) for k, t in history]
    kinds = np.where(shape.kinds == SLOT, RAISE, shape.kinds).astype(np.int32)
    targets = np.full((V, A), -1, np.int32)
    legal = np.zeros((V, A), bool)
    base = jax.tree.map(np.asarray, _strip(jax.tree.map(jnp.asarray, root)))
    state_rows = [jax.tree.map(lambda x: x[None], base)]
    info_rows = []
    path, path_cols, in_slot = [0], [], []
    cur = 0                                                  # the node of the history prefix replayed so far
    for d, nodes in enumerate(topo.levels):
        states = state_rows[d]
        info = _chunked(_INFO, states, shape.offered[nodes])
        info_rows.append(info)
        src, acts, tgts, kids = [], [], [], []
        for j, v in enumerate(nodes):
            if topo.terminal[v]:
                continue
            seat = int(topo.actor[v])
            level, maximum = int(info["level"][j]), int(info["maximum"][j])
            raise_ok = bool(info["raise_ok"][j])
            fr = level_sizes(shape.sizes[seat], shape.level[v])
            on_path = (v == cur and len(path_cols) < len(history))
            action = history[len(path_cols)] if on_path else None
            seen = set()
            fill = None
            if on_path and action[0] == RAISE and shape.slot[v] >= 0:
                offered_legal = shape.offered[v] & info["size_legal"][j]
                size_totals = info["size_targets"][j, offered_legal]
                if action[1] not in size_totals:
                    fill = action[1]
            for a in range(A):
                c = int(topo.children[v, a])
                if c < 0:
                    continue
                k = int(shape.kinds[v, a])
                if k == FOLD:
                    t, ok = int(info["current"][j]), bool(info["fold_ok"][j])
                elif k == CALL:
                    t, ok = int(info["call_target"][j]), bool(info["call_ok"][j])
                elif k == ALLIN:
                    t, ok = maximum, raise_ok
                elif k == RAISE:
                    grid_action = _FRACTION_INDEX[fr[int(shape.frac[v, a])]]
                    t = int(info["size_targets"][j, grid_action])
                    ok = bool(info["size_legal"][j, grid_action])
                else:                                                       # SLOT
                    t = fill if fill is not None else maximum
                    ok = fill is not None and raise_ok and level < t < maximum and t not in seen
                if ok and k in (RAISE, SLOT):
                    seen.add(t)
                targets[v, a], legal[v, a] = t, ok
                step_t, index = t, ENGINE_INDEX[k]
                if k in (RAISE, SLOT) and t >= maximum:                     # a dead size: stepped as the all-in
                    step_t, index = maximum, config.ALLIN
                src.append(j)
                acts.append(index)
                tgts.append(step_t)
                kids.append(c)
            if on_path:
                hit = [a for a in range(A) if topo.children[v, a] >= 0 and legal[v, a] and kinds[v, a] == action[0]
                       and (action[0] in (FOLD, CALL) or targets[v, a] == action[1])]
                if not hit:
                    raise OffTree(f"action {KIND_NAMES[action[0]]} {action[1]} of seat {seat} has no column at node {v} "
                                  f"(legal: {[(KIND_NAMES[kinds[v, a]], int(targets[v, a])) for a in range(A) if legal[v, a]]})")
                a = hit[0]
                path_cols.append(a)
                in_slot.append(bool(fill is not None and shape.kinds[v, a] == SLOT))
                cur = int(topo.children[v, a])
                path.append(cur)
        if not src:
            break
        nxt = _chunked(_STEP, jax.tree.map(lambda x: x[np.asarray(src)], states), np.asarray(acts, np.int32),
                       np.asarray(tgts, np.int32))
        order = np.argsort(kids, kind="stable")
        if not np.array_equal(np.asarray(kids)[order], topo.levels[d + 1]):
            raise RuntimeError(f"replay: the children of level {d} are not the next level of the topology")
        state_rows.append(jax.tree.map(lambda x: x[order], nxt))
    if len(path_cols) != len(history):
        raise OffTree(f"the history continues past a boundary of the tree ({len(path_cols)} of {len(history)} actions)")
    states = jax.tree.map(lambda *x: np.concatenate(x), *state_rows)
    info = jax.tree.map(lambda *x: np.concatenate(x), *info_rows)
    alive = np.zeros(V, bool)
    alive[0] = True
    for v in range(1, V):
        alive[v] = alive[topo.parent[v]] and legal[topo.parent[v], topo.label[v]]
    n_live = (~np.asarray(states.folded)).sum(1)
    term = np.asarray(states.terminal)
    moved_on = (np.asarray(states.street) > street) & ~term
    kind = np.where(n_live == 1, TR.FOLD, np.where(moved_on, TR.LEAF, TR.SHOWDOWN if street == 3 else TR.ALLIN_RUNOUT))
    boundary = term | moved_on
    bad_actor = alive & ~topo.terminal & ((np.asarray(states.to_act) != topo.actor) | boundary)
    bad_boundary = alive & topo.terminal & ((~boundary) | (n_live != topo.live.sum(1)))
    if bad_actor.any() or bad_boundary.any():
        raise NotImplementedError(f"the state does not replay onto its topology ({int(bad_actor.sum())} actors, "
                                  f"{int(bad_boundary.sum())} boundaries differ on alive nodes): a table the engine "
                                  "plays differently from the deep root (more than two players with unequal stacks?)")
    return LiveTree(shape, kinds, targets, legal, alive, states, info, kind.astype(np.int32), tuple(history),
                    np.asarray(path, np.int32), np.asarray(path_cols, np.int32), np.asarray(in_slot, bool),
                    time.perf_counter() - t0)


def bucket_of(root):
    """The size bucket of a street-start state (config.size_bucket: the effective stack in BB preflop,
    the SPR postflop)."""
    root = jax.tree.map(np.asarray, root)
    street = int(root.street)
    live = ~root.folded
    if street == 0:
        return config.size_bucket(0, depth_bb=int((root.stack + root.bet_total)[live].min()) / config.BB)
    return config.size_bucket(street, spr=int(root.stack[live].min()) / int(root.bet_total.sum()))


def build_tree(root: State, hero: int, history=(), bucket=None, sizes=None, cap=None) -> LiveTree:
    """The live tree of a real street-start state: its topology (shape_for) replayed with the history."""
    root = jax.tree.map(np.asarray, root)
    bucket = bucket_of(root) if bucket is None else bucket
    return replay(shape_for(int(root.street), bucket, int(root.stack.shape[0]), hero, sizes, cap), root, history)


def visible_board(state):
    """[5] the board cards dealt by the state's street (-1 for the rest), host ints."""
    shown = {0: 0, 1: 3, 2: 4, 3: 5}[int(np.asarray(state.street))]
    board = np.asarray(state.board).astype(np.int32).copy()
    board[shown:] = -1
    return board


_FLOP_CACHE = Path("results/cache/flop_equity_v1")


class MissingArtifact(RuntimeError):
    """An artifact the live path needs is not there (a flop class missing from the cache): a counted
    error, never a computation from scratch mid-hand (T5 measured 2.2-5.3 CPU s per flop matrix)."""


def flop_cache_status(read_only=None, cache=_FLOP_CACHE):
    """(classes present of 1755, total bytes, sha1 of the sorted file names) of the flop cache(s) - the
    ready gate records it; incomplete = not ready."""
    names = {f"{c[0]}_{c[1]}_{c[2]}.npy" for c in FL.FLOPS}
    found, size = {}, 0
    for folder in ((Path(read_only),) if read_only else ()) + (Path(cache),):
        if folder.exists():
            for f in folder.iterdir():
                if f.name in names and f.name not in found:
                    found[f.name] = f.stat().st_size
    return len(found), sum(found.values()), hashlib.sha1("\n".join(sorted(found)).encode()).hexdigest()[:12]


def flop_equity(flop, cache=_FLOP_CACHE, read_only=None, compute_missing=False):
    """The exact all-in matrix W [1326, 1326] of a flop (poker/solver/runout.py): read from the canonical
    class representative's cached matrix and permuted back (suit symmetry: W_flop[h, o] =
    W_canon[p(h), p(o)] for the suit permutation p that maps the flop onto its representative), so
    a cache of the 1755 classes covers every flop. `read_only`: a cache directory that is only read
    (another tree's cache). A class in neither raises MissingArtifact (the play path verifies the cache
    at the ready gate) unless compute_missing=True (the loop's behaviour: computed and written to `cache`)."""
    flop = np.sort(np.asarray(flop, np.int32)[:3])
    canon = FL.canonical(flop)
    perm = None
    for p in range(len(FL.CARD_PERM)):
        if np.array_equal(np.sort(FL.CARD_PERM[p, flop]), canon):
            perm = p
            break
    if perm is None:
        raise RuntimeError(f"flop {flop.tolist()}: no suit permutation maps it onto its class representative {canon.tolist()}")
    name = f"{canon[0]}_{canon[1]}_{canon[2]}.npy"
    path = None
    for folder in ((Path(read_only),) if read_only else ()) + (Path(cache),):
        if (folder / name).exists():
            path = folder / name
            break
    if path is None:
        if not compute_missing:
            raise MissingArtifact(f"flop class {name} is missing from the cache ({read_only or ''} / {cache})")
        w = np.asarray(runout.flop_equity_cached(canon, cache=Path(cache)), np.float32)
    else:
        w = np.load(path)
    hand_perm = C.PAIR_INDEX[FL.CARD_PERM[perm][C.HANDS[:, 0]], FL.CARD_PERM[perm][C.HANDS[:, 1]]]
    return w[np.ix_(hand_perm, hand_perm)]


_DISJOINT = ~(C.MEMBER.T.astype(np.int32) @ C.MEMBER.astype(np.int32)).astype(bool)        # [1326, 1326]
_BINOM7 = np.asarray([[math.comb(c, k + 1) for c in range(52)] for k in range(7)], np.int64)   # [7, 52]


def turn_equity(board4, table=None):
    """The all-in matrix W [1326, 1326] of a turn board (poker/solver/runout.equity_matrix_turn: win + tie / 2
    over the 44 river cards that avoid both hands) on the host from the rank7 table (poker/solver/rank7.py,
    int16 classes by combinadic index; `table`: the array, memory-mapped or not, or its path - the
    file is read through the page cache, nothing is copied onto the device), falling back to the
    device computation without the table when there is none. float32, the same sum as the device
    version (checks/s8_2.py compares them)."""
    board4 = np.sort(np.asarray(board4, np.int32)[:4])
    if table is None:
        return np.asarray(runout.equity_matrix_turn(jnp.asarray(board4, jnp.int32)))
    if isinstance(table, (str, Path)):
        table = np.load(table, mmap_mode="r")
    hands = C.HANDS
    total = np.zeros((H, H), np.float32)
    on_board = np.zeros(52, bool)
    on_board[board4] = True
    for c in range(52):
        if on_board[c]:
            continue
        board5 = np.sort(np.append(board4, c))
        cards = np.sort(np.concatenate([hands, np.broadcast_to(board5, (H, 5))], axis=1), axis=1)      # [H, 7]
        idx = np.sum(_BINOM7[np.arange(7)[None, :], cards], axis=1)
        idx = np.clip(idx, 0, len(table) - 1)                  # a blocked hand repeats a card: masked below
        ranks = np.asarray(table[idx], np.int32)
        ranks[C.MEMBER[board5].any(axis=0)] = -1
        ok = ranks >= 0
        r_h, r_o = ranks[:, None], ranks[None, :]
        score = (r_h > r_o).astype(np.float32) + 0.5 * (r_h == r_o).astype(np.float32)
        total += np.where(ok[:, None] & ok[None, :], score, 0.0)
    return np.where(_DISJOINT, total / 44.0, 0.0).astype(np.float32)


def default_turn_equity(rank7_path=None):
    """The turn all-in matrix builder of this device: on a GPU the device computation
    (runout.equity_matrix_turn with the rank7 table on the device when the file exists, else the evaluator: 0.2 s on
    the 1070 against 1.5 s of the host loop; compiled at the warm-up), on the CPU the host loop over the memory-mapped
    rank7 table (turn_equity). board4 -> W [H, H] float32 (numpy); the attribute `where` says which."""
    from poker.solver import rank7
    if jax.default_backend() == "gpu":
        table = rank7.cached_table()

        def device_builder(board4):
            b = jnp.asarray(np.sort(np.asarray(board4, np.int32)[:4]), jnp.int32)
            return np.asarray(runout.equity_matrix_turn(b, table))

        device_builder.where = "device"
        return device_builder
    path = Path(rank7_path) if rank7_path is not None else rank7.TABLE_PATH
    table = np.load(path, mmap_mode="r") if path.exists() else None

    def host_builder(board4):
        return turn_equity(board4, table)

    host_builder.where = "host (rank7 memmap)" if table is not None else "host (device evaluator)"
    return host_builder


class CacheUnusable(RuntimeError):
    """The persistent compilation cache directory cannot be used (RAM-backed, or a nearly full disk): the ready gate
    refuses - a bot that compiles into RAM loses its warm-up at every restart."""


def mount_fstype(path):
    """The filesystem type of the mount holding `path` (/proc/mounts; None when unknown)."""
    path = str(path)
    best = None
    try:
        for line in open("/proc/mounts"):
            parts = line.split()
            if len(parts) >= 3 and (path == parts[1] or path.startswith(parts[1].rstrip("/") + "/")):
                if best is None or len(parts[1]) > len(best[0]):
                    best = (parts[1], parts[2])
    except OSError:
        return None
    return None if best is None else best[1]


def compilation_cache_dir(explicit=None, backend=None, min_free_mb=512):
    """The persistent JAX compilation cache of this process (config.JAX_CACHE_DIR / POKER_JAX_CACHE, or `explicit`),
    one subfolder per backend and grid: created, checked (not on tmpfs / ramfs, at least `min_free_mb` free) and
    installed into jax.config; CacheUnusable otherwise. Returns (directory, free MB)."""
    base = Path(explicit if explicit is not None else config.JAX_CACHE_DIR).expanduser()
    path = base / f"{backend or jax.default_backend()}_grid{config.GRID}"
    path.mkdir(parents=True, exist_ok=True)
    fstype = mount_fstype(path.resolve())
    if fstype in ("tmpfs", "ramfs", "devtmpfs"):
        raise CacheUnusable(f"compilation cache {path} is on {fstype} (RAM): refused; point POKER_JAX_CACHE at a disk")
    st = os.statvfs(path)
    free_mb = st.f_bavail * st.f_frsize / 2 ** 20
    if free_mb < min_free_mb:
        raise CacheUnusable(f"compilation cache {path}: {free_mb:.0f} MB free < {min_free_mb} MB: refused")
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    return path, free_mb


def spot_batch(tree: LiveTree, ranges, rake_level=0.0, equity=None, fixed=None, dtype=jnp.float32, gadget=None):
    """The device batch (one spot, leading axis 1) poker/solver/pcfr.py expects (poker/solver/situations.prepare
    for a live tree): tree data in units of the STREET-START pot, ranges [n, H] normalised over the
    hands the visible board allows, rank tables (river) or the all-in equity matrix (turn, flop:
    `equity`, else computed here; preflop: given), the critics' public inputs, the frozen played
    strategies: fixed = {node: pi [H, A]} (pcfr: fixed_mask / fixed_pi), and the re-solving gadget
    (pcfr: gadget_ev / gadget_mask): gadget = {"ev": [n, H] whole-hand EV ceilings per hand of every seat in
    street-start pot units, "mask": [n] bool, the seats that get one} or None (no gadget: the keys are present
    with the mask off, so every live spot compiles to the same program)."""
    states = tree.states
    board = visible_board(tree.node_state(0))
    valid = C.board_valid(board[board >= 0]) if (board >= 0).any() else np.ones(H, bool)
    r = np.asarray(ranges, np.float64) * valid[None, :]
    total = r.sum(1, keepdims=True)
    if not np.all(total > 0):
        raise ValueError("a range has no mass on the hands the board allows")
    r = r / total
    contrib = np.asarray(states.bet_total, np.float64)                                 # [V, n]
    pot = contrib.sum(1)
    scale = float(pot[0])
    rake = np.asarray(_chunked(_RAKE, states), np.float64)
    bps, cap = int(np.asarray(states.rake_bps)[0]), int(np.asarray(states.rake_cap)[0])
    one = lambda x: jnp.asarray(np.asarray(x)[None])
    out = {"legal": one(tree.legal & (tree.topo.children >= 0)), "alive": one(tree.alive),
           "contrib": one((contrib / scale).astype(np.float32)), "pot": one((pot / scale).astype(np.float32)),
           "behind": one((np.asarray(states.stack) / scale).astype(np.float32)),
           "street_bet": one((np.asarray(states.bet_street) / scale).astype(np.float32)),
           "ranges": jnp.asarray(r[None], dtype), "valid": jnp.asarray(valid[None], dtype),
           "board": one(board.astype(np.int32)),
           "board1h": jnp.asarray(np.sum(np.eye(52, dtype=np.float32)[board[board >= 0]], axis=0)[None], dtype),
           "pot_bb": jnp.asarray([scale / config.BB], dtype),
           "stack_pot": jnp.asarray([float(np.asarray(states.stack)[0].min()) / scale], dtype),
           "rake": one((rake / scale).astype(np.float32)),
           "rake_level": jnp.asarray([rake_level], dtype), "rake_rate": jnp.asarray([bps / 500.0], dtype),
           "rake_cap": jnp.asarray([cap / scale], dtype),
           "pot_chips": one(pot.astype(np.int32)), "rake_bps": jnp.asarray([bps], jnp.int32),
           "rake_cap_chips": jnp.asarray([cap], jnp.int32)}
    vi = len(tree.internal)
    mask = np.zeros(vi, bool)
    pi = np.zeros((vi, H, tree.kinds.shape[1]), np.float32)
    for v, row in (fixed or {}).items():
        i = int(tree.st.to_internal[v])
        if i < 0:
            raise ValueError(f"node {v} is not a decision")
        mask[i] = True
        pi[i] = np.asarray(row, np.float32)
    out["fixed_mask"], out["fixed_pi"] = one(mask), one(pi)
    n = tree.n
    if gadget is None:                 # the same host->device transfers as a gadget spot: the warm-up compiles their staging
        out["gadget_ev"], out["gadget_mask"] = jnp.asarray(np.zeros((1, n, H), dtype)), jnp.asarray(np.zeros((1, n), bool))
    else:
        ev, gm = np.asarray(gadget["ev"], np.float64), np.asarray(gadget["mask"], bool)
        if ev.shape != (n, H) or gm.shape != (n,) or not np.isfinite(ev).all():
            raise ValueError(f"gadget: ev [{n}, {H}] finite and mask [{n}] expected, got {ev.shape}, {gm.shape}")
        if gm[tree.hero]:
            raise ValueError("gadget: the hero carries his range, never a ceiling")
        # host conversions, then plain device transfers: a dtype conversion on the device would be the one compile of
        # the first gadget solve after the warm-up (measured: 3 compile events, 8 ms), and the play path compiles nothing
        out["gadget_ev"] = jnp.asarray(np.asarray((ev * valid[None, :])[None], dtype))
        out["gadget_mask"] = jnp.asarray(np.asarray(gm[None], bool))
    if tree.street == 3:
        out["tables"] = SI._tables(jnp.asarray(board[None], jnp.int32))
    elif tree.street == 2:
        w = runout.equity_matrix_turn(jnp.asarray(board[:4], jnp.int32)) if equity is None else jnp.asarray(equity)
        out["equity"] = jnp.asarray(w, dtype)[None]
    elif tree.street == 1:
        w = flop_equity(board[:3]) if equity is None else equity
        out["equity"] = jnp.asarray(w, dtype)[None]
    elif equity is not None:
        out["equity"] = jnp.asarray(equity, dtype)[None]
    else:
        raise ValueError("a preflop solve needs the exact preflop matrix (runout.preflop_equity_cached)")
    return out


_COMPILE = [0.0, 0]


def _on_duration(event, duration, *args, **kwargs):
    if event.startswith("/jax/core/compile"):
        _COMPILE[0] += float(duration)
        _COMPILE[1] += 1


jax.monitoring.register_event_duration_secs_listener(_on_duration)


def compile_seconds():
    """Seconds this process has spent tracing, lowering and compiling so far (jax.monitoring)."""
    return _COMPILE[0]


def compile_events():
    """Compilations so far in this process (jax.monitoring /jax/core/compile events): the play path
    must compile at warm-up only, never at inference (checks/s8_1.py asserts no event after the warm-up)."""
    return _COMPILE[1]


_AVERAGE = jax.jit(jax.vmap(SV.average_policy))


@dataclass
class Solution:
    tree: LiveTree
    sigma: np.ndarray             # [Vi, H, A] the average strategy at every internal node (0 on absent columns)
    iterations: int
    e: float = None               # certificate: NashConv / n in street-start-pot units (certificate=True), else None
    seconds: float = 0.0          # wall time of the solve call this spot was part of (without the tree build)
    cpu_seconds: float = 0.0
    compile_seconds: float = 0.0  # compile time inside that call (0 once the topology is compiled)
    batch_size: int = 1
    batch: dict = field(default=None, repr=False)
    follow: np.ndarray = field(default=None, repr=False)   # [n, H] the gadget seats' average Follow probability (1 without a gadget)

    def strategy(self, v):
        """[H, A] the average strategy at node v (internal; zero on dead columns)."""
        return self.sigma[self.tree.st.to_internal[v]]


def solve_batch(trees, ranges, iterations: int, chunk: int = 50, aux=None, rake_level=0.0, equity=None, fixed=None,
                certificate: bool = False, keep_batch: bool = False, check_finite: bool = True, ladder=LADDER, gadget=None):
    """PCFR+ on several live trees of ONE topology in one vmapped call (concurrent decisions of many
    tables), exactly `iterations` iterations each (chunk per device call); returns a Solution per tree.
    ranges / equity / fixed: one entry per tree (equity: the all-in matrix of the board; fixed: {node:
    pi} of our played nodes). aux: the next street's critic (critics.net_critic) when the trees have
    street ends. certificate=True also computes the exact exploitability of every average strategy.
    check_finite (default on): the average policy, the root's counterfactual values of the last iterate
    (pcfr's state) and the input ranges of every spot must be finite, else NonFinite (counted by the
    player, never played). The batch
    is padded with copies of its first spot up to the next step of `ladder` (the closed set of compiled
    shapes, warmed up at start), the padded rows discarded: B enters no shape."""
    if not trees:
        return []
    keys = {t.key for t in trees}
    if len(keys) != 1:
        raise ValueError(f"solve_batch takes trees of one topology, got {len(keys)}")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    chunk = max(1, min(int(chunk), int(iterations)))
    if iterations % chunk:
        raise ValueError(f"iterations {iterations} must be a multiple of the chunk {chunk}")
    st = trees[0].st
    if len(st.leaves) and aux is None:
        raise ValueError("a tree with street ends needs the next street's critic (aux)")
    B = len(trees)
    if B > ladder[-1]:
        raise ValueError(f"a batch of {B} is above the ladder's top {ladder[-1]} (the trap's size cap)")
    B_pad = min(step for step in ladder if step >= B)      # the shape compiled at warm-up: topology x ladder
    equity = [None] * B if equity is None else list(equity)
    fixed = [None] * B if fixed is None else list(fixed)
    gadget = [None] * B if gadget is None else list(gadget)
    spots = [spot_batch(t, r, rake_level, e, f, gadget=g) for t, r, e, f, g in zip(trees, ranges, equity, fixed, gadget)]
    padded = spots + [spots[0]] * (B_pad - B)                 # copies of the first spot, discarded after the solve
    batch = padded[0] if B_pad == 1 else jax.tree.map(lambda *x: jnp.concatenate(x), *padded)
    # the compiled step depends on the chunk only (cap is the stopping rule of pcfr.solve_cfr, unused here):
    # one compile per (topology, chunk, ladder step) whatever T
    cfg = SV.SolverConfig(cap=chunk, e_target=0.0, check_every=chunk)
    c0 = compile_seconds()
    t0, p0 = time.perf_counter(), time.process_time()
    step = SV.make_cfr_chunk(st, cfg)
    state = jax.vmap(lambda _: SV.init_cfr_state(st, batch["ranges"].dtype))(jnp.arange(B_pad))
    for _ in range(iterations // chunk):
        state = step(state, batch, aux)
    sigma = np.asarray(_AVERAGE(state["avg"], batch["legal"][:, st.internal]))
    follow = np.asarray(SV.average_follow(state))                                   # [B_pad, n, H]
    if check_finite:
        values = np.asarray(state["root"])                     # the root's counterfactual values of the last iterate
        ranges_in = np.asarray(batch["ranges"])
        bad = [b for b in range(B) if not (np.isfinite(sigma[b]).all() and np.isfinite(values[b]).all()
                                            and np.isfinite(ranges_in[b]).all() and np.isfinite(follow[b]).all())]
        del values
        if bad:
            raise NonFinite(f"non-finite policy, root values or ranges in {len(bad)} of {B} spots (first {bad[:4]})")
    es = [None] * B
    if certificate:
        ex = SV.cached("ex_live", st, cfg, lambda: jax.jit(jax.vmap(
            lambda s, spot, a: SV.exploitability(SV.average_logits(s["avg"], spot, st), spot, st, cfg, a,
                                                 follow=SV.average_follow(s))["nash_conv"] / st.n,
            in_axes=(0, 0, None))))
        es = [float(x) for x in np.asarray(ex(state, batch, aux))]
        for e in es:
            if not np.isfinite(e) or e < -SV.E_TOL:
                raise RuntimeError(f"live solve: E {e} is not finite or negative - a solver bug, not convergence")
    wall, cpu, comp = time.perf_counter() - t0, time.process_time() - p0, compile_seconds() - c0
    return [Solution(t, np.ascontiguousarray(sigma[b]), int(iterations), es[b], wall, cpu, comp, B_pad,
                     spots[b] if keep_batch else None, np.ascontiguousarray(follow[b])) for b, t in enumerate(trees)]


def solve(tree: LiveTree, ranges, iterations: int, chunk: int = 50, aux=None, rake_level=0.0, equity=None, fixed=None,
          certificate: bool = False, keep_batch: bool = False, check_finite: bool = True, gadget=None) -> Solution:
    """solve_batch for one tree."""
    return solve_batch([tree], [ranges], iterations, chunk, aux, rake_level, [equity], [fixed], certificate, keep_batch,
                       check_finite, gadget=[gadget])[0]


def exploitability_of(tree: LiveTree, sigma, ranges, aux=None, rake_level=0.0, equity=None, batch=None, gadget=None,
                      follow=None):
    """Exact exploitability (NashConv / n, street-start-pot units) of an arbitrary strategy profile
    sigma [Vi, H, A] (probabilities over columns) on a live tree; checks. With `gadget` (spot_batch's) and the gadget
    seats' Follow probabilities `follow` [n, H] it is the exploitability of the gadget game."""
    st = tree.st
    batch = spot_batch(tree, ranges, rake_level, equity, gadget=gadget) if batch is None else batch
    cfg = SV.SolverConfig()
    fn = SV.cached("ex_sigma", st, cfg, lambda: jax.jit(jax.vmap(
        lambda z, spot, a, f: SV.exploitability(z, spot, st, cfg, a, follow=f)["nash_conv"] / st.n,
        in_axes=(0, 0, None, 0))))
    z = jnp.log(jnp.maximum(jnp.asarray(sigma, jnp.float32), 1e-30))[None]
    f = jnp.ones((1, st.n, H), jnp.float32) if follow is None else jnp.asarray(follow, jnp.float32)[None]
    return float(np.asarray(fn(z, batch, aux, f))[0])


def _dummy_root(n, street, menu):
    """A street-start state of `n` players in the menu `menu` (config.size_bucket decides): the first
    depth of a ladder of depths that falls into the menu; a warm-up spot, nothing else."""
    for depth_bb in (400, 200, 100, 60, 40, 30, 15, 8, 4):
        if street == 0:
            root = TR.preflop_root(n, depth_bb * config.BB)
        else:
            root = TR.street_root(n, street, 6 * config.BB, depth_bb * config.BB)
        if bucket_of(root) == menu:
            return root
    raise ValueError(f"no depth of the warm-up ladder falls into the menu {menu!r} on street {street}")


def warm_up(iterations, chunk=50, critics=None, flop=None, turn=None, preflop=None, field=None, ladder=None, log=None,
            certificate=False):
    """Compile the solver of every topology of the field (config.regime_field: every table size, every seat
    of it, every street, every menu) at every batch size of the ladder before the first hand: one
    solve_batch per (n, street, menu, seat, ladder step) on copies of a dummy spot (uniform ranges; the
    all-in matrices `flop` [H, H], `turn` [H, H], `preflop` [H, H] for the streets that need one; critics
    {street: aux}), with the play-time options (finiteness check on, no certificate). The set of compiled
    shapes is then closed: the field, the ladder, the chunk and T per street are config, nothing else enters
    a shape. Returns [(key, n, street, menu, seat, batch, nodes, decisions, seconds, compile_seconds)]."""
    field = field or FIELD
    ladder = ladder or LADDER
    out = []
    critics = critics or {}
    shown = {0: 0, 1: 3, 2: 4, 3: 5}
    for n in field["players"]:
        for street in field["streets"]:
            for menu in field["menus"]:
                T = iterations.get((street, menu), iterations.get(street))
                if T is None:
                    continue
                root = _dummy_root(n, street, menu)
                eq = preflop if street == 0 else flop if street == 1 else turn if street == 2 else None
                board = jnp.asarray([0, 14, 28, 3, 17][:shown[street]] + [-1] * (5 - shown[street]), jnp.int32)
                root = jax.tree.map(np.asarray, root.replace(board=board))
                for seat in field["seats"][n]:
                    tree = build_tree(root, seat, (), menu)
                    for B in ladder:
                        t0 = time.perf_counter()
                        sols = solve_batch([tree] * B, [np.full((n, H), 1.0 / H)] * B, min(T, chunk), chunk,
                                           aux=critics.get(street), equity=[eq] * B, ladder=ladder, certificate=certificate)
                        row = (tree.key, n, street, menu, seat, B, int(tree.size), int(len(tree.internal)),
                               time.perf_counter() - t0, sols[0].compile_seconds)
                        out.append(row)
                        if log is not None:
                            log(f"warm-up {tree.key} x{B}: {tree.size} nodes, {len(tree.internal)} decisions, {row[8]:.1f} s "
                                f"(compile {sols[0].compile_seconds:.1f} s)")
    return out


def ready_line(rows, seconds, from_cache, field=None, flop_cache=None):
    """The one log line a bot prints when warm_up is done: the field covered, the compiled set, the flop
    cache's status (flop_cache_status) when a cache directory is given."""
    field = field or FIELD
    topologies = len({r[0] for r in rows})
    sizes = sorted({r[5] for r in rows})
    cache = ""
    if flop_cache is not None:
        n, size, digest = flop_cache_status(flop_cache)
        cache = f"; flop cache {n}/{len(FL.FLOPS)} classes, {size / 1e9:.2f} GB, list sha1 {digest}"
    return (f"ready: n={list(field['players'])}, menus={list(field['menus'])}, streets={list(field['streets'])}, "
            f"ladder={sizes}, {topologies} topologies x {len(sizes)} batch sizes compiled in {seconds:.0f} s "
            f"(compile {sum(r[9] for r in rows):.0f} s; from cache: {'yes' if from_cache else 'no'}){cache}")
