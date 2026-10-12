"""Public tree of one betting round for 2..6 players, built and replayed by the project engine.

Topology: every action sequence the engine allows from a very deep root of the given street,
with at most `cap` aggressive actions (fraction raises and ALLIN) per street; allin_after_cap
keeps ALLIN available after the cap (fraction raises stop); `sizes` restricts the fraction raises
per raise level and `no_open_limp` forbids preflop calls in an unopened pot except the SB's
complete and the BB's check (the rules by table size: config.tree_rule). Nodes that are not decisions are
boundaries: a fold (one player left), a showdown (river), or the end of the street before the
river - the next street's first decision (a leaf, valued by the next street's critic) or, when
nobody can act any more, an all-in runout. A real spot (pot, stack) replays the same sequences
through poker.envs.transition.step; actions the engine forbids there are masked and their subtrees
are dead (zero reach); with real stacks a deep street end can be an all-in runout (and the
other way round), so replay() reports the kind of every boundary per spot.
Seats follow the engine: seat 0 is the button; postflop seat 1 acts first (BB in HU, SB 3-way).
Two sizes with the same chips at a decision are one action, resolved within the tree's own sizes
(Topology.offered, poker/envs/actions.py action_table): the lowest index keeps it.
"""

import contextlib
import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config, guards
from poker.envs.actions import action_table
from poker.envs.betting import init_betting
from poker.envs.evaluator import rake_paid
from poker.envs.transition import next_to_act, step

# The builder's root (build_topology): pot DEEP_POT and deep_stack(n) chips behind every seat - within the engine's
# int32 proof for n seats (static sniff S13: with 10**8 behind, (pot + to_call) * 5 of the 2.5-pot size of grid 13 reached
# 93 % / 116 % / 140 % of int32 at the decisions of the 4- / 5- / 6-player rule trees, results/guards/s13;
# the wrapped targets were masked by luck: no raise open there, or wrapped below the minimum raise, which lies above the
# stack) and deep enough that no fraction raise of the capped tree comes near the stack (checked while building), so
# the topology does not depend on it.
DEEP_POT, DEEP_STACK = 1200, 10**8
AGGRESSIVE = np.zeros(config.NUM_ACTIONS, dtype=bool)
AGGRESSIVE[2:] = True                                      # fraction raises and ALLIN
FOLD, SHOWDOWN, ALLIN_RUNOUT, LEAF = 0, 1, 2, 3            # boundary kinds (replay "kind")
STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def preflop_root(n: int, stack, rake_bps=0, rake_cap=0, history=False):
    """Engine State before the first preflop decision: blinds posted, every seat `stack` chips.
    history=True (stack a Python int) records decisions like a dealt hand (player and checks)."""
    if history:
        state = init_betting(n, jnp.full(n, stack, dtype=jnp.int32), history=True,
                             max_stack=max(int(stack), config.BB + 1))
    else:
        state = init_betting(n, jnp.full(n, stack, dtype=jnp.int32), history=False)
    return state.replace(rake_bps=jnp.asarray(rake_bps, jnp.int32), rake_cap=jnp.asarray(rake_cap, jnp.int32))


def street_root(n: int, street: int, pot, stack, rake_bps=0, rake_cap=0):
    """Engine State at the start of a street; postflop the pot is split equally and stacks are
    equal; preflop the blinds are the pot (`pot` is ignored). The rake terms of the hand
    (config.rake_terms for its level and players dealt) ride along."""
    if street not in STREETS:
        raise ValueError(f"street must be 0 (preflop), 1 (flop), 2 (turn) or 3 (river), got {street}")
    if street == 0:
        return preflop_root(n, stack, rake_bps, rake_cap)
    state = init_betting(n)
    pot = jnp.asarray(pot, jnp.int32)
    state = state.replace(
        stack=jnp.full(n, stack, dtype=jnp.int32),
        bet_street=jnp.zeros(n, jnp.int32),
        bet_total=jnp.full(n, pot // n, dtype=jnp.int32),
        folded=jnp.zeros(n, bool), acted=jnp.zeros(n, bool),
        last_raise=jnp.int32(config.BB), street=jnp.int32(street), terminal=jnp.bool_(False),
        to_act=jnp.int32(0), history_len=jnp.int32(0),
        rake_bps=jnp.asarray(rake_bps, jnp.int32), rake_cap=jnp.asarray(rake_cap, jnp.int32),
    )
    return state.replace(to_act=next_to_act(state))


def river_root(n: int, pot, stack, rake_bps=0, rake_cap=0):
    return street_root(n, 3, pot, stack, rake_bps, rake_cap)


@dataclass(frozen=True)
class Topology:
    n: int
    parent: np.ndarray        # [V] (-1 at root)
    label: np.ndarray         # [V] action taken at the parent (-1 at root)
    actor: np.ndarray         # [V] seat to act, -1 at boundaries
    children: np.ndarray      # [V, A] child index or -1
    cap_ok: np.ndarray        # [V, A] action allowed by the raise cap and present in topology
    live: np.ndarray          # [V, n] not folded
    terminal: np.ndarray      # [V] boundary (fold, showdown, all-in runout or street end)
    levels: tuple             # node indices grouped by depth, root first
    street: int = 3
    leaf: np.ndarray = None   # [V] the deep tree reaches the next street here (street < 3)
    offered: np.ndarray = None  # [V, A] fraction raises the tree offers (its sizes at the node's raise level,
    #                             none at the cap); None: every grid size (the engine's whole-grid table)

    @property
    def size(self):
        return len(self.parent)

    @property
    def internal(self):
        return np.flatnonzero(~self.terminal)

    @property
    def terminals(self):
        return np.flatnonzero(self.terminal)


@functools.lru_cache(maxsize=None)
def deep_stack(n: int) -> int:
    """Chips behind every seat at the builder's root: DEEP_STACK, lowered in steps of 10**6 until the root's table
    (n seats with stack + DEEP_POT // n chips each) passes the engine's int32 proof (config.history_bound: every
    action target, masked or not, fits int32). 10**8 under grids 5 and 7 for every n <= 6 and under grid 13 for
    n <= 3; grid 13: 85, 71 and 61 million at n = 4, 5, 6."""
    stack = DEEP_STACK
    while stack > 0:
        try:
            config.history_bound(n, start=stack + DEEP_POT // n)
            return stack
        except ValueError:
            stack -= 10**6
    raise ValueError(f"no root stack for {n} seats passes the engine's int32 proof (config.history_bound)")


def _reach(frontier, done) -> int:
    """An upper bound (host int64) on the chips a fraction raise can reach at the frontier's decisions where nobody
    is all-in yet, plus its commit-merge allowance: the target is at most level + (pot + level) * the grid's largest
    fraction + the minimum-raise step (at most max(level, BB)). Below the root's stack no size is masked or merged by
    the stack, so the tree does not depend on it."""
    chips, folded = np.asarray(frontier.stack, np.int64), np.asarray(frontier.folded)
    rows = ~done & np.all(folded | (chips > 0), axis=1)
    if not rows.any():
        return 0
    pot = np.asarray(frontier.bet_total, np.int64)[rows].sum(axis=1)
    level = np.maximum(np.asarray(frontier.bet_street, np.int64)[rows].max(axis=1), config.BB)
    num, den = max(config.RAISE_FRACS, key=lambda f: f[0] / f[1])
    target = 2 * level + config.BB - (-(pot + level) * num // den)
    merge_num, merge_den = config.COMMIT_MERGE or (0, 1)
    return int(np.max(target + (pot + target) * merge_num // merge_den))


def blind_seats(n: int):
    """(SB seat, BB seat) of a full table (poker.envs.betting.init_betting: in HU the button posts SB)."""
    return (0, 1) if n == 2 else (1, 2)


_CHUNK = 2048                              # engine calls of the builder: one compiled shape per n
_TABLE = jax.jit(jax.vmap(lambda s, sizes: action_table(s, sizes)[0]))
_STEP = jax.jit(jax.vmap(step))


def _chunked(fn, states, *args):
    """fn over a batched host State (numpy leaves, leading axis) in padded chunks of _CHUNK rows;
    returns numpy leaves (gathers stay on the host: no compilation per frontier width)."""
    count = states.terminal.shape[0]
    outs = []
    for start in range(0, count, _CHUNK):
        idx = np.arange(start, min(count, start + _CHUNK))
        pad = np.concatenate([idx, np.full(_CHUNK - len(idx), idx[0])])
        out = fn(jax.tree.map(lambda x: x[pad], states), *(a[pad] for a in args))
        outs.append(jax.tree.map(lambda x: np.asarray(x)[:len(idx)], out))
    return jax.tree.map(lambda *x: np.concatenate(x), *outs)


def build_topology(n: int, cap: int, allin_after_cap: bool = False, street: int = 3,
                   sizes=None, no_open_limp: bool = False) -> Topology:
    """The public tree (module doc) for n in 2..config.TREE_MAX_PLAYERS. sizes: None = every fraction
    raise of the grid at every raise level, else a tuple per raise level (0 = the street's first
    aggressive action) of the allowed fraction-raise actions, the last entry repeating. Built level
    by level with batched engine calls (on the CPU when one is available); node order is the
    breadth-first order of one-at-a-time expansion (parents in index order, actions ascending)."""
    if not isinstance(n, int) or not 2 <= n <= config.TREE_MAX_PLAYERS:
        raise ValueError(f"the street solver supports 2..{config.TREE_MAX_PLAYERS} players, got {n!r}")
    if no_open_limp and street != 0:
        raise ValueError("no_open_limp is a preflop rule")
    fractions = set(range(2, config.ALLIN))
    if sizes is not None:
        sizes = tuple(tuple(sorted({int(a) for a in level})) for level in sizes)
        if not sizes or not all(set(level) <= fractions for level in sizes):
            raise ValueError(f"sizes must be non-empty levels of fraction-raise actions {sorted(fractions)}, got {sizes}")
    sb, bb = blind_seats(n)
    offer = np.zeros((cap + 1, config.NUM_ACTIONS), dtype=bool)      # by raise level; at the cap: none
    for level in range(cap):
        offer[level, sorted(fractions) if sizes is None else list(sizes[min(level, len(sizes) - 1)])] = True
    cpu = _cpu()
    with jax.default_device(cpu) if cpu is not None else contextlib.nullcontext():
        stack = deep_stack(n)
        root = street_root(n, street, DEEP_POT, stack)
        frontier = jax.tree.map(lambda x: np.asarray(x)[None], root)
        parent, label, actor, live, terminal, leaf, depth, raises, children = [-1], [-1], [], [], [], [], [0], [0], []
        offered = []
        first = 0                                   # index of the frontier's first node
        while frontier is not None:
            count = frontier.terminal.shape[0]
            term = np.asarray(frontier.terminal)
            at_leaf = (np.asarray(frontier.street) > street) & ~term
            done = term | at_leaf
            reach = _reach(frontier, done)
            if reach >= stack:
                raise ValueError(f"build_topology(n={n}, street={street}, cap={cap}): a fraction raise can reach "
                                 f"{reach} chips with its commit-merge allowance, the root has {stack} behind: the "
                                 f"tree would depend on the stack")
            to_act = np.asarray(frontier.to_act)
            folded = np.asarray(frontier.folded)
            offered.append(offer[np.minimum(raises[first:first + count], cap)])
            mask = np.asarray(_chunked(_TABLE, frontier, offered[-1]))
            src, acts = [], []
            for j in range(count):
                i = first + j
                terminal.append(bool(done[j]))
                leaf.append(bool(at_leaf[j]))
                live.append(~folded[j])
                actor.append(-1 if done[j] else int(to_act[j]))
                kids = [-1] * config.NUM_ACTIONS
                if not done[j]:
                    level = raises[i]
                    for a in range(config.NUM_ACTIONS):
                        if not mask[j, a]:
                            continue
                        if AGGRESSIVE[a] and level >= cap and not (allin_after_cap and a == config.ALLIN):
                            continue
                        if sizes is not None and a in fractions and a not in sizes[min(level, len(sizes) - 1)]:
                            continue
                        if no_open_limp and a == config.CHECK_CALL and level == 0 and to_act[j] not in (sb, bb):
                            continue
                        kids[a] = len(parent)
                        parent.append(i)
                        label.append(a)
                        raises.append(level + int(AGGRESSIVE[a]))
                        depth.append(depth[i] + 1)
                        src.append(j)
                        acts.append(a)
                children.append(kids)
            first += count
            if not src:
                break
            src = np.asarray(src)
            frontier = _chunked(_STEP, jax.tree.map(lambda x: x[src], frontier), np.asarray(acts, np.int32))
    children = np.asarray(children, dtype=np.int32)
    depth = np.asarray(depth)
    levels = tuple(np.flatnonzero(depth == d) for d in range(depth.max() + 1))
    return Topology(n, np.asarray(parent, np.int32), np.asarray(label, np.int32),
                    np.asarray(actor, np.int32), children, children >= 0,
                    np.asarray(live, bool), np.asarray(terminal, bool), levels,
                    street, np.asarray(leaf, bool), np.concatenate(offered))


def rule_topology(n: int, street: int, menu=None) -> Topology:
    """The teacher's tree for `n` players on `street` in size menu `menu` (None: config's unbounded menu) under the
    rules by table size (config.tree_rule): the only builder of the decided trees (poker/solver/loop.py tree_for, checks)."""
    r = config.tree_rule(n, street, menu)
    return build_topology(n, r["cap"], allin_after_cap=r["allin_after_cap"], street=street, sizes=r["sizes"],
                          no_open_limp=r["no_open_limp"])


_REPLAY = {}


def _replay_fn(topo: Topology):
    """One compiled replay per topology (cached: no recompilation per batch)."""
    if id(topo) in _REPLAY:
        return _REPLAY[id(topo)][1]
    parent = jnp.asarray(np.maximum(topo.parent, 0))
    label = jnp.asarray(np.maximum(topo.label, 0))
    term = jnp.asarray(topo.terminal)
    cap_ok = jnp.asarray(topo.cap_ok)
    actor = jnp.asarray(topo.actor)
    street = topo.street
    offered = None if topo.offered is None else jnp.asarray(topo.offered)

    def table(states):
        if offered is None:
            return jax.vmap(action_table)(states)
        return jax.vmap(action_table)(states, offered)

    def one(p, s, bps, cap):
        root = street_root(topo.n, street, p, s, bps, cap)
        buffer = jax.tree.map(lambda x: jnp.broadcast_to(x, (topo.size,) + x.shape), root)

        def body(buf, v):
            prev = jax.tree.map(lambda x: x[parent[v]], buf)
            nxt = jax.lax.cond(v == 0, lambda: prev, lambda: step(prev, label[v]))
            buf = jax.tree.map(lambda x, y: x.at[v].set(y), buf, nxt)
            return buf, None

        states, _ = jax.lax.scan(body, buffer, jnp.arange(topo.size))
        masks, targets = table(states)
        legal = masks & cap_ok & ~term[:, None]
        targets = jnp.where(legal, targets, -1)
        alive = jnp.zeros(topo.size, bool).at[0].set(True)
        for level in topo.levels[1:]:
            par, lab = topo.parent[level], topo.label[level]
            alive = alive.at[level].set(alive[par] & legal[par, lab])
        contrib = states.bet_total
        n_live = jnp.sum(~states.folded, axis=1)
        moved_on = (states.street > street) & ~states.terminal
        kind = jnp.where(n_live == 1, FOLD,
                         jnp.where(moved_on, LEAF,
                                   jnp.where(street == 3, SHOWDOWN, ALLIN_RUNOUT)))
        boundary = states.terminal | moved_on
        return {"legal": legal, "targets": targets, "alive": alive, "contrib": contrib, "pot": contrib.sum(-1),
                "rake": jax.vmap(rake_paid)(states), "kind": kind.astype(jnp.int32),
                "stack": states.stack, "street_bet": states.bet_street,
                "actor_ok": jnp.where(alive & ~term, states.to_act == actor, True),
                "terminal_ok": jnp.where(alive, boundary == term, True)}

    fn = jax.jit(jax.vmap(one))
    _REPLAY[id(topo)] = (topo, fn)          # keep topo alive so its id is never reused
    return fn


def _cpu():
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return None


def require_cpu(what: str):
    """_cpu() for engine work the GPU may not run: the fused engine scan of replay (and the critics' card
    loop, poker/solver/loop.py aux_device) does not load on Pascal GPUs - CUDA_ERROR_NOT_SUPPORTED at the first batch, memo
    section 12 (11) - and runs on the CPU device. A process whose default device is not a CPU and which has no CPU
    device (JAX_PLATFORMS=cuda) raises here, at the first call, on every card: never a silent GPU path that only a
    Pascal card would expose mid-run."""
    cpu = _cpu()
    if cpu is None and jax.default_backend() != "cpu":
        raise RuntimeError(f"{what} needs the CPU device (the fused engine scan does not load on Pascal GPUs: "
                           f"CUDA_ERROR_NOT_SUPPORTED, memo section 12 (11)); this process has only the "
                           f"{jax.default_backend()} backend (JAX_PLATFORMS={jax.config.jax_platforms!r}): run GPU "
                           f"processes with JAX_PLATFORMS=cuda,cpu")
    return cpu


def replay(topo: Topology, pot, stack, rake_bps=None, rake_cap=None):
    """Per-spot node data for a batch of spots: pot/stack int32[B] in chips, rake terms int32[B]
    (default: no rake).

    Returns dict of arrays with leading [B, V]: legal [B,V,A] (engine legality, cap and
    topology), alive [B,V] (every action on the path is legal), contrib [B,V,n] chips put in
    over the whole hand, pot [B,V], rake [B,V] chips the engine charges at terminals (0
    elsewhere), kind [B,V] of every boundary (FOLD, SHOWDOWN, ALLIN_RUNOUT, LEAF; meaningless
    at decisions), stack [B,V,n] chips behind, and consistency flags actor_ok / terminal_ok.
    Runs on the CPU device (cheap: once per batch; a GPU process without one raises: require_cpu); the fused engine scan
    does not load on Pascal GPUs (kernel parameter limit), so results are moved afterwards.
    The spots are validated on the host first (poker.envs.guards.spot_chips, ValueError): pot >= 0, chips
    behind preflop above BB and postflop at least 1, the engine's int32 proof for the table.
    """
    guards.spot_chips(topo.n, topo.street, pot, stack, "tree.replay")
    cpu = _cpu()
    pot = jnp.asarray(pot, jnp.int32)
    stack = jnp.asarray(stack, jnp.int32)
    bps = jnp.zeros_like(pot) if rake_bps is None else jnp.asarray(rake_bps, jnp.int32)
    cap = jnp.zeros_like(pot) if rake_cap is None else jnp.asarray(rake_cap, jnp.int32)
    if cpu is None:
        require_cpu("poker.solver.tree.replay")              # a GPU process without the CPU device stops here
        return _replay_fn(topo)(pot, stack, bps, cap)
    out = _replay_fn(topo)(*(jax.device_put(x, cpu) for x in (pot, stack, bps, cap)))
    return jax.device_put(out, jax.devices()[0])
