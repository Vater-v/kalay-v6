"""Diagnostic comparison of predefined size sets against the full candidate grid.

    flock /home/vater/.gpu.lock systemd-run --user --scope -p MemoryMax=6000M -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.6 PYTHONPATH=. POKER_GRID=13 \
        POKER_REGIME=1 .venv/bin/python -m poker.solver.sizing_lab --street river

Each set supplies fraction indices per raise level, with fold/call/all-in kept under the
tree's cap. The lab uses the engine's canonical action_candidates, including own-size
deduplication, minimum raises, commit merging and effective cover. A subset's minimum-raise
label may differ from the full grid's label for the same chips; this is ordinary canonical
own sizing, shared with poker/solver/tree.py. s_only measures such legal subset labels.

The translation metric is an explicit diagnostic experiment (pot units, E = NashConv / 2):

    Loss(S, spot) = E(sigma_S translated onto T_F) - E(sigma_F)

T_F contains every grid candidate; T_S contains set S. A reference must reach E_REF before
its spot contributes; undetermined spots are counted and bounded by MAX_UNDETERMINED.
For this experiment, T_F histories map to weighted T_S nodes. Legal matching actions follow;
other fraction raises split between neighbouring legal targets with pseudo-harmonic weights,
or the nearest target outside their range. Unmapped branches and policy mass are reported.
The map back to T_F uses the same label when legal, otherwise a legal label with identical
chips. The per-seat gains sum to the reported loss.

This translated-policy loss does not measure the deployed live player's loss. Live PCFR+
replays the opponent's actual chip amount exactly and solves that subgame. The lab's separate
own-sizing experiment restricts one seat's available sizes without translating its policy.
These studies run only when explicitly invoked; they do not change the runtime menus.

Evidence is written to results/sizing/<street>/: reference reports, set losses and summary.
"""

import argparse
import contextlib
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config, derived
from poker.envs.actions import action_candidates, action_table
from poker.envs.evaluator import rake_paid
from poker.envs.transition import step
from poker.solver import pcfr as SV
from poker.solver import situations as SI
from poker.solver import tree as TR
from poker.solver.cards import NUM_HANDS as H

GRID = 13
A = config.NUM_ACTIONS
FOLD, CALL, ALLIN = config.FOLD, config.CHECK_CALL, config.ALLIN
FRACTIONS = tuple(range(2, ALLIN))
PCT = {"min": (1, 100), "25": (1, 4), "33": (1, 3), "50": (1, 2), "66": (2, 3), "75": (3, 4), "100": (1, 1),
       "150": (3, 2), "200": (2, 1), "250": (5, 2)}
INDEX = {tuple(f): 2 + i for i, f in enumerate(config.RAISE_FRACS)}
LABEL = {INDEX[f]: label for label, f in PCT.items() if f in INDEX}
NAMES = {FOLD: "fold", CALL: "check/call", ALLIN: "all-in", **LABEL}

E_REF, CAP_REF = 0.001, 6000          # the reference: E_F <= 0.001 per spot or the spot is undetermined
E_SET, CAP_SET = 0.002, 3000          # a candidate set's own solve (the teacher's accuracy)
MAX_UNDETERMINED = 0.25               # more undetermined spots than this share of a set stop the street
MEAN_BAR, MAX_BAR = 0.005, 0.02       # acceptance on A, confirmed on B (pot units)
SPOTS = {"A": (4101, 24), "B": (4102, 16)}
STREETS = {"river": 3, "turn": 2, "flop": 1, "preflop": 0}
SPR_BUCKETS = ((0.0, 2.0, "<2"), (2.0, 8.0, "2-8"), (8.0, float("inf"), ">8"))

# The coordinator's named sets (2026-10-10 23:40): sizes per raise level, all-in always in.
SETS = {
    "river": (
        ("R1", (("33", "75", "150"), ("75",)), "thin/blocker, standard, polar overbet; one raise size"),
        ("R2", (("75", "150"), ("75",)), "R1 without the small bet"),
        ("R3", (("33", "75", "200"), ("75",)), "a bigger overbet: the teacher liked 2x pot with the nuts"),
        ("R4", (("50", "100"), ("100",)), "half pot and pot, the standard two-size set"),
        ("R5", (("33", "75", "150"), ("50",)), "R1 with a smaller raise"),
        ("R6", (("33", "66", "100", "150"), ("75",)), "four bet sizes"),
    ),
    "turn": (
        ("T1", (("50", "100"), ("66",)), "half pot and pot"),
        ("T2", (("33", "75"), ("66",)), "small and standard"),
        ("T3", (("50", "100", "150"), ("66",)), "T1 plus an overbet"),
        ("T4", (("50", "100"), ("100",)), "half pot and pot, the standard two-size set; a pot-sized raise"),
    ),
    "flop": (
        ("F1", (("33", "75"), ("66",)), "small and standard"),
        ("F2", (("33", "100"), ("66",)), "small and pot"),
        ("F3", (("33", "75"), ("50",)), "F1 with a smaller raise"),
        ("F4", (("25", "66"), ("66",)), "very small and two thirds"),
        ("F5", (("50", "100"), ("100",)), "half pot and pot, the standard two-size set; a pot-sized raise"),
    ),
    "preflop": (
        ("P1", (("min", "100"), ("100",), ("66",)), "open min or pot, 3-bet pot, 4-bet 2/3"),
        ("P2", (("min", "75"), ("100",), ("66",)), "open min or 3/4, 3-bet pot, 4-bet 2/3"),
        ("P3", (("min", "100"), ("75",), ("50",)), "open min or pot, 3-bet 3/4, 4-bet 1/2"),
    ),
}


ALL = tuple(PCT)
# Diagnostic sets (never candidates; selected by name with --only): a near-full set must lose little (the
# machinery's own scale), and adding the smallest sizes to a candidate shows how much of its loss comes
# from small bets and raises mapped to its nearest candidate in the diagnostic translation.
EXTRA = {
    "river": (
        ("Xno66", (tuple(x for x in ALL if x != "66"),), "every size but 66 %: neighbours 50 / 75 on both sides"),
        ("Xno250", (tuple(x for x in ALL if x != "250"),), "every size but 250 %: overbets between 200 % and all-in"),
        ("Xnomin", (tuple(x for x in ALL if x != "min"),), "every size but the min-raise: 1 % bets map to 25 %"),
        ("Xnomin25", (tuple(x for x in ALL if x not in ("min", "25")),), "every size but min and 25 %"),
        ("R1min", (("min", "33", "75", "150"), ("min", "75")), "R1 plus the min-raise at both levels"),
        ("R1min25", (("min", "25", "33", "75", "150"), ("min", "75")), "R1 plus min and 25 % bets, min-raise"),
    ),
}


def named_sets(street):
    return SETS[street] + EXTRA.get(street, ())


def sizes_of(levels):
    """Action indices per raise level from size labels."""
    return tuple(tuple(sorted(INDEX[PCT[s]] for s in level)) for level in levels)


def describe(sizes, cap):
    """Sizes per level as used under the cap (levels at or above the cap: all-in only)."""
    if sizes is None:
        return "every size at every level + all-in"
    out = [f"L{k} {{{', '.join(LABEL[a] for a in sizes[min(k, len(sizes) - 1)])}}}" for k in range(cap)]
    return "; ".join(out) + " + all-in"


def size_count(sizes, cap):
    """(sizes summed over the levels below the cap, distinct sizes)."""
    used = [sizes[min(k, len(sizes) - 1)] for k in range(cap)]
    return sum(len(x) for x in used), len({a for x in used for a in x})


# ------------------------------------------------------------------ compile time (jax.monitoring)

_COMPILE = [0.0]


def _on_duration(event, duration, *args, **kwargs):
    if event.startswith("/jax/core/compile"):
        _COMPILE[0] += float(duration)


jax.monitoring.register_event_duration_secs_listener(_on_duration)


def compile_seconds():
    """Seconds spent tracing, lowering and compiling in this process so far."""
    return _COMPILE[0]


# ------------------------------------------------------------------ trees under S-semantics

def s_table(s, ins):
    """Canonical candidates of one State for offered fraction indices ins bool[A].

    legal and targets come from the engine, with physical targets retained on masked columns
    for diagnostic mapping. engine maps a legal subset label to the full-grid action with the
    same chips when one exists. Fold/call/all-in and all metadata retain their engine semantics.
    """
    legal, targets = action_candidates(s, ins)
    mask_e, full_targets = action_table(s)
    match = ((targets[2:ALLIN, None] == full_targets[None, 2:ALLIN])
             & mask_e[None, 2:ALLIN])
    engine = jnp.arange(A, dtype=jnp.int32).at[2:ALLIN].set(
        jnp.where(jnp.any(match, axis=1), 2 + jnp.argmax(match, axis=1),
                  jnp.arange(2, ALLIN, dtype=jnp.int32)).astype(jnp.int32))
    return {"legal": legal, "targets": targets, "engine": engine, "open": mask_e[ALLIN], "engine_legal": mask_e,
            "level": derived.bet_level(s), "base": derived.pot(s) + derived.to_call(s),
            "minimum": derived.min_raise_to(s), "maximum": derived.max_raise_to(s)}


_S_TABLE = jax.jit(jax.vmap(s_table))


def _strip(state):
    """The State without history arrays (zero-length: decisions are counted, nothing is recorded)."""
    z = jnp.zeros(0, jnp.int32)
    return state.replace(history_player=z, history_action=z, history_amount=z, history_street=z)


def _ins_row(level, cap, sizes):
    row = np.zeros(A, bool)
    if level < cap:
        row[list(FRACTIONS) if sizes is None else list(sizes[min(level, len(sizes) - 1)])] = True
    return row


@dataclass(frozen=True, eq=False)
class Tree:
    topo: TR.Topology
    st: SV.Static
    ins: np.ndarray            # [V, A] fraction actions the tree offers at the node's raise level
    raise_level: np.ndarray    # [V] aggressive actions on the street before the node
    sizes: tuple               # None = every size
    cap: int
    allin_after_cap: bool
    name: str = ""

    @property
    def decisions(self):
        return int(len(self.topo.internal))


def build_tree(n, cap, allin_after_cap, street, sizes=None, no_open_limp=False, name=""):
    """poker/solver/tree.build_topology under S-semantics (module doc): the same deep root, order and rules."""
    if not isinstance(n, int) or not 2 <= n <= config.TREE_MAX_PLAYERS:
        raise ValueError(f"2..{config.TREE_MAX_PLAYERS} players, got {n!r}")
    if no_open_limp and street != 0:
        raise ValueError("no_open_limp is a preflop rule")
    if sizes is not None:
        sizes = tuple(tuple(sorted({int(a) for a in level})) for level in sizes)
        if not sizes or not all(level and set(level) <= set(FRACTIONS) for level in sizes):
            raise ValueError(f"sizes must be non-empty levels of fraction actions {FRACTIONS}, got {sizes}")
    sb, bb = TR.blind_seats(n)
    cpu = TR._cpu()
    with jax.default_device(cpu) if cpu is not None else contextlib.nullcontext():
        root = _strip(TR.street_root(n, street, TR.DEEP_POT, TR.deep_stack(n)))
        frontier = jax.tree.map(lambda x: np.asarray(x)[None], root)
        parent, label, actor, live, terminal, leaf, depth, raises, children, ins_rows = \
            [-1], [-1], [], [], [], [], [0], [0], [], []
        first = 0
        while frontier is not None:
            count = frontier.terminal.shape[0]
            term = np.asarray(frontier.terminal)
            at_leaf = (np.asarray(frontier.street) > street) & ~term
            done = term | at_leaf
            to_act = np.asarray(frontier.to_act)
            folded = np.asarray(frontier.folded)
            ins = np.stack([_ins_row(raises[first + j], cap, sizes) for j in range(count)])
            tab = TR._chunked(_S_TABLE, frontier, ins)
            legal, engine = np.asarray(tab["legal"]), np.asarray(tab["engine"])
            src, acts = [], []
            for j in range(count):
                i = first + j
                terminal.append(bool(done[j]))
                leaf.append(bool(at_leaf[j]))
                live.append(~folded[j])
                actor.append(-1 if done[j] else int(to_act[j]))
                ins_rows.append(ins[j])
                kids = [-1] * A
                if not done[j]:
                    level = raises[i]
                    for a in range(A):
                        if not legal[j, a]:
                            continue
                        if TR.AGGRESSIVE[a] and level >= cap and not (allin_after_cap and a == ALLIN):
                            continue
                        if no_open_limp and a == CALL and level == 0 and to_act[j] not in (sb, bb):
                            continue
                        kids[a] = len(parent)
                        parent.append(i)
                        label.append(a)
                        raises.append(level + int(TR.AGGRESSIVE[a]))
                        depth.append(depth[i] + 1)
                        src.append(j)
                        acts.append(int(engine[j, a]))
                children.append(kids)
            first += count
            if not src:
                break
            src = np.asarray(src)
            frontier = TR._chunked(TR._STEP, jax.tree.map(lambda x: x[src], frontier), np.asarray(acts, np.int32))
    children = np.asarray(children, dtype=np.int32)
    depth = np.asarray(depth)
    levels = tuple(np.flatnonzero(depth == d) for d in range(depth.max() + 1))
    topo = TR.Topology(n, np.asarray(parent, np.int32), np.asarray(label, np.int32), np.asarray(actor, np.int32),
                       children, children >= 0, np.asarray(live, bool), np.asarray(terminal, bool), levels, street,
                       np.asarray(leaf, bool))
    return Tree(topo, SV.Static(topo), np.asarray(ins_rows, bool), np.asarray(raises, np.int32), sizes, cap,
                allin_after_cap, name)


_REPLAY = {}


def _replay_fn(tree: Tree):
    if id(tree) in _REPLAY:
        return _REPLAY[id(tree)][1]
    topo = tree.topo
    parent = jnp.asarray(np.maximum(topo.parent, 0))
    label = jnp.asarray(np.maximum(topo.label, 0))
    term = jnp.asarray(topo.terminal)
    cap_ok = jnp.asarray(topo.cap_ok)
    actor = jnp.asarray(topo.actor)
    ins = jnp.asarray(tree.ins)
    street, n = topo.street, topo.n

    def one(p, s, bps, cap):
        root = _strip(TR.street_root(n, street, p, s, bps, cap))
        buffer = jax.tree.map(lambda x: jnp.broadcast_to(x, (topo.size,) + x.shape), root)

        def body(buf, v):
            prev = jax.tree.map(lambda x: x[parent[v]], buf)
            tab = s_table(prev, ins[parent[v]])
            # a size legal here steps as the engine action with its chips; a dead child (illegal in this
            # spot) steps with its own label exactly as poker/solver/tree.replay (zero reach either way)
            eng = jnp.where(tab["legal"][label[v]], tab["engine"][label[v]], label[v])
            nxt = jax.lax.cond(v == 0, lambda: prev, lambda: step(prev, eng))
            return jax.tree.map(lambda x, y: x.at[v].set(y), buf, nxt), None

        states, _ = jax.lax.scan(body, buffer, jnp.arange(topo.size))
        tab = jax.vmap(s_table)(states, ins)
        legal = tab["legal"] & cap_ok & ~term[:, None]
        alive = jnp.zeros(topo.size, bool).at[0].set(True)
        for level in topo.levels[1:]:
            par, lab = topo.parent[level], topo.label[level]
            alive = alive.at[level].set(alive[par] & legal[par, lab])
        contrib = states.bet_total
        n_live = jnp.sum(~states.folded, axis=1)
        moved_on = (states.street > street) & ~states.terminal
        kind = jnp.where(n_live == 1, TR.FOLD, jnp.where(moved_on, TR.LEAF, jnp.where(street == 3, TR.SHOWDOWN,
                                                                                         TR.ALLIN_RUNOUT)))
        boundary = states.terminal | moved_on
        return {"legal": legal, "alive": alive, "contrib": contrib, "pot": contrib.sum(-1),
                "rake": jax.vmap(rake_paid)(states), "kind": kind.astype(jnp.int32), "stack": states.stack,
                "street_bet": states.bet_street,
                "actor_ok": jnp.where(alive & ~term, states.to_act == actor, True),
                "terminal_ok": jnp.where(alive, boundary == term, True),
                "targets": tab["targets"], "open": tab["open"] & ~term, "level": tab["level"], "base": tab["base"],
                "minimum": tab["minimum"], "maximum": tab["maximum"],
                "s_only": tab["legal"] & ~tab["engine_legal"] & cap_ok & ~term[:, None]}

    fn = jax.jit(jax.vmap(one))
    _REPLAY[id(tree)] = (tree, fn)
    return fn


def replay(tree: Tree, spots):
    """Per-spot node data of `tree` on the CPU (host numpy, leading [B, V]): poker/solver/tree.replay's fields
    plus targets [B, V, A], open, level, base, minimum, maximum and s_only (sizes legal here that the
    engine masks as duplicates of an off-tree size)."""
    count = len(spots["pot"])
    args = [np.asarray(spots["pot"], np.int32), np.asarray(spots["stack"], np.int32),
            np.asarray(spots.get("rake_bps", np.zeros(count)), np.int32),
            np.asarray(spots.get("rake_cap", np.zeros(count)), np.int32)]
    cpu = TR.require_cpu("poker.solver.sizing_lab.replay")       # its fused scan has the engine replay's limit
    fn = _replay_fn(tree)
    if cpu is not None:
        out = fn(*(jax.device_put(jnp.asarray(x), cpu) for x in args))
    else:
        out = fn(*(jnp.asarray(x) for x in args))
    out = {k: np.asarray(v) for k, v in out.items()}
    if not (out["actor_ok"].all() and out["terminal_ok"].all()):
        raise AssertionError(f"{tree.name}: replay disagrees with the topology")
    return out


TREE_KEYS = ("legal", "alive", "contrib", "pot", "behind", "street_bet", "rake", "pot_chips")


def tree_batch(data, spots, base, dtype=jnp.float32):
    """Device batch of a tree from its replay `data` (poker/solver/situations.prepare's formulas for the
    tree-dependent keys); the tree-independent keys (ranges, tables, equity, rake terms) from `base`."""
    scale = jnp.asarray(spots["pot"], dtype)[:, None]
    out = {k: v for k, v in base.items() if k not in TREE_KEYS}
    out.update({"legal": jnp.asarray(data["legal"]), "alive": jnp.asarray(data["alive"]),
                "contrib": jnp.asarray(data["contrib"]).astype(dtype) / scale[:, :, None],
                "pot": jnp.asarray(data["pot"]).astype(dtype) / scale,
                "behind": jnp.asarray(data["stack"]).astype(dtype) / scale[:, :, None],
                "street_bet": jnp.asarray(data["street_bet"]).astype(dtype) / scale[:, :, None],
                "rake": jnp.asarray(data["rake"]).astype(dtype) / scale,
                "pot_chips": jnp.asarray(data["pot"], jnp.int32)})
    return out


def _check_down(board1h, pub, ranges, f, spot=None):
    return f


def oracle_critic():
    """Street ends valued by the check-down oracle, V = F (no network): every hand's chance-averaged
    all-in equity at the leaf (first-pass street-end values of turn, flop and preflop)."""
    return jax.tree_util.Partial(_check_down)


# ------------------------------------------------------------------ translation

def pseudo_harmonic(x, fs):
    """[(action, weight)] of a bet of fraction x over the candidates fs = sorted [(fraction, action)]:
    the nearest outside their range, else the two neighbours with pseudo-harmonic weights."""
    if x <= fs[0][0]:
        return [(fs[0][1], 1.0)]
    if x >= fs[-1][0]:
        return [(fs[-1][1], 1.0)]
    for (fa, ia), (fb, ib) in zip(fs, fs[1:]):
        if fa <= x <= fb:
            pa = (fb - x) * (1.0 + fa) / ((fb - fa) * (1.0 + x))
            return [(ia, pa), (ib, 1.0 - pa)] if pa < 1.0 else [(ia, 1.0)]
    raise AssertionError("unreachable")


def translate_spot(F: Tree, fd, S: Tree, sd):
    """The weighted T_S nodes of every T_F node of one spot (fd, sd: host replay data of this spot, T_F
    and T_S) and the policy map. Returns (rows, info): rows [K, 5] = (u internal index, v internal index,
    a_S, a_F, weight) for pi_F[u, :, a_F] += weight * sigma_S[v, :, a_S] (a_F = -1: dropped); info: W
    (the weighted sets), exact [V] (reached by same-index follows only, weight 1), alldead [Vi] (an
    alive T_F decision without a T_S node), lost [V] (branch weight dropped at the node) and counts."""
    cf, cs = F.topo.children, S.topo.children
    VF = F.topo.size
    W = [None] * VF
    W[0] = {0: 1.0}
    exact = np.zeros(VF, bool)
    exact[0] = True
    lost = np.zeros(VF)
    stats = {"follow": 0, "split": 0, "nearest": 0, "dead": 0}
    fl, ft, flev, fbase = fd["legal"], fd["targets"], fd["level"], fd["base"]
    sl, stg, slev, sbase = sd["legal"], sd["targets"], sd["level"], sd["base"]

    def moves(u, a, v):
        if sl[v, a]:
            stats["follow"] += 1
            return [(cs[v, a], 1.0)], True
        if a in (FOLD, CALL, ALLIN):
            stats["dead"] += 1
            return [], False
        cands = [i for i in range(2, A) if sl[v, i]]
        if not cands:
            stats["dead"] += 1
            return [], False
        x = (float(ft[u, a]) - float(flev[u])) / float(fbase[u])
        fs = sorted(((float(stg[v, i]) - float(slev[v])) / float(sbase[v]), i) for i in cands)
        out = pseudo_harmonic(x, fs)
        stats["split" if len(out) == 2 else "nearest"] += 1
        return [(cs[v, i], w) for i, w in out], False

    for u in range(VF):
        if F.topo.terminal[u] or not W[u]:
            continue
        for a in np.flatnonzero(cf[u] >= 0):
            c = cf[u, a]
            if not fl[u, a]:
                W[c] = {}                       # dead in this spot: zero reach, policy irrelevant
                continue
            out, drop, same = {}, 0.0, bool(exact[u])
            for v, w in W[u].items():
                mv, followed = moves(u, a, v)
                same = same and followed
                if not mv:
                    drop += w
                for v2, w2 in mv:
                    out[v2] = out.get(v2, 0.0) + w * w2
            total = sum(out.values())
            W[c] = {v: w / total for v, w in out.items()} if total > 0 else {}
            lost[c] = drop
            exact[c] = same and len(W[c]) == 1
    fi, si = F.st.to_internal, S.st.to_internal
    rows = []
    alldead = np.zeros(len(F.topo.internal), bool)
    for u in F.topo.internal:
        if not fd["alive"][u]:
            continue
        if not W[u]:
            alldead[fi[u]] = True
            continue
        for v, w in W[u].items():
            if S.topo.actor[v] != F.topo.actor[u]:
                raise AssertionError(f"T_F node {u} (seat {F.topo.actor[u]}) maps to T_S node {v} (seat {S.topo.actor[v]})")
            for a_s in np.flatnonzero(sl[v]):
                rows.append((fi[u], si[v], a_s, map_back(int(a_s), u, fd, F), w))
    rows = np.asarray(rows, np.float64).reshape(-1, 5)
    return rows, {"W": W, "exact": exact, "alldead": alldead, "lost": lost, "stats": stats}


def map_back(a_s, u, fd, F: Tree):
    """The T_F action at node u for T_S action a_s (module doc); -1 when the T_F state cannot take it."""
    fl = fd["legal"]
    if a_s in (FOLD, CALL):
        if not fl[u, a_s]:
            raise AssertionError(f"T_F node {u}: action {a_s} of a T_S node is not legal")
        return a_s
    if fl[u, a_s]:
        return a_s
    if a_s == ALLIN:
        return -1
    t = fd["targets"][u, a_s]
    if F.ins[u, a_s] and fd["open"][u] and t < fd["maximum"][u]:
        same = [k for k in FRACTIONS if fl[u, k] and fd["targets"][u, k] == t]
        if same:
            return same[0]
    return -1


def assemble(rows, sigma_s, fd, F: Tree):
    """pi_F [Vi, H, A] (float32) from the rows and sigma_S [Vi_S, H, A] (host), renormalised over the
    T_F-legal actions; uniform where no T_S node maps (dead or all-dead nodes). Returns (pi, dropped
    [Vi, H]: the probability mass of sigma_S the T_F state could not take, before renormalisation)."""
    vi = len(F.topo.internal)
    r = rows[rows[:, 3] >= 0]
    iu, iv, a_s, a_f = (r[:, k].astype(np.int64) for k in range(4))
    vals = r[:, 4:5] * sigma_s[iv, :, a_s].astype(np.float64)                         # [K, H]
    key = iu * A + a_f
    order = np.argsort(key, kind="stable")
    ks = key[order]
    acc = np.zeros((vi, H, A))
    if len(ks):
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        acc[ks[starts] // A, :, ks[starts] % A] = np.add.reduceat(vals[order], starts, axis=0)
    total = acc.sum(axis=-1)                                                          # [Vi, H]
    covered = np.zeros(vi, bool)
    covered[np.unique(rows[:, 0].astype(np.int64))] = True
    ok = (total > 1e-12) & covered[:, None]
    acc /= np.where(ok, total, 1.0)[:, :, None]
    if not ok.all():
        legal = fd["legal"][F.topo.internal]                                          # [Vi, A]
        uniform = legal / np.maximum(legal.sum(-1, keepdims=True), 1)
        ui, hi = np.nonzero(~ok)
        acc[ui, hi, :] = uniform[ui]
    dropped = np.where(covered[:, None], np.clip(1.0 - total, 0.0, 1.0), 0.0)
    return acc.astype(np.float32), dropped.astype(np.float32)


# ------------------------------------------------------------------ the own-sizing metric (no translation)

def asymmetric_legal(F: Tree, fd, sizes, cap, seat):
    """[B, V, A] legal actions of the asymmetric game on the full tree: at `seat`'s decisions fold,
    check-call, all-in and the sizes of `sizes` at the node's raise level only (each at its own chips: a
    size the full grid masks as a duplicate - clamped to the minimum raise like the lower min-raise size -
    is that lower size's action), every size of the grid at the other seat's decisions. sizes=None: the
    full game's legal actions unchanged. Same topology as the full tree: no translation anywhere."""
    legal = fd["legal"].copy()
    if sizes is None:
        return legal
    count = legal.shape[0]
    for u in F.topo.internal:
        if F.topo.actor[u] != seat:
            continue
        k = int(F.raise_level[u])
        keep = np.zeros((count, A), bool)
        keep[:, [FOLD, CALL, ALLIN]] = True
        if k < cap:
            for a in sizes[min(k, len(sizes) - 1)]:
                t = fd["targets"][:, u, a]
                ok = fd["open"][:, u] & (t < fd["maximum"][:, u])
                same = fd["legal"][:, u, 2:ALLIN] & (fd["targets"][:, u, 2:ALLIN] == t[:, None])
                has = ok & same.any(axis=1)
                keep[np.flatnonzero(has), 2 + np.argmax(same, axis=1)[has]] = True
        legal[:, u] &= keep
    return legal


# ------------------------------------------------------------------ solving and evaluation

def diagnostics(z, spot, dropped, alldead, st):
    """Reach-weighted quantities of policy z on one spot: freq [Vi, A] = the probability (over deals)
    that the actor reaches the node and takes the action; the reach-weighted mass of dropped probability
    and of all-dead nodes, and the largest own reach at an all-dead node."""
    legal = spot["legal"][st.internal]
    pi = SV.policy(z, legal)
    reach = SV.forward(pi, spot["ranges"], st)
    den = SV.opponent_mass(reach, spot.get("tables"), st)                             # [Vi, H]
    own = reach[st.internal, st.actor[st.internal], :]
    joint = own * den
    zn = jnp.sum(spot["ranges"][0] * SV.opponent_mass_root(reach, spot.get("tables"), st))
    freq = jnp.einsum("vh,vha->va", joint, pi) / zn
    return {"freq": freq, "dropped": jnp.sum(joint * dropped) / zn,
            "alldead": jnp.sum(joint * alldead[:, None]) / zn,
            "alldead_reach_max": jnp.max(jnp.where(alldead[:, None], own, 0.0))}


class Evaluator:
    """Compiled evaluations on one tree (exploitability and diagnostics), in fixed chunks."""

    def __init__(self, tree: Tree, cfg, aux, chunk):
        self.tree, self.chunk = tree, chunk
        st = tree.st
        self._ex = jax.jit(jax.vmap(lambda z, spot: SV.exploitability(z, spot, st, cfg, aux)))
        self._diag = jax.jit(jax.vmap(lambda z, spot, d, a: diagnostics(z, spot, d, a, st)))

    def _chunks(self, count):
        for lo in range(0, count, self.chunk):
            idx = np.arange(lo, min(count, lo + self.chunk))
            yield idx, np.concatenate([idx, np.full(self.chunk - len(idx), idx[-1])])

    def exploit(self, z, batch, sel):
        """{value [k, 2], best [k, 2], e [k]} of policies z [k, Vi, H, A] (host) on the spots `sel` of
        batch (checked with SV.check_exploitability)."""
        out = {"value": [], "best": [], "e": []}
        for idx, pad in self._chunks(len(sel)):
            part = jax.tree.map(lambda x: x[sel[pad]], batch)
            res = self._ex(jnp.asarray(z[pad]), part)
            res = SV.check_exploitability({k: np.asarray(v, np.float64)[:len(idx)] for k, v in res.items()})
            out["value"].append(res["value"])
            out["best"].append(res["best"])
            out["e"].append(res["nash_conv"] / 2.0)
        return {k: np.concatenate(v) for k, v in out.items()}

    def diag(self, z, batch, sel, dropped, alldead):
        outs = []
        for idx, pad in self._chunks(len(sel)):
            part = jax.tree.map(lambda x: x[sel[pad]], batch)
            res = self._diag(jnp.asarray(z[pad]), part, jnp.asarray(dropped[pad]), jnp.asarray(alldead[pad]))
            outs.append({k: np.asarray(v, np.float64)[:len(idx)] for k, v in res.items()})
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def solve_tree(tree: Tree, batch, cfg, chunk, aux, log, tag, keep_z=True):
    """SV.solve in chunks of `chunk` spots (the last padded with its last spot). Returns z [B, Vi, H, A]
    (host float32; None unless keep_z), the solver's E and iterations per spot, wall and compile seconds,
    and the E trajectory of every spot ({iteration: E} at every check of the solver, from its log)."""
    count = batch["ranges"].shape[0]
    zs, es, us, trajs = [], [], [], []
    wall = comp = 0.0
    for lo in range(0, count, chunk):
        idx = np.arange(lo, min(count, lo + chunk))
        pad = np.concatenate([idx, np.full(chunk - len(idx), idx[-1])])
        part = jax.tree.map(lambda x: x[pad], batch)
        traj = [dict() for _ in idx]
        seen = np.full(len(idx), -1)

        def on_check(info, traj=traj, seen=seen):
            u, e = np.asarray(info["u"])[:len(traj)], np.asarray(info["e"], np.float64)[:len(traj)]
            for j in range(len(traj)):
                if u[j] != seen[j] and np.isfinite(e[j]):
                    traj[j][int(u[j])] = float(e[j])
                    seen[j] = u[j]

        c0, t0 = compile_seconds(), time.perf_counter()
        res = SV.solve(part, tree.st, cfg, log=on_check, aux=aux)
        z = jax.block_until_ready(res["z"])
        dt, dc = time.perf_counter() - t0, compile_seconds() - c0
        wall += dt
        comp += dc
        if keep_z:
            zs.append(np.asarray(z)[:len(idx)])
        es.append(np.asarray(res["e"], np.float64)[:len(idx)])
        us.append(np.asarray(res["u"])[:len(idx)])
        trajs += traj
        log(f"  {tag} spots {lo}-{lo + len(idx) - 1}: iterations {us[-1].tolist()}, E max {es[-1].max():.5f}, "
            f"{dt:.0f} s (compile {dc:.0f} s)")
    return (np.concatenate(zs) if keep_z else None), np.concatenate(es), np.concatenate(us), wall, comp, trajs


TRAJECTORY_AT = (25, 50, 100, 200, 400, 800, 1600)


def trajectory_record(trajs, stops, e_stop):
    """Per spot: E at TRAJECTORY_AT (the nearest checked iteration where a check does not fall on it:
    "nearest" names it), at the stop, and the first checked iteration with E <= 0.002 and <= 0.001
    (None: not reached in this solve)."""
    out = []
    for traj, u, e in zip(trajs, stops, e_stop):
        its = np.asarray(sorted(traj)) if traj else np.zeros(0, int)
        at, nearest = {}, {}
        for k in TRAJECTORY_AT:
            if not len(its) or k > its[-1]:
                continue
            j = int(its[np.argmin(np.abs(its - k))])
            at[str(k)] = traj[j]
            if j != k:
                nearest[str(k)] = j
        first = lambda thr: next((int(i) for i in its if traj[i] <= thr), None)
        out.append({"e_at": at, "nearest_check": nearest, "stop": {"iteration": int(u), "e": float(e)},
                    "iters_to_0.002": first(0.002), "iters_to_0.001": first(0.001),
                    "checks": {str(int(i)): traj[i] for i in its}})
    return out


def trajectory_summary(records, seconds, count):
    """Medians and maxima over spots of a trajectory record list; GPU seconds per spot (solve, no compile)."""
    def stat(values):
        v = [x for x in values if x is not None]
        return {"n": len(v), "median": float(np.median(v)) if v else None, "p90": float(np.percentile(v, 90)) if v else None,
                "max": float(np.max(v)) if v else None}
    out = {"seconds_per_spot": seconds / max(count, 1),
           "iters_to_0.002": {**stat([r["iters_to_0.002"] for r in records]),
                              "not_reached": sum(r["iters_to_0.002"] is None for r in records)},
           "iters_to_0.001": {**stat([r["iters_to_0.001"] for r in records]),
                              "not_reached": sum(r["iters_to_0.001"] is None for r in records)},
           "e_at": {str(k): stat([r["e_at"].get(str(k)) for r in records]) for k in TRAJECTORY_AT}}
    nearest = sorted({f"{k}->{v}" for r in records for k, v in r["nearest_check"].items()})
    if nearest:
        out["nearest_check"] = nearest
    return out


def host_spot(data, b):
    return {k: v[b] for k, v in data.items()}


def spr_of(spots):
    return np.asarray(spots["stack"], np.float64) / np.asarray(spots["pot"], np.float64)


def freq_tables(freq, tree: Tree):
    """Reach-weighted mass per seat (the actor) and raise level of every action from freq [B, Vi, A]
    (probability over deals that the actor reaches the node and takes the action), averaged over the
    spots: {seat: {level: {"mass": {action: p}, "share_of_aggressive": {action: share}}}}."""
    internal = tree.topo.internal
    actors = tree.topo.actor[internal]
    levels = tree.raise_level[internal]
    out = {}
    for seat in sorted(set(actors.tolist())):
        out[str(seat)] = {}
        for k in sorted(set(levels.tolist())):
            m = (actors == seat) & (levels == k)
            if not m.any():
                continue
            mass = freq[:, m, :].sum(axis=1).mean(axis=0)                               # [A]
            aggr = mass[2:].sum()
            out[str(seat)][str(k)] = {
                "mass": {NAMES[a]: float(mass[a]) for a in range(A) if mass[a] > 0},
                "share_of_aggressive": {NAMES[a]: float(mass[a] / aggr) for a in range(2, A) if aggr > 0 and mass[a] > 0}}
    return out


_FINGERPRINT = {}


def fingerprint():
    """sha256 prefixes of the sources this process runs (read once: later edits do not relabel a run)."""
    if not _FINGERPRINT:
        _FINGERPRINT.update(_fingerprint())
    return dict(_FINGERPRINT)


def _fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = ("poker/envs/config.py", "poker/envs/actions.py", "poker/envs/transition.py", "poker/envs/derived.py", "poker/envs/betting.py",
             "poker/solver/tree.py", "poker/solver/pcfr.py", "poker/solver/situations.py", "poker/solver/showdown.py",
             "poker/solver/checkdown.py", "poker/solver/sizing_lab.py")
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in files}


def write_new(path: Path, payload):
    """Write JSON to a new file; never overwrite an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        json.dump(payload, fh, indent=1)


def loss_stats(loss, spr, seat):
    """mean / p90 / max of per-spot losses, by SPR bucket, and the per-seat parts."""
    out = {"n": int(len(loss)), "mean": float(np.mean(loss)), "p90": float(np.percentile(loss, 90)),
           "max": float(np.max(loss)), "min": float(np.min(loss)), "by_spr": {}}
    for lo, hi, name in SPR_BUCKETS:
        m = (spr >= lo) & (spr < hi)
        out["by_spr"][name] = ({"n": int(m.sum()), "mean": float(np.mean(loss[m])), "max": float(np.max(loss[m]))}
                               if m.any() else {"n": 0})
    out["seat_mean"] = [float(np.mean(seat[:, i])) for i in range(seat.shape[1])]
    out["seat_max"] = [float(np.max(seat[:, i])) for i in range(seat.shape[1])]
    return out


# ------------------------------------------------------------------ spot sets

REAL_RIVER = Path("results/audit_policy/task3_ranges_river_spots.npz")    # 400 river starts of the k0 Slumbot match


def real_river_spots(count=48):
    """Set C: real river starts of k0's Slumbot match (the policy audit's extraction: board, pot, stack and the
    bot's tracked public ranges of both seats at the river start; results/audit_policy/task3_ranges_river_spots.npz,
    400 hands), count of them spread evenly over SPR (sorted
    by stack / pot, positions linspace(0, 399, count) rounded). Ranges restricted to the board and normalised. No
    rake: regime 1, Slumbot's table."""
    from poker.solver import cards as C
    d = np.load(REAL_RIVER)
    order = np.argsort(d["stack"] / d["pot"], kind="stable")
    sel = order[np.linspace(0, len(order) - 1, count).round().astype(int)]
    board = d["board"][sel].astype(np.int32)
    valid = C.board_valid(board)
    ranges = d["ranges"][sel].astype(np.float32) * valid[:, None, :]
    ranges = ranges / np.maximum(ranges.sum(-1, keepdims=True), 1e-30)
    zeros = np.zeros(count, np.int32)
    return {"board": board, "pot": d["pot"][sel].astype(np.int32), "stack": d["stack"][sel].astype(np.int32),
            "valid": valid, "ranges": ranges.astype(np.float32), "street": np.full(count, 3, np.int32),
            "rake_level": np.zeros(count, np.float32), "n_dealt": np.full(count, 2, np.int32),
            "rake_bps": zeros, "rake_cap": zeros.copy(), "hand_file": d["hand_file"][sel].astype(np.int64)}


# ------------------------------------------------------------------ the lab of one street

class Lab:
    """One street: the full tree (the reference), the candidate trees and their evaluations."""

    def __init__(self, street_name, out: Path, log, chunk_full=8, chunk_set=8, only=None, count=None,
                 cost_sets=("A", "C")):
        self.name, self.street, self.out, self.log = street_name, STREETS[street_name], out, log
        self.cost_sets = tuple(cost_sets)      # candidates also solved to E_REF there (iterations, seconds)
        self.count = count                     # spots per lab set instead of SPOTS (the preflop has one spot)
        self.chunk_full, self.chunk_set = chunk_full, chunk_set
        self.aux = None if self.street == 3 else oracle_critic()
        self.cap = config.cap_for(2, self.street)                # the decided cap (config.TREE_CAPS), every street
        t0 = time.perf_counter()
        self.F = build_tree(2, self.cap, True, self.street, None, False, "full")
        self.cands = [(name, build_tree(2, self.cap, True, self.street, sizes_of(levels), False, name), levels, why)
                      for name, levels, why in (SETS[street_name] if only is None else named_sets(street_name))
                      if only is None or name in only]
        if only is not None and {c[0] for c in self.cands} != set(only):
            raise ValueError(f"unknown sets {sorted(set(only) - {c[0] for c in self.cands})} for {street_name}")
        self.tag = ""                          # suffix of the candidates' json names (diagnostic reruns)
        self.log(f"{street_name}: cap {self.cap} + all-in; full tree {self.F.topo.size} nodes ({self.F.decisions} "
                 f"decisions); " + ", ".join(f"{n} {t.topo.size} ({t.decisions})" for n, t, _, _ in self.cands)
                 + f" ({time.perf_counter() - t0:.0f} s)")
        self.cfg_ref = SV.SolverConfig(cap=CAP_REF, e_target=E_REF, check_every=50)
        self.cfg_set = SV.SolverConfig(cap=CAP_SET, e_target=E_SET, check_every=50)
        self.cfg_own = SV.SolverConfig(cap=CAP_REF, e_target=E_SET, check_every=50)
        self.evalF = Evaluator(self.F, self.cfg_ref, self.aux, chunk_full)
        self.ref = {}

    # spots ----------------------------------------------------------------
    def spots(self, which):
        if which == "C":
            if self.street != 3:
                raise ValueError("set C (real spots) exists for the river only")
            spots = real_river_spots()
        elif self.street == 0:
            seed, count = SPOTS[which]
            spots = SI.make_preflop_spots(seed, self.count or count, 2, rake="mix")
        else:
            seed, count = SPOTS[which]
            count = self.count or count
            if self.street == 3:
                spots = SI.make_spots(seed, count, 2, spr_range=(0.5, 50.0), rake="none", street=3)
            else:
                from poker.solver import loop as LP
                spots = SI.make_spots(seed, count, 2, spr_range=(0.5, 50.0), rake="none", street=self.street,
                                      pot_bb=LP.POTS[self.name], board_pool=LP.FLOP_BOARDS if self.street == 1 else 0)
        order = np.argsort(spr_of(spots), kind="stable")             # batches of similar difficulty
        return {k: v[order] for k, v in spots.items()}

    def base_batch(self, spots):
        """The official batch of the full tree (poker/solver/situations.prepare) and the lab's replay of it,
        checked equal on every tree-dependent field."""
        prep = {k: v for k, v in spots.items() if k != "hand_file"}
        equity = None
        if self.street == 0:
            from poker.solver import loop as LP
            from poker.solver import runout as RO
            if not Path(LP.PREFLOP_EQUITY).exists():
                raise FileNotFoundError(f"{LP.PREFLOP_EQUITY}: the lab never writes results/cache")
            equity = RO.preflop_equity_cached(LP.PREFLOP_EQUITY)
        if self.street == 1:
            from poker.solver import runout as RO
            for flop in np.unique(np.sort(spots["board"][:, :3], axis=1), axis=0):
                path = Path(RO.FLOP_CACHE) / f"{flop[0]}_{flop[1]}_{flop[2]}.npy"
                if not path.exists():
                    raise FileNotFoundError(f"{path}: the lab never writes results/cache")
        base = SI.prepare(prep, self.F.topo, preflop_equity=equity)
        fd = replay(self.F, spots)
        mine = tree_batch(fd, spots, base)
        for k in TREE_KEYS:
            if not np.array_equal(np.asarray(mine[k]), np.asarray(base[k])):
                raise AssertionError(f"full tree: the lab's replay differs from situations.prepare on {k}")
        return base, fd

    # the reference -----------------------------------------------------------
    def reference(self, which, spots):
        path = self.out / f"full_{which}.json"
        base, fd = self.base_batch(spots)
        if path.exists():
            ref = json.loads(path.read_text())
            self.log(f"reference {which}: loaded {path} ({ref['spots']} spots, {ref['undetermined']} undetermined)")
            self.ref[which] = ref
            return base, fd, ref
        count = len(spots["pot"])
        spr = spr_of(spots)
        e_f, best, value = np.zeros(count), np.zeros((count, 2)), np.zeros((count, 2))
        iters, e_solver = np.zeros(count, np.int64), np.zeros(count)
        freq = np.zeros((count, self.F.decisions, A))
        sanity = {"loss_max_abs": 0.0, "exact_nodes": 0, "alive_decisions": 0, "exact_max_diff": 0.0,
                  "weight_sum_max_err": 0.0}
        wall = comp = 0.0
        trajs = []
        t_all = time.perf_counter()
        internal = self.F.topo.internal
        for lo in range(0, count, self.chunk_full):
            idx = np.arange(lo, min(count, lo + self.chunk_full))
            batch = jax.tree.map(lambda x: x[idx], base)
            z, es, us, dt, dc, tr = solve_tree(self.F, batch, self.cfg_ref, self.chunk_full, self.aux, self.log,
                                               f"full {which}")
            wall, comp = wall + dt, comp + dc
            trajs += trajectory_record(tr, us, es)
            ex = self.evalF.exploit(z, base, idx)
            e_f[idx], best[idx], value[idx], iters[idx], e_solver[idx] = ex["e"], ex["best"], ex["value"], us, es
            dg = self.evalF.diag(z, base, idx, np.zeros((len(idx), self.F.decisions, H), np.float32),
                                 np.zeros((len(idx), self.F.decisions), bool))
            freq[idx] = dg["freq"]
            # sanity: S = full set -> the translation is the identity and the loss 0 up to float noise
            sig = np.asarray(jax.vmap(SV.policy)(jnp.asarray(z), jnp.asarray(fd["legal"][idx][:, internal])))
            pis = []
            for j, b in enumerate(idx):
                fdb = host_spot(fd, b)
                rows, info = translate_spot(self.F, fdb, self.F, fdb)
                pi, dropped = assemble(rows, sig[j], fdb, self.F)
                pis.append(pi)
                alive = fd["alive"][b][internal]
                ex_nodes = info["exact"][internal] & alive
                sanity["exact_nodes"] += int(ex_nodes.sum())
                sanity["alive_decisions"] += int(alive.sum())
                sanity["exact_max_diff"] = max(sanity["exact_max_diff"], float(np.abs(pi[ex_nodes] - sig[j][ex_nodes]).max()))
                sums = [abs(sum(w.values()) - 1.0) for w in info["W"] if w]
                sanity["weight_sum_max_err"] = max(sanity["weight_sum_max_err"], max(sums))
                sanity["dropped_max"] = max(sanity.get("dropped_max", 0.0), float(dropped.max()))
                if ex_nodes.sum() != alive.sum() or float(dropped.max()) > 1e-5:
                    raise AssertionError(f"S = full: spot {b}: {int(alive.sum() - ex_nodes.sum())} alive decisions "
                                         f"reached by translation, dropped mass {float(dropped.max()):.2e}")
            zt = np.log(np.maximum(np.stack(pis), 1e-30))
            et = self.evalF.exploit(zt, base, idx)["e"]
            sanity["loss_max_abs"] = max(sanity["loss_max_abs"], float(np.abs(et - ex["e"]).max()))
            self.log(f"  full {which} spots {lo}-{idx[-1]}: E_F {np.round(ex['e'], 5).tolist()}; S = full sanity: "
                     f"|loss| max {sanity['loss_max_abs']:.2e}, translated == sigma_F within {sanity['exact_max_diff']:.1e}")
        undetermined = e_f > E_REF
        alive_int = fd["alive"][:, internal]
        dec = alive_int & self.F.ins[internal][None, :, 2] & fd["open"][:, internal]
        minraise = (fd["targets"][:, internal, 2] == fd["minimum"][:, internal]) & dec
        ref = {"street": self.name, "set": which, "spots": count, "cap": self.cap, "e_ref": E_REF, "cap_ref": CAP_REF,
               "tree": {"nodes": self.F.topo.size, "decisions": self.F.decisions},
               "undetermined": int(undetermined.sum()),
               "e_f": e_f.tolist(), "e_solver": e_solver.tolist(), "iterations": iters.tolist(),
               "best": best.tolist(), "value": value.tolist(), "spr": spr.tolist(),
               "pot_bb": (np.asarray(spots["pot"]) / config.BB).tolist(),
               "stack_bb": (np.asarray(spots["stack"]) / config.BB).tolist(),
               "board": np.asarray(spots["board"]).tolist(),
               "hand_file": spots["hand_file"].tolist() if "hand_file" in spots else None,
               "solve_seconds": wall - comp, "compile_seconds": comp, "wall_seconds": time.perf_counter() - t_all,
               "min_raise_share": {"decision_nodes_with_min_raise_open": int(dec.sum()),
                                   "share_clamped_to_min_raise": float(minraise.sum() / max(dec.sum(), 1))},
               "mass": freq_tables(freq, self.F), "sanity_full_set": sanity,
               "trajectory": trajs, "cost": trajectory_summary(trajs, wall - comp, count),
               "check_every": self.cfg_ref.check_every, "fingerprint": fingerprint()}
        write_new(path, ref)
        self.log(f"reference {which}: {count} spots, E_F median {np.median(e_f):.5f} max {e_f.max():.5f}, "
                 f"undetermined {int(undetermined.sum())}, solve {wall - comp:.0f} s + compile {comp:.0f} s")
        self.ref[which] = ref
        return base, fd, ref

    # own sizing ---------------------------------------------------------------
    def _own_pass(self, name, which, base, fd, legal_seat, cfg, idx_all):
        """Both asymmetric games (seat 0 restricted, seat 1 restricted) solved chunk by chunk; our strategy =
        seat 0's from the first and seat 1's from the second, evaluated in the full game."""
        count = len(idx_all)
        internal = self.F.topo.internal
        ours0 = (self.F.topo.actor[internal] == 0)[None, :, None, None]
        out = {"e": np.zeros(count), "best": np.zeros((count, 2)), "value": np.zeros((count, 2)),
               "e_asym": np.zeros((count, 2)), "iters": np.zeros((count, 2), np.int64),
               "freq": np.zeros((count, self.F.decisions, A)), "traj": [[None, None] for _ in range(count)],
               "solve": 0.0, "compile": 0.0}
        for lo in range(0, count, self.chunk_full):
            idx = idx_all[lo:lo + self.chunk_full]
            zs = []
            for seat in (0, 1):
                part = jax.tree.map(lambda x: x[idx], {**base, "legal": jnp.asarray(legal_seat[seat])})
                z, e, u, wall, comp, tr = solve_tree(self.F, part, cfg, self.chunk_full, self.aux, self.log,
                                                     f"{name} {which} own seat {seat}")
                out["solve"] += wall - comp
                out["compile"] += comp
                out["e_asym"][lo:lo + len(idx), seat], out["iters"][lo:lo + len(idx), seat] = e, u
                for j, rec in enumerate(trajectory_record(tr, u, e)):
                    out["traj"][lo + j][seat] = rec
                zs.append(z)
            z = np.where(ours0, zs[0], zs[1])
            del zs
            ex = self.evalF.exploit(z, base, idx)
            sl = slice(lo, lo + len(idx))
            out["e"][sl], out["best"][sl], out["value"][sl] = ex["e"], ex["best"], ex["value"]
            dg = self.evalF.diag(z, base, idx, np.zeros((len(idx), self.F.decisions, H), np.float32),
                                 np.zeros((len(idx), self.F.decisions), bool))
            out["freq"][sl] = dg["freq"]
        return out

    def evaluate_own(self, cand, which, spots, base, fd, ref, refine=True):
        """loss_own(S) = E_full(sigma_us) - E_F: sigma_us plays S at its own decisions (the asymmetric
        games, solved to E_SET in their own tree), the best response has every size; no translation.
        A set whose loss lands within 0.002 of the bar (failing it) is re-solved to E_REF (the solve's own
        E only overestimates the loss: a pass at E_SET is a true pass)."""
        name, S, levels, why = cand
        path = self.out / "sets" / f"{name}_{which}_own{self.tag}.json"
        if path.exists():
            self.log(f"{name} {which} own: loaded {path}")
            return json.loads(path.read_text())
        t_all = time.perf_counter()
        count = len(spots["pot"])
        legal_seat = {seat: asymmetric_legal(self.F, fd, S.sizes, self.cap, seat) for seat in (0, 1)}
        e_f = np.asarray(ref["e_f"])
        det = e_f <= E_REF
        spr = spr_of(spots)
        gain_f = np.asarray(ref["best"]) - np.asarray(ref["value"])
        passes_ = []
        for cfg in (self.cfg_own, self.cfg_ref):
            r = self._own_pass(name, which, base, fd, legal_seat, cfg, np.arange(count))
            loss = r["e"] - e_f
            seat = ((r["best"] - r["value"]) - gain_f) / 2.0
            st = loss_stats(loss[det], spr[det], seat[det]) if det.any() else None
            passes_.append({"e_target": cfg.e_target, "loss": loss, "seat": seat, "stats": st, "run": r})
            self.log(f"{name} {which} own (asymmetric solves to {cfg.e_target}): loss mean {st['mean']:.5f} p90 "
                     f"{st['p90']:.5f} max {st['max']:.5f}; seat parts {np.round(st['seat_mean'], 5).tolist()}; E_asym "
                     f"max {r['e_asym'].max():.5f}; solve {r['solve']:.0f} s + compile {r['compile']:.0f} s")
            near = (not passes(st) and st["mean"] <= MEAN_BAR + 0.002 and st["max"] <= MAX_BAR + 0.002)
            if not (refine and near and cfg is self.cfg_own):
                break
        final = passes_[-1]
        r = final["run"]
        n_sizes, n_distinct = size_count(S.sizes, self.cap)
        res = {"name": name, "set": which, "street": self.name, "metric": "own (asymmetric games, no translation)",
               "levels": [list(x) for x in levels], "why": why, "describe": describe(S.sizes, self.cap),
               "sizes": n_sizes, "distinct_sizes": n_distinct, "tree": {"nodes": S.topo.size, "decisions": S.decisions},
               "spots": count, "determined": int(det.sum()), "e_target_final": final["e_target"],
               "loss": final["loss"].tolist(), "seat_loss": final["seat"].tolist(), "e_full": r["e"].tolist(),
               "e_f": e_f.tolist(), "e_asym": r["e_asym"].tolist(), "iterations_asym": r["iters"].tolist(),
               "spr": spr.tolist(), "stats": final["stats"],
               "passes": [{"e_target": x["e_target"], "stats": x["stats"], "loss": x["loss"].tolist(),
                           "solve_seconds": x["run"]["solve"], "compile_seconds": x["run"]["compile"]} for x in passes_],
               "mass_ours": freq_tables(r["freq"], self.F),
               "trajectory_asym": r["traj"],
               "cost_asym": {f"seat{seat}": trajectory_summary([t[seat] for t in r["traj"]], r["solve"] / 2, count)
                             for seat in (0, 1)},
               "seconds": {"solve": sum(x["run"]["solve"] for x in passes_),
                           "compile": sum(x["run"]["compile"] for x in passes_), "wall": time.perf_counter() - t_all},
               "fingerprint": fingerprint()}
        write_new(path, res)
        return res

    # cost only ---------------------------------------------------------------
    def cost(self, which, spots):
        """Every tree (full and candidates) solved to E_REF on `which` with the E trajectory: iterations to
        0.002 / 0.001, E at TRAJECTORY_AT, GPU seconds per spot. No translation, no loss (for a set whose
        loss run did not record trajectories). Writes cost_<tree>_<set>.json (new files)."""
        base, fd = self.base_batch(spots)
        count = len(spots["pot"])
        out = {}
        for name, tree, chunk in [("full", self.F, self.chunk_full)] + [(n, t, self.chunk_set) for n, t, _, _ in self.cands]:
            path = self.out / f"cost_{name}_{which}.json"
            if path.exists():
                out[name] = json.loads(path.read_text())
                self.log(f"cost {name} {which}: loaded {path}")
                continue
            batch = base if tree is self.F else tree_batch(replay(tree, spots), spots, base)
            _, e, u, wall, comp, tr = solve_tree(tree, batch, self.cfg_ref, chunk, self.aux, self.log,
                                                 f"cost {name} {which}", keep_z=False)
            rec = trajectory_record(tr, u, e)
            res = {"tree": name, "set": which, "street": self.name, "spots": count, "e_target": E_REF, "cap": CAP_REF,
                   "check_every": self.cfg_ref.check_every, "decisions": tree.decisions, "solve_seconds": wall - comp,
                   "compile_seconds": comp, "summary": trajectory_summary(rec, wall - comp, count), "trajectory": rec,
                   "spr": spr_of(spots).tolist(), "fingerprint": fingerprint()}
            write_new(path, res)
            sm = res["summary"]
            self.log(f"cost {name} {which}: {sm['seconds_per_spot']:.2f} s/spot, iterations to 0.002 median "
                     f"{sm['iters_to_0.002']['median']} (not reached {sm['iters_to_0.002']['not_reached']}), to 0.001 "
                     f"median {sm['iters_to_0.001']['median']} (not reached {sm['iters_to_0.001']['not_reached']}), "
                     f"E at 100 median {sm['e_at']['100']['median']}")
            out[name] = res
        return out

    def restrictions(self, S: Tree):
        """[V, A] allowed actions of the best responses in T_F for the loss decomposition: no_small - no
        fraction below S's smallest size of that raise level; no_large - none above its largest (all-in
        stays); s_only - the fractions of S only. The translated policy's own actions stay allowed (in
        evaluate), so it is unchanged and only the best responses lose options."""
        frac = {2 + i: n / d for i, (n, d) in enumerate(config.RAISE_FRACS)}
        out = {r: np.ones((self.F.topo.size, A), bool) for r in ("no_small", "no_large", "s_only")}
        for u in range(self.F.topo.size):
            k = int(self.F.raise_level[u])
            if k >= self.cap:
                continue
            level = S.sizes[min(k, len(S.sizes) - 1)]
            lo, hi = min(frac[a] for a in level), max(frac[a] for a in level)
            for a in FRACTIONS:
                out["no_small"][u, a] = frac[a] >= lo
                out["no_large"][u, a] = frac[a] <= hi
                out["s_only"][u, a] = a in level
        return out

    # a candidate -------------------------------------------------------------
    def evaluate(self, cand, which, spots, base, fd, ref):
        name, S, levels, why = cand
        path = self.out / "sets" / f"{name}_{which}{self.tag}.json"
        if path.exists():
            res = json.loads(path.read_text())
            self.log(f"{name} {which}: loaded {path}")
            return res
        count = len(spots["pot"])
        t_all = time.perf_counter()
        c0, t0 = compile_seconds(), time.perf_counter()
        sd = replay(S, spots)
        batch = tree_batch(sd, spots, base)
        t_replay, c_replay = time.perf_counter() - t0, compile_seconds() - c0
        z, e_s, u_s, wall, comp, tr = solve_tree(S, batch, self.cfg_set, self.chunk_set, self.aux, self.log,
                                                 f"{name} {which}")
        traj_s = trajectory_record(tr, u_s, e_s)
        cost = None
        if which in self.cost_sets:
            _, e1, u1, w1, c1, tr1 = solve_tree(S, batch, self.cfg_ref, self.chunk_set, self.aux, self.log,
                                                f"{name} {which} to {E_REF}", keep_z=False)
            rec1 = trajectory_record(tr1, u1, e1)
            cost = {"e_target": E_REF, "cap": CAP_REF, "solve_seconds": w1 - c1, "compile_seconds": c1,
                    "summary": trajectory_summary(rec1, w1 - c1, count), "trajectory": rec1}
        internal_s = S.topo.internal
        evalS = Evaluator(S, self.cfg_set, self.aux, self.chunk_set)
        c0, t0 = compile_seconds(), time.perf_counter()
        dgS = evalS.diag(z, batch, np.arange(count), np.zeros((count, S.decisions, H), np.float32),
                         np.zeros((count, S.decisions), bool))
        s_only_nodes = sd["s_only"][:, internal_s].any(-1) & sd["alive"][:, internal_s]
        s_only_mass = float(np.mean(np.sum(dgS["freq"] * sd["s_only"][:, internal_s], axis=(1, 2))))
        t_diag_s, c_diag_s = time.perf_counter() - t0, compile_seconds() - c0
        e_t, best, value = np.zeros(count), np.zeros((count, 2)), np.zeros((count, 2))
        restr = self.restrictions(S)
        legal_np = np.asarray(base["legal"])
        rest = {r: {"e": np.zeros(count), "best": np.zeros((count, 2))} for r in restr}
        dropped_mass, alldead_mass, alldead_reach = np.zeros(count), np.zeros(count), np.zeros(count)
        stats = {"follow": 0, "split": 0, "nearest": 0, "dead": 0}
        lost_max = 0.0
        t_trans = t_eval = c_eval = 0.0
        for lo in range(0, count, self.chunk_full):
            idx = np.arange(lo, min(count, lo + self.chunk_full))
            t0 = time.perf_counter()
            pis, drops, deads = [], [], []
            # sigma_S of this chunk only (a whole set of a near-full tree is ~0.6 GB of pinned host memory)
            sig = np.asarray(jax.vmap(SV.policy)(jnp.asarray(z[idx]), jnp.asarray(sd["legal"][idx][:, internal_s])))
            for j, b in enumerate(idx):
                fdb, sdb = host_spot(fd, b), host_spot(sd, b)
                rows, info = translate_spot(self.F, fdb, S, sdb)
                pi, dropped = assemble(rows, sig[j], fdb, self.F)
                pis.append(pi)
                drops.append(dropped)
                deads.append(info["alldead"])
                for k in stats:
                    stats[k] += info["stats"][k]
                lost_max = max(lost_max, float(info["lost"].max()))
            zt = np.log(np.maximum(np.stack(pis), 1e-30))
            t_trans += time.perf_counter() - t0
            c0, t0 = compile_seconds(), time.perf_counter()
            ex = self.evalF.exploit(zt, base, idx)
            dg = self.evalF.diag(zt, base, idx, np.stack(drops), np.stack(deads))
            t_eval += time.perf_counter() - t0
            c_eval += compile_seconds() - c0
            e_t[idx], best[idx], value[idx] = ex["e"], ex["best"], ex["value"]
            dropped_mass[idx], alldead_mass[idx], alldead_reach[idx] = dg["dropped"], dg["alldead"], dg["alldead_reach_max"]
            c0, t0 = compile_seconds(), time.perf_counter()
            used = np.zeros((len(idx), self.F.topo.size, A), bool)
            used[:, self.F.topo.internal] = np.stack([pi.max(axis=1) > 1e-9 for pi in pis])
            for r, mask in restr.items():
                legal_r = legal_np.copy()
                legal_r[idx] = legal_np[idx] & (mask[None] | used)
                exr = self.evalF.exploit(zt, {**base, "legal": jnp.asarray(legal_r)}, idx)
                rest[r]["e"][idx], rest[r]["best"][idx] = exr["e"], exr["best"]
            t_eval += time.perf_counter() - t0
            c_eval += compile_seconds() - c0
        e_f = np.asarray(ref["e_f"])
        loss = e_t - e_f
        gain_t = best - value
        gain_f = np.asarray(ref["best"]) - np.asarray(ref["value"])
        seat = (gain_t - gain_f) / 2.0
        det = e_f <= E_REF
        spr = spr_of(spots)
        n_sizes, n_distinct = size_count(S.sizes, self.cap)
        res = {"name": name, "set": which, "street": self.name, "levels": [list(x) for x in levels], "why": why,
               "describe": describe(S.sizes, self.cap), "sizes": n_sizes, "distinct_sizes": n_distinct,
               "tree": {"nodes": S.topo.size, "decisions": S.decisions},
               "spots": count, "determined": int(det.sum()),
               "loss": loss.tolist(), "e_translated": e_t.tolist(), "e_f": e_f.tolist(), "e_s": e_s.tolist(),
               "iterations_s": u_s.tolist(), "seat_loss": seat.tolist(), "gain_translated": gain_t.tolist(),
               "spr": spr.tolist(), "stats": loss_stats(loss[det], spr[det], seat[det]) if det.any() else None,
               "translation": {**stats, "lost_branch_weight_max": lost_max,
                               "dropped_mass_reach_weighted_max": float(dropped_mass.max()),
                               "alldead_mass_reach_weighted_max": float(alldead_mass.max()),
                               "alldead_own_reach_max": float(alldead_reach.max())},
               "s_only": {"alive_decisions_with_size_only_here": int(s_only_nodes.sum()),
                          "reach_weighted_mass_mean": s_only_mass},
               "mass_own_tree": freq_tables(dgS["freq"], S),
               "br_restricted": {r: {"loss": (v["e"] - e_f).tolist(), "e": v["e"].tolist(),
                                     "gain": (v["best"] - value).tolist(),
                                     "mean": float(np.mean((v["e"] - e_f)[det])) if det.any() else None,
                                     "max": float(np.max((v["e"] - e_f)[det])) if det.any() else None}
                                 for r, v in rest.items()},
               "trajectory_s": traj_s, "cost_s": trajectory_summary(traj_s, wall - comp, count),
               "cost_to_0.001": cost, "check_every": self.cfg_set.check_every,
               "seconds": {"replay": t_replay, "replay_compile": c_replay, "solve": wall - comp, "solve_compile": comp,
                           "diag_s": t_diag_s, "diag_s_compile": c_diag_s, "translate": t_trans,
                           "evaluate": t_eval - c_eval, "evaluate_compile": c_eval,
                           "wall": time.perf_counter() - t_all},
               "fingerprint": fingerprint()}
        write_new(path, res)
        st = res["stats"]
        if st is not None:
            self.log(f"{name} {which}: BR restricted, loss mean " + ", ".join(
                f"{r} {v['mean']:.5f}" for r, v in res["br_restricted"].items()))
            self.log(f"{name} {which}: loss mean {st['mean']:.5f} p90 {st['p90']:.5f} max {st['max']:.5f} on "
                     f"{res['determined']} spots; seat parts (0 button, 1 big blind) {np.round(st['seat_mean'], 5).tolist()}; "
                     f"E_S max {e_s.max():.5f}; solve {wall - comp:.0f} s + compile {comp:.0f} s, wall {res['seconds']['wall']:.0f} s")
        return res


# ------------------------------------------------------------------ decision and report

def passes(st):
    return st is not None and st["mean"] <= MEAN_BAR and st["max"] <= MAX_BAR


def decide(results, sets_needed):
    """The smallest candidate passing on every set of sets_needed: fewest sizes, then fewest decisions; a
    remaining tie goes to the lower mean loss on the first set (the rule does not specify it: reported)."""
    ok = []
    for name, by_set in results.items():
        if by_set and all(w in by_set and passes(by_set[w]["stats"]) for w in sets_needed):
            first = by_set[sets_needed[0]]
            ok.append((first["sizes"], first["tree"]["decisions"], first["stats"]["mean"], name))
    return sorted(ok)[0][-1] if ok else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--street", required=True, choices=list(STREETS))
    p.add_argument("--sets", default=None, help="spot sets in order (default A,B; river A,B,C)")
    p.add_argument("--only", default=None, help="comma-separated candidate names (default every set of the street)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--chunk-full", type=int, default=8)
    p.add_argument("--chunk-set", type=int, default=8)
    p.add_argument("--budget-hours", type=float, default=2.0, help="stop before a step once the run is past this")
    p.add_argument("--own", action="store_true", help="the own-sizing metric (asymmetric games) instead of translation")
    p.add_argument("--own-all", action="store_true", help="--own: every set on every spot set (default: B only for "
                   "the sets passing on A)")
    p.add_argument("--ignore-hold", action="store_true", help="run although <out>/HOLD exists (the run it makes way for)")
    p.add_argument("--tag", default="", help="suffix of the candidates' json names (a diagnostic rerun)")
    p.add_argument("--cost-only", action="store_true", help="trajectories and solve cost only (Lab.cost), no losses")
    p.add_argument("--cost-sets", default="A,C", help="sets where every candidate is also solved to E_REF")
    p.add_argument("--count", type=int, default=None, help="spots per lab set instead of SPOTS (preflop: 1, every "
                   "regime-1 preflop spot is the same 200 BB spot)")
    args = p.parse_args(argv)
    if config.GRID != GRID:
        raise SystemExit(f"POKER_GRID={GRID} required (the candidates' grid), got {config.GRID}")
    out = args.out or Path("results/sizing") / args.street
    out.mkdir(parents=True, exist_ok=True)
    hold = out / "HOLD"
    if hold.exists() and not args.ignore_hold:                     # a queued run yields the GPU lock (its owner reorders the queue)
        print(time.strftime("[%H:%M:%S]"), f"{hold} exists: nothing done ({hold.read_text().strip()})", flush=True)
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fh = open(out / f"run_{stamp}.log", "x")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()

    t_start = time.perf_counter()
    log(f"sizing lab {args.street}: devices {jax.devices()}, regime {config.REGIME}, fingerprint {fingerprint()}")
    only = None if args.only is None else {x for x in args.only.split(",") if x}       # "" = the reference only
    lab = Lab(args.street, out, log, args.chunk_full, args.chunk_set, only, args.count,
              tuple(x for x in args.cost_sets.split(",") if x))
    lab.tag = args.tag
    set_names = (args.sets or ("A,B,C" if args.street == "river" else "A,B")).split(",")
    results = {name: {} for name, _, _, _ in lab.cands}
    timing = {}
    stopped = None
    if args.cost_only:
        for w in set_names:
            lab.cost(w, lab.spots(w))
        log(f"cost-only done, wall {(time.perf_counter() - t_start) / 3600:.2f} h")
        fh.close()
        return
    if args.own:
        own = {name: {} for name, _, _, _ in lab.cands}
        for w in set_names:
            spots = lab.spots(w)
            base, fd, ref = lab.reference(w, spots)
            for cand in lab.cands:
                if (time.perf_counter() - t_start) / 3600 > args.budget_hours:
                    stopped = f"budget {args.budget_hours} h passed before {cand[0]} on {w}"
                    break
                if w != "A" and not args.own_all and "A" in own[cand[0]] and not passes(own[cand[0]]["A"]["stats"]):
                    log(f"{cand[0]} {w} own: skipped (fails on A)")
                    continue
                own[cand[0]][w] = lab.evaluate_own(cand, w, spots, base, fd, ref)
            if stopped:
                break
            del base, fd
        log(f"own-sizing done; stopped: {stopped}; wall {(time.perf_counter() - t_start) / 3600:.2f} h")
        fh.close()
        return
    for w in set_names:
        if (time.perf_counter() - t_start) / 3600 > args.budget_hours:
            stopped = f"budget {args.budget_hours} h passed before set {w}"
            break
        spots = lab.spots(w)
        base, fd, ref = lab.reference(w, spots)
        timing[f"full_{w}"] = {"solve": round(ref["solve_seconds"], 1), "compile": round(ref["compile_seconds"], 1)}
        if ref["undetermined"] > MAX_UNDETERMINED * ref["spots"]:
            stopped = f"set {w}: {ref['undetermined']} of {ref['spots']} spots undetermined (> {MAX_UNDETERMINED:.0%})"
            break
        for cand in lab.cands:
            if (time.perf_counter() - t_start) / 3600 > args.budget_hours:
                stopped = f"budget {args.budget_hours} h passed before {cand[0]} on {w}"
                break
            r = lab.evaluate(cand, w, spots, base, fd, ref)
            results[cand[0]][w] = r
            timing[f"{cand[0]}_{w}"] = {k: round(v, 1) for k, v in r["seconds"].items()}
        if stopped:
            break
        del base, fd
    if stopped:
        log("STOP " + stopped)
    timing["run_wall_hours"] = round((time.perf_counter() - t_start) / 3600, 3)
    ab = [w for w in ("A", "B") if w in set_names]
    needed = [w for w in ("A", "B", "C") if w in set_names]
    choice_ab = decide(results, ab)
    choice_all = decide(results, needed)
    summary = out / f"summary_{stamp}{args.tag}.json"
    write_new(summary, {"street": args.street, "choice_ab": choice_ab, "choice_all": choice_all,
                        "stopped": stopped, "timing": timing})
    log(f"decision: A+B {choice_ab}, all sets {choice_all}; stopped: {stopped}; summary {summary}; "
        f"wall {timing['run_wall_hours']:.2f} h")
    fh.close()


if __name__ == "__main__":
    main()
