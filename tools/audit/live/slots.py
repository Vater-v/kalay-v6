"""SLOT handling of poker/solver/subgame.py on real Slumbot-table states: an opponent's off-menu size lands in the
node's slot at its exact chip total; our own off-menu size; a preflop limp; raises beyond the cap; two off-menu
re-raises; three players.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/slots.py
Writes results/audit/live/slots.json."""
import json
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.envs.betting import init_betting
from poker.solver import subgame as SG
from poker.solver.cards import NUM_HANDS as H
from tools.audit.live.timing import street_root

R, C_, A, F = SG.RAISE, SG.CALL, SG.ALLIN, SG.FOLD
rows = []


def run(name, fn):
    try:
        rows.append({"case": name, "ok": True, **fn()})
    except Exception as e:                                                          # noqa: BLE001
        rows.append({"case": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"})


def describe(tree):
    path = []
    for i, (k, t) in enumerate(tree.history):
        v, a = int(tree.path[i]), int(tree.path_cols[i])
        path.append({"actor": int(tree.topo.actor[v]), "kind": SG.KIND_NAMES[k], "total": t, "column_target": int(tree.targets[v, a]),
                     "in_slot": bool(tree.in_slot[i])})
    v = tree.node
    cols = [(SG.KIND_NAMES[k], t) for _, k, t in tree.columns(v)] if not tree.topo.terminal[v] else "terminal"
    return {"path": path, "node_columns": cols}


flop = street_root(1)                      # pot 400, 19,800 behind, seat 1 (BB) to act
pre = jax.tree.map(np.asarray, init_betting(2, jnp.full(2, 20000, jnp.int32), history=True, max_stack=20000, rake_level=0.0))
pre = pre.replace(board=np.asarray([12, 24, 5, 30, 44], np.int32))

# hero = 0 (SB/IP): the opponent (seat 1, BB) acts first on the flop
run("opponent off-menu flop bet 137 chips (menu: 33%=132, 75%=300) -> slot at exactly 137",
    lambda: describe(SG.build_tree(flop, 0, [(R, 137)])))
run("opponent off-menu bet 137, our menu raise, his off-menu re-raise 1999 -> second slot",
    lambda: (lambda t0: describe(SG.build_tree(flop, 0, [(R, 137), (R, [t for _, k, t in t0.columns(t0.node) if k == R][0]), (R, 1999)])))(
        SG.build_tree(flop, 0, [(R, 137)])))
run("our own off-menu size (hero=1 bets 137) -> OffTree", lambda: describe(SG.build_tree(flop, 1, [(R, 137)])))
run("preflop limp by the opponent (SB calls 100), hero BB -> CALL column; our options after it",
    lambda: describe(SG.build_tree(pre, 1, [(C_, 100)])))
run("preflop opponent open to 250 (off-menu: min-raise 200 / pot 300), hero BB",
    lambda: describe(SG.build_tree(pre, 1, [(R, 250)])))


def beyond_cap():
    """Opponent raises off-menu, we raise menu sizes until our cap, he keeps raising off-menu (slots exist at every
    level), until all-in."""
    hist = []
    tree = SG.build_tree(flop, 0, hist)
    his = 150
    for step in range(12):
        v = tree.node
        if tree.topo.terminal[v]:
            break
        actor = int(tree.topo.actor[v])
        cols = tree.columns(v)
        if actor == 1:                                                            # opponent: an off-menu raise if any room
            lo = int(tree.info["minimum"][v]) if "minimum" in tree.info else None
            mx = int(tree.info["maximum"][v])
            level = int(tree.info["level"][v])
            target = int(level + max(int(tree.info["to_call"][v]), 0) + max(his, 1))
            target = max(target, level * 2 + 7)
            if target >= mx:
                hist.append((A, mx))
            else:
                hist.append((R, target))
        else:
            raises = [t for _, k, t in cols if k == R]
            hist.append((R, raises[0]) if raises else (C_, [t for _, k, t in cols if k == C_][0]))
        tree = SG.build_tree(flop, 0, hist)
    d = describe(tree)
    d["slot_count"] = int(sum(tree.in_slot))
    d["shape_cap"] = tree.shape.cap
    return d


run("raises beyond the cap: alternating our menu raises and his off-menu re-raises", beyond_cap)


def three_players(street):
    """3 players at a postflop street start (preflop's 3-way deep-root topology was killed at 4.5 GB RSS here)."""
    def f():
        s = jax.tree.map(np.asarray, init_betting(3, jnp.full(3, 20000, jnp.int32), history=True, max_stack=20000, rake_level=0.0))
        from poker.envs.transition import step_target as _st
        step_target = jax.jit(_st)
        s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(100))       # button calls
        s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(100))       # SB completes
        s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(100))       # BB checks: flop, pot 300
        for _ in range(street - 1):
            for _ in range(3):
                s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(0))
        s = jax.tree.map(np.asarray, s.replace(board=jnp.asarray([12, 24, 5, 30, 44], jnp.int32)))
        out = {"street": int(s.street)}
        tree = SG.build_tree(s, 0, [])
        out["nodes"] = int(tree.size)
        try:
            SG.solve_batch([tree], [np.full((3, H), 1 / H)], 50, 50, equity=[np.zeros((H, H), np.float32)], ladder=(1,),
                           aux=(lambda *a: None) if street < 3 else None)
            out["solve"] = "ran"
        except Exception as e:                                                      # noqa: BLE001
            out["solve"] = f"{type(e).__name__}: {str(e)[:200]}"
        return out
    return f


run("three players: flop live tree and a solve", three_players(1))
run("three players: river live tree and a solve", three_players(3))

open("results/audit/live/slots.json", "w").write(json.dumps(rows, indent=1, default=str))
for r in rows:
    print(json.dumps(r, default=str)[:900])
