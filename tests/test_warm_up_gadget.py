"""The closed compiled set with the gadget: subgame.warm_up compiles the postflop topologies of a small field; a
gadget solve of a real spot afterwards (ceilings, uniform deal, frozen hero action) must not compile anything
(SG.compile_events unchanged) and must return finite Follow probabilities. The same for the live player's
warm_gadget: the ceiling computation compiles at the warm-up, the first hand's ceilings cost no compile.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tests/test_warm_up_gadget.py
"""
import os
import time

os.environ.setdefault("POKER_REGIME", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402
import numpy as np                              # noqa: E402

from poker.envs import config                   # noqa: E402
from poker.player import live as LV             # noqa: E402
from poker.solver import critics as CR          # noqa: E402
from poker.solver import pcfr as SV             # noqa: E402
from poker.solver import runout                 # noqa: E402
from poker.solver import subgame as SG          # noqa: E402
from poker.solver import tree as TR             # noqa: E402
from poker.solver.cards import NUM_HANDS as H   # noqa: E402


def main():
    t0 = time.perf_counter()
    T, chunk = 100, 50
    board = np.asarray([0, 14, 28, 3, 17], np.int32)
    flop = np.asarray(runout.equity_matrix_flop(jnp.asarray(board[:3]), table=None))
    turn = np.asarray(runout.equity_matrix_turn(jnp.asarray(board[:4]), table=None))
    end = {1: CR.net_critic(CR.init_critic(jax.random.key(1), (64, 64), k=2), 0.0),
           2: CR.net_critic(CR.init_critic(jax.random.key(2), (64, 64), k=2), 0.0)}
    post = {s: CR.net_critic(CR.init_critic(jax.random.key(10 + s), (64, 64), k=2), 0.0) for s in (1, 2, 3)}
    field = {"players": (2,), "seats": {2: (0, 1)}, "streets": (1, 2, 3), "menus": ("normal",), "ladder": {"cpu": (1,), "gpu": (1,)}}
    rows = SG.warm_up({1: T, 2: T, 3: T}, chunk, end, flop, turn, None, field, ladder=(1,), certificate=True)
    print(f"warm-up: {len(rows)} topologies x ladder in {time.perf_counter() - t0:.0f} s (compile {sum(r[9] for r in rows):.0f} s)")
    solver = LV.Solver(iterations={1: T, 2: T, 3: T}, chunk=chunk, critics=end, certificate=True)   # the play default: certified
    player = LV.LivePlayer(0, solver=solver, equity=lambda f: flop, turn_equity=lambda b: turn, prefetch_equity=False,
                           gadget=True, post_critics=post)
    player.warm_gadget(flop, turn)
    # a real river spot with a gadget on seat 1 (the actor at the root; we are seat 0). The root is built here, before the
    # count starts: tree.street_root is test scaffolding (eager lax.cond), not the play path (the bot replays jitted steps)
    root = jax.tree.map(np.asarray, TR.street_root(2, 3, 8 * config.BB, 50 * config.BB).replace(board=jnp.asarray(board)))
    valid = LV.LivePlayer._valid(root)
    events0 = SG.compile_events()
    trace = [("after warm_gadget", events0)]
    tree = SG.build_tree(root, 0, [(LV.CALL, 0)], "normal")
    trace.append(("after build_tree", SG.compile_events()))
    assert int(tree.topo.actor[0]) == 1 and int(tree.topo.actor[tree.node]) == 0
    rng = np.random.default_rng(0)
    ev = np.zeros((2, H))
    ev[1] = rng.normal(0.0, 0.2, H) * valid
    gadget = {"ev": ev, "mask": np.asarray([False, True])}
    ranges = np.stack([valid / valid.sum()] * 2)
    sol = solver.solve(tree, ranges, None, None, gadget)
    trace.append(("after gadget solve", SG.compile_events()))
    print("compile events:", trace)
    assert SG.compile_events() == events0, f"a gadget solve after the warm-up compiled something: {trace}"
    assert np.isfinite(sol.follow).all() and 0 <= sol.follow.min() and sol.follow.max() <= 1
    assert sol.e is not None and np.isfinite(sol.e) and sol.e >= -SV.E_TOL, sol.e
    f = sol.follow[1][valid > 0]
    print(f"river gadget solve after the warm-up: {sol.seconds:.2f} s, no compile, certified gadget-game E {sol.e:.5f} pot; "
          f"follow of seat 1: mean {f.mean():.3f}, min {f.min():.3f}, max {f.max():.3f}")
    # the player's ceiling computation on the river compiles nothing either
    player.street_start, player.street, player.n = root, 3, 2          # what new_hand sets in a hand
    player.leaf_reach = np.stack([valid, valid])
    player.ceilings = None
    g = player._gadget(None)
    assert SG.compile_events() == events0, "the ceiling computation compiled after warm_gadget"
    assert g["mask"].tolist() == [False, True] and np.isfinite(g["ev"]).all()
    print(f"ceilings after warm_gadget: no compile, {player.stats['ceiling_seconds'][-1] * 1000:.0f} ms; ok in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
