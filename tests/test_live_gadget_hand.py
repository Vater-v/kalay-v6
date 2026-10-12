"""Plumbing of the live player's gadget over a whole hand from the flop, both seats live players, no server, no
artifacts: random-weight critics stand in for the trained ones (shapes and units only - nothing here says the
decisions are good), the flop and turn all-in matrices are computed on the device. Checks: the flop (the
scenario's root) solves without a gadget, the turn and river solves carry ceilings from the post-card critics at
the leaf, the follow statistics are recorded, every action is legal, the hand ends; gadget=False still runs.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tests/test_live_gadget_hand.py
"""
import os
import sys
import time

os.environ.setdefault("POKER_REGIME", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402
import numpy as np                              # noqa: E402

from poker.envs import config                   # noqa: E402
from poker.envs.evaluator import get_rewards    # noqa: E402
from poker.player import live as LV             # noqa: E402
from poker.solver import critics as CR          # noqa: E402
from poker.solver import runout                 # noqa: E402
from poker.solver import tree as TR             # noqa: E402


def random_critic(seed):
    params = CR.init_critic(jax.random.key(seed), (64, 64), k=2)
    return CR.net_critic(params, 0.0)


def play(gadget, T=100, seed=0, log=print):
    board = np.asarray([0, 14, 28, 3, 17], np.int32)                 # 2s 3h 4d | 5s | 6h
    hole = np.asarray([[12, 25], [38, 51]], np.int32)                 # As Ah | Ad Ac (hero seat 0 and seat 1)
    pot, stack = 6 * config.BB, 60 * config.BB
    root = jax.tree.map(np.asarray, TR.street_root(2, 1, pot, stack).replace(board=jnp.asarray(board), hole=jnp.asarray(hole)))
    # history arrays: street_root has the default capacity (history on); keep them
    t0 = time.perf_counter()
    w_flop = np.asarray(runout.equity_matrix_flop(jnp.asarray(board[:3]), table=None))
    w_turn = np.asarray(runout.equity_matrix_turn(jnp.asarray(board[:4]), table=None))
    log(f"all-in matrices in {time.perf_counter() - t0:.1f} s")
    end = {1: random_critic(1), 2: random_critic(2)}                   # flop_end / turn_end stand-ins
    post = {1: random_critic(11), 2: random_critic(12), 3: random_critic(13)}
    solver = LV.Solver(iterations={0: T, 1: T, 2: T, 3: T}, chunk=50, critics=end)
    players = [LV.LivePlayer(seat, solver=solver, equity=lambda f, w=w_flop: w, turn_equity=lambda b, w=w_turn: w,
                             prefetch_equity=False, gadget=gadget, post_critics=post if gadget else None, log=log)
               for seat in range(2)]
    for seat, p in enumerate(players):
        p.reach_min = 0.0                                             # the scripted checks below force our line
        p.new_hand(root, hand_id=f"plumb-{seat}")
    state = root
    rng = np.random.default_rng(seed)
    actions = []
    while not bool(state.terminal):
        seat = int(state.to_act)
        if int(state.street) < 3:                                    # scripted: check the flop and turn down so that
            kind, total = players[seat].choose(state, LV.CALL, int(max(np.asarray(state.bet_street))))   # the turn and river
        else:                                                        # start with a gadget; the river is played
            kind, total = players[seat].act(state, rng)
        nxt = jax.tree.map(np.asarray, LV.engine_step(state, kind, total))
        actions.append((int(state.street), seat, int(kind), int(total)))
        for p in players:
            p.observe(state, kind, total, nxt)
        state = nxt
    rewards = np.asarray(get_rewards(state))
    stats = [{k: p.stats[k] for k in ("decisions", "solves", "gadget_streets", "follow_mean", "ceiling_seconds", "floor_hits")}
             for p in players]
    return actions, rewards, stats, players


def main():
    for gadget in (True, False):
        t0 = time.perf_counter()
        actions, rewards, stats, players = play(gadget)
        streets = sorted({a[0] for a in actions})
        print(f"gadget={gadget}: {len(actions)} actions on streets {streets}, rewards {rewards.tolist()}, {time.perf_counter() - t0:.1f} s")
        for seat, s in enumerate(stats):
            print(f"  seat {seat}: {s}")
        assert rewards.sum() == 0
        assert streets == [1, 2, 3], streets
        if gadget:
            later = [st for st in streets if st > 1]
            for seat, s in enumerate(stats):
                assert s["gadget_streets"] == len(later), (s["gadget_streets"], later)
                assert len(s["follow_mean"]) >= len(later)
                for means in s["follow_mean"]:
                    assert all(0.0 <= m <= 1.0 for m in means), means
            p = players[0]
            if p.ceilings is not None:
                ev = p.ceilings["ev"]
                print(f"  last street ceilings: seat1 ev mean {ev[1].mean():+.4f} pot, min {ev[1].min():+.3f}, max {ev[1].max():+.3f}; mask {p.ceilings['mask'].tolist()}")
        else:
            assert all(s["gadget_streets"] == 0 for s in stats)
    print("ok")


if __name__ == "__main__":
    main()
