"""Unit checks of the re-solving gadget in pcfr (one HU river spot, exact leaves):
  1. a spot with the gadget keys present and the mask off solves bitwise like a spot without the keys;
  2. ceilings far below any value make the gadget seat Follow with everything: the solve equals the plain one
     up to float noise (his deal range is the plain one here);
  3. the gadget game's certificate (pcfr.exploitability with follow) is finite and non-negative, and ceilings far
     above any value drive the average Follow to ~0.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tests/test_gadget.py
"""
import os
import sys
import time

os.environ.setdefault("POKER_REGIME", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402
import numpy as np                              # noqa: E402

from poker.solver import pcfr as SV             # noqa: E402
from poker.solver import situations as SI       # noqa: E402
from poker.solver import tree as TR             # noqa: E402
from poker.solver.cards import NUM_HANDS as H   # noqa: E402


def solve(batch, st, iters, chunk=50):
    cfg = SV.SolverConfig(cap=iters, e_target=0.0, check_every=chunk)
    state = jax.vmap(lambda _: SV.init_cfr_state(st, jnp.float32))(jnp.arange(batch["ranges"].shape[0]))
    step = SV.make_cfr_chunk(st, cfg)
    for _ in range(iters // chunk):
        state = step(state, batch, None)
    z = jax.vmap(lambda s, spot: SV.average_logits(s, spot, st))(state["avg"], batch)
    e = jax.jit(jax.vmap(lambda s, spot: SV.exploitability(SV.average_logits(s["avg"], spot, st), spot, st, cfg, None,
                                                           follow=SV.average_follow(s))["nash_conv"] / st.n))(state, batch)
    return np.asarray(z), np.asarray(SV.average_follow(state)), np.asarray(e), state


def main():
    t0 = time.perf_counter()
    topo = TR.rule_topology(2, 3, "normal")
    st = SV.Static(topo)
    spots = SI.make_training_spots(7, 8, 2, street=3)
    spots = {k: v[:1] for k, v in spots.items()}
    plain = SI.prepare(spots, topo)
    T = 100
    z0, f0, e0, _ = solve(plain, st, T)
    assert np.isfinite(e0).all() and e0[0] >= -SV.E_TOL, e0
    assert np.allclose(f0, 1.0), "no gadget: follow must read 1"

    off = {**plain, "gadget_ev": jnp.zeros((1, 2, H), jnp.float32), "gadget_mask": jnp.zeros((1, 2), bool)}
    z1, f1, e1, _ = solve(off, st, T)
    assert np.array_equal(z0, z1), "mask off must not change the solve"
    assert abs(float(e1[0]) - float(e0[0])) < 1e-7, (e0, e1)
    print(f"1. mask off == no keys: bitwise equal policies, E {float(e0[0]):.6f}")

    low = {**plain, "gadget_ev": jnp.full((1, 2, H), -1e3, jnp.float32), "gadget_mask": jnp.asarray([[True, False]])}
    z2, f2, e2, _ = solve(low, st, T)
    assert np.allclose(f2[0, 0], 1.0, atol=1e-6), f2[0, 0].min()
    # the same game (he follows with everything from iteration 2 on; iteration 1 scales his range by 1/2, which
    # changes nothing in regret matching but the weights of one iteration): compare where the policies matter -
    # the plain game's exploitability of the gadget solve, and the reach-weighted policy difference
    cfg = SV.SolverConfig(cap=T, e_target=0.0, check_every=50)
    e_plain_of_z2 = float(np.asarray(jax.jit(lambda z, s: SV.exploitability(z, s, st, cfg, None)["nash_conv"] / st.n)(
        jnp.asarray(z2[0]), jax.tree.map(lambda x: x[0], plain))))
    legal = plain["legal"][0][st.internal]
    pi0, pi2 = SV.policy(jnp.asarray(z0[0]), legal), SV.policy(jnp.asarray(z2[0]), legal)
    reach0 = np.asarray(SV.forward(pi0, plain["ranges"][0], st))[st.internal]                     # [Vi, n, H]
    actors = st.actor[st.internal]
    own = reach0[np.arange(len(st.internal)), actors]                                             # [Vi, H]
    diff = float(np.sum(own[:, :, None] * np.abs(np.asarray(pi2) - np.asarray(pi0))) / own.sum())
    assert e_plain_of_z2 < 2 * float(e0[0]) + 1e-3, (e_plain_of_z2, e0)
    assert diff < 0.15, diff          # two near-equilibria of one game differ where the game is indifferent
    print(f"2. ceilings -1000 pot: follow min {float(f2[0, 0].min()):.6f}; plain-game E of the gadget solve {e_plain_of_z2:.6f} "
          f"vs plain {float(e0[0]):.6f}; reach-weighted |sigma - plain| {diff:.4f}")

    high = {**plain, "gadget_ev": jnp.full((1, 2, H), 1e3, jnp.float32), "gadget_mask": jnp.asarray([[True, False]])}
    z3, f3, e3, _ = solve(high, st, T)
    valid = np.asarray(plain["valid"])[0] > 0
    assert np.isfinite(e3).all() and e3[0] >= -SV.E_TOL, e3
    assert float(f3[0, 0][valid].max()) < 0.05, f3[0, 0][valid].max()
    print(f"3. ceilings +1000 pot: follow max {float(f3[0, 0][valid].max()):.4f}, gadget-game E {float(e3[0]):.6f} (finite, >= 0)")
    print(f"ok in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
