"""Bootstrap check (brief M2, step 1, A7): the building blocks of the end-of-street targets
(checkdown.river_cards on turn boards, checkdown.turn_cards on flop boards: shares and card-removal
weights) on the device the loop uses for them (loop.aux_device) against the CPU, 20 situations
(10 turn, 10 flop), float32: max |difference| <= 1e-6, else exit 1.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m tools.cloud.probe_aux

On Pascal the loop computes them on the CPU (the river-board scan does not load there): nothing to
compare, reported as such.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from poker.solver import checkdown as CD
from poker.solver import rank7
from poker.solver import situations as SI
from poker.solver.loop import aux_device

LIMIT = 1e-6


def main():
    dev, cpu = aux_device(), jax.devices("cpu")[0]
    if dev.platform == "cpu":
        print(f"AUX: {jax.devices()[0].device_kind}: the loop builds end-of-street targets on the CPU; nothing to compare")
        print("AUX OK")
        return
    table = rank7.load_table(log=None)
    worst, t_dev, t_cpu = 0.0, 0.0, 0.0
    for street, fn, cards in ((2, CD.river_cards, 4), (1, CD.turn_cards, 3)):
        spots = SI.make_spots(111 + street, 10, 2, spr_range=(0.5, 50.0), street=street)
        jitted = jax.jit(fn)
        for i in range(10):
            board = jnp.asarray(spots["board"][i, :cards], jnp.int32)
            ranges = jnp.asarray(spots["ranges"][i], jnp.float32)
            out = []
            for d in (dev, cpu):
                t0 = time.perf_counter()
                s, m = jitted(*(jax.device_put(x, d) for x in (board, ranges, table)))
                out.append((np.asarray(s), np.asarray(m)))
                if d is dev:
                    t_dev += time.perf_counter() - t0
                else:
                    t_cpu += time.perf_counter() - t0
            worst = max(worst, float(np.abs(out[0][0] - out[1][0]).max()), float(np.abs(out[0][1] - out[1][1]).max()))
    print(f"AUX: {dev} vs CPU, end-of-street blocks on 20 situations (10 turn boards: river_cards, 10 flop boards: "
          f"turn_cards): max |diff| {worst:.1e} (limit {LIMIT:g}); {t_dev:.1f} s on {dev.platform}, {t_cpu:.1f} s on CPU "
          f"(with compilation)", flush=True)
    if not worst <= LIMIT:
        print(f"AUX FAIL: GPU and CPU paths differ by {worst:.1e} > {LIMIT:g}")
        sys.exit(1)
    print("AUX OK")


if __name__ == "__main__":
    main()
