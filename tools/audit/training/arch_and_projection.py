"""(1) Critic architecture as implemented: parameter count, FLOPs per call, CPU forward cost.
(2) project_values: is it the W-orthogonal projection onto the chip-conservation plane, and what
    does it do to per-hand / per-slot errors?  Synthetic data on random river boards.
Output: results/audit/training/arch_projection.json
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import common  # noqa: E402  (env + git stub)

import jax, jax.numpy as jnp, numpy as np  # noqa: E402
from poker.solver import critics as CR, situations as SI  # noqa: E402
from poker.solver.cards import NUM_HANDS  # noqa: E402

out = {}
hidden = (512, 512, 512)
k = 2
params = CR.init_critic(jax.random.key(0), hidden, k)
stored = sum(int(np.prod(x.shape)) for layer in params for x in layer.values())
# parameters actually touched by a k-slot call (apply_net slices)
d_in_used = 52 + CR.PUB + k * NUM_HANDS
used = (d_in_used * hidden[0] + k * NUM_HANDS * hidden[0] + hidden[0]
        + sum(a * b + b for a, b in zip(hidden[:-1], hidden[1:]))
        + hidden[-1] * k * NUM_HANDS + k * NUM_HANDS)
macs = (d_in_used + k * NUM_HANDS) * hidden[0] + sum(a * b for a, b in zip(hidden[:-1], hidden[1:])) + hidden[-1] * k * NUM_HANDS
out["architecture"] = {"D_IN": CR.D_IN, "D_F": CR.D_F, "D_OUT": CR.D_OUT, "PUB": CR.PUB, "P_MAX": CR.P_MAX,
                       "hidden": hidden, "params_stored": stored, "params_used_hu": used,
                       "first_layer_inputs_used_hu": d_in_used + k * NUM_HANDS,
                       "mflop_per_call_hu": 2 * macs / 1e6,
                       "spec_default": "3x512 MLP ~3.3M params, 6.5 MFLOP/call (docs §2.5)"}
print(out["architecture"])

# CPU forward cost (single thread), projected critic as the solver calls it
critic = CR.net_critic(params, 0.0)
rng = np.random.default_rng(0)
timings = {}
for B in (1, 13, 52, 256):
    board = jnp.asarray(np.zeros((B, 52), np.float32))
    pub = jnp.asarray(np.tile(np.asarray(CR.public_row(jnp.float32(20.0), 0., 0., 0., jnp.asarray([1., 1] + [0] * 7),
                                                            jnp.asarray([5., 5] + [0] * 7), jnp.asarray([.5, .5] + [0] * 7))), (B, 1)))
    r = rng.random((B, k, NUM_HANDS)).astype(np.float32); r /= r.sum(-1, keepdims=True)
    f = rng.random((B, k, NUM_HANDS)).astype(np.float32)
    raw = jax.jit(lambda b, p, r, f: CR.apply_net(params, b, p, r, f))
    proj = jax.jit(lambda b, p, r, f: critic(b, p, r, f))
    for name, fn in (("raw", raw), ("projected", proj)):
        fn(board, pub, jnp.asarray(r), jnp.asarray(f)).block_until_ready()
        n = max(3, int(200 / B))
        t0 = time.perf_counter()
        for _ in range(n):
            fn(board, pub, jnp.asarray(r), jnp.asarray(f)).block_until_ready()
        dt = (time.perf_counter() - t0) / n
        timings[f"{name}_B{B}"] = {"ms_per_batch": dt * 1e3, "ms_per_call": dt * 1e3 / B,
                                   "gflops": 2 * macs * B / dt / 1e9}
print(timings)
out["cpu_forward_single_thread"] = timings

# ---------------- projection properties
spots = SI.make_training_spots(5, 64, 2, street=3, rake="none")
ranges = spots["ranges"].astype(np.float64)                       # [S, 2, H]
present = np.ones((64, 2))
w, live, ok = (np.asarray(x) for x in CR.deal_weights(jnp.asarray(ranges), jnp.asarray(present)))
T = 1.0                                                           # regime 1: no rake -> T = net0 = 1
rows = []
for s in range(64):
    ws = w[s]
    v_true = rng.normal(0.5, 0.3, (2, NUM_HANDS))
    v_true -= (np.sum(ws * v_true) - T) / 2                        # true values satisfy S(V) = T
    for kind in ("iid", "slot0_offset", "hand_dependent_offset"):
        if kind == "iid":
            e = rng.normal(0, 0.05, (2, NUM_HANDS))
        elif kind == "slot0_offset":
            e = np.zeros((2, NUM_HANDS)); e[0] += 0.1
        else:                                                      # offset concentrated on top of range
            e = np.zeros((2, NUM_HANDS)); e[0] = 0.2 * (ranges[s, 0] > np.median(ranges[s, 0][ranges[s, 0] > 0]))
        v = v_true + e
        vp = np.asarray(CR.project_values(jnp.asarray(v[None]), jnp.asarray(ranges[s:s + 1]), jnp.asarray(present[:1]),
                                          jnp.asarray([1.0]), jnp.asarray([1.0]), 0.0))[0]
        ep = vp - v_true
        wmse = lambda x: float(np.sum(ws * x ** 2))
        r = ranges[s] / ranges[s].sum(-1, keepdims=True)
        rmae = lambda x: float(np.mean(np.sum(r * np.abs(x), -1)))
        slot_mae = lambda x: [float(np.sum(r[i] * np.abs(x[i]))) for i in range(2)]
        rows.append({"spot": s, "kind": kind, "wmse_before": wmse(e), "wmse_after": wmse(ep),
                     "range_mae_before": rmae(e), "range_mae_after": rmae(ep),
                     "slot_mae_before": slot_mae(e), "slot_mae_after": slot_mae(ep),
                     "constraint_after": float(np.sum(ws * vp) - T),
                     "shift_constant_per_slot": float(np.ptp(vp - v)) < 1e-9 * 0 + 1e-6})
summary = {}
for kind in ("iid", "slot0_offset", "hand_dependent_offset"):
    rs = [x for x in rows if x["kind"] == kind]
    summary[kind] = {
        "wmse_increased_frac": float(np.mean([x["wmse_after"] > x["wmse_before"] + 1e-12 for x in rs])),
        "wmse_before_mean": float(np.mean([x["wmse_before"] for x in rs])),
        "wmse_after_mean": float(np.mean([x["wmse_after"] for x in rs])),
        "range_mae_increased_frac": float(np.mean([x["range_mae_after"] > x["range_mae_before"] + 1e-12 for x in rs])),
        "range_mae_before_mean": float(np.mean([x["range_mae_before"] for x in rs])),
        "range_mae_after_mean": float(np.mean([x["range_mae_after"] for x in rs])),
        "slot1_mae_before_mean": float(np.mean([x["slot_mae_before"][1] for x in rs])),
        "slot1_mae_after_mean": float(np.mean([x["slot_mae_after"][1] for x in rs])),
        "max_abs_constraint_after": float(np.max([abs(x["constraint_after"]) for x in rs])),
        "shift_is_constant_per_slot": bool(all(x["shift_constant_per_slot"] for x in rs)),
    }
print(summary)
out["projection_synthetic"] = summary
common.dump("arch_projection.json", out)
