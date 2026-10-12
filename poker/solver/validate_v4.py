"""Exact end-of-turn value labels, and the validator the training loop lacks (training audit 2026-10-12, F1).

For an end-of-turn situation (turn board, both ranges, pot, equal stacks) the exact V(4) of every hand of every seat is
the card-removal average over the 48 river cards of the exact river solves' root values,
    V4(h) = sum_c m_c(h) v_c(h) / sum_c m_c(h),    m_c(h) = J(w_opp with c removed)(h),
the identity the end-of-street critics rely on (checkdown.card_average; v_c in the critics' relative accounting). Nothing
in the loop computes these labels: turn_end is trained and scored against the card average of the LEARNED river critic,
so its reported error says nothing about play. This module computes exact labels (48 river solves of the rule tree per
situation, T PCFR+ iterations each) and scores against them
  - the exact check-down value F at the turn (the residual the end-of-turn network has to learn),
  - a turn_end checkpoint when one is given (--critic DIR/turn_end.npz),
  - the labels' own noise: V4 at T against V4 at 2T (--double).
Errors are the loop's range-weighted MAE per seat (own range over the valid hands) in pot units of the situation, plus
the worst error over hands with at least 10 % of the seat's largest weight (critics.prediction_metrics' conventions).

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 -m poker.solver.validate_v4 --count 3 --iters 400 --double 1 \\
        --out results/audit/v4/validate_v4.json
On a GPU this is the cheap part of the spec's gate (section 6.3); on one CPU core a situation costs 48 x T x 21 ms.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from poker.envs import config
from poker.solver import cards as C
from poker.solver import checkdown as CD
from poker.solver import critics as CR
from poker.solver import pcfr as SV
from poker.solver import runout
from poker.solver import situations as SI
from poker.solver import tree as TR
from poker.solver.cards import NUM_HANDS as H


def river_spots(sit, i):
    """The 48 river spots of situation i of a turn spot set: board + c, the ranges with the hands holding c removed
    (normalised for the solve), the same pot, stacks and rake; also the unnormalised restricted ranges (the masses)."""
    board4 = np.asarray(sit["board"][i, :4], np.int32)
    cards = np.asarray([c for c in range(52) if c not in board4], np.int32)
    keep = ~C.MEMBER[cards]                                                      # [48, H]
    ranges = np.asarray(sit["ranges"][i], np.float64)                             # [2, H]
    restricted = ranges[None] * keep[:, None, :]                                  # [48, 2, H]
    total = restricted.sum(-1, keepdims=True)
    if not np.all(total > 0):
        raise ValueError("a range has no mass on some river")
    boards = np.concatenate([np.broadcast_to(board4, (48, 4)), cards[:, None]], axis=1).astype(np.int32)
    spots = {"board": boards, "pot": np.full(48, sit["pot"][i], np.int32), "stack": np.full(48, sit["stack"][i], np.int32),
             "valid": np.stack([C.board_valid(b) for b in boards]), "ranges": (restricted / total).astype(np.float32),
             "street": np.full(48, 3, np.int32)}
    for key in ("rake_bps", "rake_cap", "rake_level"):
        if key in sit:
            spots[key] = np.full(48, np.asarray(sit[key])[i], np.asarray(sit[key]).dtype)
    return spots, cards, keep, restricted


def exact_v4(sit, i, iters, chunk=50, log=None):
    """(V4 [2, H] relative accounting, E of the 48 river solves [48], seconds) for situation i."""
    spots, cards, keep, restricted = river_spots(sit, i)
    menu = config.menu_for(3, spr=int(sit["stack"][i]) / int(sit["pot"][i]))
    topo = TR.rule_topology(2, 3, menu)
    st = SV.Static(topo)
    t0 = time.perf_counter()
    batch = SI.prepare(spots, topo)
    cfg = SV.SolverConfig(cap=iters, e_target=0.0, check_every=chunk)
    res = SV.solve_fixed(batch, st, cfg, iters)
    ex = jax.jit(jax.vmap(lambda z, s: SV.exploitability(z, s, st, cfg, None)["nash_conv"] / st.n))
    root = jax.jit(jax.vmap(lambda z, s: SV.root_values(z, s, st, cfg, None)))
    e = np.asarray(ex(res["z"], batch))
    ev = np.asarray(root(res["z"], batch), np.float64)                            # [48, 2, H] whole-hand EV per hand
    contrib0 = np.asarray(batch["contrib"])[:, 0, :]                               # [48, 2] pot units
    v_rel = ev + contrib0[:, :, None]                                              # relative accounting (critic targets)
    # masses: the opponent's restricted (unnormalised) range's compatible mass for every hand, zero for hands holding c
    mass = np.zeros((48, 2, H))
    for seat in range(2):
        opp = restricted[:, 1 - seat, :]
        mass[:, seat, :] = np.asarray(jax.vmap(CD.j_mass)(jnp.asarray(opp, jnp.float32)), np.float64) * keep
    v4 = np.asarray(CD.card_average(jnp.asarray(v_rel), jnp.asarray(mass)))      # [2, H]
    if log:
        log(f"  situation {i}: {menu} river tree, T {iters}: E median {np.median(e):.5f} max {e.max():.5f}, {time.perf_counter() - t0:.0f} s")
    return v4, e, time.perf_counter() - t0


def weighted_errors(pred, target, ranges, valid):
    """Range-weighted MAE per seat and the worst error over substantial hands (>= 10 % of the seat's largest weight)."""
    err = np.abs(pred - target) * valid[None, :]
    mae = [float((ranges[s] * err[s]).sum() / max(ranges[s].sum(), 1e-12)) for s in range(2)]
    worst = [float(err[s][(ranges[s] >= 0.1 * ranges[s].max()) & (ranges[s] > 0)].max()) for s in range(2)]
    return {"mae": mae, "worst_substantial": worst}


def checkdown_turn(sit, i):
    """The exact check-down value of every hand of both seats at the turn (the residual critics' F), [2, H]."""
    board4 = np.asarray(sit["board"][i, :4], np.int32)
    w = runout.equity_matrix_turn(jnp.asarray(np.sort(board4)), table=None)
    ranges = jnp.asarray(sit["ranges"][i], jnp.float32)
    share = CD.two(lambda x: CD.share_matrix(x, w), ranges)
    bps = int(np.asarray(sit.get("rake_bps", np.zeros(1)))[i]) if "rake_bps" in sit else 0
    cap = int(np.asarray(sit.get("rake_cap", np.zeros(1)))[i]) if "rake_cap" in sit else 0
    net = CD.net_fraction(jnp.int32(int(sit["pot"][i])), jnp.int32(bps), jnp.int32(cap))
    return np.asarray(share * net, np.float64) * np.asarray(sit["valid"][i], np.float64)[None, :], np.asarray(w)


def critic_at_turn(critic, sit, i, f):
    """A turn_end checkpoint's values at the situation, [2, H] (relative accounting), slots = seats here (equal stacks)."""
    board4 = np.asarray(sit["board"][i, :4], np.int32)
    board1h = np.sum(np.eye(52, dtype=np.float32)[board4], axis=0)
    pot = float(sit["pot"][i])
    present = np.zeros(CR.P_MAX, np.float32); present[:2] = 1
    behind = np.zeros(CR.P_MAX, np.float32); behind[:2] = float(sit["stack"][i]) / pot
    contrib = np.zeros(CR.P_MAX, np.float32); contrib[:2] = 0.5
    bps = int(np.asarray(sit["rake_bps"])[i]) if "rake_bps" in sit else 0
    cap = int(np.asarray(sit["rake_cap"])[i]) if "rake_cap" in sit else 0
    level = float(np.asarray(sit["rake_level"])[i]) if "rake_level" in sit else 0.0
    pub = CR.public_row(jnp.float32(pot / config.BB), jnp.float32(level), jnp.float32(bps / 500.0), jnp.float32(cap / pot),
                        jnp.asarray(present), jnp.asarray(behind), jnp.asarray(contrib))
    ranges = jnp.asarray(sit["ranges"][i], jnp.float32)
    return np.asarray(critic(jnp.asarray(board1h)[None], pub[None], ranges[None], jnp.asarray(f, jnp.float32)[None], None))[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=3, help="end-of-turn situations (the loop's turn spot generator)")
    p.add_argument("--seed", type=int, default=1303)
    p.add_argument("--iters", type=int, default=400, help="PCFR+ iterations per river solve")
    p.add_argument("--double", type=int, default=0, help="also label the first N situations at 2T (the noise floor)")
    p.add_argument("--critic", type=Path, help="a turn_end.npz checkpoint to score against the exact labels")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    sit = SI.make_training_spots(a.seed, a.count, 2, street=2)
    critic = None
    if a.critic is not None:
        params, meta = CR.load_critic(a.critic, name="turn_end")
        critic = CR.net_critic(params, meta["kappa"])
    out = {"iters": a.iters, "seed": a.seed, "situations": []}
    for i in range(a.count):
        pot_bb, spr = float(sit["pot"][i]) / config.BB, float(sit["stack"][i]) / float(sit["pot"][i])
        log(f"situation {i}: board {np.asarray(sit['board'][i, :4]).tolist()}, pot {pot_bb:.1f} BB, SPR {spr:.2f}")
        v4, e, seconds = exact_v4(sit, i, a.iters, log=log)
        ranges = np.asarray(sit["ranges"][i], np.float64)
        valid = np.asarray(sit["valid"][i], np.float64)
        f, _ = checkdown_turn(sit, i)
        rec = {"index": i, "pot_bb": pot_bb, "spr": spr, "board": np.asarray(sit["board"][i, :4]).tolist(),
               "river_E_median": float(np.median(e)), "river_E_max": float(e.max()), "seconds": seconds,
               "checkdown_vs_exact": weighted_errors(f, v4, ranges, valid),
               "v4_mean_by_seat": [float((ranges[s] * v4[s]).sum()) for s in range(2)],
               "zero_sum_check": float(sum((ranges[s] * (v4[s] - f[s])).sum() for s in range(2)))}
        log(f"  check-down vs exact V4: range-weighted MAE {rec['checkdown_vs_exact']['mae']}, worst substantial "
            f"{rec['checkdown_vs_exact']['worst_substantial']} pot")
        if critic is not None:
            pred = critic_at_turn(critic, sit, i, f)
            rec["critic_vs_exact"] = weighted_errors(pred, v4, ranges, valid)
            log(f"  turn_end critic vs exact V4: {rec['critic_vs_exact']}")
        if i < a.double:
            v4b, e2, s2 = exact_v4(sit, i, 2 * a.iters, log=log)
            rec["noise_T_vs_2T"] = weighted_errors(v4, v4b, ranges, valid)
            rec["river_E_median_2T"] = float(np.median(e2))
            log(f"  labels at T vs 2T: {rec['noise_T_vs_2T']}")
        out["situations"].append(rec)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(out, indent=1))
    log("done")


if __name__ == "__main__":
    main()
