"""Measure three-player flop runout sampling error against all 1176 completions.

Uses the current spot sampler, three SPR fixtures, float32 completion terms and
chunks of eight. Reports weighted error and standard error for 48 sampled pairs;
this is a sampling diagnostic. The full computation can take several minutes.
From the repository root: python -m tools.measure_runout_sampling [--out FILE]
Default output: results/reports/runout_sampling.json. CPU is the default backend.
"""
import argparse
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
LIVE = (True, True, True)
CHUNK = 8


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reports/runout_sampling.json")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import json

    import jax
    import jax.numpy as jnp
    import numpy as np

    from poker.solver import cards as C
    from poker.solver import pcfr as SV
    from poker.solver import showdown as sd
    from poker.solver import situations as SI

    spots = SI.make_spots(1404, 3, 3, spr_range=(0.5, 50.0), rake="mix", street=1)
    spots["stack"][:] = np.round(np.asarray([1.0, 4.0, 20.0]) * spots["pot"]).astype(np.int32)
    member = np.asarray(C.MEMBER)
    terms = jax.jit(lambda r, k, t: SV._completion_terms(r, k, t, LIVE))
    rows = []
    t_start = time.perf_counter()
    for i in range(3):
        board = np.sort(spots["board"][i, :3]).astype(np.int32)
        r = jnp.asarray(spots["ranges"][i], jnp.float32)
        pairs = SI.flop_pairs(board, [7, *board, int(spots["pot"][i])])
        rest = np.setdiff1d(np.arange(52), board)
        every = np.asarray([(a, b) for k, a in enumerate(rest) for b in rest[k + 1:]], np.int32)
        if len(every) != 1176:
            raise RuntimeError(f"expected 1176 flop completions, got {len(every)}")
        S = np.zeros((1176, 3, C.NUM_HANDS), np.float32)
        M = np.zeros((1176, 3, C.NUM_HANDS), np.float32)
        for lo in range(0, 1176, CHUNK):
            sl = every[lo:lo + CHUNK]
            tab = SI.pair_tables(jnp.asarray(board), jnp.asarray(sl))
            k = jnp.asarray(~(member[sl[:, 0]] | member[sl[:, 1]]))
            ts_, tm_ = terms(r, k, tab)
            S[lo:lo + CHUNK], M[lo:lo + CHUNK] = np.asarray(ts_), np.asarray(tm_)
            if lo == 0:
                print(f"  spot {i}: first chunk done ({time.perf_counter() - t_start:.0f} s)", flush=True)
        S64, M64 = S.astype(np.float64), M.astype(np.float64)
        sum_m = M64.sum(0)
        exact = np.where(sum_m > 1e-30, S64.sum(0) / np.maximum(sum_m, 1e-30), 0.0)
        sel = np.asarray([np.flatnonzero((every[:, 0] == a) & (every[:, 1] == b))[0] for a, b in pairs])
        s48, m48 = S64[sel], M64[sel]
        den = m48.sum(0)
        share48 = np.where(den > 1e-30, s48.sum(0) / np.maximum(den, 1e-30), 0.0)
        resid = np.sum((s48 - share48[None] * m48) ** 2, axis=0)
        var = (1.0 - 48 / 1176) * 48 / 47 * resid
        se = np.where(den > 1e-30, np.sqrt(var) / np.maximum(den, 1e-30), 0.0)
        spr = float(spots["stack"][i] / spots["pot"][i])
        allin_pot = 1.0 + 3.0 * spr
        per = []
        rn = np.asarray(r, np.float64)
        for p in range(3):
            b, c = [j for j in range(3) if j != p]
            mass = np.asarray(sd.mass_three(jnp.asarray(rn[b]), jnp.asarray(rn[c])))
            w = rn[p] * mass
            w = w / w.sum()
            err = np.abs(share48[p] - exact[p])
            ok = w > 0
            order = np.argsort(err[ok])
            cum = np.cumsum(w[ok][order])
            p90 = float(err[ok][order][min(np.searchsorted(cum, 0.9), ok.sum() - 1)])
            per.append({"mean_abs_err": float(np.sum(w * err)), "p90_abs_err": p90, "max_abs_err": float(err[ok].max()),
                        "mean_se": float(np.sum(w * se[p])), "mean_abs_err_pots_allin": float(np.sum(w * err)) * allin_pot,
                        "p90_abs_err_pots_allin": p90 * allin_pot, "within_1.96se": float(np.sum(w * (err <= 1.96 * se[p])))})
        rows.append({"spot": i, "board": board.tolist(), "spr": spr, "allin_pot_pots": allin_pot, "pairs": 48, "players": per})
        print(f"spot {i} flop {board.tolist()} SPR {spr:.1f} (all-in pot {allin_pot:.1f} pots): "
              + "; ".join(f"p{p}: mean|err| {x['mean_abs_err']:.4f} p90 {x['p90_abs_err']:.4f} max {x['max_abs_err']:.4f} "
                          f"(= {x['mean_abs_err_pots_allin']:.3f} / {x['p90_abs_err_pots_allin']:.3f} pots at the all-in), SE {x['mean_se']:.4f}, "
                          f"cover {x['within_1.96se']:.2f}" for p, x in enumerate(per)) + f"  ({time.perf_counter() - t_start:.0f} s)", flush=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
