"""Side measurement: how well-defined are oracle CFV labels?
Solve the same 1000 random turn situations with CFR+ for 50..1000 iterations
and compare per-hand CFVs with the 1000-iteration solution."""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from common import setup, save, Log
from valuefn import reachable_x, sample_situations, solve_situations

if __name__ == "__main__":
    log = Log("label_noise.log")
    cfg, B, T = setup()
    d = sample_situations(cfg, B, 1000, 2, reachable_x(T, 1), np.random.default_rng(0))
    p0, p1 = d[2], d[3]
    U = {it: solve_situations(cfg, B, *d, iters=it) for it in (50, 100, 200, 400, 1000)}
    rows = []
    for it in (50, 100, 200, 400):
        e0 = (U[it][0] - U[1000][0]) ** 2; e1 = (U[it][1] - U[1000][1]) ** 2
        m = B.mask[d[0]]
        allh = np.sqrt(((e0 + e1) * m).sum() / (2 * m.sum()))
        inr = np.sqrt((e0 * (p0 > 1e-3)).sum() + (e1 * (p1 > 1e-3)).sum()) / np.sqrt((p0 > 1e-3).sum() + (p1 > 1e-3).sum())
        rw = np.sqrt(((e0 * p0).sum(1) + (e1 * p1).sum(1)).mean() / 2)
        rows.append(dict(iters=it, label_expl_x=U[it][2], rmse_all=float(allh), rmse_in_range=float(inr),
                         rmse_rangew=float(rw)))
        log(rows[-1])
    save("label_noise.json", dict(ref_iters=1000, ref_expl_x=U[1000][2], rows=rows))
