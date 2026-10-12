"""E2a: turn-boundary value function V_turn.
Oracle labels = exact CFR+ solve of the turn street for random (board, pot,
ranges); MLPs for N in {200, 2000, 20000}: residual over checkdown (default),
raw targets/features (ablation), range-weighted loss (ablation).
Also computes EXACT flop-boundary labels for a test set (used in E2b)."""
import os, sys, time, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
from e2_common import *
from e2_common import _train  # underscore names are not exported by import *
from common import save, Log
from valuefn import reachable_x, sample_situations

if __name__ == "__main__":
    log = Log("e2a.log")
    out = dict(K=K, label_iters=LABEL_ITERS, steps=STEPS, n_val=N_VAL, n_test=N_TEST)
    rng = np.random.default_rng(12345)
    xs_turn, xs_flop = reachable_x(T, 1), reachable_x(T, 0)
    out["x_turn"], out["x_flop"] = list(xs_turn), list(xs_flop)
    n_tot = NS[-1] + N_VAL + N_TEST
    dturn = sample_situations(CFG, B, n_tot, 2, xs_turn, rng)
    with mp.get_context("fork").Pool(4) as pool:
        t0 = time.time()
        turn, turn_expl, turn_cpu = label_parallel(dturn, LABEL_ITERS, pool, chunk=1000)
        out["turn_label"] = dict(n=n_tot, wall_s=time.time() - t0, cpu_s=turn_cpu,
                                 cpu_ms_per_situation=1000 * turn_cpu / n_tot,
                                 mean_label_expl_x_units=turn_expl)
        log("turn labels", out["turn_label"])
        save_data("e2_turn_data.npz", turn)
        tr_all = sub(turn, 0, NS[-1]); val = sub(turn, NS[-1], NS[-1] + N_VAL)
        test = sub(turn, NS[-1] + N_VAL, n_tot)
        onp = on_policy(2)
        out["n_onpolicy_turn"] = len(onp[0])
        jobs = [(N, res, rw, sub(tr_all, 0, N), val, 1) for (N, res, rw) in NET_VARIANTS]
        nets = pool.map(_train, jobs)
        VT, rows = {}, []
        for (N, res, rw, *_), (V, vl, dt) in zip(jobs, nets):
            VT[vname(N, res, rw)] = V
            rows.append(dict(name=vname(N, res, rw), N=N, residual=res, range_weighted=rw,
                             train_s=dt, val_loss=vl, **errors(V, test, onp)))
            log("V_turn", rows[-1])
        rows.append(dict(name="checkdown", N=0, **errors(CD, test, onp)))
        log("V_turn", rows[-1])
        out["V_turn"] = rows
        # exact flop-boundary test labels (full flop+turn solve), for E2b
        dft = sample_situations(CFG, B, N_FLOP_TEST, 1, xs_flop, rng)
        t0 = time.time()
        flop_test, fexpl, fcpu = label_parallel(dft, LABEL_ITERS, pool, chunk=25)
        out["flop_exact_label"] = dict(n=N_FLOP_TEST, wall_s=time.time() - t0, cpu_s=fcpu,
                                       cpu_ms_per_situation=1000 * fcpu / N_FLOP_TEST,
                                       mean_label_expl_x_units=fexpl)
        log("flop exact labels", out["flop_exact_label"])
        save_data("e2_flop_test.npz", flop_test)
    with open(os.path.join(RES, "value_nets_turn.pkl"), "wb") as fh:
        pickle.dump(VT, fh)
    save("e2a.json", out)
    log("done")
