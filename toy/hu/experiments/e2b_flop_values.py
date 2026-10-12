"""E2b: flop-boundary value function V_flop, labels BOOTSTRAPPED DeepStack-style:
solve only the flop street, leaves valued by V_turn (from E2a); M_flop =
min(N, FLOP_CAP) situations.  Tested against exact flop labels (E2a)."""
import os, sys, time, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
from e2_common import *
from e2_common import _train  # underscore names are not exported by import *
from common import save, Log
from valuefn import reachable_x, sample_situations, ByStreetV

if __name__ == "__main__":
    log = Log("e2b.log")
    VT = pickle.load(open(os.path.join(RES, "value_nets_turn.pkl"), "rb"))
    flop_test = load_data("e2_flop_test.npz")
    onp = on_policy(1)
    out = dict(flop_cap=FLOP_CAP, n_val_flop=N_VAL_FLOP, label_iters=LABEL_ITERS,
               n_onpolicy_flop=len(onp[0]))
    rng = np.random.default_rng(777)
    nmax = min(NS[-1], FLOP_CAP)
    dflop = sample_situations(CFG, B, nmax + N_VAL_FLOP, 1, reachable_x(T, 0), rng)
    VF, rows = {}, []
    with mp.get_context("fork").Pool(4) as pool:
        for (N, res, rw) in NET_VARIANTS:
            name = vname(N, res, rw)
            Vleaf = ByStreetV(B, CD, VT[name])
            m = min(N, FLOP_CAP)
            dsel = tuple(np.concatenate([z[:m], z[nmax:]]) for z in dflop)
            t0 = time.time()
            lab, bexpl, bcpu = label_parallel(dsel, LABEL_ITERS, pool, V=Vleaf, chunk=250)
            lw = time.time() - t0
            V, vl, dt = _train((N, res, rw, sub(lab, 0, m), sub(lab, m, m + N_VAL_FLOP), 2))
            VF[name] = V
            # quality of the bootstrapped labels themselves on the exact test set
            bl, _, _ = label_parallel(flop_test[:4], LABEL_ITERS, pool, V=Vleaf, chunk=25)
            boot = lambda b, x, p0, p1, bl=bl: (bl[4], bl[5])
            rows.append(dict(name=name, N=N, M_flop=m, residual=res, range_weighted=rw,
                             boot_label_wall_s=lw, boot_label_cpu_s=bcpu,
                             boot_cpu_ms_per_situation=1000 * bcpu / len(dsel[0]),
                             boot_label_rmse_vs_exact=rmse(B, boot, flop_test),
                             boot_label_rmse_rangew_vs_exact=rmse_w(boot, flop_test),
                             train_s=dt, val_loss=vl, **errors(V, flop_test, onp, "exact_test")))
            log("V_flop", rows[-1])
    rows.append(dict(name="checkdown", N=0, **errors(CD, flop_test, onp, "exact_test")))
    log("V_flop", rows[-1])
    out["V_flop"] = rows
    with open(os.path.join(RES, "value_nets_flop.pkl"), "wb") as fh:
        pickle.dump(VF, fh)
    save("e2b.json", out)
    log("done")
