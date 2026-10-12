"""Shared pieces of experiment E2 (continual re-solving with a value function)."""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from common import setup, RES
from cfr import Solver, values
from tree import DEC
from valuefn import CheckdownV, solve_situations, train_value_net, rmse

K = int(os.environ.get("E2_K", 300))                 # CFR+ iterations per re-solve
LABEL_ITERS = int(os.environ.get("E2_LABEL_ITERS", 300))
NS = [int(v) for v in os.environ.get("E2_NS", "200,2000,20000").split(",")]
FLOP_CAP = int(os.environ.get("E2_FLOP_CAP", 2000))  # bootstrapped flop situations
N_VAL, N_TEST, N_FLOP_TEST, N_VAL_FLOP = [int(v) for v in os.environ.get("E2_NTEST", "1000,1000,300,300").split(",")]
STEPS = int(os.environ.get("E2_STEPS", 6000))        # Adam steps per net
# value-net variants: (N, residual, range_weighted)
NET_VARIANTS = [(N, True, False) for N in NS] + [(NS[-1], False, False), (NS[-1], True, True)]

CFG, B, T = setup()
CD = CheckdownV(B)


def _label(args):
    d, iters, V = args
    t0 = time.time()
    u0, u1, e = solve_situations(CFG, B, *d, iters=iters, V_leaf=V, chunk=1000)
    return u0, u1, e, time.time() - t0


def label_parallel(d, iters, pool, V=None, chunk=500):
    """Label situations d=(bid,x,p0,p1) in parallel chunks. Returns the
    labelled tuple, mean label exploitability (x units) and total CPU seconds."""
    bid, x, p0, p1 = d
    jobs = [((bid[a:a + chunk], x[a:a + chunk], p0[a:a + chunk], p1[a:a + chunk]), iters, V)
            for a in range(0, len(bid), chunk)]
    res = pool.map(_label, jobs)
    u0 = np.concatenate([r[0] for r in res]); u1 = np.concatenate([r[1] for r in res])
    expl = float(np.average([r[2] for r in res], weights=[len(j[0][0]) for j in jobs]))
    return (bid, x, p0, p1, u0, u1), expl, sum(r[3] for r in res)


def sub(d, a, b):
    return tuple(z[a:b] for z in d)


def save_data(name, d):
    np.savez(os.path.join(RES, name), *d)


def load_data(name):
    z = np.load(os.path.join(RES, name))
    return tuple(z[f"arr_{i}"] for i in range(len(z.files)))


def on_policy(street):
    """Boundary states of the reference equilibrium (E1 blueprint) at the
    start of `street`, labelled with the blueprint's exact normalised CFVs."""
    e0 = np.load(os.path.join(RES, "blueprint_eff0.npy")); e1 = np.load(os.path.join(RES, "blueprint_eff1.npy"))
    S = Solver(T, [T.bmask[:1].copy()] * 2)
    v0, v1, r0, r1 = values(S, e0, e1)
    roots = np.nonzero((T.kind == DEC) & (T.street == street) &
                       np.array([s.n_act == 0 for s in T.states]))[0]
    M0, M1 = r0[roots].sum(1), r1[roots].sum(1)
    ok = (M0 > 1e-4) & (M1 > 1e-4)
    roots, M0, M1 = roots[ok], M0[ok], M1[ok]
    x = T.contrib[roots, 0]; cs = T.cscale[roots]
    p0 = r0[roots] / M0[:, None]; p1 = r1[roots] / M1[:, None]
    u0 = v0[roots] / (cs * x * M1)[:, None]; u1 = v1[roots] / (cs * x * M0)[:, None]
    return (T.board_id[roots], x, p0, p1, u0, u1)


def rmse_w(V, d):
    """Range-weighted RMSE (x units): sqrt(mean_situations sum_h p_i(h) err_i(h)^2 / 2)."""
    bid, x, p0, p1, u0, u1 = d
    a0, a1 = V(bid, x, p0, p1)
    m = B.mask[bid]
    e = (((a0 - u0) ** 2) * p0 * m).sum(1) + (((a1 - u1) ** 2) * p1 * m).sum(1)
    return float(np.sqrt(e.mean() / 2))


def errors(V, test, onp, exact_key="test"):
    return {f"{exact_key}_rmse": rmse(B, V, test), f"{exact_key}_rmse_rangew": rmse_w(V, test),
            "onpolicy_rmse": rmse(B, V, onp), "onpolicy_rmse_rangew": rmse_w(V, onp)}


def _train(args):
    N, residual, rw, data, val, seed = args
    t0 = time.time()
    V, vl = train_value_net(B, CFG, data, val, residual=residual, steps=STEPS, seed=seed,
                            range_weighted=rw)
    return V, vl, time.time() - t0


def vname(N, residual, rw):
    return f"{N}" + ("" if residual else "_raw") + ("_rw" if rw else "")
