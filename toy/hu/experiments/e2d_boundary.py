"""E2d: which boundary's value function hurts?  Agents that use the exact
(oracle) leaf at one street boundary and an approximate leaf (checkdown or the
best learned net) at the other.  Same K and extraction as E2c."""
import os, sys, time, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
from e2_common import *
from common import save, Log, mA, pct_pot
from cfr import exploitability
from valuefn import ByStreetV
from resolve import extract_agent

VARIANTS = {}


def _extract(args):
    name, a = args
    t0 = time.time()
    eff, stats = extract_agent(T, a, VARIANTS[name], safe=True, iters=K)
    return name, a, eff, time.time() - t0


if __name__ == "__main__":
    log = Log("e2d.log")
    VT = pickle.load(open(os.path.join(RES, "value_nets_turn.pkl"), "rb"))
    VF = pickle.load(open(os.path.join(RES, "value_nets_flop.pkl"), "rb"))
    approx = {"checkdown": ("checkdown", ByStreetV(B, CD, CD)),
              "learned20000_rw": ("learned", ByStreetV(B, VF["20000_rw"], VT["20000_rw"]))}
    for nm, lv in approx.items():
        VARIANTS[f"{nm}@preflop + oracle@flop"] = {0: lv, 1: ("oracle", None)}
        VARIANTS[f"oracle@preflop + {nm}@flop"] = {0: ("oracle", None), 1: lv}
    jobs = [(n, a) for n in VARIANTS for a in (0, 1)]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(_extract, jobs)
    S = Solver(T, [T.bmask[:1].copy()] * 2)
    out = {}
    for name in VARIANTS:
        r = {rr[1]: rr for rr in res if rr[0] == name}
        e = exploitability(S, r[0][2], r[1][2])
        out[name] = dict(expl_mA=mA(e["expl"]), expl_pct_pot=pct_pot(e["expl"]),
                         wall_s=r[0][3] + r[1][3])
        log(f"{name:40s} expl {mA(e['expl']):8.1f} mA/h ({pct_pot(e['expl']):5.2f}% pot)")
    save("e2d.json", dict(K=K, agents=out))
    log("done")
