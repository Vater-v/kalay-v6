"""E2c: continual re-solving agents.  Extract each agent's complete strategy
in both seats (resolve.extract_agent) and score it with an exact best response
on the full game.  Diagnostic: 'hybrid' = agent's strategy on ONE street,
reference equilibrium (E1) on the others -> which street's decisions leak."""
import os, sys, time, pickle
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # before numpy is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
from e2_common import *
from common import save, load, Log, mA, pct_pot
from cfr import exploitability
from valuefn import ByStreetV
from resolve import extract_agent

VARIANTS = {}


def _extract(args):
    name, a = args
    leaf, V, safe = VARIANTS[name]
    t0 = time.time()
    eff, stats = extract_agent(T, a, leaf, V, safe=safe, iters=K)
    return name, a, eff, stats, time.time() - t0


if __name__ == "__main__":
    log = Log("e2c.log")
    VT = pickle.load(open(os.path.join(RES, "value_nets_turn.pkl"), "rb"))
    VF = pickle.load(open(os.path.join(RES, "value_nets_flop.pkl"), "rb"))
    VARIANTS["oracle_naive"] = ("oracle", None, False)
    VARIANTS["oracle_safe"] = ("oracle", None, True)
    VARIANTS["checkdown_safe"] = ("checkdown", ByStreetV(B, CD, CD), True)
    for (N, res, rw) in NET_VARIANTS:
        nm = vname(N, res, rw)
        VARIANTS[f"learned{nm}_safe"] = ("learned", ByStreetV(B, VF[nm], VT[nm]), True)
    nm = vname(NS[-1], True, False)
    VARIANTS[f"learned{nm}_naive"] = ("learned", ByStreetV(B, VF[nm], VT[nm]), False)
    jobs = [(n, a) for n in VARIANTS for a in (0, 1)]
    jobs.sort(key=lambda j: 0 if j[0].startswith("oracle") else 1)
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(_extract, jobs)
    S = Solver(T, [T.bmask[:1].copy()] * 2)
    bp = [np.load(os.path.join(RES, "blueprint_eff0.npy")), np.load(os.path.join(RES, "blueprint_eff1.npy"))]
    gv = load("e1.json")["game_value_P0"]
    agents = {}
    for name in VARIANTS:
        r = {rr[1]: rr for rr in res if rr[0] == name}
        eff = [r[0][2], r[1][2]]
        np.save(os.path.join(RES, f"agent_{name}_eff0.npy"), eff[0])
        np.save(os.path.join(RES, f"agent_{name}_eff1.npy"), eff[1])
        e = exploitability(S, eff[0], eff[1])
        st = r[0][3] + r[1][3]
        per_street = {}
        for s in range(3):
            ph = [p for p in st if p["street"] == s]
            n = sum(p["n_resolves"] for p in ph); sec = sum(p["seconds"] for p in ph)
            per_street[s] = dict(n_resolves=n, seconds=sec, ms_per_resolve_batched=1000 * sec / max(n, 1))
        hybrid = {}
        for s in range(3):  # agent on street s only
            m = (T.street == s)[:, None]
            h = [np.where(m, eff[p], bp[p]) for p in (0, 1)]
            hybrid[s] = mA(exploitability(S, h[0], h[1])["expl"])
        agents[name] = dict(expl=e["expl"], expl_mA=mA(e["expl"]), expl_pct_pot=pct_pot(e["expl"]),
                            seat0_exploitable_by_mA=mA(e["BR1"] + gv), seat1_exploitable_by_mA=mA(e["BR0"] - gv),
                            wall_s_seat0=r[0][4], wall_s_seat1=r[1][4], per_street=per_street,
                            hybrid_only_street_expl_mA=hybrid)
        log(f"agent {name:28s} expl {mA(e['expl']):8.1f} mA/h ({pct_pot(e['expl']):5.2f}% pot) "
            f"hybrid(pre,flop,turn)={[round(hybrid[s], 1) for s in range(3)]}")
    e1 = load("e1.json")
    ref = [r for r in e1["cfr_plus"] if r["iter"] == K]
    save("e2c.json", dict(K=K, agents=agents, full_solve_same_K=ref[0] if ref else None,
                          full_solve_final=e1["cfr_plus"][-1]))
    log("done")
