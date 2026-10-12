"""E1: exact solve of the full game with vectorised CFR+ vs vanilla CFR.

Exploitability = (BR0 + BR1)/2 in antes/hand (exact best responses on the full
public tree).  Vanilla CFR gets the same wall-clock budget as CFR+.
Saves the CFR+ average strategy as the reference ("blueprint") for E3/E4.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from common import setup, save, mA, pct_pot, peak_rss_mb, Log, RES
from cfr import Solver, exploitability

ITERS = int(os.environ.get("E1_ITERS", 2000))
CHECK = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000]


def run(plus, linear, budget_s=None, iters=None, log=None):
    cfg, B, T = setup()
    s = Solver(T, [T.bmask[:1].copy(), T.bmask[:1].copy()], plus=plus, linear=linear)
    rows, solve_t = [], 0.0
    for c in CHECK:
        if iters is not None and c > iters:
            break
        t0 = time.time()
        s.iterate(c - s.it)
        solve_t += time.time() - t0
        e = exploitability(s)
        rows.append(dict(iter=s.it, seconds=solve_t, expl=e["expl"], expl_mA=mA(e["expl"]),
                         expl_pct_pot=pct_pot(e["expl"]), BR0=e["BR0"], BR1=e["BR1"], EV0=e["EV0"]))
        if log:
            log(f"{'CFR+' if plus else 'CFR '} it {s.it:5d} t {solve_t:7.1f}s expl {mA(e['expl']):9.3f} mA/h "
                f"({pct_pot(e['expl']):.3f}% pot)")
        if budget_s is not None and solve_t >= budget_s:
            break
    return s, rows, solve_t


if __name__ == "__main__":
    log = Log("e1.log")
    cfg, B, T = setup()
    summ = T.summary()
    log("game", summ)
    sp, rows_p, tp = run(True, True, iters=ITERS, log=log)
    arrays = sum(a.nbytes for a in sp.eff + sp.R + sp.Ssum) + sp.t.eff_template.nbytes + sp.t.bmask.nbytes
    np.save(os.path.join(RES, "blueprint_eff0.npy"), sp.avg_eff(0))
    np.save(os.path.join(RES, "blueprint_eff1.npy"), sp.avg_eff(1))
    sv, rows_v, tv = run(False, False, budget_s=tp, log=log)
    out = dict(game=summ, cfr_plus=rows_p, vanilla=rows_v,
               sec_per_iter_plus=tp / rows_p[-1]["iter"], sec_per_iter_vanilla=tv / rows_v[-1]["iter"],
               solver_array_MB=arrays / 2**20, peak_rss_MB=peak_rss_mb(),
               game_value_P0=rows_p[-1]["EV0"])
    # first iteration count reaching 1% and 0.1% of the pot-ante
    for name, rows in (("plus", rows_p), ("vanilla", rows_v)):
        for thr in (1.0, 0.1):
            hit = [r for r in rows if r["expl_pct_pot"] < thr]
            out[f"{name}_first_check_below_{thr}pct"] = (hit[0]["iter"], hit[0]["seconds"]) if hit else None
    save("e1.json", out)
    log("saved", {k: v for k, v in out.items() if k not in ("cfr_plus", "vanilla")})
