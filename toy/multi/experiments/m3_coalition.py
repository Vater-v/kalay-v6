"""M3: "opponents are weather" against a colluding coalition (seats 1+2 vs seat 0).

For two fixed colluding joint strategies (coalition.py) we measure seat 0's exact EV when it
  (i)   keeps playing the frozen self-play strategy sigma*_0 from M2,
  (ii)  plays an exact best response to the (known) coalition  -> the ceiling,
  (iii) runs CFR+ for seat 0 only, with the coalition as a fixed environment and full
        counterfactual feedback (fresh start, and warm start from the M2 self-play regrets),
  (iv)  runs MODEL-FREE outcome-sampling regret matching: one real hand per iteration, only the
        observed payoff and its own sampling probabilities (no opponent model).
"""
import time

import numpy as np

from common import MAIN, save, table, ma, RESULTS
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import CFRPlus, evaluate, normalize, best_response_strategy
from coalition import build
from mccfr import OutcomeSamplingLearner

CFR_CHECKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
OS_CHECKS = [1000, 10_000, 100_000, 300_000, 1_000_000]


def ev0(tree, sigma, ex):
    return float(evaluate(tree, sigma, ex)["ev"][0])


def iters_to_close(log, frozen, best, frac):
    for it, ev in log:
        if ev - frozen >= frac * (best - frozen):
            return it
    return None


def main():
    tree = PublicTree(MAIN)
    ex = TerminalEvaluator(tree, "exact")
    sig_star = np.load(f"{RESULTS}/m2_sigma_3p.npy")
    R_star = np.load(f"{RESULTS}/m2_regrets_3p.npy")
    base = evaluate(tree, sig_star, ex)
    print(f"self-play sigma*: EV per seat {np.round(base['ev'], 4)}, delta(mA) {np.round(1000 * base['delta'], 2)}")
    out = dict(selfplay_ev=base["ev"].tolist(), selfplay_delta=base["delta"].tolist())
    summary = []
    for kind in ("softplay", "whipsaw"):
        coal = build(tree, sig_star, kind)
        e = evaluate(tree, coal, ex)
        frozen, best = float(e["ev"][0]), float(e["br"][0])
        coal_ev = float(e["ev"][1] + e["ev"][2])
        # coalition's own exploitability by seat 0 = best - frozen; also, how much the
        # coalition gains vs self-play
        print(f"\n== coalition {kind}: seat0 frozen EV {frozen:.4f}, BR EV {best:.4f}, "
              f"coalition EV {coal_ev:.4f}")
        res = dict(frozen=frozen, best=best, coalition_ev=coal_ev)
        # (iii) CFR+ for seat 0 only, fresh and warm
        for start in ("fresh", "warm"):
            s = CFRPlus(tree, ex, sigma0=coal, players=[0])
            if start == "warm":
                mine = tree.dec_edges[tree.dec_owner == 0]
                s.R[mine] = R_star[mine]
                s.sigma[mine] = normalize(tree, R_star)[mine]
            cur_log, avg_log = [], []
            t0, clock = time.perf_counter(), 0.0
            for it in range(1, CFR_CHECKS[-1] + 1):
                t1 = time.perf_counter(); s.iterate(1); clock += time.perf_counter() - t1
                if it in CFR_CHECKS:
                    cur_log.append((it, ev0(tree, s.sigma, ex)))
                    avg_log.append((it, ev0(tree, s.average_strategy(), ex)))
            res[f"cfr_{start}"] = dict(cur=cur_log, avg=avg_log, seconds=clock,
                                       close50=iters_to_close(avg_log, frozen, best, 0.5),
                                       close90=iters_to_close(avg_log, frozen, best, 0.9),
                                       close99=iters_to_close(avg_log, frozen, best, 0.99),
                                       cur_close90=iters_to_close(cur_log, frozen, best, 0.9))
            print(f"  CFR+ {start}: " + ", ".join(f"t={a}: avg {b:.4f}/cur {c:.4f}"
                                                for (a, b), (_, c) in zip(avg_log, cur_log)))
        # (iv) model-free outcome sampling, one hand per iteration
        R0 = np.zeros_like(R_star)
        mine = tree.dec_edges[tree.dec_owner == 0]
        R0[mine] = R_star[mine] / MAIN.n_deals          # CFR units -> per-hand units
        for start, eps in (("fresh", 0.6), ("fresh", 0.2), ("warm", 0.2), ("warm", 0.05)):
            L = OutcomeSamplingLearner(tree, coal, me=0, eps=eps, seed=7,
                                       R0=R0 if start == "warm" else None)
            log = []
            t1 = time.perf_counter()
            done = 0
            for h in OS_CHECKS:
                while done < h:
                    L.play_hand(); done += 1
                log.append(dict(hands=h, avg=ev0(tree, L.average_strategy(), ex),
                                cur=ev0(tree, L.current_strategy(), ex),
                                realised=L.realised / L.hands, sec=time.perf_counter() - t1))
                print(f"  OS {start} eps={eps} hands={h:>8d}: avg EV {log[-1]['avg']:.4f} cur EV "
                      f"{log[-1]['cur']:.4f} realised/hand {log[-1]['realised']:.4f}", flush=True)
            res[f"os_{start}_eps{eps}"] = log
        out[kind] = res
        summary.append((kind, res))
    # tables
    rows = []
    for kind, r in summary:
        rows.append([kind, round(out["selfplay_ev"][0], 4), round(r["frozen"], 4), round(r["best"], 4),
                     round(r["coalition_ev"], 4),
                     round(r["cfr_fresh"]["avg"][-1][1], 4), round(r["cfr_warm"]["avg"][-1][1], 4),
                     r["cfr_fresh"]["close90"], r["cfr_fresh"]["close99"],
                     r["cfr_warm"]["close90"], r["cfr_warm"]["close99"]])
    hdr = ["coalition", "seat0 EV self-play", "(i) frozen sigma*", "(ii) best response",
           "coalition EV (vs frozen)", "(iii) CFR+ fresh avg @2000", "(iii) CFR+ warm avg @2000",
           "fresh: iters to 90% of gap", "fresh: 99%", "warm: 90%", "warm: 99%"]
    print(table(hdr, rows))
    out["table"] = dict(h=hdr, rows=rows)
    save("m3", out)


if __name__ == "__main__":
    main()
