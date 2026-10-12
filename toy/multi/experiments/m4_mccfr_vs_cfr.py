"""M4: external-sampling MCCFR vs vectorised CFR+ at equal wall-clock.

Both algorithms use the same public tree and the same exact exploitability evaluation.
Clocks count only algorithm time (evaluation at checkpoints is excluded).
Work counters: CFR+ touches N passes x (M nodes x D cards) entries per iteration (plus the
terminal evaluation); ES-MCCFR touches `visits` nodes (each one scalar infoset).
Main comparison on the 3-player game; a short sweep over player count and deck size to see in
which direction the crossover moves.
"""
import time

import numpy as np

from common import MAIN, save, table, ma
from game import GameConfig
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import CFRPlus, evaluate
from mccfr import ESMCCFR


def run_cfr(tree, ex, budgets):
    s = CFRPlus(tree, ex)
    clock, out = 0.0, []
    for B in budgets:
        while clock < B:
            t0 = time.perf_counter(); s.iterate(1); clock += time.perf_counter() - t0
        e = evaluate(tree, s.average_strategy(), ex)
        out.append(dict(budget=B, sec=clock, iters=s.t, nashconv=e["nashconv"],
                        delta=e["delta"].tolist()))
    return out


def run_es(tree, ex, budgets, seed):
    s = ESMCCFR(tree, seed=seed, linear_avg=True)
    clock, out = 0.0, []
    for B in budgets:
        while clock < B:
            t0 = time.perf_counter(); s.iterate(100); clock += time.perf_counter() - t0
        e = evaluate(tree, s.average_strategy(), ex)
        out.append(dict(budget=B, sec=clock, iters=s.t, visits=s.visits, nashconv=e["nashconv"],
                        delta=e["delta"].tolist()))
    return out


def compare(cfg, budgets, seeds):
    tree = PublicTree(cfg)
    ex = TerminalEvaluator(tree, "exact")
    print(f"\n== {cfg.describe()}  public nodes {tree.M}, infosets {tree.stats()['infosets']}", flush=True)
    c = run_cfr(tree, ex, budgets)
    es = [run_es(tree, ex, budgets, sd) for sd in seeds]
    rows = []
    for k, B in enumerate(budgets):
        es_nc = [e[k]["nashconv"] for e in es]
        rows.append([B, c[k]["iters"], ma(c[k]["nashconv"]), es[0][k]["iters"],
                     f"{es[0][k]['visits']:.2e}", ma(np.mean(es_nc)),
                     "CFR+" if c[k]["nashconv"] < np.mean(es_nc) else "ES",
                     round(float(np.mean(es_nc) / c[k]["nashconv"]), 1)])
    hdr = ["budget s", "CFR+ iters", "CFR+ NashConv mA", "ES iters", "ES node visits",
           f"ES NashConv mA (mean of {len(seeds)} seeds)", "winner", "ES/CFR+ ratio"]
    print(table(hdr, rows), flush=True)
    return dict(config=cfg.describe(), nodes=tree.M, infosets=tree.stats()["infosets"],
                cfr=c, es=es, h=hdr, rows=rows)


if __name__ == "__main__":
    res = dict(main=compare(MAIN, [5, 15, 45, 90], seeds=[0, 1]))
    sweep = []
    for cfg in (GameConfig(2, 6, 2, 15, 2), GameConfig(3, 4, 2, 15, 2), GameConfig(3, 8, 2, 15, 2),
                GameConfig(4, 4, 2, 15, 1)):
        sweep.append(compare(cfg, [3, 10, 30], seeds=[0]))
    res["sweep"] = sweep
    rows = []
    for r in [res["main"]] + sweep:
        last = r["rows"][-1]
        rows.append([r["config"], r["nodes"], r["infosets"], last[0], last[2], last[5], last[6], last[7]])
    hdr = ["game", "public nodes", "infosets", "budget s", "CFR+ NashConv mA", "ES NashConv mA",
           "winner", "ES/CFR+"]
    print(table(hdr, rows))
    res["summary"] = dict(h=hdr, rows=rows)
    save("m4", res)
