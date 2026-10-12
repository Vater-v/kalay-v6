"""T3 (targeted): what can be certified in 3-player poker?

(a) Self-play multiplicity: CFR+ self-play (exact card removal) from K random
    initialisations (random initial regrets + random update order), 2000 iterations
    each.  Per run: NashConv, per-seat EV.  Then cross-run MIXES of average marginals
    (seat i from run a, the others from run b; and three different runs): NashConv.
(b) Card-sharing coalition (team.py): for every seat i,
      * coalition maxmin v_i (2p zero-sum CFR+, bracketed),
      * security level of each self-play run's sigma_i  = min over ANY behaviour of
        the other two seats (incl. collusion with shared cards),
      * coalition gap v_i - security(sigma_i).
    Fold-everything bound: check/fold guarantees -1 ante (the ante) per hand.
"""
import time
import multiprocessing as mp

import numpy as np

from common import MAIN, save, table, ma, RESULTS
from tree import PublicTree
from terminal import TerminalEvaluator
from cfr import CFRPlus, evaluate, normalize
from team import TeamGame

K = 8
ITERS = 2000
TEAM_ITERS = 1500
TREE = PublicTree(MAIN)


def selfplay(seed):
    rng = np.random.default_rng(seed)
    ex = TerminalEvaluator(TREE, "exact")
    s = CFRPlus(TREE, ex)
    if seed > 0:
        R0 = np.zeros_like(s.R)
        R0[TREE.dec_edges] = rng.exponential(1.0, (len(TREE.dec_edges), MAIN.deck)) * 30.0
        s.R = R0
        s.sigma = normalize(TREE, R0)
        s.players = list(rng.permutation(MAIN.n_players))
    t0 = time.time()
    s.iterate(ITERS)
    sig = s.average_strategy()
    e = evaluate(TREE, sig, ex)
    return ("sp", seed, sig, e["ev"], e["delta"], e["nashconv"], time.time() - t0, s.players)


def team(hero):
    g = TeamGame(TREE, hero)
    t0 = time.time()
    avg, log = g.solve(TEAM_ITERS, checkpoints=(100, 300, 1000, TEAM_ITERS))
    return ("team", hero, avg["H"], log, time.time() - t0)


def run(job):
    kind, x = job
    return selfplay(x) if kind == "sp" else team(x)


def embed_C(g, sigma):
    """Independent (non-sharing) coalition strategy from a 3p sigma, in team fiber."""
    sc = np.ones((TREE.M, g.F))
    for a, j in enumerate(g.opps):
        e = TREE.dec_edges[TREE.dec_owner == j]
        sc[e] = sigma[e][:, g.cards[a]]
    return sc


if __name__ == "__main__":
    ex = TerminalEvaluator(TREE, "exact")
    # correctness: team value of an embedded independent coalition == 3p EV
    sig_star = np.load(f"{RESULTS}/m2_sigma_3p.npy")
    base = evaluate(TREE, sig_star, ex)
    G = [TeamGame(TREE, h) for h in range(3)]
    chk = [G[h].value(sig_star, embed_C(G[h], sig_star)) - base["ev"][h] for h in range(3)]
    print("embedding check (team value - 3p EV):", chk, flush=True)
    assert max(abs(c) for c in chk) < 1e-10
    jobs = [("team", h) for h in range(3)] + [("sp", s) for s in range(K)]
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(run, jobs)
    sp = sorted([r for r in res if r[0] == "sp"], key=lambda r: r[1])
    tm = sorted([r for r in res if r[0] == "team"], key=lambda r: r[1])
    out = dict(selfplay=[], team=[], mixes=[])
    rows = []
    for r in sp:
        _, seed, sig, ev, d, nc, secs, order = r
        secu = [G[h].value(sig, None if False else np.ones((TREE.M, G[h].F)), br="C") for h in range(3)]
        out["selfplay"].append(dict(seed=seed, order=[int(o) for o in order], ev=ev.tolist(), delta=d.tolist(),
                                    nashconv=nc, security=secu, seconds=secs))
        rows.append([seed, str([int(o) for o in order]), *[round(float(v), 4) for v in ev], ma(nc),
                     *[round(s, 4) for s in secu]])
    print(table(["run", "update order", "EV0", "EV1", "EV2", "NashConv mA", "sec0", "sec1", "sec2"], rows))
    # strategy distance between runs: reach-weighted L1 of action probs, per seat
    sigs = [r[2] for r in sp]
    reaches = [None] * K
    from cfr import reach_all
    for k in range(K):
        reaches[k] = reach_all(TREE, sigs[k])
    dist = np.zeros((3, K, K))
    for i in range(3):
        e = TREE.dec_edges[TREE.dec_owner == i]
        par = TREE.parent[e]
        for a in range(K):
            for b in range(K):
                w = 0.5 * (reaches[a][i, par] + reaches[b][i, par])
                dist[i, a, b] = float((np.abs(sigs[a][e] - sigs[b][e]) * w).sum() /
                                      (w.sum() / TREE.n_children[par].mean()))
    out["strategy_distance_max"] = [float(dist[i].max()) for i in range(3)]
    out["strategy_distance_mean"] = [float(dist[i][np.triu_indices(K, 1)].mean()) for i in range(3)]
    # mixes
    mixrows = []
    for a in range(K):
        for b in range(K):
            if a == b:
                continue
            for i in range(3):
                m = sigs[b].copy()
                e = TREE.dec_edges[TREE.dec_owner == i]
                m[e] = sigs[a][e]
                ev = evaluate(TREE, m, ex)
                out["mixes"].append(dict(kind="one_from_a", a=a, b=b, seat=i, nashconv=ev["nashconv"],
                                         delta=ev["delta"].tolist(), ev=ev["ev"].tolist()))
    for (a, b, c) in [(0, 1, 2), (3, 4, 5), (5, 6, 7), (1, 4, 7), (2, 5, 0)]:
        m = sigs[a].copy()
        for i, src in ((1, b), (2, c)):
            e = TREE.dec_edges[TREE.dec_owner == i]
            m[e] = sigs[src][e]
        ev = evaluate(TREE, m, ex)
        out["mixes"].append(dict(kind="three_runs", a=a, b=b, c=c, nashconv=ev["nashconv"],
                                 delta=ev["delta"].tolist(), ev=ev["ev"].tolist()))
    nc_pure = [r["nashconv"] for r in out["selfplay"]]
    nc_mix1 = [m["nashconv"] for m in out["mixes"] if m["kind"] == "one_from_a"]
    nc_mix3 = [m["nashconv"] for m in out["mixes"] if m["kind"] == "three_runs"]
    out["summary"] = dict(nashconv_runs_mA=[ma(min(nc_pure)), ma(np.median(nc_pure)), ma(max(nc_pure))],
                          nashconv_mix1_mA=[ma(min(nc_mix1)), ma(np.median(nc_mix1)), ma(max(nc_mix1))],
                          nashconv_mix3_mA=[ma(min(nc_mix3)), ma(np.median(nc_mix3)), ma(max(nc_mix3))],
                          ev_range_per_seat=[[float(min(r["ev"][i] for r in out["selfplay"])),
                                              float(max(r["ev"][i] for r in out["selfplay"]))] for i in range(3)])
    print("summary", out["summary"], "dist", out["strategy_distance_mean"], out["strategy_distance_max"])
    # team results
    trows = []
    for r in tm:
        _, h, sigH, log, secs = r
        lo, hi = log[-1][1], log[-1][2]
        sec_runs = [s["security"][h] for s in out["selfplay"]]
        out["team"].append(dict(hero=h, log=[list(map(float, l)) for l in log], maxmin_lower=lo, maxmin_upper=hi,
                                seconds=secs, selfplay_security=sec_runs,
                                selfplay_ev=[s["ev"][h] for s in out["selfplay"]]))
        trows.append([h, round(lo, 4), round(hi, 4), round(min(sec_runs), 4), round(max(sec_runs), 4),
                      round(float(np.mean([s["ev"][h] for s in out["selfplay"]])), 4), round(secs)])
        print(f"hero {h}: log {log}", flush=True)
    print(table(["seat", "maxmin lower (sec of avg)", "maxmin upper (BR to avg C)", "self-play sec min",
                 "self-play sec max", "self-play EV mean", "s"], trows))
    save("t3", out)
