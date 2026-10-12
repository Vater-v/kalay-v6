"""T2 (targeted): pre-chance street-boundary labels.

A pre-chance situation sits at the END of flop betting, before the turn card:
(flop board b, pot share x, normalised ranges p0, p1).  Its exact normalised
value (same units as valuefn.py: u_i(h) * cs * x * M_opp = absolute CFV) is

  u_pre_i(h) = 1/(n_left - 2) * sum_{c != h, c not on board} (1 - p_opp(c)) * u_i^c(h)

where n_left = cards not on the board, u_i^c is the exact normalised value of the
turn subgame after card c with ranges p^c = p*[.!=c]/(1-p(c)).  (Each (h, h') pair
sees the turn card uniformly among the n_left - 2 cards not held by either player.)

Measured:
 (1) identity check with the checkdown value (no solver): W[b] p == formula over W[b+c].
 (2) bias of the naive averages a pipeline might use instead:
       naive_legal: 1/n_left * sum_{c != h} u^c(h)       (claim's literal "uniform over legal cards with 1[c not in h]")
       naive_noth : 1/(n_left-1) * sum_{c != h} u^c(h)   (uniform over cards not in h, normalised values)
 (3) single-card estimator L_c(h) = n_left/(n_left-2) * 1[c!=h] * (1-p_opp(c)) * u^c(h),
     c uniform over the n_left cards: unbiased; its noise vs the signal.
 (4) equal-compute regression: the same number of turn-subgame solves spent as
     m = 9 (all cards, exact labels), m = 3 or m = 1 sampled cards per situation
     (correspondingly 3x / 9x more situations).  Identical MLPs (residual over the
     exact flop checkdown, zero-sum projection, range-weighted loss).  Scored on held-out
     situations with exact labels.
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import multiprocessing as mp
import numpy as np
from common import setup, save, Log
from valuefn import (sample_situations, solve_situations, reachable_x, CheckdownV,
                     train_value_net)

CFG, B, T = setup()
N = CFG.n_cards
ITERS = int(os.environ.get("T2_ITERS", 300))
NFULL = [int(v) for v in os.environ.get("T2_NFULL", "500,2000").split(",")]
NTEST = int(os.environ.get("T2_NTEST", 400))
STEPS = int(os.environ.get("T2_STEPS", 6000))
XS = reachable_x(T, 1)


def expand(bid, x, p0, p1, cards_per):
    """Turn-subgame situations for chosen cards. cards_per[j] = list of cards."""
    tb, tx, tp0, tp1, own, cc = [], [], [], [], [], []
    for j in range(len(bid)):
        b = B.boards[bid[j]]
        for c in cards_per[j]:
            q = []
            for p in (p0[j], p1[j]):
                r = p.copy(); r[c] = 0.0
                q.append(r / r.sum() if r.sum() > 0 else B.mask[B.id(b + (c,))] / B.mask[B.id(b + (c,))].sum())
            tb.append(B.id(b + (c,))); tx.append(x[j]); tp0.append(q[0]); tp1.append(q[1])
            own.append(j); cc.append(c)
    return (np.array(tb), np.array(tx, float), np.array(tp0), np.array(tp1),
            np.array(own), np.array(cc))


def _solve(args):
    tb, tx, tp0, tp1 = args
    u0, u1, e = solve_situations(CFG, B, tb, tx, tp0, tp1, iters=ITERS, chunk=1500)
    return u0, u1, e


def solve_par(tb, tx, tp0, tp1, pool, chunk=1500):
    jobs = [(tb[a:a + chunk], tx[a:a + chunk], tp0[a:a + chunk], tp1[a:a + chunk])
            for a in range(0, len(tb), chunk)]
    res = pool.map(_solve, jobs)
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
            float(np.mean([r[2] for r in res])))


def legal_cards(bid):
    return [[c for c in range(N) if c not in B.boards[b]] for b in bid]


def aggregate(n_sit, p0, p1, own, cc, u0, u1, nleft):
    """Exact pre-chance label (all cards) or the unbiased m-card estimate (any subset):
    mean over the given cards of L_c(h)."""
    out = [np.zeros((n_sit, N)), np.zeros((n_sit, N))]
    cnt = np.zeros(n_sit)
    np.add.at(cnt, own, 1)
    for i, (u, popp) in enumerate(((u0, p1), (u1, p0))):
        notc = (np.arange(N)[None, :] != cc[:, None]).astype(float)
        L = (nleft / (nleft - 2)) * notc * (1 - popp[own, cc])[:, None] * u
        np.add.at(out[i], own, L)
        out[i] /= cnt[:, None]
    return out


def naive(n_sit, own, cc, u0, u1, nleft):
    res = {}
    for name, div in (("naive_legal", nleft), ("naive_noth", nleft - 1)):
        o = [np.zeros((n_sit, N)), np.zeros((n_sit, N))]
        notc = (np.arange(N)[None, :] != cc[:, None]).astype(float)
        for i, u in enumerate((u0, u1)):
            np.add.at(o[i], own, notc * u)
            o[i] /= div
        res[name] = o
    return res


def rw_rmse(pred, truth, p0, p1, bid):
    m = B.mask[bid]
    e = (((pred[0] - truth[0]) ** 2) * p0 * m).sum(1) + (((pred[1] - truth[1]) ** 2) * p1 * m).sum(1)
    return float(np.sqrt(e.mean() / 2))


def _train(args):
    name, data, val, test = args
    t0 = time.time()
    V, vl = train_value_net(B, CFG, data, val, residual=True, steps=STEPS, range_weighted=True)
    bid, x, p0, p1, u0, u1 = test
    a0, a1 = V(bid, x, p0, p1)
    m = B.mask[bid]
    rmse_all = float(np.sqrt((((a0 - u0) ** 2 + (a1 - u1) ** 2) * m).sum() / (2 * m.sum())))
    return name, rw_rmse((a0, a1), (u0, u1), p0, p1, bid), rmse_all, time.time() - t0


if __name__ == "__main__":
    log = Log("t2.log")
    rng = np.random.default_rng(7)
    out = {}
    # (1) checkdown identity
    bid, x, p0, p1 = sample_situations(CFG, B, 300, 1, XS, rng)
    cards = legal_cards(bid)
    tb, tx, tp0, tp1, own, cc = expand(bid, x, p0, p1, cards)
    cd0, cd1 = CheckdownV(B)(tb, tx, tp0, tp1)
    nleft = N - 1
    agg = aggregate(len(bid), p0, p1, own, cc, cd0, cd1, nleft)
    ex0, ex1 = CheckdownV(B)(bid, x, p0, p1)
    m = B.mask[bid]
    nv = naive(len(bid), own, cc, cd0, cd1, nleft)
    out["checkdown_identity_maxabs"] = float(max(np.abs((agg[0] - ex0) * m).max(), np.abs((agg[1] - ex1) * m).max()))
    for k, v in nv.items():
        out[f"checkdown_{k}_maxabs"] = float(np.abs((v[0] - ex0) * m).max())
        out[f"checkdown_{k}_rw_rmse"] = rw_rmse(v, (ex0, ex1), p0, p1, bid)
    log("checkdown identity", {k: round(v, 6) if isinstance(v, float) else v for k, v in out.items()})

    with mp.get_context("fork").Pool(4) as pool:
        # (2)+(4) pools of situations
        nf = max(NFULL)
        sets = {}
        t0 = time.time()
        # full-card pool (exact labels): training pool and test set
        for name, n in (("full", nf), ("test", NTEST)):
            bid, x, p0, p1 = sample_situations(CFG, B, n, 1, XS, rng)
            tb, tx, tp0, tp1, own, cc = expand(bid, x, p0, p1, legal_cards(bid))
            u0, u1, e = solve_par(tb, tx, tp0, tp1, pool)
            sets[name] = (bid, x, p0, p1, own, cc, u0, u1)
            log(f"solved {name}: {n} situations, {len(tb)} turn subgames, label expl {e:.5f} x, {time.time()-t0:.0f}s")
        # sampled pools: m cards per situation, n = nf*9/m situations
        for mm in (3, 1):
            n = nf * nleft // mm
            bid, x, p0, p1 = sample_situations(CFG, B, n, 1, XS, rng)
            ch = [list(rng.choice(c, size=mm, replace=False)) for c in legal_cards(bid)]
            tb, tx, tp0, tp1, own, cc = expand(bid, x, p0, p1, ch)
            u0, u1, e = solve_par(tb, tx, tp0, tp1, pool)
            sets[f"m{mm}"] = (bid, x, p0, p1, own, cc, u0, u1)
            log(f"solved m={mm}: {n} situations, {len(tb)} turn subgames, {time.time()-t0:.0f}s")

        # (2) bias of naive averages + (3) single-card noise, on the test set
        bid, x, p0, p1, own, cc, u0, u1 = sets["test"]
        truth = aggregate(len(bid), p0, p1, own, cc, u0, u1, nleft)
        nv = naive(len(bid), own, cc, u0, u1, nleft)
        m = B.mask[bid]
        held = [(p0 > 1e-3) & (m > 0), (p1 > 1e-3) & (m > 0)]
        for k, v in nv.items():
            rel = np.concatenate([(v[i] - truth[i])[held[i]] for i in (0, 1)])
            tru = np.concatenate([truth[i][held[i]] for i in (0, 1)])
            out[f"{k}_rw_rmse"] = rw_rmse(v, truth, p0, p1, bid)
            out[f"{k}_mean_abs_err_held"] = float(np.abs(rel).mean())
            out[f"{k}_ratio_sum"] = float((np.concatenate([v[i][held[i]] for i in (0, 1)]) * np.sign(tru)).sum()
                                          / np.abs(tru).sum())
        # single-card noise: per (situation, hand) variance across c of L_c, range-weighted
        notc = (np.arange(N)[None, :] != cc[:, None]).astype(float)
        noise = 0.0; sig = 0.0
        for i, (u, popp, pown) in enumerate(((u0, p1, p0), (u1, p0, p1))):
            L = (nleft / (nleft - 2)) * notc * (1 - popp[own, cc])[:, None] * u
            # average over the nleft equiprobable draws (all legal cards are present)
            dev = (L - truth[i][own]) ** 2
            var = np.zeros((len(bid), N)); np.add.at(var, own, dev); var /= nleft
            noise += float((var * pown * m).sum(1).mean())
            mu = (truth[i] * pown * m).sum(1, keepdims=True)
            sig += float((((truth[i] - truth[i].mean(0)) ** 2) * pown * m).sum(1).mean())
        out["single_card_noise_rw_rms"] = float(np.sqrt(noise / 2))
        out["signal_rw_rms_across_situations"] = float(np.sqrt(sig / 2))
        cdv = CheckdownV(B)(bid, x, p0, p1)
        out["checkdown_vs_truth_rw_rmse"] = rw_rmse(cdv, truth, p0, p1, bid)
        test = (bid, x, p0, p1, truth[0], truth[1])
        log("bias/noise", {k: round(v, 5) for k, v in out.items() if not k.startswith("checkdown_identity")})

        # (4) equal-compute training
        jobs = []
        for nfull in NFULL:
            for name, mm in (("m9_exact", 9), ("m3", 3), ("m1", 1)):
                key = "full" if mm == 9 else f"m{mm}"
                bidA, xA, p0A, p1A, ownA, ccA, u0A, u1A = sets[key]
                nsit = nfull * nleft // mm
                sel = ownA < nsit
                lab = aggregate(nsit, p0A[:nsit], p1A[:nsit], ownA[sel], ccA[sel], u0A[sel], u1A[sel], nleft)
                d = (bidA[:nsit], xA[:nsit], p0A[:nsit], p1A[:nsit], lab[0], lab[1])
                nv_ = max(1, nsit // 10)
                train = tuple(z[nv_:] for z in d); val = tuple(z[:nv_] for z in d)
                jobs.append((f"{name}@{nfull*nleft}solves", train, val, test))
        res = pool.map(_train, jobs)
    out["training"] = []
    for name, rw, ra, s in res:
        out["training"].append(dict(variant=name, test_rw_rmse=rw, test_rmse_all=ra, train_s=s))
        log(f"train {name:22s} test range-w RMSE {rw:.4f}  unweighted {ra:.4f}  ({s:.0f}s)")
    save("t2.json", out)
    log("done")
