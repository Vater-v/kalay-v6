"""Showdown and revealed-card profile from completed Slumbot records.

From the repository root: python -m tools.slumbot_showdowns HANDS_DIR [OUT_JSON] [--rank7 RANK7_NPY]
Default output: results/reports/slumbot_showdowns.json.
Uses the existing rank7 cache (memory-mapped); preflop equity uses 40000 seeded
Monte Carlo completions. Turn/flop comparisons enumerate completions exactly.
"""
import glob
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from math import comb

import numpy as np

from poker.arena.slumbot.protocol import parse_history, BIG_BLIND

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hands_dir", type=Path)
    ap.add_argument("out_json", type=Path, nargs="?", default=ROOT / "results/reports/slumbot_showdowns.json")
    ap.add_argument("--rank7", type=Path, default=ROOT / "results/cache/rank7_v1.npy")
    args = ap.parse_args(argv)
    hands_dir, out_json = str(args.hands_dir), args.out_json
    rank7_path = args.rank7
    out_json.parent.mkdir(parents=True, exist_ok=True)
    TABLE = np.load(rank7_path, mmap_mode="r")
    BIN = np.asarray([[comb(c, k) for c in range(52)] for k in range(8)], np.int64)
    RANKS, SUITS = "23456789TJQKA", "shdc"


    def pid_to_eng(pid):
        rank, suit = divmod(int(pid), 4)
        return suit * 13 + rank


    def str_to_eng(s):
        return SUITS.index(s[1]) * 13 + RANKS.index(s[0])


    def rank7(cards7):
        """cards7 [R, 7] engine ids -> int rank class (higher = better)."""
        c = np.sort(np.asarray(cards7, np.int64), axis=1)
        idx = BIN[np.arange(1, 8)[None, :], c].sum(1)
        return np.asarray(TABLE[idx], np.int32)


    RNG = np.random.default_rng(12345)


    def equity(hero2, bot2, board, mc=40000):
        """P(hero beats bot) + tie/2 over completions of the board (exact to the turn; preflop Monte Carlo)."""
        known = set(hero2) | set(bot2) | set(board)
        deck = np.asarray([c for c in range(52) if c not in known], np.int64)
        need = 5 - len(board)
        if need == 0:
            comps = np.zeros((1, 0), np.int64)
        elif need <= 2:
            comps = np.asarray(list(combinations(deck, need)), np.int64)
        else:
            comps = np.stack([RNG.choice(deck, need, replace=False) for _ in range(mc)])
        R = len(comps)
        b = np.concatenate([np.tile(np.asarray(board, np.int64), (R, 1)), comps], axis=1)
        rh = rank7(np.concatenate([np.tile(np.asarray(hero2, np.int64), (R, 1)), b], axis=1))
        rb = rank7(np.concatenate([np.tile(np.asarray(bot2, np.int64), (R, 1)), b], axis=1))
        return float(np.mean(rh > rb) + 0.5 * np.mean(rh == rb)), R


    def se(xs):
        n = len(xs)
        if n < 2:
            return float("nan")
        m = sum(xs) / n
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


    files = sorted(glob.glob(hands_dir + "/*.json"))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("status") != "completed":
            continue
        c = d["result"]["completed"][0]
        last = d["result"]["requests"][-1]["response"]
        rows.append(dict(pos=c["client_pos"], action=c["action"], hole=[pid_to_eng(x) for x in c["hole_cards"]],
                         board=[pid_to_eng(x) for x in c["board"]], win=c["winnings_chips"], base=c.get("baseline_winnings_chips"),
                         bot=[str_to_eng(x) for x in last["bot_hole_cards"]] if last.get("bot_hole_cards") else None,
                         decisions=d.get("decisions", [])))
    N = len(rows)
    print("hands", N, "with bot cards", sum(r["bot"] is not None for r in rows))

    allin = []            # dicts per all-in confrontation
    river_face = []       # hero decisions on the river facing a bet
    showdown = []
    base_pairs = []
    rake_hyp = []         # hypothetical NL5 rake paid by hero (chips), per hand
    for r in rows:
        h = parse_history(r["action"])
        hero = r["pos"]
        win = r["win"]
        if r["base"] is not None:
            base_pairs.append((win, r["base"]))
        # hypothetical rake NL5 (5%, cap 2.6 BB for 2 dealt), flop dealt
        if len(r["board"]) >= 3:
            inv = h.invested
            contested = 2 * min(inv)                     # uncalled part returned
            rk = min(contested * 500 // 10000, int(2.6 * BIG_BLIND))
            rake_hyp.append(rk if win > 0 else 0)
        else:
            rake_hyp.append(0)
        bot = r["bot"]
        if bot is None:
            continue
        full = len(r["board"]) == 5
        if h.terminal_reason == "showdown" and min(h.stacks) == 0 and full:
            last = h.events[-1]
            st = last.street
            board_known = r["board"][: (0, 3, 4, 5)[st]]
            eq, R = equity(r["hole"], bot, board_known)
            pot = sum(h.invested)
            ev = eq * pot - h.invested[hero]
            # who shoved: the last raise event's actor
            raises = [e for e in h.events if e.kind == "raise"]
            shover = raises[-1].actor if raises else None
            allin.append(dict(street=st, eq=eq, ev=ev, win=win, pot=pot, hero_shoved=(shover == hero), runouts=R,
                              pos="bb" if hero == 0 else "sb",
                              n_raises_street=sum(1 for e in h.events if e.street == st and e.kind == "raise")))
        if h.terminal_reason == "showdown" and full:
            rh = int(rank7(np.asarray([r["hole"] + r["board"]]))[0])
            rb = int(rank7(np.asarray([bot + r["board"]]))[0])
            showdown.append(dict(pot=sum(h.invested), win=win, hero_best=(rh > rb), tie=(rh == rb), pos="bb" if hero == 0 else "sb",
                                 allin=min(h.stacks) == 0))
        # river decisions facing a bet
        if full:
            rh = int(rank7(np.asarray([r["hole"] + r["board"]]))[0])
            rb = int(rank7(np.asarray([bot + r["board"]]))[0])
            for e in h.events:
                if e.street == 3 and e.actor == hero and e.to_call_before > 0:
                    pot_before_bet = e.pot_before - e.to_call_before
                    x = e.to_call_before / pot_before_bet if pot_before_bet else float("inf")
                    river_face.append(dict(kind=e.kind, x=x, hero_best=(rh > rb), tie=(rh == rb), to_call=e.to_call_before,
                                           pot_before=e.pot_before, win=win, n_raises=sum(1 for q in h.events if q.street == 3 and q.kind == "raise" and h.events.index(q) < h.events.index(e))))

    out = {"hands": N}
    # --- all-in
    def agg(items):
        if not items:
            return None
        ev = [a["ev"] for a in items]
        ac = [a["win"] for a in items]
        return dict(n=len(items), mean_equity=float(np.mean([a["eq"] for a in items])),
                    pot_weighted_equity=float(sum(a["eq"] * a["pot"] for a in items) / sum(a["pot"] for a in items)),
                    ev_bb100_contrib=100 * sum(ev) / N / BIG_BLIND, actual_bb100_contrib=100 * sum(ac) / N / BIG_BLIND,
                    luck_bb100_contrib=100 * (sum(ac) - sum(ev)) / N / BIG_BLIND,
                    actual_per_hand_bb=float(np.mean(ac)) / BIG_BLIND, ev_per_hand_bb=float(np.mean(ev)) / BIG_BLIND,
                    se_actual_bb100_contrib=100 * se(ac) * len(ac) / N / BIG_BLIND)

    out["allin_all"] = agg(allin)
    out["allin_by_street"] = {str(s): agg([a for a in allin if a["street"] == s]) for s in range(4)}
    out["allin_by_shover"] = {"hero_shoved": agg([a for a in allin if a["hero_shoved"]]), "bot_shoved": agg([a for a in allin if not a["hero_shoved"]])}
    out["allin_by_pos"] = {p: agg([a for a in allin if a["pos"] == p]) for p in ("bb", "sb")}
    out["allin_by_street_shover"] = {f"st{s} {'hero' if hs else 'bot'}_shoved": agg([a for a in allin if a["street"] == s and a["hero_shoved"] == hs]) for s in range(4) for hs in (True, False)}
    eqs = [a["eq"] for a in allin]
    out["allin_equity_hist"] = {f"{lo:.1f}-{lo + 0.1:.1f}": int(sum(lo <= e < lo + 0.1 for e in eqs)) for lo in np.arange(0, 1, 0.1)}
    # --- showdown
    def sd_agg(items):
        if not items:
            return None
        return dict(n=len(items), won_pct=100 * np.mean([i["hero_best"] for i in items]), tie_pct=100 * np.mean([i["tie"] for i in items]),
                    bb100_contrib=100 * sum(i["win"] for i in items) / N / BIG_BLIND, mean_bb=float(np.mean([i["win"] for i in items])) / BIG_BLIND)
    out["showdown_all"] = sd_agg(showdown)
    out["showdown_non_allin"] = sd_agg([s for s in showdown if not s["allin"]])
    out["showdown_by_pot"] = {}
    for lo, hi in ((0, 6), (6, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 401)):
        out["showdown_by_pot"][f"{lo:03d}-{hi:03d}"] = sd_agg([s for s in showdown if lo <= s["pot"] / BIG_BLIND < hi])
    out["showdown_by_pos"] = {p: sd_agg([s for s in showdown if s["pos"] == p]) for p in ("bb", "sb")}
    # --- river facing a bet
    def rf_agg(items):
        if not items:
            return None
        folds = [i for i in items if i["kind"] == "fold"]
        calls = [i for i in items if i["kind"] == "call"]
        raises = [i for i in items if i["kind"] == "raise"]
        d = dict(n=len(items), fold_pct=100 * len(folds) / len(items), call_pct=100 * len(calls) / len(items), raise_pct=100 * len(raises) / len(items))
        if folds:
            d["folded_winner_pct"] = 100 * np.mean([f["hero_best"] for f in folds])
            d["folded_tie_pct"] = 100 * np.mean([f["tie"] for f in folds])
            # ex-post value of calling instead (chips): win -> pot_before (incl. bet) ; lose -> -to_call
            d["fold_expost_call_value_bb100_contrib"] = 100 * sum((f["pot_before"] if f["hero_best"] else (0.5 * f["pot_before"] - 0.5 * f["to_call"] if f["tie"] else -f["to_call"])) for f in folds) / N / BIG_BLIND
        if calls:
            d["call_won_pct"] = 100 * np.mean([c["hero_best"] for c in calls])
            d["call_tie_pct"] = 100 * np.mean([c["tie"] for c in calls])
        xs = [i["x"] for i in items if math.isfinite(i["x"])]
        d["breakeven_bluff_pct_mean"] = 100 * float(np.mean([x / (1 + 2 * x) for x in xs])) if xs else None
        return d
    out["river_facing_bet_all"] = rf_agg(river_face)
    out["river_facing_bet_by_size"] = {}
    for lo, hi, name in ((0, 0.4, "<0.4"), (0.4, 0.6, "0.4-0.6 (half)"), (0.6, 0.85, "0.6-0.85"), (0.85, 1.2, "0.85-1.2 (pot)"), (1.2, 2.5, "1.2-2.5"), (2.5, 1e9, ">2.5 / allin")):
        out["river_facing_bet_by_size"][name] = rf_agg([i for i in river_face if lo <= i["x"] < hi])
    out["river_facing_bet_first_bet_only"] = rf_agg([i for i in river_face if i["n_raises"] == 1])
    out["river_facing_raise"] = rf_agg([i for i in river_face if i["n_raises"] >= 2])
    # --- baseline
    if base_pairs:
        w = np.asarray([p[0] for p in base_pairs], float)
        b = np.asarray([p[1] for p in base_pairs], float)
        vals = Counter(int(x) for x in b)
        out["baseline"] = dict(n=len(b), mean_bb100=100 * b.mean() / BIG_BLIND, se_bb100=100 * b.std(ddof=1) / math.sqrt(len(b)) / BIG_BLIND,
                               corr_with_winnings=float(np.corrcoef(w, b)[0, 1]), zero_pct=100 * np.mean(b == 0),
                               equal_to_winnings_pct=100 * np.mean(w == b), top_values=vals.most_common(12),
                               adjusted_bb100=100 * (w - b).mean() / BIG_BLIND, adjusted_se_bb100=100 * (w - b).std(ddof=1) / math.sqrt(len(b)) / BIG_BLIND,
                               winnings_se_bb100=100 * w.std(ddof=1) / math.sqrt(len(w)) / BIG_BLIND)
    # --- rake
    out["rake_hypothetical_NL5"] = dict(hero_paid_bb100=100 * sum(rake_hyp) / N / BIG_BLIND, hands_with_flop=int(sum(len(r["board"]) >= 3 for r in rows)))
    json.dump(out, open(out_json, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
