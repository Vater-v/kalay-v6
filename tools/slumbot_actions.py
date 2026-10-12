"""Action and loss profile from completed Slumbot hand records.

From the repository root: python -m tools.slumbot_actions HANDS_DIR [OUT_JSON]
Default output: results/reports/slumbot_actions.json.
Groups action strings, bet sizes and per-hand loss; does not run a policy.
"""
import glob
import json
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from poker.arena.slumbot.protocol import parse_history, BIG_BLIND

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hands_dir", type=Path)
    ap.add_argument("out_json", type=Path, nargs="?", default=ROOT / "results/reports/slumbot_actions.json")
    args = ap.parse_args(argv)
    hands_dir, out_json = str(args.hands_dir), args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
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
        rows.append(dict(n=c["number"], pos=c["client_pos"], action=c["action"], hole=c["hole_cards"], board=c["board"],
                         win=c["winnings_chips"], base=c.get("baseline_winnings_chips"), bot=last.get("bot_hole_cards"),
                         decisions=d.get("decisions", [])))
    N = len(rows)
    print(f"hands {N}")


    def se(xs):
        n = len(xs)
        if n < 2:
            return float("nan")
        m = sum(xs) / n
        v = sum((x - m) ** 2 for x in xs) / (n - 1)
        return math.sqrt(v / n)


    def stat(xs, total=N):
        """count, bb/100 (mean), SE, share of total loss in bb/100 units (sum/total*100)."""
        xs = [x / BIG_BLIND for x in xs]
        n = len(xs)
        return dict(hands=n, bb100=(100 * sum(xs) / n) if n else None, se=(100 * se(xs)) if n > 1 else None,
                    contrib_bb100=100 * sum(xs) / total)


    groups = defaultdict(list)
    sizes = Counter()
    sizes_pre = Counter()
    hero_sizes = Counter()
    lines = defaultdict(list)
    bucket_by_street = defaultdict(Counter)
    over_cap_hands = []
    allin_hands = []
    base_vals = []
    deltas = []
    for r in rows:
        h = parse_history(r["action"])
        hero = r["pos"]
        pos = "bb" if hero == 0 else "sb"
        win = r["win"]
        deltas.append(win)
        if r["base"] is not None:
            base_vals.append(r["base"])
        reason = h.terminal_reason
        final_street = h.street
        last = h.events[-1] if h.events else None
        who = "none"
        if reason == "fold":
            who = "hero_fold" if last.actor == hero else "opp_fold"
        key = (pos, reason + ("" if reason == "showdown" else ":" + who), final_street)
        groups[("terminal", key)].append(win)
        groups[("pos", pos)].append(win)
        groups[("terminal_all", reason + ("" if reason == "showdown" else ":" + who))].append(win)
        pot = sum(h.invested)
        pot_bb = pot / BIG_BLIND
        for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 25), (25, 50), (50, 100), (100, 200), (200, 401)):
            if lo <= pot_bb < hi:
                groups[("pot", f"{lo:03d}-{hi:03d}")].append(win)
                groups[("pot_pos", pos, f"{lo:03d}-{hi:03d}")].append(win)
                break
        # bet sizes (fraction of pot after call), raises only
        raises_this_street = Counter()
        for e in h.events:
            if e.kind == "raise":
                base = e.pot_before + e.to_call_before
                level = e.street_total - e.contribution + e.to_call_before
                x = (e.street_total - level) / base if base else 0
                allin = e.stacks_before[e.actor] == e.contribution
                k = "allin" if allin else f"{x:.2f}"
                # coarse bins
                if allin:
                    b = "allin"
                else:
                    b = min(((abs(x - v), name) for v, name in ((0.25, "1/4"), (0.333, "1/3"), (0.5, "1/2"), (0.667, "2/3"), (0.75, "3/4"), (1.0, "pot"), (1.5, "1.5p"), (2.0, "2p"), (3.0, "3p+"))))[1]
                if e.actor == hero:
                    hero_sizes[(e.street, b)] += 1
                else:
                    (sizes_pre if e.street == 0 else sizes)[(e.street, b, raises_this_street[e.street])] += 1
                raises_this_street[e.street] += 1
        # preflop line
        pre = [e for e in h.events if e.street == 0]
        pl = "/".join(("H" if e.actor == hero else "O") + {"raise": "r", "call": "c", "check": "k", "fold": "f"}[e.kind] for e in pre)
        lines[(pos, pl)].append(win)
        # all-in involvement
        if any(e.kind == "raise" and e.stacks_before[e.actor] == e.contribution for e in h.events):
            allin_hands.append(win)
        # over cap: any street with >= 3 raises (cap 2) where hero had to act after
        for st in range(4):
            n_r = sum(1 for e in h.events if e.street == st and e.kind == "raise")
            if n_r >= 3:
                over_cap_hands.append(win)
                groups[("over_cap_street", st)].append(win)
                break
        # decision buckets per street
        # map decisions to streets by walking hero events
        hero_events = [e for e in h.events if e.actor == hero]
        for e, dcs in zip(hero_events, r["decisions"]):
            bucket_by_street[e.street][dcs["bucket"]] += 1

    out = {"hands": N, "overall": stat(deltas), "baseline_adjusted": None}
    if base_vals and len(base_vals) == N:
        adj = [r["win"] - r["base"] for r in rows]
        out["baseline_adjusted"] = stat(adj)
        out["baseline_mean_bb100"] = 100 * sum(base_vals) / N / BIG_BLIND
        out["baseline_nonzero"] = sum(1 for b in base_vals if b != 0)
    out["groups"] = {}
    for k, v in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out["groups"][" ".join(str(x) for x in k)] = stat(v)
    out["allin_hands"] = stat(allin_hands)
    out["over_cap_hands"] = stat(over_cap_hands)
    out["preflop_lines"] = {" ".join(k): stat(v) for k, v in sorted(lines.items(), key=lambda kv: -len(kv[1]))[:40]}
    out["opp_sizes_postflop"] = {f"st{k[0]} {k[1]} r{k[2]}": v for k, v in sorted(sizes.items())}
    out["opp_sizes_preflop"] = {f"st{k[0]} {k[1]} r{k[2]}": v for k, v in sorted(sizes_pre.items())}
    out["hero_sizes"] = {f"st{k[0]} {k[1]}": v for k, v in sorted(hero_sizes.items())}
    out["hero_bucket_by_street"] = {str(k): dict(v) for k, v in sorted(bucket_by_street.items())}
    json.dump(out, open(out_json, "w"), indent=1)


    def show(name, d):
        print(f"\n== {name}")
        for k, v in d.items():
            if isinstance(v, dict) and "hands" in v:
                print(f"{k:45s} n={v['hands']:6d}  bb/100={v['bb100'] if v['bb100'] is None else round(v['bb100'], 1)!s:>8}  "
                      f"SE={v['se'] if v['se'] is None else round(v['se'], 1)!s:>6}  contrib={v['contrib_bb100']:+7.2f}")
            else:
                print(f"{k:45s} {v}")


    print("overall", out["overall"])
    print("baseline_adjusted", out["baseline_adjusted"], "baseline mean bb/100", out.get("baseline_mean_bb100"), "nonzero", out.get("baseline_nonzero"))
    show("groups", out["groups"])
    print("allin", out["allin_hands"]); print("over_cap", out["over_cap_hands"])
    show("preflop lines (top 40)", out["preflop_lines"])
    show("opp sizes preflop", out["opp_sizes_preflop"])
    show("opp sizes postflop", out["opp_sizes_postflop"])
    show("hero sizes", out["hero_sizes"])
    show("hero bucket by street", out["hero_bucket_by_street"])


if __name__ == "__main__":
    main()
