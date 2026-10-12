"""Independent brute-force 7-card evaluator vs poker.envs.evaluator.evaluate_hand.

Reference: best of the 21 five-card subsets, each ranked by a plain-Python 5-card ranker returning
(category, tiebreak tuple). The engine score must be an order isomorphism of the reference key:
equal keys <=> equal scores, and key order == score order. Card id = suit * 13 + rank (rank 0 = deuce).
Sets: random 7-card hands, every 7-card rank multiset with no flush possible (suits assigned with <= 4 per
suit), flush-heavy random hands (5/6/7 suited), hand-written adversarial cases, and board-plays pairs.
"""
import itertools, json, random, sys, time
import numpy as np

def rank5(cards):
    ranks = sorted((c % 13 for c in cards), reverse=True)
    suits = [c // 13 for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_high = 3          # wheel: five-high
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [g[1] for g in groups]
    gr = [g[0] for g in groups]
    if straight_high is not None and flush:
        return (8, (straight_high,))
    if shape[0] == 4:
        return (7, (gr[0], gr[1]))
    if shape == [3, 2]:
        return (6, (gr[0], gr[1]))
    if flush:
        return (5, tuple(ranks))
    if straight_high is not None:
        return (4, (straight_high,))
    if shape[0] == 3:
        return (3, tuple(gr))
    if shape == [2, 2, 1]:
        return (2, tuple(gr))
    if shape[0] == 2:
        return (1, tuple(gr))
    return (0, tuple(ranks))

def rank7(cards):
    return max(rank5(c) for c in itertools.combinations(cards, 5))

def main():
    import jax, jax.numpy as jnp
    from poker.envs.evaluator import evaluate_hand
    ev = jax.jit(jax.vmap(evaluate_hand))
    rng = random.Random(12345)
    sets = {}
    sets["random"] = [rng.sample(range(52), 7) for _ in range(30000)]
    # every rank multiset of 7 cards (each rank <= 4 times), suits assigned round-robin so no suit has 5 cards
    ms = []
    for combo in itertools.combinations_with_replacement(range(13), 7):
        cnt = {r: combo.count(r) for r in set(combo)}
        if max(cnt.values()) > 4:
            continue
        cards, used = [], {}
        k = 0
        for r in combo:
            s = used.get(r, 0)
            used[r] = s + 1
            # suit of the j-th copy of rank r: (j + offset) % 4 rotates to avoid 5 in one suit
            cards.append(((s + k) % 4) * 13 + r)
            k += 1
        if len(set(cards)) != 7:
            # fall back: plain suits 0..3 per copy
            cards = []
            used = {}
            for r in combo:
                s = used.get(r, 0); used[r] = s + 1; cards.append(s * 13 + r)
        ms.append(cards)
    sets["rank_multisets"] = ms
    fl = []
    for _ in range(20000):
        m = rng.choice([5, 6, 7]); suit = rng.randrange(4)
        suited = [suit * 13 + r for r in rng.sample(range(13), m)]
        rest = rng.sample([c for c in range(52) if c // 13 != suit], 7 - m)
        fl.append(suited + rest)
    sets["flush_heavy"] = fl
    sf = []
    for _ in range(3000):   # straight-flush stress: 5 consecutive suited ranks (incl. wheel) + 2 random
        suit = rng.randrange(4); lo = rng.randrange(-1, 9)
        rs = [(lo + i) % 13 if lo >= 0 else [12, 0, 1, 2, 3][i] for i in range(5)]
        base = [suit * 13 + r for r in rs]
        rest = rng.sample([c for c in range(52) if c not in base], 2)
        sf.append(base + rest)
    sets["straight_flush"] = sf
    C = lambda s: [ {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}[t[0]] + 13*'cdhs'.index(t[1]) for t in s.split()]
    adv = {
        "wheel": "Ac 2d 3h 4s 5c 9d Kh",
        "six_high_straight": "2d 3h 4s 5c 6c 9d Kh",
        "wheel_plus_six": "Ac 2d 3h 4s 5c 6d Kh",
        "broadway": "Ac Kd Qh Js Tc 2d 3h",
        "flush6": "Ah 9h 7h 5h 3h 2h Kc",
        "flush7": "Ah Kh 9h 7h 5h 3h 2h",
        "flush_vs_straight": "4h 5h 6h 7c 8h 2h Kd",
        "sf_wheel": "Ah 2h 3h 4h 5h Kh Qh",
        "sf_6high_with_wheel": "Ah 2h 3h 4h 5h 6h Kd",
        "two_trips": "Ac Ad Ah Kc Kd Kh 2s",
        "trips_two_pairs": "Ac Ad Ah Kc Kd Qh Qs",
        "quads_kicker": "Ac Ad Ah As Kc Qd Jh",
        "quads_pair_kicker": "2c 2d 2h 2s 3c 3d Kh",
        "quads_trips": "2c 2d 2h 2s Ac Ad Ah",
        "three_pairs": "Ac Ad Kc Kd Qc Qd 2h",
        "three_pairs_low_kicker": "4c 4d 3c 3d 2c 2d 5h",
        "counterfeit_two_pair": "7c 7d 9h 9s Qc Qd Ah",
        "board_quads": "9c 9d 9h 9s 2c 3d 4h",
        "straight_and_trips": "5c 5d 5h 6s 7c 8d 9h",
    }
    sets["adversarial"] = [C(v) for v in adv.values()]
    # board-plays ties: two hands share a 5-card board that is the best hand for both (e.g. a board straight/flush)
    bp = []
    for _ in range(4000):
        cards = rng.sample(range(52), 9)
        bp.append(cards[:2] + cards[4:9]); bp.append(cards[2:4] + cards[4:9])
    sets["shared_board_pairs"] = bp

    out = {}
    all_keys, all_scores = [], []
    t0 = time.time()
    for name, hands in sets.items():
        arr = np.asarray(hands, np.int32)
        assert all(len(set(h)) == 7 for h in hands), name
        scores = [int(x) for x in np.asarray(ev(jnp.asarray(arr)))]
        keys = [rank7(h) for h in hands]
        cat_mismatch = sum(int(scores[i] >> 20) != keys[i][0] for i in range(len(hands)))
        out[name] = {"hands": len(hands), "category_mismatches": cat_mismatch}
        all_keys += keys; all_scores += scores
        if name == "adversarial":
            out[name]["cases"] = {k: {"ref": list(map(lambda x: x if isinstance(x, int) else list(x), keys[i])),
                                      "engine_category": int(scores[i] >> 20), "engine_score": int(scores[i])}
                                  for i, k in enumerate(adv)}
        if name == "shared_board_pairs":
            sign_bad = 0
            for i in range(0, len(hands), 2):
                a = (keys[i] > keys[i+1]) - (keys[i] < keys[i+1])
                b = (scores[i] > scores[i+1]) - (scores[i] < scores[i+1])
                sign_bad += a != b
            out[name]["pair_order_mismatches"] = sign_bad
            out[name]["ties_in_ref"] = sum(keys[i] == keys[i+1] for i in range(0, len(hands), 2))
    # global order isomorphism over all hands
    pairs = sorted(zip(all_keys, all_scores))
    viol_order = 0; viol_eq = 0
    for (k1, s1), (k2, s2) in zip(pairs, pairs[1:]):
        if k1 == k2 and s1 != s2: viol_eq += 1
        if k1 < k2 and not s1 < s2: viol_order += 1
    out["global"] = {"hands": len(pairs), "distinct_ref_keys": len(set(all_keys)), "distinct_scores": len(set(all_scores)),
                     "equal_key_different_score": viol_eq, "order_violations_adjacent": viol_order,
                     "seconds": round(time.time() - t0, 1)}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cases"} for k, v in out.items()}, indent=1))
    path = sys.argv[1] if len(sys.argv) > 1 else "results/audit/envs/eval_check.json"
    json.dump(out, open(path, "w"), indent=1)

if __name__ == "__main__":
    main()
