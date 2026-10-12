"""Independent reference pieces for the Slumbot audit (no import from poker/arena/slumbot).

- parse_action(): a re-implementation of ParseAction of https://slumbot.com/sample_api.py (the official
  sample client; reconstructed from the published file - the host was unreachable from this box, the
  fragments that GitHub code search returns for copies of it were used to confirm: client_pos 0 = big
  blind (second to act preflop, first postflop), 1 = small blind; sample response
  {'old_action': '', 'action': 'b200', 'client_pos': 0, 'hole_cards': ['Ac', '9d'], 'board': [], 'token': ...};
  "An all-in can contain streets with no action. For example: b20000c///").
- best5(): a brute-force 7-card evaluator (best of the 21 five-card subsets), for showdown winnings.
- card strings 'Ac', 'Td', ...
"""
from itertools import combinations

SMALL_BLIND, BIG_BLIND, STACK_SIZE, NUM_STREETS = 50, 100, 20000, 4
RANKS, SUITS = "23456789TJQKA", "cdhs"


def parse_action(action):
    """Slumbot's ParseAction: dict(st, pos, street_last_bet_to, total_last_bet_to, last_bet_size, last_bettor)
    or {'error': ...}. pos -1 = hand over."""
    st, street_last_bet_to, total_last_bet_to = 0, BIG_BLIND, BIG_BLIND
    last_bet_size, last_bettor, pos = BIG_BLIND - SMALL_BLIND, 0, 1
    sz = len(action)
    out = lambda: dict(st=st, pos=pos, street_last_bet_to=street_last_bet_to, total_last_bet_to=total_last_bet_to,
                       last_bet_size=last_bet_size, last_bettor=last_bettor)
    if sz == 0:
        return out()
    ends = False
    i = 0
    while i < sz:
        if st >= NUM_STREETS:
            return {"error": "Unexpected error"}
        c = action[i]
        i += 1
        if c == "k":
            if last_bet_size > 0:
                return {"error": "Illegal check"}
            if ends:
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != "/":
                        return {"error": "Missing slash"}
                    i += 1
                if st == NUM_STREETS - 1:
                    pos = -1
                else:
                    pos, st = 0, st + 1
                street_last_bet_to, ends = 0, False
            else:
                pos, ends = (pos + 1) % 2, True
        elif c == "c":
            if last_bet_size == 0:
                return {"error": "Illegal call"}
            if total_last_bet_to == STACK_SIZE:
                if i != sz:
                    for _ in range(st, NUM_STREETS - 1):
                        if i == sz:
                            return {"error": "Missing slash (end of string)"}
                        c2 = action[i]
                        i += 1
                        if c2 != "/":
                            return {"error": "Missing slash"}
                if i != sz:
                    return {"error": "Extra characters at end of action"}
                st, pos, last_bet_size = NUM_STREETS - 1, -1, 0
                return out()
            if ends:
                if st < NUM_STREETS - 1 and i < sz:
                    if action[i] != "/":
                        return {"error": "Missing slash"}
                    i += 1
                if st == NUM_STREETS - 1:
                    pos = -1
                else:
                    pos, st = 0, st + 1
                street_last_bet_to, ends = 0, False
            else:
                pos, ends = (pos + 1) % 2, True
            last_bet_size, last_bettor = 0, -1
        elif c == "f":
            if last_bet_size == 0:
                return {"error": "Illegal fold"}
            if i != sz:
                return {"error": "Extra characters at end of action"}
            pos = -1
            return out()
        elif c == "b":
            j = i
            while i < sz and "0" <= action[i] <= "9":
                i += 1
            if i == j:
                return {"error": "Missing bet size"}
            new_to = int(action[j:i])
            new_size = new_to - street_last_bet_to
            remaining = STACK_SIZE - total_last_bet_to
            if last_bet_size > 0:
                min_size = max(last_bet_size, BIG_BLIND)
            else:
                min_size = BIG_BLIND
            if min_size > remaining:
                min_size = remaining
            if new_size < min_size:
                return {"error": "Bet too small"}
            if new_size > remaining:
                return {"error": "Bet too big"}
            last_bet_size = new_size
            street_last_bet_to = new_to
            total_last_bet_to += last_bet_size
            last_bettor = pos
            pos, ends = (pos + 1) % 2, True
        else:
            return {"error": "Unexpected character in action"}
    return out()


def card_str(i):
    """0..51 -> 'Ac' style (rank-major, any fixed suit order)."""
    return RANKS[i // 4] + SUITS[i % 4]


def _rank5(cards):
    rs = sorted((RANKS.index(c[0]) for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(rs), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_high = 3
    counts = sorted(((rs.count(r), r) for r in uniq), reverse=True)
    shape = [c for c, _ in counts]
    order = [r for _, r in counts]
    if straight_high is not None and flush:
        return (8, straight_high)
    if shape == [4, 1]:
        return (7, *order)
    if shape == [3, 2]:
        return (6, *order)
    if flush:
        return (5, *rs)
    if straight_high is not None:
        return (4, straight_high)
    if shape == [3, 1, 1]:
        return (3, *order)
    if shape == [2, 2, 1]:
        return (2, *order)
    if shape == [2, 1, 1, 1]:
        return (1, *order)
    return (0, *rs)


def best5(cards7):
    return max(_rank5(c) for c in combinations(cards7, 5))
