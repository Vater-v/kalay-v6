"""The Slumbot card and history path, offline: protocol payloads (the shapes of slumbot.com/sample_api.py) through
parse_response -> LiveSlumbotBot._hand_start / _replay -> the engine state the live player decides at, and the
protocol's own serializer back. No solver, no network, no artifacts: the bot object is built without __init__.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tests/test_slumbot_replay.py
"""
import itertools
import os
import sys
import threading
import traceback

os.environ.setdefault("POKER_REGIME", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                      # noqa: E402
import numpy as np              # noqa: E402

from poker.arena.slumbot import protocol as P                      # noqa: E402
from poker.arena.slumbot.live_bot import LiveSlumbotBot             # noqa: E402
from poker.envs import config                                       # noqa: E402
from poker.envs.betting import init_betting                         # noqa: E402
from poker.envs.transition import step_target                       # noqa: E402
from poker.player import live as LV                                 # noqa: E402
from poker.solver import subgame as SG                              # noqa: E402

RANKS, SUITS = "23456789TJQKA", "shdc"


def bare_bot():
    """A LiveSlumbotBot with only the fields _hand_start / _replay read (no critics, no server)."""
    bot = LiveSlumbotBot.__new__(LiveSlumbotBot)
    bot.n = 2
    base = init_betting(2, jax.numpy.full(2, P.STACK_SIZE, jax.numpy.int32), history=True, max_stack=P.STACK_SIZE,
                        rake_level=0.0)
    bot.base = jax.tree.map(lambda x: np.array(x, copy=True), base)
    bot._step = jax.jit(step_target)
    bot.lock = threading.Lock()
    bot.stats = {"replays": 0}
    return bot


def payload(action, client_pos, hole, board, winnings=None, token="t0k3n"):
    d = {"old_action": "", "action": action, "client_pos": client_pos, "hole_cards": list(hole), "board": list(board),
         "token": token}
    if winnings is not None:
        d["winnings"] = winnings
    return d


def card_str(c):
    """engine card (suit * 13 + rank) -> protocol string."""
    return RANKS[c % 13] + SUITS[c // 13]


CASES = [
    # (action, client_pos, hole, board): non-terminal, the client to act
    ("", 1, ["Ac", "Kd"], []),                                    # SB (button) opens the hand
    ("b200", 0, ["Th", "Ts"], []),                                 # BB facing an open
    ("b200b600", 1, ["Ac", "Kd"], []),                            # SB facing a 3-bet
    ("b200b600b1800", 0, ["Th", "Ts"], []),                       # BB facing a 4-bet
    ("b200c/", 0, ["Th", "Ts"], ["7h", "8s", "2c"]),              # flop, BB acts first (with the separator)
    ("b200c", 0, ["Th", "Ts"], ["7h", "8s", "2c"]),               # flop, without the separator
    ("b200c/k", 1, ["Ac", "Kd"], ["7h", "8s", "2c"]),             # flop, SB after a check
    ("b200c/kb150", 0, ["Th", "Ts"], ["7h", "8s", "2c"]),         # flop, BB facing an odd 1.5 BB bet
    ("b200c/kb150b450", 1, ["Ac", "Kd"], ["7h", "8s", "2c"]),     # flop, SB facing a raise
    ("b200c/kk/", 0, ["Th", "Ts"], ["7h", "8s", "2c", "Qd"]),     # turn
    ("b200c/kk/b300c/", 0, ["Th", "Ts"], ["7h", "8s", "2c", "Qd", "2d"]),   # river
    ("b200c/kk/b300c/k", 1, ["Ac", "Kd"], ["7h", "8s", "2c", "Qd", "2d"]),
    ("b200c/kk/b300c/kb1234", 0, ["Th", "Ts"], ["7h", "8s", "2c", "Qd", "2d"]),       # any size
    ("ck/", 0, ["2c", "3d"], ["Ah", "Ad", "Ac"]),                  # limp, check: flop, BB acts first
    ("c", 0, ["Jc", "Jd"], []),                                    # BB facing a limp
    ("ck/b100", 1, ["Ac", "Kd"], ["7h", "8s", "2c"]),             # flop: BB bets the minimum, SB to act
    ("b300c/b250b19700", 0, ["Th", "Ts"], ["7h", "8s", "2c"]),     # flop: SB shoves (19700 = all of its chips) over a bet
    ("b20000", 0, ["Th", "Ts"], []),                              # BB facing an open shove
]
REJECTED = [                                                       # (action, client_pos, hole, board): parse_response must refuse
    ("b300c/b250b19700", 1, ["Ac", "Kd"], ["7h", "8s", "2c"]),     # not the client's turn
    ("b200c/", 0, ["Th", "Ts"], []),                                # the flop is due but no board
    ("b200c/", 0, ["Th", "Th"], ["7h", "8s", "2c"]),               # a duplicate card
    ("b200c/", 0, ["Th", "Ts"], ["7h", "8s", "Ts"]),               # a hole card on the board
    ("b200c/", 0, ["1h", "Ts"], ["7h", "8s", "2c"]),               # not a card
    ("b200b250", 0, ["Th", "Ts"], []),                              # an under-minimum raise that is not all-in
]
TERMINAL = [
    ("b200f", 1, ["Ac", "Kd"], [], 100),                          # BB folds to the open: SB wins the BB
    ("b200f", 0, ["Th", "Ts"], [], -100),
    ("b20000c", 0, ["Th", "Ts"], ["7h", "8s", "2c", "Qd", "2d"], 20000),      # called all-in, no separators
    ("b20000c///", 1, ["Ac", "Kd"], ["7h", "8s", "2c", "Qd", "2d"], -20000),  # with all the separators
    ("b200c/kk/kk/kk", 0, ["Th", "Ts"], ["7h", "8s", "2c", "Qd", "2d"], 0),   # showdown chop
]


def check_case(bot, action, client_pos, hole, board, winnings=None):
    resp = P.parse_response(payload(action, client_pos, hole, board, winnings))
    hero = P.SEAT[client_pos]
    # 1. cards: protocol ids -> engine ids -> back to the strings Slumbot sent
    eng_hole = [P.our_card(c) for c in resp.hole_cards]
    eng_board = [P.our_card(c) for c in resp.board]
    assert [card_str(c) for c in eng_hole] == hole, (eng_hole, hole)
    assert [card_str(c) for c in eng_board] == board, (eng_board, board)
    assert all(0 <= c < 52 for c in eng_hole + eng_board) and len(set(eng_hole + eng_board)) == len(eng_hole + eng_board)
    if resp.state.terminal:
        return {"terminal": resp.state.terminal_reason, "winnings": resp.winnings}
    # 2. the replay onto the engine
    steps, state, hero2 = bot._replay(resp)
    assert hero2 == hero
    assert int(state.to_act) == hero, (int(state.to_act), hero)
    assert int(state.street) == resp.state.street
    assert np.asarray(state.hole)[hero].tolist() == eng_hole
    vis = SG.visible_board(state)
    assert vis[vis >= 0].tolist() == eng_board, (vis, eng_board)
    assert [int(state.stack[P.SEAT[p]]) for p in (0, 1)] == list(resp.state.stacks)
    assert [int(state.bet_street[P.SEAT[p]]) for p in (0, 1)] == list(resp.state.street_bets)
    # 3. the hand index the player would use, and the legality of every candidate reply
    h = LV.hand_index(np.asarray(state.hole)[hero])
    assert 0 <= h < 1326
    info = LV._info(state)
    replies = []
    if info["fold_ok"]:
        replies.append(P.serialize_action(resp.state, "fold"))
    replies.append(P.serialize_action(resp.state, "check_call"))
    if info["raise_ok"]:
        for total in sorted({info["minimum"], (info["minimum"] + info["maximum"]) // 2, info["maximum"]}):
            assert LV.legal(state, LV.ALLIN if total == info["maximum"] else LV.RAISE, total), (action, total)
            replies.append(P.serialize_action(resp.state, "raise", raise_to=int(total)))
    # 4. every reply parses back into a history that continues this one
    for r in replies:
        sep = "/" if resp.state.separator_pending else ""
        nxt = P.parse_history(resp.state.action + sep + r)
        assert nxt.events[:len(resp.state.events)] == resp.state.events
    return {"street": resp.state.street, "hero": hero, "events": len(steps), "replies": replies, "hand": h}


def main():
    bot = bare_bot()
    failures = 0
    for case in CASES + TERMINAL:
        try:
            out = check_case(bot, *case)
            print("ok  ", case[0] or "<start>", f"pos{case[1]}", out)
        except Exception as e:                                   # noqa: BLE001
            failures += 1
            print("FAIL", case[0] or "<start>", f"pos{case[1]}", type(e).__name__, str(e)[:300])
            traceback.print_exc(limit=3)
    for case in REJECTED:
        try:
            P.parse_response(payload(*case))
        except P.ProtocolError as e:
            print("rejected as it must:", case[0], f"pos{case[1]}", case[2], case[3], "->", str(e)[:60])
        else:
            failures += 1
            print("FAIL: accepted a malformed response", case)
    # every card string round-trips
    for r, s in itertools.product(RANKS, SUITS):
        c = P.our_card(P.card_id(r + s))
        assert card_str(c) == r + s and 0 <= c < 52
    print("card strings: 52 round-trip ok")
    print("replays:", bot.stats["replays"], "failures:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
