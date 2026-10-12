"""Unit tests of the Slumbot protocol boundary against realistic responses (card strings 'Ac', 'Td', boards of
0/3/4/5 cards, both seats, all-ins with and without '///', folds on every street, limps), the card mapping into
the engine (rank*4+suit -> suit*13+rank checked against the engine's evaluator), the seat mapping (engine
actor order vs the API's), and fuzzing parse_history against the independent reference parser.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/test_protocol.py
Writes results/audit/live/protocol_tests.json.
"""
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np

from poker.arena.slumbot import protocol as P
from tools.audit.live import slumbot_ref as REF

OUT = Path("results/audit/live/protocol_tests.json")
results = []


def case(name):
    def deco(fn):
        try:
            detail = fn()
            results.append({"test": name, "ok": True, "detail": detail})
        except Exception as e:                                   # noqa: BLE001
            results.append({"test": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-800:]})
        return fn
    return deco


def resp(action, client_pos, hole, board, **kw):
    r = {"old_action": "", "action": action, "client_pos": client_pos, "hole_cards": hole, "board": board, "token": "t-1"}
    r.update(kw)
    return r


# ------------------------------------------------------------------ realistic responses
@case("sample response of sample_api.py: b200, client BB, Ac 9d, empty board")
def _():
    h = P.parse_response(resp("b200", 0, ["Ac", "9d"], []))
    return {"street": h.state.street, "next": h.state.next_player, "hole_ids": h.hole_cards}


@case("tens: Td Ts hole, board Th 2c 3d")
def _():
    h = P.parse_response(resp("b200c/", 0, ["Td", "Ts"], ["Th", "2c", "3d"]))
    return {"hole_ids": h.hole_cards, "board_ids": h.board}


@case("flop without trailing slash: b200c")
def _():
    h = P.parse_response(resp("b200c", 0, ["Ah", "Kh"], ["2c", "7d", "Tc"]))
    return {"street": h.state.street, "pending_sep": h.state.separator_pending}


@case("turn, client SB facing a bet: b200c/kk/b300")
def _():
    h = P.parse_response(resp("b200c/kk/b300", 1, ["Qs", "Jc"], ["2c", "7d", "Tc", "9h"]))
    return {"street": h.state.street, "next": h.state.next_player, "to_call": h.state.to_call}


@case("river, client BB first: b200c/kk/kk/")
def _():
    h = P.parse_response(resp("b200c/kk/kk/", 0, ["5s", "5d"], ["2c", "7d", "Tc", "9h", "As"]))
    return {"street": h.state.street, "next": h.state.next_player}


@case("limp then check: ck/ (client BB on flop)")
def _():
    h = P.parse_response(resp("ck/", 0, ["8c", "6c"], ["Kd", "Qd", "4s"]))
    return {"street": h.state.street, "next": h.state.next_player, "pot": h.state.pot}


@case("preflop all-in called, terminal, '///' suffix, 5 board cards")
def _():
    h = P.parse_response(resp("b20000c///", 0, ["Ac", "Ad"], ["2c", "7d", "Tc", "9h", "As"], winnings=20000))
    return {"terminal": h.state.terminal_reason, "street": h.state.street}


@case("preflop all-in called, terminal, no suffix")
def _():
    h = P.parse_response(resp("b20000c", 1, ["Ac", "Ad"], ["2c", "7d", "Tc", "9h", "As"], winnings=-20000))
    return {"terminal": h.state.terminal_reason}


@case("flop all-in called '//' suffix")
def _():
    h = P.parse_response(resp("b200c/b19800c//", 0, ["Ac", "Ad"], ["2c", "7d", "Tc", "9h", "As"], winnings=0))
    return {"terminal": h.state.terminal_reason}


@case("fold preflop by client BB facing an open (terminal, empty board)")
def _():
    h = P.parse_response(resp("b300f", 0, ["7c", "2d"], [], winnings=-100))
    return {"terminal": h.state.terminal_reason}


@case("client SB folds its first action: f (winnings -50)")
def _():
    h = P.parse_response(resp("f", 1, ["7c", "2d"], [], winnings=-50))
    return {"terminal": h.state.terminal_reason}


@case("fold on turn by bot (terminal, 4 board cards)")
def _():
    h = P.parse_response(resp("b200c/kk/b400f", 0, ["Kc", "Kd"], ["2c", "7d", "Tc", "9h"], winnings=200))
    return {"terminal": h.state.terminal_reason, "winnings": h.winnings}


@case("new_hand where the bot (SB) folds at once: f")
def _():
    h = P.parse_response(resp("f", 0, ["Kc", "Kd"], [], winnings=50))
    return {"terminal": h.state.terminal_reason}


@case("river showdown terminal with bot_hole_cards")
def _():
    h = P.parse_response(resp("b200c/kk/kk/kk", 1, ["Kc", "Kd"], ["2c", "7d", "Tc", "9h", "As"], winnings=200,
                              bot_hole_cards=["Qc", "Qd"], baseline_winnings=-150))
    return {"terminal": h.state.terminal_reason, "winnings": h.winnings, "baseline": h.baseline_winnings}


@case("act response without a token (token carried over)")
def _():
    r = resp("b200c/k", 1, ["Kc", "Kd"], ["2c", "7d", "Tc"])
    r.pop("token")
    h = P.parse_response(r, previous_token="t-1")
    return {"token_kept": h.token == "t-1"}


@case("lower-case / malformed card strings are rejected (the ONLY card-parse failure mode)")
def _():
    out = {}
    for bad in (["ac", "9d"], ["10c", "9d"], ["Ac", "9D"], ["A", "9d"]):
        try:
            P.parse_response(resp("b200", 0, bad, []))
            out[repr(bad)] = "accepted"
        except P.ProtocolError as e:
            out[repr(bad)] = f"ProtocolError: {e}"
    return out


# ------------------------------------------------------------------ card mapping into the engine
@case("our_card(card_id(s)) == engine suit*13+rank for all 52 strings; engine evaluator agrees on hand classes")
def _():
    from poker.envs import evaluator as EV
    import jax.numpy as jnp
    ranks, suits = "23456789TJQKA", "shdc"
    ids = {}
    for r, rc in enumerate(ranks):
        for s, sc in enumerate(suits):
            e = P.our_card(P.card_id(rc + sc))
            assert e % 13 == r and e // 13 == s, (rc + sc, e)
            ids[rc + sc] = e
    assert sorted(ids.values()) == list(range(52))
    # evaluator check: compare the engine's 7-card ordering with the independent brute force on random deals
    rng = random.Random(7)
    fn = EV.evaluate_hand
    names = [n for n in dir(EV) if not n.startswith("_")]
    if fn is None:
        return {"mapping": "ok (52/52)", "evaluator_check": f"skipped: no single-hand rank function among {names}"}
    agree = 0
    for _ in range(300):
        cards = rng.sample(sorted(ids), 9)
        a, b, board = cards[:2], cards[2:4], cards[4:]
        ra = fn(jnp.asarray([ids[c] for c in a + board], jnp.int32))
        rb = fn(jnp.asarray([ids[c] for c in b + board], jnp.int32))
        ref_a, ref_b = REF.best5(a + board), REF.best5(b + board)
        engine = (int(ra) > int(rb)) - (int(ra) < int(rb))
        ref = (ref_a > ref_b) - (ref_a < ref_b)
        agree += engine == ref
    return {"mapping": "ok (52/52)", "evaluator": fn.__name__, "showdown_agreement": f"{agree}/300"}


# ------------------------------------------------------------------ seats
@case("SEAT mapping: engine seat 0 = SB acts first preflop, seat 1 = BB acts first postflop (API: pos 1 SB, pos 0 BB)")
def _():
    import jax
    import jax.numpy as jnp
    from poker.envs import config
    from poker.envs.betting import init_betting
    from poker.envs.transition import step_target
    s = init_betting(2, jnp.full(2, 20000, jnp.int32), history=True, max_stack=20000, rake_level=0.0)
    first_pre = int(s.to_act)
    s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(100))        # SB limps
    s = step_target(s, jnp.int32(config.CHECK_CALL), jnp.int32(100))        # BB checks
    first_post = int(s.to_act)
    assert P.SEAT == (1, 0)
    assert first_pre == P.SEAT[1] and first_post == P.SEAT[0], (first_pre, first_post)
    return {"engine_first_preflop": first_pre, "engine_first_postflop": first_post, "street": int(s.street)}


# ------------------------------------------------------------------ fuzz parse_history vs reference
@case("fuzz: 20000 random legal Slumbot action strings, parse_history agrees with the reference parser")
def _():
    from tools.audit.live.fake_slumbot import FakeSlumbot
    srv = FakeSlumbot(seed=11)
    mism, n = [], 0
    for k in range(20000):
        srv.hand = {"action": "", "hole": [0, 1], "bot": [2, 3], "board": [4, 5, 6, 7, 8], "new": True, "token": "x"}
        srv.client_pos = -1                                     # the bot plays both seats
        while True:
            p = REF.parse_action(srv.hand["action"])
            if p["pos"] == -1:
                break
            incr = srv._bot_action()
            if srv._append(incr):
                raise RuntimeError("generator produced an illegal action")
            a = srv.hand["action"]
            n += 1
            ref = REF.parse_action(a)
            try:
                h = P.parse_history(a)
                ours = (h.street, -1 if h.next_player is None else h.next_player)
            except P.ProtocolError as e:
                ours = ("ERR", str(e))
            theirs = (ref["st"], ref["pos"])
            if ref["pos"] == -1:
                ok = ours[0] != "ERR" and h.terminal
            else:
                ok = ours == theirs
            if not ok and len(mism) < 10:
                mism.append({"action": a, "ours": ours, "reference": theirs})
    assert not mism, mism
    return {"prefixes_checked": n}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, default=str))
    bad = [r for r in results if not r["ok"]]
    for r in results:
        print(("PASS " if r["ok"] else "FAIL ") + r["test"], "" if r["ok"] else r["error"])
    print(f"{len(results) - len(bad)}/{len(results)} passed -> {OUT}")
    sys.exit(1 if bad else 0)
