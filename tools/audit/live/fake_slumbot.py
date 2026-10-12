"""A simulated Slumbot server (transport.post(endpoint, body) -> dict) built on the independent reference
parser (slumbot_ref.parse_action) and evaluator (slumbot_ref.best5). Card strings 'Ac', 'Td', ...; the
bot (Slumbot's seat) plays random legal actions incl. off-menu sizes, limps, all-ins and folds; responses
carry old_action, action, client_pos, hole_cards, board, token and, at the end, winnings,
bot_hole_cards and baseline_winnings. Streets in the action string end with '/' (as in 'b200c/kk/kb100c/'),
and a called all-in carries '///'-style separators with probability `allin_slashes`."""
import random

from tools.audit.live.slumbot_ref import BIG_BLIND, STACK_SIZE, best5, card_str, parse_action

NEW_HAND = "/slumbot/api/new_hand"
ACT = "/slumbot/api/act"


def contributions(action):
    """(totals per protocol position [0=BB, 1=SB]) of an action string, independent of the parser under test."""
    totals_prev = [0, 0]
    bets = [BIG_BLIND, BIG_BLIND // 2]
    i, pos = 0, 1
    while i < len(action):
        c = action[i]
        i += 1
        if c == "/":
            totals_prev = [totals_prev[p] + bets[p] for p in (0, 1)]
            bets = [0, 0]
            pos = 0
            continue
        if c == "b":
            j = i
            while i < len(action) and action[i].isdigit():
                i += 1
            bets[pos] = int(action[j:i])
        elif c == "c":
            bets[pos] = max(bets)
        pos = 1 - pos
    return [totals_prev[p] + bets[p] for p in (0, 1)]


class FakeSlumbot:
    def __init__(self, seed=0, allin_slashes=0.5, odd_sizes=0.35, limp=0.25, trailing_slash=0.85, token_every_act=False):
        self.rng = random.Random(seed)
        self.allin_slashes, self.odd_sizes, self.limp, self.trailing_slash = allin_slashes, odd_sizes, limp, trailing_slash
        self.token_every_act = token_every_act
        self.hand = None
        self.client_pos = 1
        self.log = []

    # ---------------------------------------------------------------- game helpers
    def _visible(self, st, over_allin):
        return [card_str(c) for c in self.hand["board"][: (5 if over_allin else (0, 3, 4, 5)[st])]]

    def _append(self, incr):
        a = self.hand["action"]
        p = parse_action(a)
        if "error" in p:
            raise RuntimeError(f"server state broken: {a!r} {p}")
        new = a + incr
        q = parse_action(new)
        if "error" in q:
            return q["error"]
        # close a street with '/' (Slumbot style), unless the hand is over
        if q["pos"] != -1 and q["st"] > p["st"] and self.rng.random() < self.trailing_slash:
            new += "/"
        elif q["pos"] == -1 and incr == "c" and p["total_last_bet_to"] == STACK_SIZE and q["st"] == 3:
            if self.rng.random() < self.allin_slashes:
                new += "/" * (3 - p["st"])
        if "error" in parse_action(new):
            raise RuntimeError(f"server appended an invalid suffix: {new!r}")
        self.hand["action"] = new
        return None

    def _bot_action(self):
        a = self.hand["action"]
        p = parse_action(a)
        facing = p["last_bet_size"] > 0
        remaining = STACK_SIZE - p["total_last_bet_to"]
        lo_size = min(max(p["last_bet_size"], BIG_BLIND) if facing else BIG_BLIND, remaining)
        lo, hi = p["street_last_bet_to"] + lo_size, p["street_last_bet_to"] + remaining
        pot = 2 * p["total_last_bet_to"]
        r = self.rng.random()
        opts = []
        if facing:
            opts += ["f"] * 2 + ["c"] * 5
        else:
            opts += ["k"] * 5
        if remaining > 0 and hi >= lo:
            opts += ["b"] * 4
        choice = self.rng.choice(opts)
        if p["st"] == 0 and a == "" and r < self.limp:
            choice = "c"
        if choice != "b":
            return choice
        x = self.rng.random()
        if x < 0.15:
            to = hi                                                 # all-in
        elif x < 0.25:
            to = lo                                                 # min bet / min raise
        elif x < 0.25 + self.odd_sizes:
            to = self.rng.randint(lo, hi)                          # arbitrary chip amount
        else:
            frac = self.rng.choice([0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0])
            to = int(p["street_last_bet_to"] + frac * pot)
        to = min(max(to, lo), hi)
        return f"b{to}"

    def _run_bot(self):
        while True:
            p = parse_action(self.hand["action"])
            if p["pos"] == -1 or p["pos"] == self.client_pos:
                return
            incr = self._bot_action()
            err = self._append(incr)
            if err:
                raise RuntimeError(f"bot produced an illegal action {incr!r} after {self.hand['action']!r}: {err}")

    def _response(self, old):
        a = self.hand["action"]
        p = parse_action(a)
        over = p["pos"] == -1
        allin = over and contributions(a)[0] == contributions(a)[1] == STACK_SIZE
        r = {"old_action": old, "action": a, "client_pos": self.client_pos,
             "hole_cards": [card_str(c) for c in self.hand["hole"]], "board": self._visible(p["st"], allin)}
        if self.hand["new"] or self.token_every_act:
            r["token"] = self.hand["token"]
        if over:
            tot = contributions(a)
            me, bot = self.client_pos, 1 - self.client_pos
            if a.endswith("f"):
                folder = self._last_actor(a)
                w = -tot[me] if folder == me else tot[bot]
            else:
                hb = best5([card_str(c) for c in self.hand["hole"] + self.hand["board"]])
                bb = best5([card_str(c) for c in self.hand["bot"] + self.hand["board"]])
                w = tot[bot] if hb > bb else (-tot[me] if hb < bb else 0)
                r["bot_hole_cards"] = [card_str(c) for c in self.hand["bot"]]
            r["winnings"] = w
            r["baseline_winnings"] = 0
        self.hand["new"] = False
        return r

    @staticmethod
    def _last_actor(a):
        """Protocol position of the player who made the last action of `a`."""
        p = parse_action(a[:-1].rstrip("/") if a[:-1].endswith("/") else a[:-1])
        return p["pos"]

    # ---------------------------------------------------------------- transport
    def post(self, endpoint, body):
        if endpoint == NEW_HAND:
            deck = list(range(52))
            self.rng.shuffle(deck)
            self.client_pos = 1 - self.client_pos
            self.hand = {"hole": deck[:2], "bot": deck[2:4], "board": deck[4:9], "action": "", "new": True,
                         "token": body.get("token") or "tok-%08x" % self.rng.getrandbits(32)}
            self._run_bot()
            r = self._response("")
            self.log.append(r)
            return r
        if endpoint == ACT:
            if self.hand is None or body.get("token") != self.hand["token"]:
                return {"error_msg": "bad token"}
            old = self.hand["action"]
            p = parse_action(old)
            if p["pos"] != self.client_pos:
                return {"error_msg": "not your turn"}
            err = self._append(body["incr"])
            if err:
                return {"error_msg": f"Illegal action: {err}"}
            self._run_bot()
            r = self._response(old)
            self.log.append(r)
            return r
        raise ValueError(endpoint)
