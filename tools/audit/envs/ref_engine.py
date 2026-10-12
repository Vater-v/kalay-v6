"""Plain-Python NLHE reference (independent of poker/envs): betting rules, side pots, uncalled bets, odd chips, rake.

Rules written from the TDA / Robert's-rules text, not from the engine:
- seat 0 is the button; heads-up the button posts the SB and acts first preflop, last postflop;
  3+ players: SB = first occupied seat after the button, BB = next, first preflop actor = next after BB;
  postflop the first occupied seat with chips after the button acts first.
- a raise must be at least the last full raise size (initially BB each street; preflop the level is at least BB);
  an all-in that raises by less is a short raise: it does not change the minimum increment and does not reopen
  betting to a player who has already acted unless the total he now faces since his last action is >= the last
  full raise; nobody may raise when no other seat in the hand has chips behind.
- uncalled excess returns to the top contributor; pots layer by the contributions of the players still in;
  each pot is split separately; odd chips go to winners in seat order starting left of the button.
- rake (poker/envs/config.py comment, re-implemented): only if the flop was dealt (a preflop all-in that runs out
  counts); base = all contributions minus the uncalled excess; rake = min(floor(base*bps/1e4), cap);
  each pot pays floor(rake * pot / base), the main (first non-empty) pot pays the remainder.
"""

class Ref:
    def __init__(self, stacks, active, sb=50, bb=100):
        n = len(stacks)
        self.n, self.sb, self.bb = n, sb, bb
        self.active = list(active)
        self.stack = [s if a else 0 for s, a in zip(stacks, active)]
        self.bet = [0] * n
        self.total = [0] * n
        self.folded = [not a for a in active]
        self.street = 0
        self.flop_dealt = False
        occ = [i for i in range(n) if active[i]]
        assert occ[0] == 0
        if len(occ) == 2:
            sbs, bbs = 0, occ[1]
        else:
            sbs, bbs = occ[1], occ[2]
        self._post(sbs, sb); self._post(bbs, bb)
        self.sb_seat, self.bb_seat = sbs, bbs
        self.level = bb                    # preflop level is the full BB
        self.last_full = bb
        self.acted = [False] * n
        self.seen = [0] * n                # level the player faced/matched at his last action
        self.terminal = False
        self.to_act = self._next_from(bbs, first_street_actor=False)

    def _post(self, i, amt):
        a = min(amt, self.stack[i])
        self.stack[i] -= a; self.bet[i] += a; self.total[i] += a

    def can_act(self, i):
        return not self.folded[i] and self.stack[i] > 0

    def _needs(self, i):
        if not self.can_act(i):
            return False
        owes = self.bet[i] < self.level
        others = sum(self.can_act(j) for j in range(self.n) if j != i)
        return owes or (not self.acted[i] and others >= 1)

    def _next_from(self, seat, first_street_actor):
        for k in range(1, self.n + 1):
            j = (seat + k) % self.n
            if self._needs(j):
                return j
        return -1

    def to_call(self):
        return max(0, self.level - self.bet[self.to_act])

    def legal(self):
        i = self.to_act
        tc = self.to_call()
        others_with_chips = any(self.can_act(j) for j in range(self.n) if j != i)
        reopened = (not self.acted[i]) or (self.level - self.seen[i] >= self.last_full)
        allin_to = self.bet[i] + self.stack[i]
        can_raise = others_with_chips and reopened and allin_to > self.level
        min_to = self.level + self.last_full
        return {"fold": tc > 0, "call_to": self.bet[i] + min(tc, self.stack[i]), "raise": can_raise,
                "min_to": min_to, "allin_to": allin_to}

    def act(self, kind, to=None):
        i = self.to_act
        L = self.legal()
        if kind == "fold":
            assert L["fold"]; self.folded[i] = True
        elif kind == "call":
            add = L["call_to"] - self.bet[i]
            self.stack[i] -= add; self.bet[i] += add; self.total[i] += add
        else:
            assert L["raise"] and (L["min_to"] <= to <= L["allin_to"] or to == L["allin_to"]) and to > self.level
            add = to - self.bet[i]
            self.stack[i] -= add; self.bet[i] += add; self.total[i] += add
            inc = to - self.level
            if inc >= self.last_full:
                self.last_full = inc
                self.acted = [False] * self.n
            self.level = to
        self.acted[i] = True
        self.seen[i] = self.level
        self._advance(i)

    def _advance(self, i):
        alive = [j for j in range(self.n) if not self.folded[j]]
        if len(alive) == 1:
            self.terminal = True; self.to_act = -1; return
        nxt = self._next_from(i, False)
        if nxt != -1:
            self.to_act = nxt; return
        # street over
        while True:
            if self.street == 3:
                self.terminal = True; self.to_act = -1; return
            self.street += 1; self.flop_dealt = True
            self.bet = [0] * self.n; self.level = 0; self.last_full = self.bb
            self.acted = [False] * self.n; self.seen = [0] * self.n
            first = self._next_from(0, True)
            if first == -1:      # nobody (or one player) can act: run it out
                self.street = 3
                self.terminal = True; self.to_act = -1; return
            self.to_act = first; return

    def settle(self, keys, bps=0, cap=0):
        """keys: comparable hand strengths per seat (larger wins). Returns net profit per seat and rake."""
        n = self.n
        contrib = list(self.total)
        order = sorted(range(n), key=lambda j: contrib[j], reverse=True)
        top, second = order[0], order[1]
        uncalled = contrib[top] - contrib[second]
        payout = [0] * n
        payout[top] += uncalled
        contrib[top] -= uncalled
        live = [j for j in range(n) if not self.folded[j]]
        levels = sorted(set(contrib[j] for j in live))
        assert max(contrib) <= levels[-1], "folded money above the top live level"
        pots, prev = [], 0
        for L in levels:
            amt = sum(min(c, L) - min(c, prev) for c in contrib)
            elig = [j for j in live if contrib[j] >= L]
            pots.append([amt, elig]); prev = L
        base = sum(p[0] for p in pots)
        rake = 0
        if self.flop_dealt and bps:
            rake = min(base * bps // 10000, cap)
        if rake:
            shares = [rake * p[0] // base for p in pots]
            main = next(k for k, p in enumerate(pots) if p[0] > 0)
            shares[main] += rake - sum(shares)
            for k in range(len(pots)):
                pots[k][0] -= shares[k]
        seat_order = [(1 + k) % n for k in range(n)]
        for amt, elig in pots:
            if amt == 0:
                continue
            best = max(keys[j] for j in elig)
            win = [j for j in seat_order if j in elig and keys[j] == best]
            q, r = divmod(amt, len(win))
            for k, j in enumerate(win):
                payout[j] += q + (1 if k < r else 0)
        return [payout[j] - self.total[j] for j in range(n)], rake
