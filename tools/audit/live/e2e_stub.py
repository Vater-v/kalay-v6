"""End-to-end run of the real Slumbot path (client.run_match -> protocol.parse_response -> LiveSlumbotBot.choose
-> _replay on the engine -> LivePlayer.observe/act -> subgame.build_tree/replay with SLOT columns ->
serialize_action) against the simulated server (fake_slumbot.py), with ONLY the solve replaced by a stub that
returns a random strategy on the real LiveTree (no critics, no equity caches exist on this box). Every hand's
failure is recorded with its type, message and history.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/e2e_stub.py [HANDS] [SEED]
Writes results/audit/live/e2e_stub.json.
"""
import collections
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from poker.arena.slumbot import client as CL
from poker.arena.slumbot import live_bot as LB
from poker.arena.slumbot.protocol import STACK_SIZE
from poker.envs import config
from poker.envs.betting import init_betting
from poker.envs.transition import step_target
from poker.player import live as LV
from poker.solver import subgame as SG
from poker.solver.cards import NUM_HANDS as H
from tools.audit.live.fake_slumbot import FakeSlumbot

OUT = Path("results/audit/live/e2e_stub.json")


class StubServer:
    """The SolveServer interface ServerClient uses; a random (Dirichlet) strategy per hand at every internal node."""

    def __init__(self, seed=0):
        self.solver = SimpleNamespace(chunk=50, critics={}, iterations=dict(config.PLAY_ITERATIONS))
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.keys = collections.Counter()
        self.batches, self.errors, self.handovers = [], {}, []

    def request(self, tree, ranges, fixed=None, equity=None, timeout=None):
        self.calls += 1
        self.keys[tree.key] += 1
        legal = (tree.legal & (tree.topo.children >= 0))[tree.internal]             # [Vi, A]
        g = self.rng.gamma(0.7, size=(legal.shape[0], H, legal.shape[1])) * legal[:, None, :]
        sigma = g / np.maximum(g.sum(-1, keepdims=True), 1e-30)
        for v, row in (fixed or {}).items():
            sigma[tree.st.to_internal[v]] = row
        return SG.Solution(tree, sigma.astype(np.float32), 1, None, 0.0, 0.0, 0.0, 1)

    def close(self):
        pass


def make_bot(seed):
    bot = LB.LiveSlumbotBot.__new__(LB.LiveSlumbotBot)
    bot.regime = LB.check_live_regime()
    bot.log = None
    bot.n = 2
    bot.server = StubServer(seed)
    bot.budget_seconds = None
    bot.eps = LV.DEFAULT_EPS
    bot.table = None                                            # no preflop table on this box: preflop is "solved" by the stub
    bot.flop_equity = lambda f: None
    bot.turn_equity = lambda b: None
    bot.preflop_equity = np.zeros((H, H), np.float32)
    base = init_betting(2, jnp.full(2, STACK_SIZE, jnp.int32), history=True, max_stack=STACK_SIZE, rake_level=0.0)
    bot.base = jax.tree.map(lambda x: np.array(x, copy=True), base)
    bot._step = jax.jit(step_target)
    bot.local = threading.local()
    bot.lock = threading.Lock()
    bot.stats = {"hands": 0, "decisions": 0, "events": 0, "failures": 0, "replays": 0, "new_hand_by_signature": 0,
                 "new_hand_by_history": 0, "forced_own": 0}
    bot.failures, bot.decision_seconds = [], []
    bot.decisions = bot.batches = 0
    bot.max_observed_batch = 1
    bot.policy_id = "audit-stub"
    return bot


def main(hands=300, seed=0):
    bot = make_bot(seed)
    srv = FakeSlumbot(seed=seed)
    rng = np.random.default_rng(seed)
    out = {"hands_requested": hands, "completed": 0, "failures": [], "failure_types": collections.Counter()}
    t0 = time.perf_counter()
    token = None
    slot_hits = 0
    for k in range(hands):
        try:
            res = CL.run_match(lambda r: bot.choose(r, rng).increment, srv, hands=1, hand_cap=1, policy_id="audit-stub",
                               initial_token=token)
            token = srv.hand["token"]
            out["completed"] += 1
        except CL.MatchInterrupted as e:
            reason = e.result.unresolved.reason
            last = bot.failures[-1] if bot.failures else None
            out["failures"].append({"hand": k, "reason": reason, "bot_failure": last,
                                    "server_action": srv.hand["action"], "client_pos": srv.client_pos})
            out["failure_types"][(last or {}).get("error", reason[:60])] += 1
            token = srv.hand["token"]
    player_stats = {}
    for th in [bot.local]:
        p = getattr(th, "player", None)
        if p is not None:
            player_stats = {k: (v if not isinstance(v, (set, list)) else len(v)) for k, v in p.stats.items()
                            if k not in ("solve_seconds", "solve_cpu_seconds", "build_seconds", "decision_seconds", "equity_seconds")}
    out.update(seconds=time.perf_counter() - t0, bot_stats=bot.stats, stub_calls=bot.server.calls,
               topologies=dict(bot.server.keys), last_player_stats=player_stats,
               decision_seconds={"mean": float(np.mean(bot.decision_seconds)), "p90": float(np.percentile(bot.decision_seconds, 90)),
                                 "max": float(np.max(bot.decision_seconds))} if bot.decision_seconds else None,
               shapes={k: list(v) for k, v in SG.shapes().items()})
    out["failure_types"] = dict(out["failure_types"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("failures", "shapes", "topologies")}, indent=1, default=str))
    for f in out["failures"][:8]:
        print("FAIL", f["reason"][:200], "|", f["server_action"], "| pos", f["client_pos"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
