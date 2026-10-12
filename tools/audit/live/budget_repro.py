"""Reproduces the way a Slumbot evaluation dies: the production runner (parallel.run_parallel -> client.run_match ->
LiveSlumbotBot.choose -> LivePlayer -> ServerClient -> SolveServer with the 4 s budget) against the simulated server,
on this CPU (run.py sets JAX_PLATFORMS=cpu by default, run.py:13). River decisions go through the REAL SolveServer
and Solver (exact river, T = config.PLAY_ITERATIONS[3] = 300, warmed up first); the other streets use the random
stub of e2e_stub.py (their critics and caches do not exist here). Then the same output directory is resumed.

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/budget_repro.py
Writes results/audit/live/budget_repro.json (and the runner's evidence under results/audit/live/budget_run/)."""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from poker.arena.slumbot import parallel as PA
from poker.envs import config
from poker.player import live as LV
from poker.solver.cards import NUM_HANDS as H
from tools.audit.live.e2e_stub import StubServer, make_bot
from tools.audit.live.fake_slumbot import FakeSlumbot

OUT = Path("results/audit/live/budget_run")


class Hybrid(StubServer):
    def __init__(self, real, budget):
        super().__init__(0)
        self.real, self.budget, self.real_calls = real, budget, 0
        self.solver = real.solver

    def request(self, tree, ranges, fixed=None, equity=None, timeout=None):
        if tree.street == 3:
            self.real_calls += 1
            return self.real.request(tree, ranges, fixed, equity, timeout=timeout)
        return super().request(tree, ranges, fixed, equity, timeout)


def main():
    field = {"players": (2,), "seats": {2: (0, 1)}, "streets": (3,), "menus": tuple(config.STREET_SIZES), "ladder": {"cpu": (1,), "gpu": (1,)}}
    solver = LV.Solver(iterations=dict(config.PLAY_ITERATIONS), chunk=50, critics={})
    real = LV.SolveServer(solver, window_seconds=0.3)
    t0 = time.perf_counter()
    real.warm_up(None, None, None, field=field, cache_dir=False)
    warm = time.perf_counter() - t0
    bot = make_bot(0)
    bot.budget_seconds = 4.0
    bot.server = Hybrid(real, 4.0)
    shutil.rmtree(OUT, ignore_errors=True)
    out = {"warm_up_seconds_river_only": warm, "ready_line": real.ready_line, "budget_seconds": 4.0}
    t0 = time.perf_counter()
    report = PA.run_parallel(bot, output=OUT, hands=60, workers=2, seed=1, hand_cap=60,
                             transport_factory=lambda i: FakeSlumbot(seed=100 + i), log_every=10)
    out["first_run"] = {k: report.get(k) for k in ("status", "completed_hands", "unresolved_hands", "bb_per_100")}
    out["first_run_seconds"] = time.perf_counter() - t0
    out["bot_failures"] = bot.failures
    out["server_errors"] = dict(real.errors)
    out["real_river_requests"] = bot.server.real_calls
    out["unresolved_records"] = []
    for f in sorted((OUT / "hands").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("status") != "completed":
            out["unresolved_records"].append({"number": d["number"], "status": d["status"],
                                              "reason": (d.get("result") or {}).get("unresolved", {}).get("reason")})
    try:
        PA.run_parallel(bot, output=OUT, hands=60, workers=2, seed=1, hand_cap=60, resume=True,
                        transport_factory=lambda i: FakeSlumbot(seed=200 + i))
        out["resume"] = "resumed"
    except Exception as e:                                                       # noqa: BLE001
        out["resume"] = f"{type(e).__name__}: {e}"
    real.close()
    Path("results/audit/live/budget_repro.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "bot_failures"}, indent=1, default=str)[:3000])
    for f in bot.failures[:3]:
        print("FAILURE", f["error"], f["message"][:200], f["street"], f["history"])


if __name__ == "__main__":
    main()
