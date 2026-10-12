"""Live Slumbot runner with explicit current-game critics and their canonical preflop table.

    JAX_PLATFORMS=cuda,cpu PYTHONPATH=. .venv/bin/python -m poker.arena.slumbot.run \
        --critics DIR --preflop-table FILE --hands 200 --out results/slumbot/live_200

The workers start after bot.warm_up(). Failed decisions stay unresolved, stop the match and
block resume. Per-hand evidence is in <out>/hands; summary.json and live_summary.json retain
runner and solver diagnostics. Slumbot is heads-up, 200 BB, no rake (POKER_REGIME=1).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from poker.arena.slumbot.live_bot import LiveSlumbotBot  # noqa: E402
from poker.arena.slumbot.parallel import run_parallel  # noqa: E402
from poker.arena.slumbot.util import atomic_json  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--critics", type=Path, required=True, help="critics of the current game")
    p.add_argument("--preflop-table", type=Path, required=True, help="canonical table generated from these critics")
    p.add_argument("--flop-cache", type=Path, default=Path("results/cache/flop_equity_v1"))
    p.add_argument("--rank7", type=Path, default=Path("results/cache/rank7_v1.npy"))
    p.add_argument("--preflop-equity", type=Path, default=Path("results/cache/preflop_equity_v1.npy"))
    p.add_argument("--window", type=float, default=0.3, help="the trap collector's window in seconds")
    p.add_argument("--dev-size", type=int, default=0, help="developer toggle: the trap waits for N requests, no timer")
    p.add_argument("--budget", type=float, default=4.0, help="wall-clock budget per decision in seconds")
    p.add_argument("--jax-cache", type=Path, help="the persistent compilation cache base (default config.JAX_CACHE_DIR)")
    p.add_argument("--hands", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-hands", type=int)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args(argv)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    bot = LiveSlumbotBot(critics_dir=args.critics, preflop_table=args.preflop_table, flop_cache=args.flop_cache, rank7=args.rank7,
                         preflop_equity=args.preflop_equity, window_seconds=args.window, dev_size=args.dev_size,
                         budget_seconds=args.budget, log=log, jax_cache=args.jax_cache)
    log(f"slumbot live: {args.hands} hands, {args.workers} workers, game {bot.metadata['game_hash'][:12]}, "
        f"T {bot.metadata['iterations']}, window {args.window} s, dev_size {args.dev_size}, budget {args.budget} s, "
        f"device {bot.metadata['device']}, id {bot.policy_id[:12]}")
    bot.warm_up()                                                          # the ready gate: workers only after it
    if not bot.ready:
        raise SystemExit("live bot: not ready after the warm-up")

    def bot_stats():
        atomic_json(args.out / "live_summary.json", bot.summary())
        atomic_json(args.out / "bot_stats.json", {"stats": dict(bot.stats), "decisions": bot.decisions})

    def progress(s):
        bot_stats()
        rate, se = s.get("bb_per_100"), s.get("standard_error_bb_per_100")
        rate_text = "n/a" if rate is None else f"{rate:+.1f}"
        se_text = "n/a" if se is None else f"{se:.1f}"
        log(f"  {s.get('completed_hands')} hands: {rate_text} bb/100 "
            f"(SE {se_text}), {s.get('elapsed_seconds', 0):.0f} s"
            + f"; failures {bot.stats['failures']}")

    try:
        report = run_parallel(bot, output=args.out, hands=args.hands, workers=args.workers, seed=args.seed,
                              hand_cap=args.hands, resume=args.resume, max_new_hands=args.max_new_hands,
                              log_every=args.log_every, on_progress=progress)
    finally:
        bot_stats()
        bot.close()
    log(json.dumps({k: v for k, v in report.items() if not isinstance(v, (dict, list))}))
    live = bot.summary()
    log(f"live: failures {live['stats']['failures']}, decisions {live['decision_seconds']}, server batches "
        f"{live['server']['batches'][:20]}..., errors {live['server']['errors']}")
    if report["status"] == "unresolved":
        raise SystemExit(f"slumbot live: unresolved hands {report['unresolved_hands']}")

if __name__ == "__main__":
    main()
