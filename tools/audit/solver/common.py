"""Spot construction for the solver audit (no writes outside the scratchpad / results/audit/solver)."""
import os
from pathlib import Path
import numpy as np
from poker.envs import config
from poker.solver import cards as C, situations as SI, runout as RO

SCRATCH = Path(os.environ.get("AUDIT_SCRATCH",
                              "/tmp/claude-0/-home-user-kalay-v6/721b9d69-ee04-59bb-a6cc-2f7cda70f7ae/scratchpad/cache"))
_ORIG = RO.flop_equity_cached


def no_cache_write():
    """Flop matrices from/to the scratch cache only (never results/cache)."""
    SI.runout.flop_equity_cached = lambda flop: _ORIG(flop, cache=SCRATCH / "flop_equity")


def spots(n, street, sprs, seed=7, pot_bb=10.0, board=None):
    """Equal-stack spots, no rake, random ranges (situations.random_range), given SPRs."""
    rng = np.random.default_rng(seed)
    count = len(sprs)
    k = SI.BOARD_CARDS[street]
    boards = np.full((count, 5), -1, np.int32)
    for i in range(count):
        b = rng.choice(52, k, replace=False) if board is None else np.asarray(board[:k])
        boards[i, :k] = b
    valid = C.board_valid(boards[:, :k])
    ranges = np.stack([[SI.random_range(rng, valid[i], 0.2) for _ in range(n)] for i in range(count)]).astype(np.float32)
    pot = np.full(count, int(round(pot_bb * config.BB / n)) * n, np.int32)
    stack = np.maximum(np.round(np.asarray(sprs) * pot), config.BB).astype(np.int32)
    return {"board": boards, "pot": pot, "stack": stack, "valid": valid, "ranges": ranges,
            "street": np.full(count, street, np.int32)}
