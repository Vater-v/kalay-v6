"""Shared setup: the game configuration used by every experiment, units, IO."""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

from game import GameConfig, Boards          # noqa: E402
from tree import full_game_tree              # noqa: E402

# The game used everywhere: 5 ranks x 2 suits = 10 cards, stacks 12 antes.
CFG = GameConfig(n_ranks=5, n_suits=2, ante=1.0, stack=12.0,
                 bet_menu=((0.5, 1.0), (1.0,), ()))
POT_ANTE = 2 * CFG.ante  # the initial pot


def setup():
    B = Boards(CFG)
    T = full_game_tree(CFG, B)
    return CFG, B, T


def mA(x):
    """antes/hand -> milli-antes/hand (the analogue of mbb/hand)."""
    return 1000.0 * x


def pct_pot(x):
    return 100.0 * x / POT_ANTE


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


class Log:
    def __init__(self, name):
        self.f = open(os.path.join(RES, name), "w")
        self.t0 = time.time()

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        line = f"[{time.time() - self.t0:7.1f}s] {s}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()
