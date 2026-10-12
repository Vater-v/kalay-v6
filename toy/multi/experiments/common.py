"""Shared helpers for the experiment scripts (paths, the default game, result I/O)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
os.makedirs(RESULTS, exist_ok=True)

from game import GameConfig  # noqa: E402

# The main game used by M1-M4: 3 players, 12 cards (6 ranks x 2 suits), stacks 15 antes,
# pot-bet + all-in, raise cap 2 per street.
MAIN = GameConfig(n_players=3, n_ranks=6, n_suits=2, stack=15, raise_cap=2)


def save(name, obj):
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)


def table(headers, rows):
    """Markdown table as a string (also printed by callers)."""
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def ma(x):
    """antes -> milli-antes (the 'mbb/hand' analogue: 1 ante plays the role of the big blind)."""
    return round(1000.0 * float(x), 2)
