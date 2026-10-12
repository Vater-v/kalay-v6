"""menu_for (host) vs menu_index (traced, float32) at the MENU_BOUNDS edges +- 1 chip, over a range of pots up to the
deep-root sizes the solver uses (poker/solver/subgame.py DEEP = 1e8 chips). Usage: menu_bounds_check.py OUT.json"""
import json, sys
from poker.envs import config
rows, bad = [], []
for stack in (3999, 4000, 4001, 40000 - 1, 40000):
    a = config.menu_for(0, depth_bb=stack / config.BB); b = config.MENUS[int(config.menu_index(0, stack, 150))]
    rows.append({"street": 0, "stack": stack, "menu_for": a, "menu_index": b})
    if a != b: bad.append(rows[-1])
for pot in (1, 3, 7, 150, 1201, 33333, 2_000_000, 4_194_305, 16_777_217, 50_000_001):
    for d in (-1, 0, 1):
        stack = 2 * pot + d
        a = config.menu_for(1, spr=stack / pot); b = config.MENUS[int(config.menu_index(1, stack, pot))]
        rows.append({"street": 1, "pot": pot, "stack": stack, "menu_for": a, "menu_index": b})
        if a != b: bad.append(rows[-1])
out = {"checked": len(rows), "disagreements": bad}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
