"""Sensitivity of betting_check.py: re-run it with deliberately wrong reference rules and confirm it reports mismatches.
M1: a short all-in reopens betting for everyone. M3: rake charged on preflop folds too. Usage: mutation_check.py HANDS OUT.json"""
import json, sys
sys.path.insert(0, "tools/audit/envs")
import ref_engine, betting_check
hands = int(sys.argv[1]); out = {}
orig_legal, orig_settle, orig_act = ref_engine.Ref.legal, ref_engine.Ref.settle, ref_engine.Ref.act
def m1_act(self, kind, to=None):
    lvl, lf = self.level, self.last_full
    orig_act(self, kind, to)
    if kind == "raise" and to - lvl < lf and not self.terminal:
        self.acted = [False] * self.n
def m3_settle(self, keys, bps=0, cap=0):
    fd = self.flop_dealt; self.flop_dealt = True
    try: return orig_settle(self, keys, bps, cap)
    finally: self.flop_dealt = fd
for name, patch in (("M1_short_allin_reopens", ("act", m1_act)), ("M3_rake_without_flop", ("settle", m3_settle))):
    setattr(ref_engine.Ref, patch[0], patch[1])
    st = betting_check.run(hands, 99)
    out[name] = {"hands": st["hands"], "mismatches": len(st["mismatches"]), "first": st["mismatches"][0]["detail"][:200] if st["mismatches"] else None}
    setattr(ref_engine.Ref, "act", orig_act); setattr(ref_engine.Ref, "settle", orig_settle)
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
