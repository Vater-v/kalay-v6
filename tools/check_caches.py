"""Check exact cache hashes and matrix structure without regenerating the tables.

From the repository root: python -m tools.check_caches [--out FILE]
Reads results/cache by default; writes results/reports/cache_check.json and its
SHA-256 manifest. Optional --gpu N recomputes N random flop classes. A completed
diagnostic with ok=false exits 1; unreadable required tables raise their error.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

RECORDED = {   # Expected v1 hashes, also recorded by the cloud cache manifest.
    "rank7_v1.npy": "a8c26aae129d98050681c214e4fa3dba21a382fcbf74b100f0272fe1cb4ba2ed",
    "preflop_equity_v1.npy": "c2d416dc644df3895445b33b4945bfb69f16c51bfa3e1a5e335ec162602528f0",
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "results/cache"
SIZE_1326 = 128 + 1326 * 1326 * 4


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


_DISJOINT = None


def identity(w, valid_hands):
    """max |W + W^T - 1| on valid pairs, max |W| on the others, value range."""
    global _DISJOINT
    if _DISJOINT is None:                      # integer matmul in numpy is slow: once, in float
        from poker.solver.cards import MEMBER
        _DISJOINT = (MEMBER.T.astype(np.float32) @ MEMBER.astype(np.float32)) == 0
    valid = _DISJOINT & valid_hands[:, None] & valid_hands[None, :]
    s = w.astype(np.float64) + w.T.astype(np.float64)
    return {
        "max_dev_valid": float(np.max(np.abs(s[valid] - 1.0))),
        "max_abs_invalid": float(np.max(np.abs(w[~valid]))) if (~valid).any() else 0.0,
        "valid_pairs": int(valid.sum()),
        "finite": bool(np.isfinite(w).all()),
        "min": float(w.min()), "max": float(w.max()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/reports/cache_check.json")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="directory holding the exact tables and flop cache")
    ap.add_argument("--gpu", type=int, default=0, help="recompute this many random flop classes")
    args = ap.parse_args(argv)
    CACHE = args.cache_dir
    from poker.solver.cards import MEMBER, NUM_HANDS
    from poker.solver.flops import FLOPS
    t0 = time.time()
    out = {"tables": {}, "flop": {}}

    for name, want in RECORDED.items():
        path = CACHE / name
        got = sha256_file(path)
        out["tables"][name] = {"size": path.stat().st_size, "sha256": got, "recorded": want, "match": got == want}
        print(f"{name}: size {path.stat().st_size} sha256 {got} {'MATCH' if got == want else 'MISMATCH'}", flush=True)

    pre = np.load(CACHE / "preflop_equity_v1.npy")
    ident = identity(pre, np.ones(NUM_HANDS, bool))
    out["tables"]["preflop_equity_v1.npy"].update(shape=list(pre.shape), dtype=str(pre.dtype), identity=ident)
    print(f"preflop: shape {pre.shape} {pre.dtype} identity {ident}", flush=True)
    del pre

    fdir = CACHE / "flop_equity_v1"
    names = sorted(p.name for p in fdir.iterdir())
    expected = sorted(f"{a}_{b}_{c}.npy" for a, b, c in FLOPS.tolist())
    extra = sorted(set(names) - set(expected))
    missing = sorted(set(expected) - set(names))
    print(f"flop dir: {len(names)} entries, expected {len(expected)}, missing {len(missing)}, extra {len(extra)}", flush=True)
    out["flop"].update(entries=len(names), expected=len(expected), missing=missing, extra=extra)

    manifest, worst_valid, worst_invalid, bad = {}, 0.0, 0.0, []
    lo, hi = 1.0, 0.0
    for i, (a, b, c) in enumerate(FLOPS.tolist()):
        path = fdir / f"{a}_{b}_{c}.npy"
        if not path.exists():
            continue
        size = path.stat().st_size
        manifest[path.name] = sha256_file(path)
        try:
            w = np.load(path)
        except Exception as e:                                   # noqa: BLE001 - report every unreadable file
            bad.append((path.name, f"load: {e}"))
            continue
        if size != SIZE_1326 or w.shape != (NUM_HANDS, NUM_HANDS) or w.dtype != np.float32:
            bad.append((path.name, f"size {size} shape {w.shape} dtype {w.dtype}"))
            continue
        valid_hands = ~MEMBER[[a, b, c]].any(axis=0)
        ident = identity(w, valid_hands)
        if not ident["finite"] or ident["max_dev_valid"] > 1e-6 or ident["max_abs_invalid"] != 0.0:
            bad.append((path.name, ident))
        worst_valid = max(worst_valid, ident["max_dev_valid"])
        worst_invalid = max(worst_invalid, ident["max_abs_invalid"])
        lo, hi = min(lo, ident["min"]), max(hi, ident["max"])
        if i % 250 == 0:
            print(f"  {i}/1755 checked, {time.time() - t0:.0f} s", flush=True)
    out["flop"].update(checked=len(manifest), bad=bad, worst_identity_dev_valid=worst_valid,
                       worst_abs_invalid=worst_invalid, value_range=[lo, hi], file_size=SIZE_1326)
    all_bytes = hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(manifest.items())).encode()).hexdigest()
    out["flop"]["manifest_sha256"] = all_bytes
    print(f"flop matrices: {len(manifest)} checked, {len(bad)} bad, identity max dev {worst_valid:.3g}, "
          f"invalid max {worst_invalid}, range [{lo}, {hi}], manifest sha256 {all_bytes}", flush=True)
    for item in bad[:20]:
        print("  BAD", item, flush=True)

    if args.gpu:
        import jax
        import jax.numpy as jnp
        from poker.solver import runout
        rng = np.random.default_rng(20261009)
        picks = rng.choice(len(FLOPS), size=args.gpu, replace=False)
        rec = []
        for k in picks.tolist():
            flop = FLOPS[k]
            w_new = np.asarray(runout.equity_matrix_flop(jnp.asarray(flop, jnp.int32)), np.float32)
            w_old = np.load(fdir / f"{flop[0]}_{flop[1]}_{flop[2]}.npy")
            diff = float(np.max(np.abs(w_new.astype(np.float64) - w_old)))
            rec.append({"flop": flop.tolist(), "bit_identical": bool(np.array_equal(w_new, w_old)), "max_abs_diff": diff})
        out["flop"]["recomputed"] = {"device": str(jax.devices()[0]), "classes": rec,
                                     "all_bit_identical": all(r["bit_identical"] for r in rec),
                                     "max_abs_diff": max(r["max_abs_diff"] for r in rec)}
        print(f"recomputed {len(rec)} flop classes on {jax.devices()[0]}: bit identical "
              f"{sum(r['bit_identical'] for r in rec)}/{len(rec)}, max |diff| {out['flop']['recomputed']['max_abs_diff']:.3g}",
              flush=True)

    out["wall_s"] = round(time.time() - t0, 1)
    ok = (all(t["match"] for t in out["tables"].values() if "match" in t) and not missing and not extra and not bad
          and out["tables"]["preflop_equity_v1.npy"]["identity"]["max_dev_valid"] <= 1e-6
          and out["tables"]["preflop_equity_v1.npy"]["identity"]["max_abs_invalid"] == 0.0
          and (not args.gpu or out["flop"]["recomputed"]["all_bit_identical"]))
    out["ok"] = bool(ok)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).with_suffix(".manifest").write_text("".join(f"{v}  {k}\n" for k, v in sorted(manifest.items())))
    print(f"CACHE {'OK' if ok else 'FAIL'} ({out['wall_s']} s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
