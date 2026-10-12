"""Rebuild the committed Rust fixture without modifying the oracle checkout.

Run with the native Python/Cargo of the oracle checkout, e.g. on Windows:
  python tools/generate_oracle.py --rust-crate C:/projects/kalay-v2/rs \
      --cargo C:/Users/Vater/.cargo/bin/cargo.exe
Only generation needs Rust; python -m checks.s2_4 uses the committed fixture.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

COUNT = 1_000_000
SEED = 0x724E6144506F6B65


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(crate):
    paths = [crate / name for name in ("Cargo.toml", "Cargo.lock", "build.rs")]
    paths += [p for p in (crate / "src").rglob("*") if p.is_file()]
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.relative_to(crate).as_posix()):
        digest.update(path.relative_to(crate).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-crate", required=True, type=Path)
    parser.add_argument("--cargo", default=shutil.which("cargo"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.cargo:
        parser.error("Cargo not found; pass --cargo (generation only)")
    crate = args.rust_crate.resolve()
    repo = Path(__file__).resolve().parents[1]
    output = args.output or repo / "checks/data/rust_oracle_1m.bin"
    output.parent.mkdir(parents=True, exist_ok=True)
    before = source_hash(crate)
    env = os.environ.copy()
    env["PATH"] = str(Path(args.cargo).resolve().parent) + os.pathsep + env["PATH"]
    rust_version = subprocess.check_output([str(Path(args.cargo).with_name("rustc.exe" if os.name == "nt" else "rustc")), "--version"], env=env, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="poker-huy-oracle-") as directory:
        project = Path(directory)
        (project / "src").mkdir()
        helper = repo / "tools/oracle.rs"
        shutil.copyfile(helper, project / "src/main.rs")
        # JSON quoting is also valid TOML quoting for this normalized path.
        manifest = '[package]\nname = "poker_huy_oracle"\nversion = "0.1.0"\nedition = "2021"\n\n[dependencies]\npoker_engine = { path = ' + json.dumps(crate.as_posix()) + ', default-features = false }\n'
        (project / "Cargo.toml").write_text(manifest, encoding="utf-8")
        fixture = project / "oracle.bin"
        env["CARGO_TARGET_DIR"] = str(project / "target")
        subprocess.run([args.cargo, "run", "--offline", "--release", "--manifest-path", str(project / "Cargo.toml"), "--", str(fixture), str(COUNT), str(SEED)], check=True, env=env)
        if source_hash(crate) != before:
            raise RuntimeError("Oracle sources changed during generation")
        metadata = {
            "format": "PHUYOR01: 8-byte magic, LE u64 count/seed, rows uint8[7] + LE uint16 rank",
            "count": COUNT, "seed": SEED, "fixed_anchor_count": 16,
            "sampler": "SplitMix64 v1; unbiased rejection + partial Fisher-Yates",
            "jax_card_encoding": "suit*13+rank; suits cdhs, rank 2..A",
            "rust_card_conversion": "((jax_id % 13) << 2) | (3 - jax_id // 13)",
            "rank_orientation": "smaller Rust rank is stronger; range 1..7462",
            "oracle": "kalay-v2 poker_engine::table::eval::eval_nlh_river",
            "oracle_source_sha256": before,
            "oracle_evaluator_sha256": sha256(crate / "src/table/eval.rs"),
            "oracle_card_sha256": sha256(crate / "src/table/card.rs"),
            "oracle_lookup_sha256": sha256(crate / "src/lookup.bin"),
            "generator_rust_sha256": sha256(helper),
            "cargo_lock_sha256": sha256(project / "Cargo.lock"),
            "rustc": rust_version, "fixture_sha256": sha256(fixture),
        }
        shutil.copyfile(fixture, output)
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {output} ({output.stat().st_size:,} bytes)")
    print(f"SHA256 {metadata['fixture_sha256']}")


if __name__ == "__main__":
    main()
