"""The code fingerprint: which code made an artifact - its provenance,
not its game (the game is config.game_record / game_hash, compared at every load; this never replaces it).

    PYTHONPATH=. python3 -m poker.envs.fingerprint [--json] [--check results/pack/PACKED.json]

code_fingerprint() = sha256 over the source files of ROOTS and EXTRA (sorted paths, each file content-hashed; the
per-file hashes are recorded in "files"), with the commit and a dirty flag. Where the files come from ("source"):
- "git": a checkout of this tree (git ls-files; the working-tree content is hashed, so an edit changes the
  fingerprint); dirty = any `git status --porcelain` line under ROOTS/EXTRA, untracked files included;
- "packed": no git, but the results/pack/PACKED.json that tools/cloud/pack.sh puts into the tarball of a rented box: its
  file list is hashed as found; dirty = the content differs from the packer's fingerprint, a file of ROOTS is not in
  the list, or the pack itself was dirty (pack.sh --allow-dirty);
- "files": neither (a copied tree): every file under ROOTS, no commit, dirty (provenance unknown).
Next to the digest, not in it, "runtime": Python, the numerical packages and the NVIDIA driver and GPUs of the process
(static sniff S7: a layout proves which numpy / JAX a shard or a resume ran with; poker/solver/loop.py check_code compares
Python and the packages on a resume).
Standard library only (the box checks it before its venv exists). One value per process and tree - the code the
process loaded - computed at the first call. config.game_meta() stores it next to the game record in every artifact
that carries the record; tools/cloud/gate.py compares the packed fingerprint, the box's and the artifacts'.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

VERSION = 1
ROOTS = ("poker", "checks", "tools")
EXTRA = ("requirements.lock",)          # the packages the box installs from: part of the code that runs
REPO = Path(__file__).resolve().parents[2]
PACKED = Path("results/pack") / "PACKED.json"
LISTED_DIRTY = 50                       # dirty paths kept in a record (dirty_count has them all)
RUNTIME_PACKAGES = ("numpy", "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cuda-nvcc-cu12")
_CACHE = {}


def runtime() -> dict:
    """Python, RUNTIME_PACKAGES (None when absent) and the NVIDIA driver and GPUs (nvidia-smi; none without one)."""
    from importlib import metadata
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60, check=True).stdout
        gpus = [", ".join(x.strip() for x in line.split(",")) for line in rows.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        gpus = []
    return {"python": platform.python_version(), "packages": packages, "gpus": gpus,
            "driver": gpus[0].split(", ")[1] if gpus and len(gpus[0].split(", ")) > 1 else None}


def _file_sha256(path: Path) -> str:
    if path.is_symlink():                                   # as git stores a link: its target text
        return hashlib.sha256(os.readlink(path).encode()).hexdigest()
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(files: dict) -> str:
    """sha256 over the sorted (path, content sha256) pairs: the fingerprint of a file map."""
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(f"{path}\0{files[path]}\n".encode())
    return h.hexdigest()


def _walk(root: Path) -> list:
    """Every file under ROOTS (and EXTRA) of a tree without git, bytecode caches aside."""
    out = [p for p in EXTRA if (root / p).is_file()]
    for top in ROOTS:
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root)
            if path.is_file() and "__pycache__" not in rel.parts and path.suffix not in (".pyc", ".pyo"):
                out.append(rel.as_posix())
    return sorted(out)


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout.decode()


def _from_git(root: Path):
    if Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve() != root.resolve():
        raise LookupError("not the repository of this tree")
    listed = [p for p in _git(root, "ls-files", "-z", "--", *ROOTS, *EXTRA).split("\0") if p]
    status = [line for line in _git(root, "status", "--porcelain", "--untracked-files=all", "--", *ROOTS,
                                    *EXTRA).splitlines() if line]
    return listed, _git(root, "rev-parse", "HEAD").strip(), status


def compute(root=REPO) -> dict:
    """The fingerprint of the tree at `root` now (code_fingerprint: once per process)."""
    root = Path(root).resolve()
    packed = None
    try:
        listed, commit, dirty_paths = _from_git(root)
        source = "git"
    except (OSError, subprocess.CalledProcessError, LookupError):
        path = root / PACKED
        packed = json.loads(path.read_text()).get("fingerprint") if path.is_file() else None
        if packed and packed.get("files"):
            source, commit = "packed", packed.get("commit")
            listed = sorted(packed["files"])
            dirty_paths = [f"?? {p}" for p in sorted(set(_walk(root)) - set(listed))]
        else:
            source, commit, listed = "files", None, _walk(root)
            dirty_paths = ["no git checkout and no results/pack/PACKED.json with a fingerprint: provenance unknown"]
    files = {p: _file_sha256(root / p) for p in sorted(listed)}
    if packed is not None and source == "packed":
        dirty_paths = [f" M {p}" for p in listed if files[p] != packed["files"][p]] + dirty_paths
        if packed.get("dirty"):
            dirty_paths += [f"(packed dirty) {p}" for p in packed.get("dirty_paths", [])] or ["(packed dirty)"]
    return {"version": VERSION, "sha256": digest(files), "commit": commit, "dirty": bool(dirty_paths),
            "dirty_count": len(dirty_paths), "dirty_paths": dirty_paths[:LISTED_DIRTY], "source": source,
            "roots": list(ROOTS + EXTRA), "files": files, "runtime": runtime()}


def code_fingerprint(root=REPO) -> dict:
    """The fingerprint of the code this process loaded (computed at the first call for `root`)."""
    key = str(Path(root).resolve())
    if key not in _CACHE:
        _CACHE[key] = compute(root)
    return _CACHE[key]


def describe(fp) -> str:
    """One line of a fingerprint record (an artifact's or this process's)."""
    if not isinstance(fp, dict) or "sha256" not in fp:
        return "code fingerprint not recorded (made before task #34)"
    state = ("clean" if not fp.get("dirty") else
             f"DIRTY {fp.get('dirty_count', len(fp.get('dirty_paths', [])))}: " + ", ".join(fp.get("dirty_paths", [])[:3]))
    return (f"code {fp['sha256'][:12]} (commit {str(fp.get('commit'))[:12]}, {state}, {len(fp.get('files', {}))} files, "
            f"{fp.get('source')})")


def describe_runtime(fp) -> str:
    """One line of a record's runtime: Python, the packages, the driver and the GPUs."""
    rt = (fp or {}).get("runtime") or {}
    p = rt.get("packages") or {}
    return (f"python {rt.get('python')}, numpy {p.get('numpy')}, jax {p.get('jax')} / jaxlib {p.get('jaxlib')}, cuda "
            f"plugin {p.get('jax-cuda12-plugin')}, nvcc {p.get('nvidia-cuda-nvcc-cu12')}, driver {rt.get('driver')}, "
            f"GPUs {rt.get('gpus')}")


def runtime_differences(a, b) -> dict:
    """{name: (a's, b's)} of Python and RUNTIME_PACKAGES where two records differ (the driver and the GPUs aside: a
    shard or a resume may run on another box)."""
    ra, rb = (a or {}).get("runtime") or {}, (b or {}).get("runtime") or {}
    pa, pb = ra.get("packages") or {}, rb.get("packages") or {}
    out = {"python": (ra.get("python"), rb.get("python"))} if ra.get("python") != rb.get("python") else {}
    out.update({k: (pa.get(k), pb.get(k)) for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)})
    return out


def differences(a, b) -> list:
    """Paths whose content differs between two fingerprint records ('+' only in b, '-' only in a)."""
    fa, fb = (a or {}).get("files", {}), (b or {}).get("files", {})
    return ([p for p in sorted(set(fa) & set(fb)) if fa[p] != fb[p]] + [f"-{p}" for p in sorted(set(fa) - set(fb))]
            + [f"+{p}" for p in sorted(set(fb) - set(fa))])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", type=Path, default=REPO)
    p.add_argument("--json", action="store_true", help="the whole record")
    p.add_argument("--check", type=Path, help="exit 1 unless this tree's fingerprint equals the one in this PACKED.json")
    args = p.parse_args(argv)
    fp = code_fingerprint(args.root)
    print(json.dumps(fp, indent=1, sort_keys=True) if args.json else describe(fp))
    if args.check:
        packed = json.loads(args.check.read_text()).get("fingerprint")
        if not packed:
            print(f"FINGERPRINT FAIL: {args.check} has no fingerprint (a pack from before task #34)")
            return 1
        if packed["sha256"] != fp["sha256"]:
            diff = differences(packed, fp)
            print(f"FINGERPRINT FAIL: this tree is {describe(fp)}, {args.check} packed {describe(packed)}; differing: "
                  + ", ".join(diff[:8]) + (f" (+{len(diff) - 8} more)" if len(diff) > 8 else ""))
            return 1
        print(f"FINGERPRINT OK: equal to {args.check} ({fp['sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
