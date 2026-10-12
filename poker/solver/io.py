"""Shared worker logging, claims, atomic result files, and job barriers."""

import os
import socket
import time
from pathlib import Path

import numpy as np

WORKER = os.environ.get("WORKER_ID", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))


def log(msg):
    print(time.strftime("%H:%M:%S"), f"[w{WORKER}]", msg, flush=True)


def claim(out: Path, name: str) -> bool:
    """True if this process now owns work item `name` (exclusive create; first one wins)."""
    path = out / "claims" / f"{name}.claim"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} pid {os.getpid()} gpu {WORKER} {time.time():.0f}\n".encode())
    os.close(fd)
    return True


def write_text(path: Path, text: str):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def savez(path: Path, **arrays):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def publish_once(path: Path, text: str) -> str:
    """Write `text` to `path` unless another process already did; return the content that won."""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    try:
        os.link(tmp, path)
    except FileExistsError:
        pass
    finally:
        tmp.unlink()
    return path.read_text()


def wait_for(paths, what):
    """Phase barrier: block until every path exists (other workers finishing their items)."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        log(f"waiting for {len(missing)} {what} of other workers")
    while missing:
        time.sleep(5)
        missing = [p for p in missing if not p.exists()]
