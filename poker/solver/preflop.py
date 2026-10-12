"""Build the HU 200 BB no-rake preflop cache from an explicitly selected new checkpoint.

The topology, masks and chip targets are the teacher's canonical engine replay.
The cached strategy is tied to the exact pre_end bytes and current game. A solve
that misses its error target does not publish a playable table.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np

from poker.envs import config
from poker.solver import critics as CR
from poker.solver import pcfr as SV
from poker.solver import runout
from poker.solver import situations as SI
from poker.solver import tree as TR
from poker.solver.cards import HANDS

EQUITY_CACHE = Path(__file__).resolve().parents[2] / "results/cache/preflop_equity_v1.npy"


def prepare_root():
    """The regime's one preflop root and canonical replay, shared with critic training."""
    if config.REGIME != 1 or config.PLAYERS != (2,) or config.GRID != config.STREET_SIZES_GRID:
        raise ValueError("preflop cache requires current grid 13 and regime 1: HU, 200 BB, no rake")
    spots = SI.make_preflop_spots(0, 1, 2, rake="mix")
    depth = int(spots["stack"][0])
    menu = config.menu_for(0, depth_bb=depth / config.BB)
    topo = TR.rule_topology(2, 0, menu)
    replay = TR.replay(topo, spots["pot"], spots["stack"], spots["rake_bps"], spots["rake_cap"])
    if not np.asarray(replay["actor_ok"]).all() or not np.asarray(replay["terminal_ok"]).all():
        raise RuntimeError("canonical preflop replay changed actors or boundaries")
    return spots, topo, replay, menu


def table_arrays(topo, replay, policy, values):
    """Serialize canonical replay, with no second implementation of bet sizing."""
    legal = np.asarray(replay["legal"])[0]
    policy = np.asarray(policy, np.float32)
    expected = (len(topo.internal), len(HANDS), config.NUM_ACTIONS)
    if policy.shape != expected or not np.isfinite(policy).all() or np.any(policy < 0):
        raise ValueError(f"invalid preflop strategy, expected {expected}, got {policy.shape}")
    if np.any(policy * ~legal[topo.internal, None, :]):
        raise ValueError("preflop strategy has mass on illegal actions")
    reached = np.asarray(replay["alive"])[0, topo.internal]
    if not np.allclose(policy[reached].sum(-1), 1.0, rtol=0, atol=2e-6):
        raise ValueError("preflop strategy does not sum to one at live decision nodes")
    return {
        "policy": policy, "internal": topo.internal.astype(np.int32),
        "parent": topo.parent, "label": topo.label, "actor": topo.actor,
        "children": topo.children, "terminal": topo.terminal, "leaf": topo.leaf,
        "legal": legal, "targets": np.asarray(replay["targets"])[0],
        "street_bet": np.asarray(replay["street_bet"])[0],
        "contrib": np.asarray(replay["contrib"])[0], "pot": np.asarray(replay["pot"])[0],
        "root_values": np.asarray(values, np.float32), "hands": HANDS.astype(np.int32),
    }


def publish_table(out, arrays, *, critic_path, critic_meta, spot, solver, seconds):
    """Write once; metadata binds this table to its bytes, game and continuation model."""
    out = Path(out)
    if out.suffix != ".npz":
        raise ValueError("--out must end in .npz")
    meta_path = out.with_suffix(".json")
    if out.exists() or meta_path.exists():
        raise FileExistsError(f"{out} or {meta_path} exists; select a new output")
    if not np.isfinite(solver["error"]) or solver["error"] > solver["target_error"]:
        raise ValueError(f"preflop solve error {solver['error']} exceeds {solver['target_error']}; table not published")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    info = {
        "format": "preflop-canonical-v1", **config.game_meta(),
        "spot": spot, "solver": solver, "seconds": seconds,
        "critic": {"path": str(Path(critic_path).resolve()),
                   "sha256": hashlib.sha256(Path(critic_path).read_bytes()).hexdigest(),
                   "name": critic_meta["name"], "kappa": critic_meta["kappa"]},
        "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "scope": "local error of the canonical preflop tree with this critic; not full-game exploitability",
    }
    with meta_path.open("x") as stream:
        json.dump(info, stream, indent=1)
        stream.write("\n")
    return info


def build(critics_dir, out, *, target_error=0.001, max_iterations=10000, chunk=50):
    if not np.isfinite(target_error) or target_error <= 0:
        raise ValueError("target error must be finite and positive")
    if chunk <= 0 or max_iterations <= 0 or max_iterations % chunk:
        raise ValueError("max iterations must be a positive multiple of chunk")
    out = Path(out)
    if out.suffix != ".npz" or out.exists() or out.with_suffix(".json").exists():
        raise ValueError("select a new .npz output and unused adjacent .json")
    started = time.perf_counter()
    critic_path = Path(critics_dir) / "pre_end.npz"
    params, meta = CR.load_critic(critic_path, name="pre_end")
    critic = CR.net_critic(params, meta["kappa"])
    spots, topo, replay, menu = prepare_root()
    equity = runout.preflop_equity_cached(EQUITY_CACHE)
    batch = SI.prepare(spots, topo, preflop_equity=equity)
    static = SV.Static(topo)
    cfg = SV.SolverConfig(cap=max_iterations, e_target=target_error, check_every=chunk)
    result = SV.solve(batch, static, cfg, aux=critic)
    diagnostic = SV.check_exploitability(jax.tree.map(np.asarray, jax.vmap(
        lambda z, spot: SV.exploitability(z, spot, static, cfg, critic))(result["z"], batch)))
    error = float(diagnostic["nash_conv"][0] / 2)
    if not np.isfinite(error) or error > target_error:
        raise RuntimeError(f"preflop error {error:.6g} exceeds {target_error:g} after "
                           f"{int(result['u'][0])} iterations; no table published")
    policy = SV.policy(result["z"][0], batch["legal"][0, topo.internal])
    values = jax.jit(lambda z, spot: SV.root_values(z, spot, static, cfg, critic))(
        result["z"][0], jax.tree.map(lambda x: x[0], batch))
    arrays = table_arrays(topo, replay, policy, values)
    return publish_table(out, arrays, critic_path=critic_path, critic_meta=meta,
                         spot={"stack_chips": int(spots["stack"][0]), "pot_chips": int(spots["pot"][0]),
                               "rake_bps": int(spots["rake_bps"][0]), "menu": menu, "ranges": "uniform"},
                         solver={"method": "pcfr+", "error": error, "target_error": target_error,
                                 "iterations": int(result["u"][0]), "max_iterations": max_iterations,
                                 "chunk": chunk}, seconds=time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critics", type=Path, required=True, help="new checkpoint directory containing pre_end.npz")
    parser.add_argument("--out", type=Path, required=True, help="new .npz table; adjacent .json binds its model and game")
    parser.add_argument("--target-error", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=50)
    args = parser.parse_args(argv)
    info = build(args.critics, args.out, target_error=args.target_error,
                 max_iterations=args.max_iterations, chunk=args.chunk)
    print(json.dumps({"table": str(args.out), "solver": info["solver"], "seconds": info["seconds"]}))


if __name__ == "__main__":
    main()
