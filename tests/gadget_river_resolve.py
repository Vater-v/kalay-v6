"""Safe re-solving on the real HU river tree: trunk + subgame, exact leaves, no critic.

The full river game (config.tree_rule(2, 3, menu)) is solved on a training spot. The trunk is the
first two actions (seat 1 checks, seat 0 bets); the subgame starts where seat 1 (the hero) faces the bet.
The hero's subgame strategy is re-solved three ways and spliced back into the full strategy; the exact
exploitability of the spliced profile in the FULL game (pcfr.exploitability, pot units; also mbb/hand)
is the result:
  naive   - the opponent's range at the subgame root = his equilibrium reach (what the live path
            carried before this fix: a range, no ceilings)
  gadget  - our range + the opponent's per-hand counterfactual values at the subgame root from the
            full solve (exact ceilings): the CFR-D / DeepStack gadget of pcfr
  gadget + error - the same ceilings with iid Gaussian noise of sigma pot (what a critic's regression
            error does to the re-solve), a constant bias, or the check-down EV in place of the values
  rangegadget - the gadget with the opponent's equilibrium reach as his deal range instead of the uniform one
            (Burch et al.'s original form: Terminate can only REMOVE hands from the carried range), exact and
            with the same errors: which deal is more robust to ceiling error
  crushed  - the carried range is WRONG the way a Bayesian update with a likelihood floor makes it (spec audit,
            carried_range_demo): the 30 % of his betting range with the lowest equity (his bluffs) is kept at 1e-3 of
            its weight. "naive_crushed" re-solves on that range alone, "rangegadget_crushed" uses it as the deal
            with exact ceilings, "mixgadget_crushed" uses half crushed range + half uniform; the uniform-deal gadget
            above does not see the range at all. This decides which deal the live player defaults to.
The gap "spliced E - full E" is the price of the re-solving method; every number is exact (no sampling).

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \\
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \\
    python3 tests/gadget_river_resolve.py --spots 2 --iters 800 --out results/audit/gadget/river_resolve.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POKER_REGIME", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                      # noqa: E402
import jax.numpy as jnp                         # noqa: E402
import numpy as np                              # noqa: E402

from poker.envs import config                   # noqa: E402
from poker.solver import checkdown as CD        # noqa: E402
from poker.solver import pcfr as SV             # noqa: E402
from poker.solver import situations as SI       # noqa: E402
from poker.solver import tree as TR             # noqa: E402
from poker.solver.cards import NUM_HANDS as H   # noqa: E402


def subtree(topo: TR.Topology, root: int):
    """The subgame topology rooted at node `root` of `topo` and the map new index -> old index."""
    order = [root]
    for v in order:
        for c in topo.children[v]:
            if c >= 0:
                order.append(int(c))
    old = np.asarray(order, np.int32)
    new_of = {int(o): i for i, o in enumerate(old)}
    V, A = len(old), topo.children.shape[1]
    parent = np.asarray([-1] + [new_of[int(topo.parent[o])] for o in old[1:]], np.int32)
    label = np.asarray([-1] + [int(topo.label[o]) for o in old[1:]], np.int32)
    children = np.full((V, A), -1, np.int32)
    for i, o in enumerate(old):
        for a in range(A):
            c = int(topo.children[o, a])
            if c >= 0:
                children[i, a] = new_of[c]
    depth = np.zeros(V, np.int32)
    for i in range(1, V):
        depth[i] = depth[parent[i]] + 1
    levels = tuple(np.flatnonzero(depth == d) for d in range(depth.max() + 1))
    sub = TR.Topology(topo.n, parent, label, topo.actor[old].copy(), children, children >= 0, topo.live[old].copy(),
                      topo.terminal[old].copy(), levels, topo.street, None if topo.leaf is None else topo.leaf[old].copy(),
                      None if topo.offered is None else topo.offered[old].copy())
    return sub, old


def slice_spot(batch, b, old, ranges, gadget_ev=None, gadget_mask=None):
    """One spot of a prepared batch restricted to the subtree nodes `old`, with new root ranges [n, H]."""
    take = lambda k: jnp.asarray(np.asarray(batch[k])[b][old])
    out = {"legal": take("legal"), "alive": take("alive"), "contrib": take("contrib"), "pot": take("pot"),
           "behind": take("behind"), "street_bet": take("street_bet"), "rake": take("rake"),
           "ranges": jnp.asarray(ranges, jnp.float32), "valid": jnp.asarray(np.asarray(batch["valid"])[b]),
           "tables": jax.tree.map(lambda x: x[b], batch["tables"])}
    for k in ("board", "board1h", "pot_bb", "stack_pot", "rake_level", "rake_rate", "rake_cap", "pot_chips", "rake_bps",
              "rake_cap_chips"):
        out[k] = jnp.asarray(np.asarray(batch[k])[b])
    if gadget_ev is not None:
        out["gadget_ev"] = jnp.asarray(gadget_ev, jnp.float32)
        out["gadget_mask"] = jnp.asarray(gadget_mask, bool)
    return jax.tree.map(lambda x: x[None], out)


def solve(batch, st, iters, chunk=50):
    cfg = SV.SolverConfig(cap=iters, e_target=0.0, check_every=chunk)
    state = jax.vmap(lambda _: SV.init_cfr_state(st, jnp.float32))(jnp.arange(1))
    step = SV.make_cfr_chunk(st, cfg)
    for _ in range(iters // chunk):
        state = step(state, batch, None)
    z = jax.vmap(lambda s, spot: SV.average_logits(s, spot, st))(state["avg"], batch)
    follow = SV.average_follow(state)
    return z, follow, state, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spots", type=int, default=2)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--sub-iters", type=int, default=None)
    p.add_argument("--menu", default="normal")
    p.add_argument("--seed", type=int, default=1101)
    p.add_argument("--noise", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--bias", type=float, nargs="+", default=[-0.05, 0.05])
    p.add_argument("--trunk-frac", type=int, nargs=2, default=[3, 4], metavar=("NUM", "DEN"),
                   help="the trunk bet of seat 0 as a pot fraction of the river menu (default 3/4; 1/3 and 3/2 also exist)")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    sub_iters = a.sub_iters or a.iters
    topo = TR.rule_topology(2, 3, a.menu)
    st = SV.Static(topo)
    spots = SI.make_training_spots(a.seed, max(8 * a.spots, 32), 2, street=3)
    menus = np.asarray([config.menu_for(3, spr=int(s) / int(q)) for s, q in zip(spots["stack"], spots["pot"])])
    idx = np.flatnonzero(menus == a.menu)[:a.spots]
    spots = {k: v[idx] for k, v in spots.items()}
    batch = SI.prepare(spots, topo)
    cfg = SV.SolverConfig(cap=a.iters, e_target=0.0, check_every=50)
    ex_nodes = jax.jit(lambda z, spot: SV.exploitability(z, spot, st, cfg, None, nodes=True))
    ex = jax.jit(jax.vmap(lambda z, spot: SV.exploitability(z, spot, st, cfg, None)))
    # the trunk: root (seat 1) checks -> c1 (seat 0) bets the 3/4 pot (grid action of (3, 4)) -> c2: seat 1 faces the bet
    bet_action = 2 + list(config.RAISE_FRACS).index(tuple(a.trunk_frac))
    c1 = int(topo.children[0, config.CHECK_CALL])
    c2 = int(topo.children[c1, bet_action])
    assert c1 >= 0 and c2 >= 0 and int(topo.actor[0]) == 1 and int(topo.actor[c1]) == 0 and int(topo.actor[c2]) == 1
    hero, opp = 1, 0
    sub, old = subtree(topo, c2)
    sst = SV.Static(sub)
    full_int = st.to_internal[old[~sub.terminal]]        # full internal index of every subgame decision, in sub order
    results = {"menu": a.menu, "iters": a.iters, "sub_iters": sub_iters, "trunk": f"seat1 check, seat0 bet {a.trunk_frac[0]}/{a.trunk_frac[1]} pot",
               "subgame_nodes": int(sub.size), "full_nodes": int(topo.size), "spots": []}
    t_all = time.perf_counter()
    for b in range(len(idx)):
        one = jax.tree.map(lambda x: x[b:b + 1], batch)
        t0 = time.perf_counter()
        z_full, _, _, _ = solve(one, st, a.iters)
        e_full = float(np.asarray(ex(z_full, one)["nash_conv"])[0]) / 2
        nodes = ex_nodes(z_full[0], jax.tree.map(lambda x: x[0], one))
        on = np.asarray(nodes["on"])                                   # [V, n, H] on-policy CFVs
        pi = SV.policy(z_full[0], one["legal"][0][st.internal])
        reach = np.asarray(SV.forward(pi, one["ranges"][0], st))      # [V, n, H]
        r_hero, r_opp = reach[c2, hero], reach[c2, opp]
        valid = np.asarray(one["valid"][0])
        mass_hero = np.asarray(SV.j_mass(jnp.asarray(r_hero), jax.tree.map(lambda x: x[0], one["tables"])))
        cfv_opp = on[c2, opp]
        ev_opp = np.where(mass_hero > 1e-30, cfv_opp / np.maximum(mass_hero, 1e-30), 0.0) * valid    # whole-hand EV per hand
        pot_bb = float(spots["pot"][b]) / config.BB
        spr = float(spots["stack"][b]) / float(spots["pot"][b])
        hero_root = r_hero / r_hero.sum()
        uniform = valid / valid.sum()
        # the check-down EV of the opponent's hands at c2 (the crude V the toy measured at 533 mA/h)
        share = np.asarray(CD.share_river(jnp.asarray(hero_root, jnp.float32), jax.tree.map(lambda x: x[0], one["tables"])))
        contrib_opp = float(np.asarray(one["contrib"])[0, c2, opp])
        pot_c2 = float(np.asarray(one["pot"])[0, c2])
        cd_ev = (share * pot_c2 - contrib_opp) * valid
        rec = {"spot": int(idx[b]), "pot_bb": pot_bb, "spr": spr, "E_full": e_full, "solve_s": time.perf_counter() - t0,
               "hero_mass_at_c2": float(r_hero.sum()), "opp_mass_at_c2": float(r_opp.sum()), "methods": {}}
        rng = np.random.default_rng([a.seed, b])
        configs = [("naive_range", None), ("gadget_exact", np.zeros(H))]
        configs += [(f"gadget_noise_{s:g}", rng.normal(0.0, s, H)) for s in a.noise]
        configs += [(f"gadget_bias_{c:+g}", np.full(H, c)) for c in a.bias]
        configs += [("gadget_checkdown_ceiling", cd_ev - ev_opp), ("gadget_zero_ceiling", -ev_opp)]
        # the wrong carried range: his lowest-equity 30 % of betting mass (the bluffs) crushed to 1e-3 of its weight
        eq_opp = np.asarray(CD.share_river(jnp.asarray(hero_root, jnp.float32), jax.tree.map(lambda x: x[0], one["tables"])))
        order = np.argsort(np.where(r_opp > 0, eq_opp, np.inf))
        cum = np.cumsum(r_opp[order]) / max(r_opp.sum(), 1e-30)
        bluffs = order[: int(np.searchsorted(cum, 0.30))]
        crushed = r_opp.copy(); crushed[bluffs] *= 1e-3
        crushed_root = crushed / max(crushed.sum(), 1e-30)
        configs += [("naive_crushed", None), ("rangegadget_crushed", np.zeros(H)), ("rangegadget_crushed_noise_0.01", rng.normal(0.0, 0.01, H)),
                    ("mixgadget_crushed", np.zeros(H)), ("mixgadget_crushed_noise_0.01", rng.normal(0.0, 0.01, H))]
        configs += [("rangegadget_exact", np.zeros(H))]
        configs += [(f"rangegadget_noise_{s:g}", rng.normal(0.0, s, H)) for s in a.noise if s in (0.01, 0.02, 0.05)]
        configs += [(f"rangegadget_bias_{c:+g}", np.full(H, c)) for c in a.bias]
        opp_root = r_opp / max(r_opp.sum(), 1e-30)
        for name, err in configs:
            t1 = time.perf_counter()
            if "crushed" in name:
                deal = 0.5 * crushed_root + 0.5 * uniform if name.startswith("mixgadget") else crushed_root
            elif name.startswith("rangegadget"):
                deal = opp_root
            else:
                deal = uniform
            if err is None:
                ranges = np.stack([crushed_root if "crushed" in name else opp_root, hero_root])
                sb = slice_spot(one, 0, old, ranges)
            else:
                ranges = np.stack([deal, hero_root])
                ev = np.zeros((2, H), np.float32)
                ev[opp] = (ev_opp + err) * valid
                sb = slice_spot(one, 0, old, ranges, ev, np.asarray([True, False]))
            z_sub, follow, state, scfg = solve(sb, sst, sub_iters)
            e_sub = float(np.asarray(jax.jit(lambda z, s, f: SV.exploitability(z, s, sst, scfg, None, follow=f)["nash_conv"])(
                z_sub[0], jax.tree.map(lambda x: x[0], sb), follow[0]))) / 2
            z_spliced = np.asarray(z_full).copy()
            z_spliced[0, full_int] = np.asarray(z_sub)[0]
            e_spliced = float(np.asarray(ex(jnp.asarray(z_spliced), one)["nash_conv"])[0]) / 2
            f = np.asarray(follow)[0, opp]
            rec["methods"][name] = {"E_spliced": e_spliced, "gap": e_spliced - e_full, "gap_mbb_per_hand": 1000 * (e_spliced - e_full) * pot_bb,
                                    "E_subgame_own": e_sub, "follow_mean": float((f * ranges[opp]).sum()),
                                    "follow_min": float(f[valid > 0].min()) if err is not None else None,
                                    "err_rms": float(np.sqrt(np.mean((err * valid) ** 2))) if err is not None else 0.0,
                                    "seconds": time.perf_counter() - t1}
            print(f"spot {idx[b]} pot {pot_bb:.1f} BB spr {spr:.2f} E_full {e_full:.5f} | {name:28s} E_spliced {e_spliced:.5f} "
                  f"gap {e_spliced - e_full:+.5f} pot = {1000 * (e_spliced - e_full) * pot_bb:+.1f} mbb/hand; own E {e_sub:.5f}; "
                  f"follow mean {rec['methods'][name]['follow_mean']:.3f}; {time.perf_counter() - t1:.1f} s", flush=True)
        results["spots"].append(rec)
    results["seconds"] = time.perf_counter() - t_all
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "spots"}))


if __name__ == "__main__":
    main()
