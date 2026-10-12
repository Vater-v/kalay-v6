"""Count the fused kernels per PCFR+ iteration of the live flop / river solve (compiled HLO of pcfr's chunk on
this CPU backend: the body of the fori_loop). A GPU backend fuses differently, so this is an order-of-magnitude
input for a kernel-launch-bound estimate of one iteration on the GTX 1070 (no GPU here).

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/hlo_count.py
Writes results/audit/live/hlo_count.json."""
import json, re
import jax, jax.numpy as jnp, numpy as np
from tools.audit.live.timing import street_root
from poker.solver import subgame as SG, pcfr as SV, critics as CR
from poker.solver.cards import NUM_HANDS as H

out = {}
for st in (1, 3):
    root = street_root(st)
    tree = SG.build_tree(root, 1, ())
    eq = None if st == 3 else np.random.default_rng(0).random((H, H), np.float32)
    aux = CR.net_critic(CR.init_critic(jax.random.key(0), (512, 512, 512)), 0.0) if st == 1 else None
    spot = SG.spot_batch(tree, np.full((2, H), 1 / H), 0.0, eq, None)
    cfg = SV.SolverConfig(cap=50, e_target=0.0, check_every=50)
    step = SV.make_cfr_chunk(tree.st, cfg)
    state = jax.vmap(lambda _: SV.init_cfr_state(tree.st, jnp.float32))(jnp.arange(1))
    txt = step.lower(state, spot, aux).compile().as_text()
    comps = re.split(r"\n(?=\S)", txt)
    # the while body: the computation called as body= of the while instruction
    body_names = re.findall(r"body=%?([\w.\-]+)", txt)
    counts = {}
    for name in set(body_names):
        for c in comps:
            if re.match(rf"%?{re.escape(name)}\b", c.strip()) or c.strip().startswith(name) or (" " + name + " ") in c.split("\n")[0]:
                counts[name] = {"fusion_calls": len(re.findall(r"= \S+ fusion\(", c)), "dots": len(re.findall(r" dot\(", c)),
                                "custom_calls": len(re.findall(r"custom-call\(", c)), "instructions": c.count("\n")}
    out[["pre", "flop", "turn", "river"][st]] = {"nodes": int(tree.size), "decisions": int(len(tree.internal)),
                                                   "leaves": int(len(tree.st.leaves)), "while_bodies": counts,
                                                   "total_fusion_kernels_in_module": len(re.findall(r"= \S+ fusion\(", txt))}
print(json.dumps(out, indent=1))
open("results/audit/live/hlo_count.json", "w").write(json.dumps(out, indent=1))
