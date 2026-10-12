"""What the likelihood floor buys and what it does not: a HU river re-solve (exact terminals, the live path's
solve_batch at the play T = 300) whose carried opponent range is WRONG in the way range-only re-solving gets it
wrong - an earlier solve's likelihoods said the opponent never takes this line with his weak hands, he does.

True opponent range: uniform over the hands the board allows (our range: uniform). Carried range: the weakest
`air` fraction of his hands (by showdown rank on this board) down-weighted by eps^k, the live path's floor after
k floored updates (live.py:787-802: r' = r * max(L, eps) / Z). Our strategy sigma_c from the carried-range solve
is then evaluated on the TRUE ranges: the opponent's best-response value against it minus his value in the
true-range equilibrium (pcfr.exploitability 'best' / 'value', pot units) - the local loss on this street (an
illustration of the mechanism, not a whole-game exploitability; the gadget is not implemented in the engine,
so it cannot be measured here).

    POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. python3 tools/audit/live/river_floor.py
Writes results/audit/live/river_floor.json."""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from poker.solver import cards as C
from poker.solver import pcfr as SV
from poker.solver import subgame as SG
from poker.solver.cards import NUM_HANDS as H
from tools.audit.live.timing import street_root

T = 300


def main():
    root = street_root(3)
    board = SG.visible_board(root)
    valid = C.board_valid(board[board >= 0]).astype(np.float64)
    ranks = np.asarray(C.hand_ranks(jnp.asarray(board, jnp.int32)))
    hero = 1                                                         # BB, first to act on the river
    opp = 0
    tree = SG.build_tree(root, hero, ())
    true = np.tile(valid / valid.sum(), (2, 1))
    st = tree.st
    cfg = SV.SolverConfig()
    ex = jax.jit(lambda z, spot: SV.exploitability(z, spot, st, cfg, None))
    spot_true = jax.tree.map(lambda x: x[0], SG.spot_batch(tree, true))

    def evaluate(sigma_ours, sigma_eq):
        """sigma_ours at our nodes, sigma_eq at his (his part is irrelevant to his best-response value)."""
        actors = st.actor[st.internal]
        mixed = np.where((actors == hero)[:, None, None], sigma_ours, sigma_eq)
        z = jnp.log(jnp.maximum(jnp.asarray(mixed, jnp.float32), 1e-30))
        r = ex(z, spot_true)
        return float(np.asarray(r["best"])[opp]), float(np.asarray(r["value"])[opp])

    t0 = time.perf_counter()
    eq = SG.solve_batch([tree], [true], T, 50, ladder=(1,), certificate=True)[0]
    v_star_best, v_star = evaluate(eq.sigma, eq.sigma)
    out = {"T": T, "board": board.tolist(), "pot_chips": int(root.bet_total.sum()), "behind": root.stack.tolist(),
           "nodes": int(tree.size), "equilibrium_E_pot": eq.e, "opp_value_equilibrium": v_star,
           "opp_best_vs_equilibrium": v_star_best, "rows": []}
    valid_idx = np.flatnonzero(valid > 0)
    order = valid_idx[np.argsort(ranks[valid_idx], kind="stable")]
    for air in (0.3, 0.5):
        weak = order[: int(air * len(order))]
        for k, eps in ((1, 1e-3), (2, 1e-3), (3, 1e-3), (1, 1e-9), (0, None)):
            carried = true.copy()
            if eps is not None:
                carried[opp, weak] *= eps ** k
                carried[opp] /= carried[opp].sum()
            sol = SG.solve_batch([tree], [carried], T, 50, ladder=(1,))[0]
            best, _ = evaluate(sol.sigma, eq.sigma)
            row = {"air_fraction": air, "floored_updates_k": k, "eps": eps,
                   "air_mass_in_carried_range": float(carried[opp, weak].sum()),
                   "opp_best_response_value": best, "local_loss_pot": best - v_star,
                   "local_loss_mbb_per_hand": 1000 * (best - v_star) * int(root.bet_total.sum()) / 100}
            out["rows"].append(row)
            print(row, flush=True)
    out["seconds"] = time.perf_counter() - t0
    open("results/audit/live/river_floor.json", "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
