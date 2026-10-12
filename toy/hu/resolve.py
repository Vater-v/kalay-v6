"""Continual re-solving agent + extraction of its complete strategy.

The agent never stores a strategy for the whole game.  At each of its
decisions it solves only the CURRENT STREET (from the most recent public node
where it holds opponent information) with:
  * our range        pi_a(R)   (exact: product of our own past action probs)
  * opponent info    either
       naive : an opponent range q(R) carried forward from the last re-solve
       safe  : opponent counterfactual values w(R, h') from the last re-solve,
               enforced with the DeepStack gadget (opponent may "terminate" at
               R with value w(h') for every hand h')
  * leaf values      V at the next-street boundary: 'checkdown', 'learned',
                     or 'oracle' = the remaining game is kept below the leaf
                     and solved jointly (exact; in the limit identical to calling
                     an exact solver at every leaf on every iteration)

Re-solve roots on a street: the street root S for the agent's first decision
on that street (the opponent may already have acted at S), otherwise the
agent's decision node itself (opponent CFVs/range there come from the
previous re-solve).  After a re-solve we keep, for the next re-solves,
  - opponent CFVs and range at the agent's next decision nodes on this street
  - opponent CFVs and range at every next-street root (via the leaves / V).

EXTRACTION: because a re-solve only needs information produced by earlier
re-solves on the same path, all re-solves at the same (street, k-th agent
decision on that street) are independent and are batched into ONE forest
solve.  Visiting all phases in order yields the agent's complete strategy,
which is then scored with an exact best response on the full game.
"""
from __future__ import annotations

import time
import numpy as np

from tree import PublicTree, DEC, LEAF, CHANCE
from cfr import Solver, values, forward
from valuefn import LeafEvaluator


def _agent_counts(T: PublicTree, a: int):
    """count[n] = number of agent (player a) actions taken earlier on the same
    street along the path from the street root (or forest root) to n."""
    cnt = np.zeros(T.M, dtype=np.int64)
    par, kind, pl = T.parent, T.kind, T.player
    for n in range(T.n_roots, T.M):
        p = par[n]
        if kind[p] == CHANCE:
            cnt[n] = 0
        else:
            cnt[n] = cnt[p] + (1 if (kind[p] == DEC and pl[p] == a) else 0)
    return cnt


def extract_agent(full: PublicTree, a: int, leaf: str, V=None, safe=True, iters=300,
                  log=None):
    """Run the continual re-solving agent for seat `a` at every public node.
    leaf in {'oracle','checkdown','learned'}; V = normalised value fn
    (ByStreetV) for the depth-limited variants.  `leaf` may also be a dict
    {street: (leaf, V)} to use a different leaf source per street.
    Returns (eff_a, stats)."""
    N, o = full.N, 1 - a
    cnt_full = _agent_counts(full, a)
    eff_a = full.eff_template.copy()
    Ea = full.E[a]
    eff_a[Ea] = np.repeat(1.0 / full.E_cnt[a], full.E_cnt[a])[:, None]
    w_store, q_store = {}, {}
    st_full = full.states
    is_street_root = np.array([s.n_act == 0 for s in st_full]) & (full.kind == DEC)
    stats = []
    for street in range(full.cfg.n_streets):
        for k in range(3):
            if k == 0:
                roots = np.nonzero(is_street_root & (full.street == street))[0]
            else:
                roots = np.nonzero((full.kind == DEC) & (full.player == a) &
                                   (full.street == street) & (cnt_full == k))[0]
            if len(roots) == 0:
                continue
            t0 = time.time()
            pi_a = forward(full, eff_a, full.bmask[:1])[roots]
            lkind, Vs = leaf.get(street, ("oracle", None)) if isinstance(leaf, dict) else (leaf, V)
            depth_limited = lkind != "oracle"
            F = PublicTree(full.cfg, full.B, [st_full[r] for r in roots],
                           list(full.cscale[roots]), depth_limited=depth_limited)
            game_root = street == 0 and k == 0
            if game_root:  # the true start of the game: opponent range is known exactly
                q0 = full.bmask[roots]
                gadget = None
            elif safe:
                q0 = full.bmask[roots]
                gadget = dict(player=o, w=np.stack([w_store[r] for r in roots]))
            else:
                q0 = np.stack([q_store[r] for r in roots])
                gadget = None
            reach = [None, None]
            reach[a], reach[o] = pi_a, q0
            lf = LeafEvaluator(F, Vs) if (depth_limited and len(F.L)) else None
            S = Solver(F, reach, leaf_fn=lf, gadget=gadget)
            S.iterate(iters)
            e = [S.avg_eff(0), S.avg_eff(1)]
            gf = S.avg_follow() if gadget is not None else None
            v0, v1, r0, r1 = values(S, e[0], e[1], g_follow=gf)
            vo, ro = (v0, r0) if o == 0 else (v1, r1)
            # ---- write the agent's strategy for the decisions this phase owns
            cntF = _agent_counts(F, a)
            own = np.nonzero((F.kind == DEC) & (F.player == a) &
                             (F.street == street) & (cntF == 0))[0]
            if k > 0:
                own = own[own < F.n_roots]
            keys = F.states
            for n in own:
                fn = full.key_index[keys[n].key]
                cs, cc = F.child_start[n], F.n_child[n]
                fs, fc = full.child_start[fn], full.n_child[fn]
                assert cc == fc
                eff_a[fs: fs + fc] = e[a][cs: cs + cc]
            # ---- store opponent info for the next re-solves on this street
            nxt = np.nonzero((F.kind == DEC) & (F.player == a) &
                             (F.street == street) & (cntF == 1))[0]
            for n in nxt:
                fn = full.key_index[keys[n].key]
                w_store[fn] = vo[n].copy(); q_store[fn] = ro[n].copy()
            # ---- ... and at the next-street roots
            if depth_limited and lf is not None:
                c0, c1, pr0, pr1 = lf.cfv_pairs(r0[F.L], r1[F.L])
                co, pro = (c0, pr0) if o == 0 else (c1, pr1)
                for j, key in enumerate(lf.next_keys):
                    fn = full.key_index.get(key)
                    if fn is not None:
                        w_store[fn] = co[j]; q_store[fn] = pro[j]
            elif not depth_limited:
                nr = np.nonzero((F.kind == DEC) & (F.street == street + 1) &
                                np.array([s.n_act == 0 for s in F.states]))[0]
                for n in nr:
                    fn = full.key_index[keys[n].key]
                    w_store[fn] = vo[n].copy(); q_store[fn] = ro[n].copy()
            dt = time.time() - t0
            stats.append(dict(street=street, k=k, n_resolves=len(roots), nodes=F.M,
                              seconds=dt, per_resolve=dt / len(roots)))
            if log:
                log(f"  seat {a} street {street} phase {k}: {len(roots)} re-solves, "
                    f"{F.M} nodes, {dt:.1f}s")
    return eff_a, stats
