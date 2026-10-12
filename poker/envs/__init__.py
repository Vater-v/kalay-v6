"""NLHE engine: one path for 2..12 seats and any stack depth (rules, side pots, history, payouts).

Callers use poker.envs.table ({legal_mask, step, rewards}, TableSampler) or the rule functions in
betting/actions/transition directly; legality of an applied action is the caller's precondition.
"""
