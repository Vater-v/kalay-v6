"""Heads-up matches against Slumbot (slumbot.com, NLHE 50/100, 200 BB, no rake).

protocol.py, client.py, metrics.py and parallel.py are the user's earlier project kalay-v2
(evaluation/slumbot{,_client,_metrics,_parallel}.py, sha256 617bb516…, 8cbaea4e…, 92087958…,
a1a1137d…), copied 2026-10-10 with only the import paths changed (algo.artifacts -> util.py): the
exact public history parser, the HTTP client that never retries a mutating request, per-hand
evidence and resume. live_bot.py replays that exact history through our engine and live solver.
run.py starts a match with explicitly selected current critics and their preflop table.
"""
