# Engine tests and measurements added in the audit stage

Run from the repository root with

    export POKER_REGIME=1 JAX_PLATFORMS=cpu PYTHONPATH=. OPENBLAS_NUM_THREADS=1
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

| Script | What it checks | Needs | Time (1 CPU core) |
|---|---|---|---|
| `tests/test_slumbot_replay.py` | Slumbot protocol payloads -> engine replay -> candidate replies -> protocol; malformed responses refused; 52 card strings round-trip | nothing | 1 min |
| `tests/test_gadget.py` | the re-solving gadget in pcfr: mask off == no gadget (bitwise), ceilings far below -> plain solve, ceilings far above -> Follow 0, finite certificate | nothing (one river spot) | 1 min |
| `tests/test_warm_up_gadget.py` | the closed compiled set: a gadget solve and the ceiling computation after `subgame.warm_up` / `warm_gadget` compile nothing | nothing (random critics) | 1-2 min |
| `tests/test_live_gadget_hand.py` | a whole hand from the flop through `LivePlayer` with random critics: ceilings at the turn and river, Follow statistics, legal actions; `gadget=False` still runs | nothing (flop and turn matrices computed, ~15 s) | 2-3 min |
| `tests/gadget_river_resolve.py` | the measurement of section 2.3 of docs/engine-audit.md: range-only vs gadget re-solving (uniform or range deal) spliced into the exact river game, with ceiling noise and bias; `--trunk-frac 1 3` / `3 4` / `3 2` picks the trunk bet | nothing | 5 min per trunk for 3 spots at T 800 |

The user's own `checks/` package (10k lines) was not in the upload; these scripts do not replace it.
