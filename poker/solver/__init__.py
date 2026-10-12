"""Vector public-tree solver, critics and the training loop.

Full float32 matmuls for every process that imports the solver (loop, learning curves,
arena, player): Ampere and newer GPUs default to TF32 for float32 products
(measured on a 4090: 5e-5 relative error against 8e-7 at "highest"; the 1070 has no TF32), and the
solver's values, its certificates, the critics and E must mean the same on every card.
tools/cloud/bootstrap.sh probes it on every rented GPU (matmul 1326 x 1326 against float64).
"""

import jax

jax.config.update("jax_default_matmul_precision", "highest")
