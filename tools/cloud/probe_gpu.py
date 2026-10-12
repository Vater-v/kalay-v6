"""GPU probe of a rented box (tools/cloud/bootstrap.sh runs it once per GPU).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m tools.cloud.probe_gpu

JAX sees the GPU (name, compute capability), float32 matmul throughput, and the TF32 probe: a 1326 x 1326 float32 matmul of uniform [0, 1) inputs against
the float64 product of the same float32 inputs, error relative to the largest entry.

Threshold 1e-5. Full float32 accumulation depends on the order inside the cuBLAS kernel: measured
5.2e-7 (RTX 5090 Laptop, "highest") up to 2.5e-6 (GTX 1070, any precision; Pascal has no TF32), i.e.
5e-7 ... 3e-6. TF32 inputs (the default precision on Ampere and newer) give 4.7e-5 (RTX 5090
Laptop) and 5.4e-5 (RTX 4090). 1e-5 is 4x above the worst
full-fp32 number and 5x below TF32.

Two checks, exit 1 if either fails:
1. at the pipeline's precision (importing solver sets jax_default_matmul_precision = "highest")
   the error is < 1e-5: the box computes the same solves, certificates and critics as the 1070;
2. on cards with compute capability >= 8.0 (TF32 hardware) the error at the card's default
   precision is >= 1e-5: otherwise the probe would detect nothing there.
The ratio default / pipeline is printed: ~1 on Pascal, >> 1 on Ampere and newer.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import poker.solver  # noqa: F401  (the pipeline's matmul precision)

LIMIT = 1e-5
TF32_FROM = 8.0                    # compute capability of the first TF32 hardware (Ampere)


def matmul_error(precision=None, n=1326):
    rng = np.random.default_rng(0)
    a = rng.random((n, n)).astype(np.float32)
    b = rng.random((n, n)).astype(np.float32)
    want = a.astype(np.float64) @ b.astype(np.float64)
    if precision is None:
        got = jax.jit(lambda x, y: x @ y)(a, b)
    else:
        with jax.default_matmul_precision(precision):
            got = jax.jit(lambda x, y: x @ y)(a, b)
    return float(np.abs(np.asarray(got, np.float64) - want).max() / np.abs(want).max())


def main():
    dev = jax.devices()[0]
    if dev.platform != "gpu":
        print(f"PROBE FAIL: JAX sees no GPU: {jax.devices()}")
        sys.exit(1)
    cc = getattr(dev, "compute_capability", None)
    if cc is None:
        print(f"PROBE FAIL: {dev} reports no compute capability: the TF32 check cannot be decided")
        sys.exit(1)
    x = jnp.ones((8192, 8192), jnp.float32)
    f = jax.jit(lambda a: a @ a)
    f(x).block_until_ready()
    t = time.perf_counter()
    f(x).block_until_ready()
    tflops = 2 * 8192 ** 3 / (time.perf_counter() - t) / 1e12
    err, raw = matmul_error(), matmul_error("default")
    print(f"{dev} {dev.device_kind} (compute capability {cc}): fp32 matmul {tflops:.1f} TFLOP/s; "
          f"1326^2 matmul error {err:.1e} at the pipeline's precision ({jax.config.jax_default_matmul_precision}), "
          f"{raw:.1e} at the card's default, ratio {raw / err:.1f}", flush=True)
    failed = False
    if not err < LIMIT:
        print(f"PROBE FAIL: float32 matmul error {err:.1e} >= {LIMIT:g} at the pipeline's precision: do not run here")
        failed = True
    if float(cc) >= TF32_FROM and not raw >= LIMIT:
        print(f"PROBE FAIL: compute capability {cc} (TF32 hardware) but the default precision error {raw:.1e} < "
              f"{LIMIT:g}: the probe would not detect TF32 here")
        failed = True
    if failed:
        sys.exit(1)
    print("PROBE OK")


if __name__ == "__main__":
    main()
