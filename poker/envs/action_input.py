"""Strict scalar integer action input; never silently narrow a host value."""

import numbers

import jax
import jax.numpy as jnp
import numpy as np


def action_input(value, *, label="action"):
    """Retain array integer dtype until range checks; bool/float/vector fail statically.

    Host integers must fit int32 before JAX conversion (including NumPy int64
    when JAX x64 is disabled). Device uint32/int64 values are checked before
    narrowing by the worker or public transition. Existing device int64/uint64
    requires current x64=True: promotion with x64=False can truncate before a
    range comparison. Check only static dtype/config before conversion; no
    device value is read here.
    """
    message = f"{label} must be a scalar integer (not bool); batch through vmap"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, numbers.Integral):
        if not -(1 << 31) <= int(value) < (1 << 31):
            raise ValueError(f"host {label} must fit signed int32 before conversion")
        return jnp.asarray(value, dtype=jnp.int32)
    if isinstance(value, np.ndarray):
        if value.shape != () or value.dtype.kind not in "iu":
            raise ValueError(message)
        if not -(1 << 31) <= int(value) < (1 << 31):
            raise ValueError(f"host {label} must fit signed int32 before conversion")
        return jnp.asarray(value, dtype=jnp.int32)
    dtype = getattr(value, "dtype", None)
    if (
        dtype is not None
        and dtype.kind in "iu"
        and dtype.itemsize > 4
        and not jax.config.x64_enabled
    ):
        raise ValueError(
            f"{label} dtype {dtype} requires jax_enable_x64=True; "
            "use int32 inputs when x64 is disabled"
        )
    action = jnp.asarray(value)
    if action.shape != () or action.dtype.kind not in "iu":
        raise ValueError(message)
    return action


def fits_int32(action):
    """Value guard before narrowing a wider or unsigned device action."""
    if action.dtype.kind == "u" and action.dtype.itemsize >= 4:
        return action <= (1 << 31) - 1
    if action.dtype.kind == "i" and action.dtype.itemsize > 4:
        return (action >= -(1 << 31)) & (action <= (1 << 31) - 1)
    return jnp.bool_(True)
