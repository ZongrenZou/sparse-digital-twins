import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def kernel_fn(x, z, xp, zp, lx, lz, active_modes):
    term_z = 1
    for i in range(zp.shape[1]):
        term_z = term_z * (
            1 + jnp.exp(-((z[:, i : i + 1] - zp[:, i : i + 1].T) ** 2) / 2 / lz**2)
        )
    term_x = 1
    for i in range(xp.shape[1]):
        term_x = term_x * (
            1
            + jnp.exp(-((x[:, i : i + 1] - xp[:, i : i + 1].T) ** 2) / 2 / lx**2)
            * active_modes[i]
        )
    return term_z * term_x
