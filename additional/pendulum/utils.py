import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def kernel_fn(x, z, xp, zp, lx, lz, active_modes, is_periodic=None):
    """
    Product kernel over (x, z).

    For each input dimension i:
        - if is_periodic[i] is True: uses periodic kernel
              k_i = 1 + exp(-2*sin^2((x_i - x_i')/2) / lx[i]^2) * active_modes[i]
          appropriate for angular variables theta in [-pi, pi]
        - otherwise: uses standard Gaussian kernel
              k_i = 1 + exp(-(x_i - x_i')^2 / (2*lx[i]^2)) * active_modes[i]

    is_periodic: list of bools, length = x.shape[1]
                 if None, all variables treated as non-periodic (original behavior)
    """
    if not isinstance(lx, list):
        lx = x.shape[1] * [lx]
    if is_periodic is None:
        is_periodic = x.shape[1] * [False]

    term_z = 1
    for i in range(zp.shape[1]):
        term_z = term_z * (
            1 + jnp.exp(-((z[:, i : i + 1] - zp[:, i : i + 1].T) ** 2) / 2 / lz**2)
        )

    term_x = 1
    for i in range(xp.shape[1]):
        if is_periodic[i]:
            # Periodic kernel: geodesic distance on circle
            diff = x[:, i : i + 1] - xp[:, i : i + 1].T
            term_x = term_x * (
                1 + jnp.exp(-2 * jnp.sin(diff / 2) ** 2 / lx[i] ** 2) * active_modes[i]
            )
        else:
            # Standard Gaussian kernel
            term_x = term_x * (
                1
                + jnp.exp(
                    -((x[:, i : i + 1] - xp[:, i : i + 1].T) ** 2) / 2 / lx[i] ** 2
                )
                * active_modes[i]
            )
    return term_z * term_x
