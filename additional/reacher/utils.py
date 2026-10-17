import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp


@jax.jit
def loss_function(_params, data, active_modes):
    lx = _params["lx"]
    lz = _params["lz"]
    gamma = jnp.exp(_params["log_gamma"])

    ## step 1: perform the regression with all of the data
    x, z, y = data

    K = kernel_fn(
        x,
        z,
        x,
        z,
        lx,
        lz,
        active_modes,
    )
    y_regressed = K @ jnp.linalg.solve(K + gamma * np.eye(K.shape[0]), y)
    error = y - y_regressed

    ## step 2: compute the Hat matrix
    H = K @ jnp.linalg.inv(K + gamma * np.eye(K.shape[0]))
    H = jnp.linalg.diagonal(H)

    ## step 3: compute the cross validation loss
    error = error.reshape([-1, y.shape[1]])
    H = H.reshape([-1, 1])
    H = jnp.tile(H, [1, error.shape[1]])
    cv = jnp.mean((error / (1 - H)) ** 2, axis=0)
    cv = jnp.sum(cv)
    return cv


def kernel_fn(x, z, xp, zp, lx, lz, active_modes):
    if not isinstance(lx, list):
        lx = x.shape[1] * [lx]
    term_z = 1
    for i in range(zp.shape[1]):
        term_z = term_z * (
            1 + jnp.exp(-((z[:, i : i + 1] - zp[:, i : i + 1].T) ** 2) / 2 / lz**2)
        )
    term_x = 1
    for i in range(xp.shape[1]):
        term_x = term_x * (
            1
            + jnp.exp(-((x[:, i : i + 1] - xp[:, i : i + 1].T) ** 2) / 2 / lx[i] ** 2)
            * active_modes[i]
        )
    return term_z * term_x
