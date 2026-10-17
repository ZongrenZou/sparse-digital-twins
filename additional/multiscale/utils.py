import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial


@jax.jit
def kernel_fn(x, z, xp, zp, lx, lz, active_modes):
    active_modes = jnp.asarray(active_modes)
    zd = z[:, None, :] - zp[None, :, :]
    term_z = jnp.prod(1 + jnp.exp(-(zd**2) / (2 * lz**2)), axis=-1)
    xd = x[:, None, :] - xp[None, :, :]
    term_x = 1 + jnp.exp(-(xd**2) / (2 * lx**2)) * active_modes
    return term_z, term_x


@partial(jax.jit, static_argnames=("index",))
def prune_step(X, Z, Y, lx, lz, log_gamma, active_modes, index):
    gamma = jnp.exp(log_gamma)
    Kz, Kx = kernel_fn(X, Z, X, Z, lx, lz, active_modes)
    K = Kz * jnp.prod(Kx, axis=-1)
    n = K.shape[0]
    L = jnp.linalg.cholesky(K + gamma * jnp.eye(n))
    yb = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y))
    signal2 = yb.T @ K @ yb
    noise2 = yb.T @ (gamma * jnp.eye(n)) @ yb

    # tr(yb.T @ D_j @ yb) = sum_{ij} D_j[i,j] G[i,j] with G = yb @ yb.T (one matmul, vectorized over j)
    Delta = K[..., None] * (1.0 - 1.0 / Kx)
    G = yb @ yb.T
    quads = jnp.sum(Delta * G[..., None], axis=(0, 1))
    j_idx = jnp.arange(active_modes.shape[0])
    valid = (j_idx != index) & (active_modes == 1.0)
    activations = jnp.where(valid, quads, 1e12)
    ratio = noise2 / (noise2 + signal2)
    return ratio, activations
