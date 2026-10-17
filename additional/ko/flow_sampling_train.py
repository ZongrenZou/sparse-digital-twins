from dataclasses import dataclass
from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
import scipy.io as sio
import numpy as np
import diffrax as dfx

from ckpt_io import load_model


class VelocityMLP(nn.Module):
    """v_theta(t, z, x): concat [z, t, x] -> MLP -> R^d."""

    hidden_dims: Tuple[int, ...]
    dim_out: int

    @nn.compact
    def __call__(self, t, z, x):
        # t: () or (B,), z: (d,) or (B,d), x: (k,) or (B,k)
        # Broadcast t to match batch of z/x if needed
        if t.ndim == 0:
            t = t[None] if z.ndim == 1 else jnp.broadcast_to(t, (z.shape[0],))
        t = t[..., None]  # (B,1) or (1,1)

        # Ensure rank alignment
        if z.ndim == 1:
            z = z[None, :]  # (1,d)
        if x.ndim == 1:
            x = x[None, :]  # (1,k)

        h = jnp.concatenate([z, t, x], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)  # (B, d)
        return out if out.shape[0] > 1 else out[0]


@dataclass
class ModelCfg:
    hidden_dims: Tuple[int, ...] = (256, 256, 256)


@dataclass
class SolverCfg:
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 1_000_000


def sample(
    key,
    params,
    x,
    z0=None,
    M=None,
):
    # Build once
    term = dfx.ODETerm(lambda t, y, args: velocity(params, model, t, y, args))

    def solve_single(z0i, x):
        sol = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=z0i,
            args=x,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=1_000_000,
        )
        return sol.ys

    # If you want n samples for the SAME x:
    # z0 = jr.normal(key, [x.shape[0], 4])
    if z0 is None:
        z0 = np.random.normal(size=[x.shape[0], 1])
        z0 = jnp.array(z0)
    else:
        # z0 = jnp.tile(z0, [M, 1])
        z0 = jnp.tile(z0[None, :, :], [M, 1, 1])
        z0 = jnp.transpose(z0, [1, 0, 2])
        z0 = z0.reshape([-1, z0.shape[-1]])
    x_fixed = x  # (k,)
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x_fixed)
    return zT, z0


def velocity(params, model, t, z, x):
    """Single-trajectory: t:(), z:(d,), x:(k,) -> (d,)"""
    return model.apply({"params": params}, t, z, x)


if __name__ == "__main__":

    params, model, extras = load_model("./checkpoints/cfm_3", VelocityMLP)
    x_mu = extras["x_mu"]
    x_sd = extras["x_sd"]
    y_mu = extras["y_mu"]
    y_sd = extras["y_sd"]

    data = sio.loadmat("./data/data.mat")

    x_data = data["x_data_train"].reshape([3, -1]).T

    key = jr.PRNGKey(72819)
    z_samples, z0_samples = sample(
        key,
        params,
        x=(x_data - x_mu) / x_sd,
        z0=None,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    sio.savemat(
        "./outputs/samples_3.mat",
        {
            "y_samples": y_samples,
            "z_samples": z0_samples,
            "x_samples": x_data,
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
