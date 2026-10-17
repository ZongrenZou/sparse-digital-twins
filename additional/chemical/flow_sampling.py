from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import diffrax as dfx

from ckpt_io import load_model
import models


@dataclass
class SolverCfg:
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 1_000_000


def sample(
    key,
    params,
    x,
    model,
    z0=None,
    M=None,
):
    # Build once
    term = dfx.ODETerm(lambda t, y, args: velocity(params, model, t, y, args))

    @jax.jit
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
        z0 = np.random.normal(size=[x.shape[0], 4])
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

    params, model, extras = load_model("./checkpoints/cfm_ssa_v4", models.VelocityMLP)
    x_mu = extras["x_mu"]
    x_sd = extras["x_sd"]
    y_mu = extras["y_mu"]
    y_sd = extras["y_sd"]

    data = sio.loadmat("./data/data_ssa_v4.mat")
    x_data = data["Xs"].reshape([-1, 4])

    M = 5000
    idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)[:M]
    x_samples = x_data[idx, :]

    N = 10
    z0 = np.random.normal(size=[N, 4])
    x = np.tile(x_samples, [N, 1]).reshape([N * M, x_samples.shape[1]])
    key = jr.PRNGKey(6371)
    z_samples, z0_samples = sample(
        key,
        params,
        x=(x - x_mu) / x_sd,
        model=model,
        z0=z0,
        M=M,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    sio.savemat(
        "./outputs/samples_ssa_v4.mat",
        {
            "y_samples": y_samples,
            "z_samples": z0_samples,
            "x_samples": x,
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
