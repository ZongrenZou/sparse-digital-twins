import numpy as np
import jax
import jax.numpy as jnp
import scipy.io as sio
import numpy as np
import diffrax as dfx

from ckpt_io import load_model
import models


def sample(
    params,
    x,
    model,
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

    z0 = np.random.normal(size=[x.shape[0], 1])
    z0 = jnp.array(z0)
    x_fixed = x  # (k,)
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x_fixed)
    return zT, z0


def velocity(params, model, t, z, x):
    """Single-trajectory: t:(), z:(d,), x:(k,) -> (d,)"""
    return model.apply({"params": params}, t, z, x)


if __name__ == "__main__":

    params, model, extras = load_model("./checkpoints/cfm_1", models.VelocityMLP)
    # params, model, extras = load_model("./checkpoints/cfm_2", models.VelocityMLP)
    # params, model, extras = load_model("./checkpoints/cfm_3", models.VelocityMLP)
    x_mu = extras["x_mu"]
    x_sd = extras["x_sd"]
    y_mu = extras["y_mu"]
    y_sd = extras["y_sd"]

    data = sio.loadmat("./data/data_train.mat")
    x_data = data["x_data"]
    M = x_data.shape[0]
    N = 100
    x = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])

    z_samples, z0_samples = sample(
        params,
        x=(x - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    sio.savemat(
        "./data_for_pruning/samples_1.mat",
        # "./data_for_pruning/samples_2.mat",
        # "./data_for_pruning/samples_3.mat",
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
