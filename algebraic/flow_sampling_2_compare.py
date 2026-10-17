import numpy as np
import jax
import jax.numpy as jnp
import scipy.io as sio
import numpy as np
import diffrax as dfx
import matplotlib.pyplot as plt


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

    params, model, extras = load_model(
        "./checkpoints/cfm_2_compare", models.VelocityMLP
    )
    x_mu = extras["x_mu"]
    x_sd = extras["x_sd"]
    y_mu = extras["y_mu"]
    y_sd = extras["y_sd"]

    data = sio.loadmat("./data/data_test.mat")
    x_data = data["x_data"]

    M = x_data.shape[0]
    N = 100
    x_samples = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])
    z_samples, z0_samples = sample(
        params,
        x=(x_samples - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    tmp = x_samples.reshape([100, 500, -1])
    x3 = tmp[0, :, 2]
    idx = np.argsort(x3)

    plt.figure(dpi=200)
    plt.plot(x_samples[:, 2:3], y_samples, "cx", alpha=1, label="Generated samples", markersize=10)
    plt.plot(
        np.hstack([np.zeros([1]), x3[idx]]),
        np.hstack([np.zeros([1]), np.sqrt(x3[idx])]),
        "k-",
        linewidth=2,
    )
    plt.plot(
        x3[idx], -np.sqrt(x3[idx]), "k-", linewidth=2, label="Exact: $\pm\sqrt{x_3}$"
    )
    # plt.title("$y_2|x_1, x_2, x_3$ without pruning", fontsize=20)
    # plt.xlabel("$x_3$", fontsize=20)
    plt.legend(fontsize=15, frameon=False, loc=2)
    plt.ylim([-1.1, 1.1])
    plt.xlim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig("./figs/no_prune.png")

    params, model, extras = load_model(
        "./checkpoints/cfm_2_pruned_compare", models.VelocityMLP
    )
    x_mu = extras["x_mu"]
    x_sd = extras["x_sd"]
    y_mu = extras["y_mu"]
    y_sd = extras["y_sd"]

    x_data = x_data[:, 2:3]
    N = 100
    x_samples = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])
    z_samples, z0_samples = sample(
        params,
        x=(x_samples - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    tmp = x_samples.reshape([100, 500, -1])
    x3 = tmp[0, :, 0]
    idx = np.argsort(x3)

    plt.figure(dpi=200)
    plt.plot(x_samples[:, 0:1], y_samples, "c*", alpha=1, label="Generated samples", markersize=10)
    plt.plot(
        np.hstack([np.zeros([1]), x3[idx]]),
        np.hstack([np.zeros([1]), np.sqrt(x3[idx])]),
        "k-",
        linewidth=2,
    )
    plt.plot(
        x3[idx], -np.sqrt(x3[idx]), "k-", linewidth=2, label="Exact: $\pm\sqrt{x_3}$"
    )
    # plt.title("$y_2|x_1, x_2, x_3$ without pruning", fontsize=20)
    # plt.xlabel("$x_3$", fontsize=20)
    plt.legend(fontsize=15, frameon=False, loc=2)
    plt.ylim([-1.1, 1.1])
    plt.xlim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig("./figs/prune.png")

    print("End main.")
