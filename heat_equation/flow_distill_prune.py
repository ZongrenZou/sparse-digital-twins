import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import numpy as np
import diffrax as dfx
import matplotlib.pyplot as plt


from ckpt_io import save_model
import models
import utils


def sample(
    params,
    x,
    model,
):  
    # build velocity function
    def velocity(params, model, t, z, x):
        return model.apply({"params": params}, t, z, x)
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
    x_fixed = x
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x_fixed)
    return zT, z0


def discover(index, params, X, Y, Z):

    ## define active modes
    active_modes = jnp.ones(N)
    names = []
    for i in range(N - 2):
        names += ["$x_{" + str(i + 1) + "}$"]
    names += ["$u_1$", "$u_2$"]
    target_name = "$\Delta x_{}$".format(str(index))

    pruned_names = []
    ratios = []

    Xj = jnp.asarray(X)
    Zj = jnp.asarray(Z)
    Yj = jnp.asarray(Y)

    for i in range(N + 1):
        ## then we perform the regression and compute activations
        lx = params["lx"]
        lz = params["lz"]
        log_gamma = params["log_gamma"]

        ratio, activations = utils.prune_step(
            Xj,
            Zj,
            Yj,
            lx,
            lz,
            log_gamma,
            active_modes,
        )

        if i < N:
            ## finally, we prune the variable with the lowest energy
            idx = int(jnp.argmin(activations))
            print("The {}th mode has the lowest energy.".format(str(idx)))
            active_modes = active_modes.at[idx].set(0.0)
            pruned_names += [names[idx]]

        ratios += [ratio]

    ratios = [v.reshape([]) for v in ratios]
    np.save(f"./figs/ratios_{index}.npy", np.array(ratios))
    # # we plot the results
    plt.figure(figsize=[20, 5])
    plt.title("Finding ancestors of " + target_name)
    plt.plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    plt.ylabel("The increment of the noise-to-signal ratio")
    plt.xlabel("Pruned variables")
    plt.tight_layout()
    plt.savefig("./figs/ratios_{}.png".format(str(index)), dpi=300)
    plt.close()


if __name__ == "__main__":
    data = sio.loadmat("./data/data.mat")
    x_data = data["X"]
    u_data = data["U"]
    y_data = data["X_next"] - data["X"]

    x_data = np.concatenate([x_data, u_data], axis=-1)
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    X_train = (x_data - x_mu) / x_sd


    seed = 9999  # for reproducibility
    np.random.seed(seed)

    # prune only x1 and x10
    for index in [1, 100]:
        y_mu = np.mean(y_data[:, index-1: index], axis=0)
        y_sd = np.std(y_data[:, index-1: index], axis=0)
        Y_train = (y_data[:, index-1: index] - y_mu) / y_sd

        key = np.random.randint(0, 1_000_000_000)
        key = jr.PRNGKey(key)

        ############ Training ############
        params, model = models.train_cfm_flax(
            key,
            X_train,
            Y_train,
            model_cfg=models.ModelCfg(
                hidden_dims=[64, 64, 64, 64],
            ),
            train_cfg=models.TrainCfg(
                lr=1e-4,
                batch_size=1_000,
                steps=1_000,
            ),
        )

        save_model(
            ckpt_dir=f"./checkpoints/cfm_{index}",
            params=params,
            model=model,
            model_cfg=models.ModelCfg(hidden_dims=[64, 64, 64, 64]),
            extras={
                "x_mu": x_mu,
                "x_sd": x_sd,
                "y_mu": y_mu,
                "y_sd": y_sd,
            },
        )

        ############ Sampling ############
        M = x_data.shape[0]
        N = 1 # repeat data N times
        x = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])
        z_samples, z0_samples = sample(
            params,
            x=(x - x_mu) / x_sd,
            model=model,
        )
        y_samples = z_samples[:, 0, :] * y_sd + y_mu
        x_samples = x
        z_samples = z0_samples

        ############ Pruning ############
        params = {
            "lx": 1.0,
            "lz": 1.0,
            "log_gamma": jnp.log(1.0),
        }
        N = 102
        M = 5_000

        idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
        x_train = x_samples[idx[:M]]
        z_train = z_samples[idx[:M]]
        y_train = y_samples[idx[:M]]

        X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
        Z = (z_train - np.mean(z_train, axis=0)) / np.std(z_train, axis=0)

        discover(index, params, X, Y, Z)

    print("End main.")
