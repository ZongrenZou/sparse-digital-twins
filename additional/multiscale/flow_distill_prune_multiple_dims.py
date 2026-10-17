import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import diffrax as dfx
import matplotlib.pyplot as plt


import models
import utils


def sample(
    params,
    x,
    model,
):
    # define velocity function
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


def discover(index, X, Y, Z, params, m):

    ## define active modes
    active_modes = jnp.ones(m)
    names = []
    for i in range(m):
        names += ["$x_{" + str(i + 1) + "}$"]
    target_name = "$\Delta x_{" + str(index+1) + "}$"

    pruned_names = []
    ratios = []
    import time

    t0 = time.time()
    Xj = jnp.asarray(X)
    Zj = jnp.asarray(Z)
    Yj = jnp.asarray(Y)

    for i in range(m):

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
            index,
        )

        if i < m - 1:
            ## finally, we prune the variable with the lowest energy
            idx = int(jnp.argmin(activations))
            print("The {}th mode has the lowest energy.".format(str(idx)))
            active_modes = active_modes.at[idx].set(0.0)
            pruned_names += [names[idx]]

        ratios += [ratio]

    t1 = time.time()
    print("The time cost is: ", t1 - t0)

    ratios = [v.reshape([]) for v in ratios]

    # # we plot the results
    fig, ax = plt.subplots(1, 2, figsize=[30, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax[0].plot(ratios, "k-o")
    ax[0].set_title("The noise-to-signal ratio")
    ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax[1].set_title("The increment of the noise-to-signal ratio")
    ax[1].set_xlabel("Pruned variables")

    plt.tight_layout()
    fig.savefig(f"./figs/ratios_m_{m}_{index+1}.png", dpi=200)
    plt.close()


if __name__ == "__main__":

    for m in [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        data = sio.loadmat(f"./data/data_{m}.mat")
        x_data = data["Xs"]
        y_data = data["dXs"][:, 0:1]

        x_mu = np.mean(x_data, axis=0)
        x_sd = np.std(x_data, axis=0)
        y_mu = np.mean(y_data, axis=0)
        y_sd = np.std(y_data, axis=0)
        X_train = (x_data - x_mu) / x_sd
        Y_train = (y_data - y_mu) / y_sd

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
                steps=2_000,
            ),
        )

        ############ Sampling ############
        z_samples, z0_samples = sample(
            params,
            x=(x_data - x_mu) / x_sd,
            model=model,
        )
        y_samples = z_samples[:, 0, :] * y_sd + y_mu
        x_samples = x_data
        z_samples = z0_samples

        ############ Pruning ############
        params = {
            "lx": 5.0,
            "lz": 1.0,
            "log_gamma": jnp.log(1.0),
        }
        M = 2000

        idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
        x_train = x_samples[idx[:M]]
        z_train = z_samples[idx[:M]]
        y_train = y_samples[idx[:M]]

        X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
        Z = z_train

        discover(0, X, Y, Z, params, m)


    print("End main.")
