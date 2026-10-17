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
        out = model.apply({"params": params}, t, z, x)
        return out.reshape(-1) 

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

    z0 = np.random.normal(size=[x.shape[0], 3])
    z0 = jnp.array(z0)
    x_fixed = x
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x_fixed)
    return zT, z0


def discover(index, X, Y, Z, params, N, run_idx):
    Y = Y[:, index - 1 : index]

    ## define active modes
    active_modes = N * [1.0]
    names = ["x{}".format(str(i + 1)) for i in range(N)]
    target_name = "dx{}/dt".format(str(index))

    pruned_names = []
    ratios = []

    for i in range(N + 1):

        ## then we perform the regression and compute activations
        lx = params["lx"]
        lz = params["lz"]
        gamma = jnp.exp(params["log_gamma"])

        K = utils.kernel_fn(
            X,
            Z,
            X,
            Z,
            lx,
            lz,
            active_modes,
        )
        yb = jnp.linalg.solve(K + gamma * np.eye(K.shape[0]), Y)

        signal2 = yb.T @ K @ yb
        noise2 = yb.T @ (gamma * np.eye(K.shape[0])) @ yb

        activations = N * [1e12]
        for j in range(N):
            if active_modes[j] == 1:
                _active_modes = active_modes.copy()
                _active_modes[j] = 0
            else:
                continue
            _K = 1 * K - utils.kernel_fn(
                X,
                Z,
                X,
                Z,
                lx,
                lz,
                _active_modes,
            )
            rkhs_norm2 = yb.T @ _K @ yb
            print(
                "The {}th variable's activation: ".format(str(j)),
                rkhs_norm2.reshape([]),
            )
            activations[j] = rkhs_norm2.reshape([])

        if i < N:
            ## finally, we prune the variable with the lowest energy
            idx = np.argmin(activations)
            print("The {}th mode has the lowest energy.".format(str(idx)))
            active_modes[idx] = 0
            pruned_names += [names[idx]]

        ratios += [noise2 / (noise2 + signal2)]

    ratios = [v.reshape([]) for v in ratios]

    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax[0].plot(ratios, "k-o")
    ax[0].set_title("The noise-to-signal ratio")
    ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax[1].set_title("The increment of the noise-to-signal ratio")
    ax[1].set_xlabel("Pruned variables")
    fig.savefig(f"./figs/ratios_{index}_{run_idx}.png", dpi=200)
    plt.close()

    np.savetxt(f"./outputs/ratios_{index}_{run_idx}.txt", np.array(ratios))


if __name__ == "__main__":
    data = sio.loadmat("./data/data.mat")
    x_data = data["X"]
    y_data = data["dX"]

    ## normalize the data
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    y_mu = np.mean(y_data, axis=0)
    y_sd = np.std(y_data, axis=0)
    X_train = (x_data - x_mu) / x_sd
    Y_train = (y_data - y_mu) / y_sd

    for run_idx in range(10):
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
                epochs=1_000,
            ),
        )

        ############ Sampling ############
        M = x_data.shape[0]
        N = 100
        x = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])

        z_samples, z0_samples = sample(
            params,
            x=(x - x_mu) / x_sd,
            model=model,
        )
        y_samples = z_samples[:, 0, :] * y_sd + y_mu

        ############ Pruning ############
        params = {
            "lx": 1.0,
            "lz": 1.0,
            "log_gamma": 0.0,
        }
        N = 3
        M = 5_000

        idx = np.random.choice(x.shape[0], x.shape[0], replace=False)
        x_train = x[idx[:M]]
        z_train = z0_samples[idx[:M]]
        y_train = y_samples[idx[:M]]

        X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
        Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
        Z = (z_train - np.mean(z_train, axis=0)) / np.std(z_train, axis=0)

        ## do it for one
        discover(1, X, Y, Z, params, N, run_idx)
        discover(2, X, Y, Z, params, N, run_idx)
        discover(3, X, Y, Z, params, N, run_idx)

    print("End main.")
