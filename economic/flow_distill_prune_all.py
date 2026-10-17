import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import numpy as np
import diffrax as dfx
import matplotlib.pyplot as plt


import models


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


def discover(index, params, X, Y, Z, X_val, Y_val, Z_val):
    ## define active modes
    active_modes = N * [1.0] + 5 * [1.0]
    names = [
        "GPR at\n one day",
        "GPR at\n one week",
        "GPR at\n two weeks",
        "GPR at\n three weeks",
        "GPR at\n four weeks",
    ]

    target_name = "return of the oil price at step n: $\log(P_{n+1}) - \log(P_n)$"

    pruned_names = []
    ratios = []

    for i in range(N + 1):

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

        y_pred = K @ yb
        err_train = jnp.mean((y_pred - Y) ** 2)

        K_val = utils.kernel_fn(
            X_val,
            Z_val,
            X,
            Z,
            lx,
            lz,
            active_modes,
        )
        y_pred_val = K_val @ yb
        err_val = jnp.mean((y_pred_val - Y_val) ** 2)
        print("train error: ", err_train)
        print("validation error: ", err_val)

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
                rkhs_norm2.reshape([]) / signal2.reshape([]),
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

    # fig, ax = plt.subplots(1, 2, figsize=[25, 5])
    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax[0].plot(ratios, "k-o")
    ax[0].set_title("The noise-to-signal ratio")
    ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax[1].set_title("The increment of the noise-to-signal ratio")
    ax[1].set_xlabel("Pruned variables")
    plt.tight_layout()
    fig.savefig("./figs/ratios_phase_1986_2026.png", dpi=200)


if __name__ == "__main__":
    data = sio.loadmat("./data/data_oil_gpr.mat")["data"]
    N = data.shape[0]
    # the beginning of 1986 to 2026-04
    oil = data[:, 0:1]
    gpr = data[:, 1:2]
    r_oil = np.log(oil[1:]) - np.log(oil[:-1])
    log_gpr = np.log(gpr[:-1])

    y_data = r_oil[20:, :]
    x_data = np.concatenate(
        [
            r_oil[19:-1, :],
            r_oil[15:-5, :],
            r_oil[10:-10, :],
            r_oil[5:-15, :],
            r_oil[0:-20, :],
            log_gpr[20:, :],
            log_gpr[16:-4, :],
            log_gpr[11:-9, :],
            log_gpr[6:-14, :],
            log_gpr[1:-19, :],
        ],
        axis=-1,
    )


    ## normalize the data
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    y_mu = np.mean(y_data, axis=0)
    y_sd = np.std(y_data, axis=0)
    X_train = (x_data - x_mu) / x_sd
    Y_train = (y_data - y_mu) / y_sd

    np.random.seed(1234)
    key = np.random.randint(0, 1_000_000_000)
    key = jr.PRNGKey(key)

    ############ Training ############
    params, model = models.train_cfm_flax(
        key,
        X_train,
        Y_train,
        model_cfg=models.ModelCfg(
            hidden_dims=[64, 64],
        ),
        train_cfg=models.TrainCfg(
            lr=1e-4,
            batch_size=1_000,
            steps=5_000,
        ),
    )

    ############ Sampling ############
    x_samples = x_data
    M = x_samples.shape[0]
    N = 100
    x_samples = np.tile(x_samples, [N, 1])
    z_samples, z0_samples = sample(
        params,
        x=(x_samples - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu
    z_samples = z0_samples

    ############ Discovering ############
    import utils

    N = 5
    M = 10_000
    params = {
        "lx": 5.0,
        "lz": 5.0,
        "log_gamma": 1.0,
    }
    not_pruned_x = x_samples[:, 0:5]
    x_samples = np.concatenate([x_samples[:, 5:], not_pruned_x], axis=1)


    idx = np.random.choice(x_samples.shape[0], x_samples.shape[0], replace=False)
    x_train = x_samples[idx[:M], :]
    z_train = z_samples[idx[:M], :]
    y_train = y_samples[idx[:M], :]
    x_val = x_samples[idx[M : 2 * M]]
    z_val = z_samples[idx[M : 2 * M]]
    y_val = y_samples[idx[M : 2 * M]]

    X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    Z = z_train
    X_val = (x_val - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    Y_val = (y_val - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    Z_val = z_val

    discover(1, params, X, Y, Z, X_val, Y_val, Z_val)


    print("End main.")
