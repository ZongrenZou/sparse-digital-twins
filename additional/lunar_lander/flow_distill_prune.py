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


def discover(name, params, X, Y, Z):

    ## define active modes
    active_modes = N * [1.0]
    names = [
        "x",
        "y",
        "vx",
        "vy",
        "angle",
        "va",
        "leg 1",
        "leg 2",
        "a_1",
        "a_2",
    ]
    target_name = "$\Delta " + name + "$"

    pruned_names = []
    ratios = []

    for i in range(N + 1):

        ## then we perform the regression and compute activations
        lx = params["lx"]
        lz = params["lz"]
        gamma = jnp.exp(params["log_gamma"])
        # print("The log_gamma is: ", params["log_gamma"])

        K = utils.kernel_fn(
            X,
            Z,
            X,
            Z,
            lx,
            lz,
            active_modes,
        )
        L = jnp.linalg.cholesky(K + gamma * jnp.eye(K.shape[0]))
        yb = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y))

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
    plt.tight_layout()
    fig.savefig("./figs/ratios_{}.png".format(name), dpi=200)

    # np.savetxt("./outputs/ratios_{}.txt".format(str(index)), np.array(ratios))


if __name__ == "__main__":
    data = sio.loadmat("./data/lunar_lander_impulse_data_0.mat")
    actions = data["actions"]
    states = data["states"]
    delta_vx = data["delta_vx"].reshape([-1, 1])
    delta_vy = data["delta_vy"].reshape([-1, 1])
    delta_va = data["delta_angvel"].reshape([-1, 1])

    x_data = np.concatenate(
        [states, actions],
        axis=-1,
    )
    y_data = np.concatenate(
        [delta_vx, delta_vy, delta_va],
        axis=-1,
    )

    seed = 81763263  # for reproducibility
    np.random.seed(seed)

    ## normalize the data
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
            steps=10_000,
        ),
    )

    save_model(
        ckpt_dir="./checkpoints/cfm",
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
    z_samples, z0_samples = sample(
        params,
        x=(x_data - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu
    x_samples = x_data
    z_samples = z0_samples

    ############ Pruning ############
    import utils
    params = {
        "lx": 10 * [5.0],
        "lz": 5.0,
        "log_gamma": jnp.log(1.0),
    }
    N = 10
    M = 10_000

    idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
    x_train = x_samples[idx[:M]]
    z_train = z_samples[idx[:M]]
    y_train = y_samples[idx[:M]]

    X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
    Z = z_train


    discover("v_x", params, X, Y[:, 0:1], Z)
    discover("v_y", params, X, Y[:, 1:2], Z)
    discover("v_a", params, X, Y[:, 2:3], Z)



    print("End main.")
