import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import scipy.io as sio
import matplotlib.pyplot as plt


import utils as utils

############# Configurations #############
params = {
    "lx": 5.0,
    "lz": 1.0,
    "log_gamma": jnp.log(0.1),
}  # for y_1 and y_3
# params = {
#     "lx": 5.0, "lz": 0.1, "log_gamma": jnp.log(1.0),
# } # for y_2

N = 3
M = 5_000
index = 1


############# Load data #############
data = sio.loadmat(f"./data_for_pruning/samples_{index}.mat")
z_data = data["z_samples"]
x_data = data["x_samples"]
y_data = data["y_samples"]


x_data = x_data.reshape([-1, x_data.shape[-1]])
z_data = z_data.reshape([-1, z_data.shape[-1]])
y_data = y_data.reshape([-1, y_data.shape[-1]])

idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)
x_train = x_data[idx[:M]]
z_train = z_data[idx[:M]]
y_train = y_data[idx[:M]]
x_val = x_data[idx[M : 2 * M]]
z_val = z_data[idx[M : 2 * M]]
y_val = y_data[idx[M : 2 * M]]

X = x_train
Y = y_train
Z = z_train

_X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
_Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
X_val = (x_val - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
_Y_val = (y_val - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
_Z = z_train
Z_val = z_val


def discover(index):

    X = _X
    Y = _Y
    Z = _Z

    ## define active modes
    active_modes = N * [1.0]
    names = ["$x_1$", "$x_2$", "$x_3$"]
    target_name = "$y_{}$".format(str(index))

    pruned_names = []
    ratios = []

    for i in range(N + 1):
        ## then we perform the regression and compute activations
        lx = params["lx"]
        # lx = np.sqrt(N+1-i) * params["lx"]
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
    # ax[0].set_ylim([-0.05, 1.05])
    # ax[1].set_ylim([-0.05, 1.05])
    plt.tight_layout()
    fig.savefig(f"./figs/ratios_{index}.png", dpi=200)

    # np.savetxt(f"./outputs/ratios_{index}.txt", np.array(ratios))


## do it for one
discover(index)

print("End main.")
