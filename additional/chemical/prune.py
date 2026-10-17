import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt

import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=4)
parser.add_argument("--M", type=int, default=5_000)
args = parser.parse_args()


############# Configurations #############
# seed = np.random.randint(0, 2**32 - 1)
# print("The random seed is: ", seed)
# seed = 2333
# np.random.seed(seed)
niter = 0
lr = 0.1
params0 = {
    "lx": 1.0,
    "lz": 1.0,
    "log_gamma": 0.0,
}
N = args.N
# index = args.index
M = args.M
print("Number of data: ", M)


############# Load data #############
data = sio.loadmat("./outputs/samples_ssa_v4.mat")
z_data = data["z_samples"]
y_data = data["y_samples"]
x_data = data["x_samples"]


idx = np.random.choice(x_data.shape[0], x_data.shape[0], replace=False)[:M]
x_train = x_data[idx, :]
z_train = z_data[idx, :]
y_train = y_data[idx, :]

_X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
_Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
_Z = z_train


def discover(index):
    X = _X
    Y = _Y[:, index - 1 : index]
    Z = _Z

    ## define active modes
    active_modes = N * [1.0]
    names = ["x{}".format(str(i + 1)) for i in range(N)]
    target_name = "dx{}".format(str(index))

    pruned_names = []
    ratios = []

    for i in range(N + 1):

        data = (X, Z, Y)

        ## then we perform the regression and compute activations
        lx = params0["lx"]
        lz = params0["lz"]
        gamma = jnp.exp(params0["log_gamma"])
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

    # we plot the results
    # fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    # plt.suptitle("Finding ancestors of " + target_name)
    # ax[0].plot(losses, "k-o")
    # ax[0].set_title("The LOO-CV GP distillation loss")
    # ax[1].plot(pruned_names, np.array(losses[1:]) - np.array(losses[:-1]), "k-o")
    # ax[1].set_title("The increment of the LOO-CV GP distillation loss")
    # ax[1].set_xlabel("Pruned variables")
    # plt.tight_layout()
    # fig.savefig("./figs/losses_{}.png".format(str(index)), dpi=200)

    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax[0].plot(ratios, "k-o")
    ax[0].set_title("The noise-to-signal ratio")
    ax[1].plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax[1].set_title("The increment of the noise-to-signal ratio")
    ax[1].set_xlabel("Pruned variables")
    plt.tight_layout()
    fig.savefig("./figs/ratios_{}.png".format(str(index)), dpi=200)

    np.savetxt("./outputs/ratios_{}.txt".format(str(index)), ratios)


## do it for one
discover(1)

print("End main.")
