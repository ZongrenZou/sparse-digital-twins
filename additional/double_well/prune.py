import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt

import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--M", type=int, default=5000)
args = parser.parse_args()


############# Configurations #############
niter = 0
lr = 0.1
# params0 = {
#     "lx": 0.5, "lz": 10.0, "log_gamma": jnp.log(1.0),
# }
# params0 = {
#     "lx": 2.0, "lz": 10.0, "log_gamma": jnp.log(1.0),
# }
params0 = {
    "lx": 1.0,
    "lz": 10.0,
    "log_gamma": jnp.log(1.0),
}
N = args.N
M = args.M
print("Number of data: ", M)


############# Load data #############
data = sio.loadmat("./outputs/samples.mat")
z_data = data["z_samples"]
x_data = data["x_samples"]
y_data = data["y_samples"]


x_data = x_data.reshape([-1, x_data.shape[-1]])
z_data = z_data.reshape([-1, 1])
y_data = y_data.reshape([-1, 1])

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

X = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
Y = (y_train - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
# Z = (z_train - np.mean(z_train, axis=0)) / np.std(z_train, axis=0)
X_val = (x_val - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
Y_val = (y_val - np.mean(y_train, axis=0)) / np.std(y_train, axis=0)
# Z_val = (z_val - np.mean(z_train, axis=0)) / np.std(z_train, axis=0)
Z = z_train
Z_val = z_val


def discover(index):

    ## define active modes
    active_modes = N * [1.0]
    names = ["x{}".format(str(i + 1)) for i in range(N)]
    target_name = "dx{}/dt".format(str(index))

    pruned_names = []
    losses = []
    losses_val = []
    losses_train = []
    ratios = []

    for i in range(N + 1):

        data = (X, Z, Y)
        params, _ = utils.learn_hyperparameters(
            data=data,
            active_modes=active_modes,
            params=params0,
            niter=niter,
            lr=lr,
        )
        loss = utils.loss_function(params, data, active_modes)
        # print("The loss is: ", loss)

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
        # yb = jnp.linalg.solve(K + gamma*np.eye(K.shape[0]), Y)
        K_val = utils.kernel_fn(
            X_val,
            Z_val,
            X,
            Z,
            lx,
            lz,
            active_modes,
        )
        Y_pred = K_val @ yb
        losses_val += [jnp.mean((Y_val - Y_pred) ** 2)]
        losses_train += [jnp.mean((Y - K @ yb) ** 2)]
        print("Training loss vs Validation loss:")
        print(losses_train[-1], losses_val[-1])

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

        losses += [loss.copy()]
        ratios += [noise2 / (noise2 + signal2)]

    ratios = [v.reshape([]) for v in ratios]

    # we plot the results
    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax[0].plot(losses, "k-o")
    ax[0].set_title("The LOO-CV GP distillation loss")
    ax[1].plot(pruned_names, np.array(losses[1:]) - np.array(losses[:-1]), "k-o")
    ax[1].set_title("The increment of the LOO-CV GP distillation loss")
    ax[1].set_xlabel("Pruned variables")
    plt.tight_layout()
    fig.savefig("./figs/losses_{}.png".format(str(index)), dpi=200)

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
    fig.savefig("./figs/ratios_{}.png".format(str(index)), dpi=200)

    fig, ax = plt.subplots(1, 2, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax[0].plot(losses_val, "k-o")
    ax[0].set_title("The VAL GP distillation loss")
    ax[1].plot(
        pruned_names, np.array(losses_val[1:]) - np.array(losses_val[:-1]), "k-o"
    )
    ax[1].set_title("The increment of the VAL GP distillation loss")
    ax[1].set_xlabel("Pruned variables")
    plt.tight_layout()
    fig.savefig("./figs/vals_{}.png".format(str(index)), dpi=200)

    np.savetxt("./outputs/ratios_{}.txt".format(str(index)), np.array(ratios))


## Do it for all

# for i in range(N):
#     print("Discovering for dx{}/dt".format(str(i)), flush=True)
#     discover(i)
#     print("")

## do it for one
discover(1)

print("End main.")
