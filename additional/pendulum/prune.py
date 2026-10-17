import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt

import utils


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--M", type=int, default=5_000)
args = parser.parse_args()


############# Configurations #############
# Still 10 individual variables but grouped by time lag:
#   group 0: (theta_t,   u_t)    -> variable indices (0, 5)
#   group 1: (theta_{t-1}, u_{t-1}) -> variable indices (1, 6)
#   group 2: (theta_{t-2}, u_{t-2}) -> variable indices (2, 7)
#   group 3: (theta_{t-3}, u_{t-3}) -> variable indices (3, 8)
#   group 4: (theta_{t-4}, u_{t-4}) -> variable indices (4, 9)
#   group 5: (theta_{t-5}, u_{t-5}) -> variable indices (0, 5)
#   group 6: (theta_{t-6}, u_{t-6}) -> variable indices (1, 6)
#   group 7: (theta_{t-7}, u_{t-7}) -> variable indices (2, 7)
#   group 8: (theta_{t-8}, u_{t-8}) -> variable indices (3, 8)
#   group 9: (theta_{t-9}, u_{t-9}) -> variable indices (4, 9)
N_vars = 20  # total individual variables
N_groups = args.N  # number of groups to prune over
M = args.M

# Map group index -> list of variable indices
groups = {k: [k, k + 10] for k in range(N_groups)}
group_names = [
    r"$(\theta_t, u_t)$",
    r"$(\theta_{t-1}, u_{t-1})$",
    r"$(\theta_{t-2}, u_{t-2})$",
    r"$(\theta_{t-3}, u_{t-3})$",
    r"$(\theta_{t-4}, u_{t-4})$",
    r"$(\theta_{t-5}, u_{t-5})$",
    r"$(\theta_{t-6}, u_{t-6})$",
    r"$(\theta_{t-7}, u_{t-7})$",
    r"$(\theta_{t-8}, u_{t-8})$",
    r"$(\theta_{t-9}, u_{t-9})$",
]

params0 = {
    "lx": N_vars * [1.0],
    "lz": 10.0,
    "log_gamma": jnp.log(1.0),
}
print("Number of groups: ", N_groups)
print("Number of data: ", M)


############# Load data #############
data = sio.loadmat("./outputs/samples.mat")
z_data = data["z_samples"]
x_data = data[
    "x_samples"
]  # (N_samples, 30): [sin_t..sin_{t-9}, cos_t..cos_{t-9}, u_t..u_{t-9}]
y_data = data["y_samples"]

x_data = x_data.reshape([-1, x_data.shape[-1]])
z_data = z_data.reshape([-1, z_data.shape[-1]])
y_data = y_data.reshape([-1, y_data.shape[-1]])

# ── Convert (sin, cos) -> theta, keep u as-is ─────────────────────────────────
# x_data columns: [sin_t, sin_{t-1}, ..., sin_{t-9},
#                  cos_t, cos_{t-1}, ..., cos_{t-9},
#                  u_t,   u_{t-1},  ..., u_{t-9}]
sin_cols = x_data[:, 0:10]  # sin(theta_t) ... sin(theta_{t-9})
cos_cols = x_data[:, 10:20]  # cos(theta_t) ... cos(theta_{t-9})
u_cols = x_data[:, 20:30]  # u_t ... u_{t-9}

theta_cols = np.arctan2(sin_cols, cos_cols)  # (N_samples, 5)  in [-pi, pi]

# New input: [theta_t, theta_{t-1}, ..., theta_{t-9}, u_t, ..., u_{t-9}]
x_data_theta = np.concatenate([theta_cols, u_cols], axis=1)  # (N_samples, 10)

# is_periodic: True for theta variables, False for u variables
is_periodic = [True] * 10 + [False] * 10  # 10 entries

# ── Train/val split ───────────────────────────────────────────────────────────
idx = np.random.choice(x_data_theta.shape[0], x_data_theta.shape[0], replace=False)
x_train = x_data_theta[idx[:M]]
z_train = z_data[idx[:M]]
y_train = y_data[idx[:M]]
x_val = x_data_theta[idx[M : 2 * M]]
z_val = z_data[idx[M : 2 * M]]
y_val = y_data[idx[M : 2 * M]]

# Normalize: theta normalized over [-pi, pi], u normalized by train stats
x_mu = np.mean(x_train, axis=0)
x_sd = np.std(x_train, axis=0)
y_mu = np.mean(y_train, axis=0)
y_sd = np.std(y_train, axis=0)

_X = (x_train - x_mu) / x_sd
_Y = (y_train - y_mu) / y_sd
X_val = (x_val - x_mu) / x_sd
_Y_val = (y_val - y_mu) / y_sd
_Z = z_train
Z_val = z_val


def discover(index):

    X = _X
    Y = _Y
    Z = _Z

    # active_modes over all 10 individual variables
    active_modes = N_vars * [1.0]
    target_name = r"$\Delta\theta$"

    pruned_names = []
    ratios = []

    for i in range(N_groups + 1):

        data = (X, Z, Y)
        params = params0.copy()

        lx = params["lx"]
        lz = params["lz"]
        gamma = jnp.exp(params["log_gamma"])

        K = utils.kernel_fn(X, Z, X, Z, lx, lz, active_modes, is_periodic)
        L = jnp.linalg.cholesky(K + gamma * jnp.eye(K.shape[0]))
        yb = jnp.linalg.solve(L.T, jnp.linalg.solve(L, Y))

        signal2 = yb.T @ K @ yb
        noise2 = yb.T @ (gamma * np.eye(K.shape[0])) @ yb

        # ── Group activation: contribution of removing the entire group ───────
        group_activations = N_groups * [1e12]
        for g in range(N_groups):
            var_indices = groups[g]

            # Skip if all variables in this group are already pruned
            if all(active_modes[j] == 0 for j in var_indices):
                continue

            # Set all variables in group g to inactive
            _active_modes = active_modes.copy()
            for j in var_indices:
                _active_modes[j] = 0

            _K = K - utils.kernel_fn(X, Z, X, Z, lx, lz, _active_modes, is_periodic)
            rkhs_norm2 = yb.T @ _K @ yb
            print(f"  Group {group_names[g]} activation: {rkhs_norm2.reshape([]):.6f}")
            group_activations[g] = rkhs_norm2.reshape([])

        if i < N_groups:
            prune_g = np.argmin(group_activations)
            print(f"Pruning group: {group_names[prune_g]}")
            for j in groups[prune_g]:
                active_modes[j] = 0
            pruned_names += [group_names[prune_g]]

        ratios += [noise2 / (noise2 + signal2)]

    ratios = [v.reshape([]) for v in ratios]

    fig, ax = plt.subplots(1, 1, figsize=[12, 5])
    plt.suptitle("Finding ancestors of " + target_name)
    ax.plot(pruned_names, np.array(ratios[1:]) - np.array(ratios[:-1]), "k-o")
    ax.set_title("Increment of noise-to-signal ratio per pruned group")
    ax.set_xlabel("Pruned group")
    plt.tight_layout()
    fig.savefig("./figs/ratios_{}.png".format(str(index)), dpi=200)
    

discover(1)

print("End main.")
