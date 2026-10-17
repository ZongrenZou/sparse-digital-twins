"""
Data generation for the stochastic Lorenz-63 system
with fluctuation-dissipation (FD) multiplicative noise.

System:
    dx1 = sigma*(x2 - x1) dt + alpha*x1 dW1
    dx2 = (x1*(rho - x3) - x2) dt + alpha*x2 dW2
    dx3 = (x1*x2 - beta*x3) dt + alpha*x3 dW3

where W1, W2, W3 are independent Wiener processes.

Discretization: Euler-Maruyama
    x_i,n+1 = x_i,n + f_i(x_n) * dt + alpha * x_i,n * sqrt(dt) * xi_i
where xi_i ~ N(0,1) are independent.

Data format:
    Input:  X_n  = (x1_n, x2_n, x3_n)         shape (N_data, 3)
    Output: dX_n = (dx1_n, dx2_n, dx3_n)       shape (N_data, 3)
where dX_n = X_{n+1} - X_n.

References:
    - Lorenz (1963), J. Atmos. Sci., 20(2):130-141.
    - Geurts, Holm, Luesink (2020), J. Stat. Phys., 179(5-6):1343-1365.
    - Kloeden & Platen (1992), Numerical Solution of SDEs, Springer.
"""

import numpy as np
from scipy.io import savemat

# Lorenz-63 parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

# Noise amplitude for FD multiplicative noise: alpha * x_i * dW_i
ALPHA = 0.5

# Time discretization
DT = 0.01  # time step
N_STEPS = 1000  # number of steps per trajectory (excluding initial condition)
BURN_IN = 200  # steps to discard at the start of each trajectory (transient)

# Number of trajectories
N_TRAJ = 50

# Initial condition sampling
# x0 ~ Uniform[-15, 15]^3 (covers the Lorenz attractor)
IC_LOW = -15.0
IC_HIGH = 15.0

# Random seed for reproducibility
# SEED = 42 # for training
SEED = 98492  # for testing

# Output file
# SAVE_PATH = "./data_train.mat"
SAVE_PATH = "./data_test.mat"


def drift(x):
    """
    Lorenz-63 drift vector f(x) = (f1, f2, f3).

    Args:
        x: array of shape (3,) or (N, 3)

    Returns:
        f: same shape as x
    """
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    f1 = SIGMA * (x2 - x1)
    f2 = x1 * (RHO - x3) - x2
    f3 = x1 * x2 - BETA * x3
    return np.stack([f1, f2, f3], axis=-1)


def em_step(x, dt, alpha, rng):
    """
    One Euler-Maruyama step for the stochastic Lorenz-63 FD system.

    dx_i = f_i(x) dt + alpha * x_i * sqrt(dt) * xi_i,  xi_i ~ N(0,1)

    Args:
        x:     current state, shape (3,)
        dt:    time step
        alpha: noise amplitude
        rng:   numpy random generator

    Returns:
        x_new: next state, shape (3,)
        dx:    increment x_new - x, shape (3,)
    """
    f = drift(x)
    xi = rng.standard_normal(3)
    dx = f * dt + alpha * x * np.sqrt(dt) * xi
    x_new = x + dx
    return x_new, dx


def simulate_trajectory(x0, n_steps, burn_in, dt, alpha, rng):
    """
    Simulate one trajectory of the stochastic Lorenz-63 FD system.

    Args:
        x0:      initial condition, shape (3,)
        n_steps: number of steps to record (after burn-in)
        burn_in: number of initial steps to discard
        dt:      time step
        alpha:   noise amplitude
        rng:     numpy random generator

    Returns:
        X:  states at each recorded step,     shape (n_steps, 3)
        dX: increments at each recorded step, shape (n_steps, 3)
    """
    x = x0.copy()

    # burn-in: evolve without recording
    for _ in range(burn_in):
        x, _ = em_step(x, dt, alpha, rng)

    # record
    X = np.zeros((n_steps, 3))
    dX = np.zeros((n_steps, 3))

    for i in range(n_steps):
        X[i] = x
        x, dX[i] = em_step(x, dt, alpha, rng)

    return X, dX


def generate_data():
    rng = np.random.default_rng(SEED)

    # Sample random initial conditions
    X0 = rng.uniform(IC_LOW, IC_HIGH, size=(N_TRAJ, 3))

    X_list = []
    dX_list = []

    print(f"Generating {N_TRAJ} trajectories x {N_STEPS} steps each...")
    for i, x0 in enumerate(X0):
        X, dX = simulate_trajectory(x0, N_STEPS, BURN_IN, DT, ALPHA, rng)
        X_list.append(X)
        dX_list.append(dX)
        if (i + 1) % 10 == 0:
            print(f"  Trajectory {i+1}/{N_TRAJ} done.")

    # Stack all trajectories into flat datasets
    X_all = np.concatenate(X_list, axis=0)  # (N_traj * N_steps, 3)
    dX_all = np.concatenate(dX_list, axis=0)  # (N_traj * N_steps, 3)

    N_data = X_all.shape[0]
    print(f"\nTotal data points: {N_data}")
    print(f"X  shape: {X_all.shape}   (input: current state)")
    print(f"dX shape: {dX_all.shape}  (output: increment)")

    # Save as .mat file (compatible with MATLAB and scipy.io.loadmat)
    savemat(
        SAVE_PATH,
        {
            "X": X_all,
            "dX": dX_all,
            "sigma": np.array([[SIGMA]]),
            "rho": np.array([[RHO]]),
            "beta": np.array([[BETA]]),
            "alpha": np.array([[ALPHA]]),
            "dt": np.array([[DT]]),
        },
    )
    print(f"\nData saved to '{SAVE_PATH}'")

    return X_all, dX_all


if __name__ == "__main__":
    X, dX = generate_data()

    print("End main.")
