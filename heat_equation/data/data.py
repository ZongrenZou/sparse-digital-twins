import numpy as np
import scipy.io as sio

np.random.seed(42)


def build_system_matrices(n=10, kappa=0.01, dt=0.5):
    h = 1.0 / (n - 1)
    r = kappa * dt / h**2

    assert r <= 0.5, f"Stability violated: r={r:.4f} > 0.5. Reduce dt or increase n."

    A_tilde = np.zeros((n, n))
    for i in range(1, n - 1):
        A_tilde[i, i - 1] = 1.0
        A_tilde[i, i] = -2.0
        A_tilde[i, i + 1] = 1.0
    A_tilde[0, 0] = -2.0
    A_tilde[0, 1] = 2.0
    A_tilde[n - 1, n - 2] = 2.0
    A_tilde[n - 1, n - 1] = -2.0

    A = np.eye(n) + r * A_tilde  # forward Euler

    B_tilde = np.zeros((n, 2))
    B_tilde[0, 0] = 1.0
    B_tilde[n - 1, 1] = 1.0
    B = (2.0 * dt / h) * B_tilde

    print("=" * 55)
    print("System matrices")
    print("=" * 55)
    print(f"  n={n}, kappa={kappa}, dt={dt}, h={h:.4f}, r={r:.4f}")
    print(f"  A: {A.shape},  sparsity {np.mean(A==0)*100:.0f}% zeros")
    print(f"  B: {B.shape},  sparsity {np.mean(B==0)*100:.0f}% zeros")
    print(f"  Spectral radius of A: {max(abs(np.linalg.eigvals(A))):.6f}")

    return A, B, h, r


def boundary_input(x, u):
    """Effective Neumann fluxes: (x(0) u_L, x(1) u_R)."""
    return np.array([u[0], u[1]])


def simulate_trajectory(A, B, X0, T, sigma, u_bound):
    n = A.shape[0]
    X_traj = np.zeros((T + 1, n))
    U_traj = np.zeros((T, 2))

    X_traj[0] = X0

    for t in range(T):
        U = np.random.uniform(-u_bound, u_bound, size=2)  # random control
        omega = sigma * np.random.randn(n)  # multiplicative noise
        X_traj[t + 1] = (
            A @ X_traj[t] + B @ boundary_input(X_traj[t], U) + X_traj[t] * omega
        )
        U_traj[t] = U

    return X_traj, U_traj


def generate_dataset(A, B, n_traj, T, sigma, x0_scale=1.0, u_bound=1.0):
    n = A.shape[0]
    N = n_traj * T

    X_all = np.zeros((N, n))
    U_all = np.zeros((N, 2))
    X_next_all = np.zeros((N, n))

    for i in range(n_traj):
        X0 = x0_scale * np.random.randn(n)  # random initial condition

        X_traj, U_traj = simulate_trajectory(A, B, X0, T, sigma=sigma, u_bound=u_bound)

        idx = slice(i * T, (i + 1) * T)
        X_all[idx] = X_traj[:T]  # X_t
        U_all[idx] = U_traj  # U_t
        X_next_all[idx] = X_traj[1 : T + 1]  # X_{t+1}

    return {"X": X_all, "U": U_all, "X_next": X_next_all}


if __name__ == "__main__":

    # ── parameters ───────────────────────────────────────────────────────────
    n = 100  # spatial grid points  (state dimension)
    kappa = 0.01  # thermal diffusivity
    dt = 0.005 # time step
    sigma = 0.01  # process noise std
    u_bound = 5  # U_t ~ Uniform[-u_bound, u_bound]^2
    x0_scale = 1.0  # X0  ~ N(0, x0_scale^2 * I)

    # ── build system ─────────────────────────────────────────────────────────
    A, B, h, r = build_system_matrices(n=n, kappa=kappa, dt=dt)
    # ── generate training data ───────────────────────────────────────────────
    # Strategy: random X0, random U_t at every step, simulate forward
    # No LQR involved.
    print("\n" + "=" * 55)
    print("Generating training data")
    print("=" * 55)
    train_data = generate_dataset(
        A,
        B,
        n_traj=200,  # 200 independent trajectories
        T=200,  # 200 steps each -> 40,000 pairs total
        sigma=sigma,
        x0_scale=x0_scale,
        u_bound=u_bound,
    )

    sio.savemat("./data.mat", train_data)

    print("End main.")
