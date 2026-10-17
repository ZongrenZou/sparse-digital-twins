import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import scipy.io as sio


@dataclass
class L96TwoScaleConfig:
    """
    Two-scale Lorenz-96 configuration.

    Equations (k = 0..K-1, j = 0..J-1; all indices periodic):
      dX_k/dt = X_{k-1}(X_{k+1} - X_{k-2}) - X_k + F - (h*c/b) * sum_j Y_{j,k}
      dY_{j,k}/dt = c*b*Y_{j+1,k}(Y_{j-1,k} - Y_{j+2,k}) - c*Y_{j,k} + (h*c/b) * X_k

    Common defaults follow Lorenz (1996) / Wilks (2005)-style setups.
    """

    K: int = 36  # number of slow variables
    J: int = 10  # fast variables per slow variable
    F: float = 10.0  # forcing on slow variables
    h: float = 1.0  # coupling strength
    c: float = 10.0  # time-scale separation (fast = c times faster)
    b: float = 10.0  # amplitude scaling of fast variables
    dt: float = 0.005  # integrator time step (smaller if c is larger)
    steps: int = 20000  # total integration steps
    store_every: int = 10  # store every N steps (controls output stride)
    spinup: int = 2000  # steps to integrate before storing (discard transients)


def _rhs(
    X: np.ndarray, Y: np.ndarray, cfg: L96TwoScaleConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Right-hand side for the two-scale L96 system.
    X shape: (K,)
    Y shape: (J, K)  (fast index first for convenient rolling)
    """
    K, J = cfg.K, cfg.J

    # Slow tendency dX/dt
    Xm1 = np.roll(X, 1)  # X_{k-1}
    Xp1 = np.roll(X, -1)  # X_{k+1}
    Xp2 = np.roll(X, -2)  # X_{k+2} -> we actually need X_{k-2}; roll +2 is -2?
    Xm2 = np.roll(X, 2)  # X_{k-2}
    dX = Xm1 * (Xp1 - Xm2) - X + cfg.F

    # Coupling term from fast to slow: -(h*c/b) * sum_j Y_{j,k}
    coupling_scale = (cfg.h * cfg.c) / cfg.b
    dX -= coupling_scale * Y.sum(axis=0)

    # Fast tendency dY/dt
    Yjp1 = np.roll(Y, -1, axis=0)  # Y_{j+1,k}
    Yjm1 = np.roll(Y, 1, axis=0)  # Y_{j-1,k}
    Yjp2 = np.roll(Y, -2, axis=0)  # Y_{j+2,k}

    dY = cfg.c * cfg.b * (Yjp1 * (Yjm1 - Yjp2)) - cfg.c * Y
    # Coupling from slow to fast: +(h*c/b) * X_k
    dY += coupling_scale * X[np.newaxis, :]

    return dX, dY


def _rk4_step(
    X: np.ndarray, Y: np.ndarray, cfg: L96TwoScaleConfig
) -> Tuple[np.ndarray, np.ndarray]:
    dt = cfg.dt

    k1x, k1y = _rhs(X, Y, cfg)
    k2x, k2y = _rhs(X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y, cfg)
    k3x, k3y = _rhs(X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y, cfg)
    k4x, k4y = _rhs(X + dt * k3x, Y + dt * k3y, cfg)

    Xn = X + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    Yn = Y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    return Xn, Yn


def simulate(
    cfg: L96TwoScaleConfig,
    X0: Optional[np.ndarray] = None,
    Y0: Optional[np.ndarray] = None,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the two-scale L96 simulation.

    Returns:
        t  : (T,) times corresponding to stored states (in model time units)
        Xs : (T, K) slow variable snapshots
        Ys : (T, J, K) fast variable snapshots

    Notes:
        - Integrates 'spinup' steps without storing, then stores every 'store_every' steps.
        - Defaults create small perturbations around X ~ F and Y ~ 0.
    """
    rng = np.random.default_rng(seed)

    # Initialize
    if X0 is None:
        X = cfg.F + 0.01 * rng.standard_normal(cfg.K)
    else:
        X = np.array(X0, dtype=float, copy=True)
        assert X.shape == (cfg.K,)

    if Y0 is None:
        Y = 0.01 * rng.standard_normal((cfg.J, cfg.K))
    else:
        Y = np.array(Y0, dtype=float, copy=True)
        assert Y.shape == (cfg.J, cfg.K)

    # Spinup (discard)
    for _ in range(cfg.spinup):
        X, Y = _rk4_step(X, Y, cfg)

    # Storage sizing
    n_kept_steps = max(cfg.steps, 0)
    stride = max(cfg.store_every, 1)
    T = (n_kept_steps // stride) + 1  # include initial post-spinup state

    Xs = np.empty((T, cfg.K), dtype=float)
    Ys = np.empty((T, cfg.J, cfg.K), dtype=float)
    ts = np.empty(T, dtype=float)

    # Store initial post-spinup state
    Xs[0] = X
    Ys[0] = Y
    ts[0] = 0.0

    # Main loop
    write_idx = 1
    time_accum = 0.0
    for n in range(1, n_kept_steps + 1):
        X, Y = _rk4_step(X, Y, cfg)
        time_accum += cfg.dt
        if (n % stride) == 0:
            Xs[write_idx] = X
            Ys[write_idx] = Y
            ts[write_idx] = time_accum
            write_idx += 1

    return ts, Xs, Ys

# m is the number of slow variables
# we take m=10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200
m = 10
cfg = L96TwoScaleConfig(
    K=m,
    J=10,
    F=10.0,
    h=0.1,
    c=10.0,
    b=10.0,
    dt=0.001,
    steps=100000,
    store_every=10,
    spinup=2000,
)


ts = []
Xs = []
Ys = []
for i in range(1):
    _t, _Xs, _Ys = simulate(cfg, seed=42)
    ts += [_t]
    Xs += [_Xs]
    Ys += [_Ys]


Xss = []
dXss = []
for i in range(1):
    dXss += [Xs[i][1:] - Xs[i][:-1, :]]
    Xss += [Xs[i][:-1, :]]


Xs = np.stack(Xss, axis=0)
dXs = np.stack(dXss, axis=0)


sio.savemat(
    f"./data_{m}.mat",
    {
        "dXs": dXs.reshape([-1, m]),
        "Xs": Xs.reshape([-1, m]),
    },
)