import json
import time
from dataclasses import dataclass
from pathlib import Path
from flax import linen as nn
from flax import serialization
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

CONTROL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentCfg:
    n: int = 100
    kappa: float = 0.01
    dt: float = 0.005
    sigma: float = 0.01
    T: int = 10_000
    H: int = 20
    S: int = 64
    m: int = 2
    u_bound: float = 5
    num_trials: int = 10
    seed: int = 23333
    x_ref_value: float = 0.0
    x0_noise_std: float = 0.1
    plot_path: str = "./figs/surrogate_mpc.png"
    all_traj_plot_path: str = "./figs/surrogate_mpc_all_traj.png"
    mpc_mean_cum_total_cost_path: str = "./figs/surrogate_mpc_mean_cum_total_cost.png"
    end_profile_plot_path: str = "./figs/surrogate_mpc_end_profile.png"
    results_mat_path: str = "./figs/surrogate_mpc_results.mat"


CFG = ExperimentCfg()

UNPRUNED_X0_STUDENT = "./checkpoints/cfm_1_student"
UNPRUNED_XN_STUDENT = "./checkpoints/cfm_100_student"
PRUNED_X0_STUDENT = "./checkpoints/cfm_1_pruned_student"
PRUNED_XN_STUDENT = "./checkpoints/cfm_100_pruned_student"


class StudentMLP(nn.Module):
    hidden_dims: tuple[int, ...]
    dim_out: int

    @nn.compact
    def __call__(self, x, z):
        if x.ndim == 1:
            x = x[None, :]
        if z.ndim == 1:
            z = z[None, :]
        h = jnp.concatenate([x, z], axis=-1)
        for width in self.hidden_dims:
            h = nn.silu(nn.Dense(width)(h))
        out = nn.Dense(self.dim_out)(h)
        return out if out.shape[0] > 1 else out[0]


def make_student_predictor(ckpt_dir: str):
    ckpt_path = (CONTROL_DIR / ckpt_dir).resolve()
    student_cfg = json.loads((ckpt_path / "student_cfg.json").read_text())
    params = serialization.msgpack_restore((ckpt_path / "params.msgpack").read_bytes())
    extras = dict(np.load(ckpt_path / "extras.npz", allow_pickle=False))

    model = StudentMLP(
        hidden_dims=tuple(student_cfg["hidden_dims"]),
        dim_out=int(student_cfg["out_dim"]),
    )
    y_mu = jnp.asarray(extras["y_mu"])
    y_sd = jnp.asarray(extras["y_sd"])

    def predict_single(x_cond, z0):
        pred = model.apply({"params": params}, x_cond, z0)
        return pred.reshape(y_mu.shape) * y_sd + y_mu

    return jax.jit(predict_single), jax.jit(jax.vmap(predict_single, in_axes=(0, 0)))


predict_unpruned_x0_single, predict_unpruned_x0_batch = make_student_predictor(
    UNPRUNED_X0_STUDENT
)
predict_unpruned_xn_single, predict_unpruned_xn_batch = make_student_predictor(
    UNPRUNED_XN_STUDENT
)
predict_pruned_x0_single, predict_pruned_x0_batch = make_student_predictor(
    PRUNED_X0_STUDENT
)
predict_pruned_xn_single, predict_pruned_xn_batch = make_student_predictor(
    PRUNED_XN_STUDENT
)


def build_system_matrices(n=10, kappa=0.01, dt=0.5):
    h = 1.0 / (n - 1)
    r = kappa * dt / h**2
    assert r <= 0.5, f"Stability violated: r={r:.4f} > 0.5"

    a_tilde = np.zeros((n, n))
    for i in range(1, n - 1):
        a_tilde[i, i - 1] = 1.0
        a_tilde[i, i] = -2.0
        a_tilde[i, i + 1] = 1.0
    a_tilde[0, 0] = -2.0
    a_tilde[0, 1] = 2.0
    a_tilde[n - 1, n - 2] = 2.0
    a_tilde[n - 1, n - 1] = -2.0

    a = np.eye(n) + r * a_tilde
    b_tilde = np.zeros((n, 2))
    b_tilde[0, 0] = 1.0
    b_tilde[n - 1, 1] = 1.0
    b = (2.0 * dt / h) * b_tilde
    return jnp.array(a), jnp.array(b)


def equilibrium_feedforward(a, b, x_ref):
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    x_np = np.asarray(x_ref)
    u_eff, *_ = np.linalg.lstsq(b_np, x_np - a_np @ x_np, rcond=None)
    u_ff = np.zeros(2, dtype=float)
    if abs(x_np[0]) > 1e-12:
        u_ff[0] = u_eff[0]
    if abs(x_np[-1]) > 1e-12:
        u_ff[1] = u_eff[1]
    return jnp.asarray(u_ff)


def boundary_input(x, u):
    """Effective Neumann fluxes: (x(0) u_L, x(1) u_R)."""
    return jnp.array([u[0], u[1]])


@jax.jit
def true_step(a, b, sigma, x, u, key):
    w = jr.normal(key, shape=x.shape)
    return a @ x + b @ boundary_input(x, u) + sigma * x * w


def make_known_mpc(a, b, H, S, q, sigma, n, u_bound, x_ref, m=2, maxiter=300):
    r = 0.01 * jnp.eye(m)

    def rollout_cost(u_flat, x0, w_scenarios):
        u_seq = u_flat.reshape(H, m)

        def step_batch(x_batch, inputs):
            u_k, w_k = inputs
            u_k = jnp.clip(u_k, -u_bound, u_bound)
            err = x_batch - x_ref
            cost_k = jnp.sum((err @ q) * err, axis=1) + u_k @ r @ u_k
            u_eff_0 = u_k[0:1]
            u_eff_1 = u_k[1:2]
            x_next = (
                x_batch @ a.T
                + u_eff_0[:, None] * b[:, 0]
                + u_eff_1[:, None] * b[:, 1]
                + sigma * x_batch * w_k
            )
            return x_next, cost_k

        x0_batch = jnp.broadcast_to(x0, (S, n))
        x_final, costs = jax.lax.scan(
            step_batch, x0_batch, (u_seq, w_scenarios.swapaxes(0, 1))
        )
        err_f = x_final - x_ref
        terminal = jnp.sum((err_f @ q) * err_f, axis=1)
        return jnp.mean(jnp.sum(costs, axis=0) + terminal)

    @jax.jit
    def solve(x0, key, u_init):
        key, subkey = jr.split(key)
        w_scenarios = jr.normal(subkey, shape=(S, H, n))

        def obj(u):
            return jax.value_and_grad(rollout_cost)(u, x0, w_scenarios)

        solver = jaxopt.LBFGS(
            fun=obj,
            value_and_grad=True,
            maxiter=maxiter,
            jit=True,
            unroll=False,
        )
        result = solver.run(u_init.flatten())
        u_opt = jnp.clip(result.params.reshape(H, m), -u_bound, u_bound)
        return u_opt, u_opt[0], key

    return solve


def make_unpruned_surrogate_mpc(
    a, b, H, S, q, sigma, n, u_bound, x_ref, m=2, maxiter=300
):
    r = 0.01 * jnp.eye(m)

    def rollout_cost(u_flat, x0, w_scenarios):
        u_seq = u_flat.reshape(H, m)

        def step_batch(x_batch, inputs):
            u_k, w_k = inputs
            u_k = jnp.clip(u_k, -u_bound, u_bound)
            u_batch = jnp.broadcast_to(u_k, (S, m))
            err = x_batch - x_ref
            cost_k = jnp.sum((err @ q) * err, axis=1) + u_k @ r @ u_k
            x_next_interior = x_batch @ a[1:-1, :].T + sigma * x_batch[:, 1:-1] * w_k[:, 1:-1]
            x_cond = jnp.concatenate([x_batch, u_batch], axis=1)
            delta_left = predict_unpruned_x0_batch(x_cond, w_k[:, 0:1])
            delta_right = predict_unpruned_xn_batch(x_cond, w_k[:, -1:])
            x_next = jnp.concatenate(
                [
                    x_batch[:, 0:1] + delta_left,
                    x_next_interior,
                    x_batch[:, -1:] + delta_right,
                ],
                axis=-1,
            )
            return x_next, cost_k

        x0_batch = jnp.broadcast_to(x0, (S, n))
        x_final, costs = jax.lax.scan(
            step_batch, x0_batch, (u_seq, w_scenarios.swapaxes(0, 1))
        )
        err_f = x_final - x_ref
        terminal = jnp.sum((err_f @ q) * err_f, axis=1)
        return jnp.mean(jnp.sum(costs, axis=0) + terminal)

    @jax.jit
    def solve(x0, key, u_init):
        key, subkey = jr.split(key)
        w_scenarios = jr.normal(subkey, shape=(S, H, n))

        def obj(u):
            return jax.value_and_grad(rollout_cost)(u, x0, w_scenarios)

        solver = jaxopt.LBFGS(
            fun=obj,
            value_and_grad=True,
            maxiter=maxiter,
            jit=True,
            unroll=False,
        )
        result = solver.run(u_init.flatten())
        u_opt = jnp.clip(result.params.reshape(H, m), -u_bound, u_bound)
        return u_opt, u_opt[0], key

    return solve


def make_pruned_surrogate_mpc(
    a, b, H, S, q, sigma, n, u_bound, x_ref, m=2, maxiter=300
):
    r = 0.01 * jnp.eye(m)

    def rollout_cost(u_flat, x0, w_scenarios):
        u_seq = u_flat.reshape(H, m)

        def step_batch(x_batch, inputs):
            u_k, w_k = inputs
            u_k = jnp.clip(u_k, -u_bound, u_bound)
            u_batch = jnp.broadcast_to(u_k, (S, m))
            err = x_batch - x_ref
            cost_k = jnp.sum((err @ q) * err, axis=1) + u_k @ r @ u_k
            
            x_next_interior = x_batch @ a[1:-1, :].T + sigma * x_batch[:, 1:-1] * w_k[:, 1:-1]
            x0_cond = jnp.concatenate([x_batch[:, 0:2], u_batch[:, 0:1]], axis=1)
            xn_cond = jnp.concatenate([x_batch[:, -2:], u_batch[:, 1:2]], axis=1)
            delta_left = predict_pruned_x0_batch(x0_cond, w_k[:, 0:1])
            delta_right = predict_pruned_xn_batch(xn_cond, w_k[:, -1:])
            x_next = jnp.concatenate(
                [
                    x_batch[:, 0:1] + delta_left,
                    x_next_interior,
                    x_batch[:, -1:] + delta_right,
                ],
                axis=-1,
            )
            return x_next, cost_k

        x0_batch = jnp.broadcast_to(x0, (S, n))
        x_final, costs = jax.lax.scan(
            step_batch, x0_batch, (u_seq, w_scenarios.swapaxes(0, 1))
        )
        err_f = x_final - x_ref
        terminal = jnp.sum((err_f @ q) * err_f, axis=1)
        return jnp.mean(jnp.sum(costs, axis=0) + terminal)

    @jax.jit
    def solve(x0, key, u_init):
        key, subkey = jr.split(key)
        w_scenarios = jr.normal(subkey, shape=(S, H, n))

        def obj(u):
            return jax.value_and_grad(rollout_cost)(u, x0, w_scenarios)

        solver = jaxopt.LBFGS(
            fun=obj,
            value_and_grad=True,
            maxiter=maxiter,
            jit=True,
            # jit=False,
            unroll=False,
            # unroll=True,
        )
        result = solver.run(u_init.flatten())
        u_opt = jnp.clip(result.params.reshape(H, m), -u_bound, u_bound)
        return u_opt, u_opt[0], key

    return solve


def simulate_mpc(
    controller, a, b, sigma, x0, T, H, m, key_dynamics, key_mpc, u_ff=None
):
    x_traj = [x0]
    u_traj = []
    x = x0
    if u_ff is None:
        u_init = jnp.zeros((H, m))
    else:
        u_init = jnp.broadcast_to(u_ff, (H, m))
    for _k in range(T):
        u_opt, u0, key_mpc = controller(x, key_mpc, u_init)
        u_init = jnp.concatenate([u_opt[1:], u_opt[-1:]], axis=0)
        # print(u0)

        key_dynamics, subkey = jr.split(key_dynamics)
        x = true_step(a, b, sigma, x, u0, subkey)
        x_traj.append(x)
        u_traj.append(u0)

    return jnp.stack(x_traj), jnp.stack(u_traj)


def warmup_controllers(controller_map, a, b, sigma, x0, H, m, u_ff, seed=0):
    """JIT-compile each controller / true_step before timed trials.

    Runs one MPC solve and one true dynamics step per controller, then
    blocks until device work finishes so compile cost is excluded from
    subsequent wall-time measurements.
    """
    print("Warming up controllers (JIT compile; not timed)...")
    if u_ff is None:
        u_init = jnp.zeros((H, m))
    else:
        u_init = jnp.broadcast_to(u_ff, (H, m))
    key = jr.PRNGKey(seed)
    for name, controller in controller_map.items():
        key, key_mpc, key_dyn = jr.split(key, 3)
        u_opt, u0, _ = controller(x0, key_mpc, u_init)
        x_next = true_step(a, b, sigma, x0, u0, key_dyn)
        jax.block_until_ready((u_opt, u0, x_next))
        print(f"  warmed: {name}")
    print("Warm-up done.")


def compute_cost(x_traj, u_traj, q, r, x_ref):
    state_cost = float(sum((x - x_ref) @ q @ (x - x_ref) for x in x_traj[:-1]))
    control_cost = float(sum(u @ r @ u for u in u_traj))
    return state_cost + control_cost, state_cost, control_cost


def summarize(name, results, baseline_total_costs):
    total_costs = np.array([r["total_cost"] for r in results], dtype=float)
    state_costs = np.array([r["state_cost"] for r in results], dtype=float)
    control_costs = np.array([r["control_cost"] for r in results], dtype=float)
    terminal_norms = np.array([r["terminal_norm"] for r in results], dtype=float)
    wall_times = np.array([r["wall_time"] for r in results], dtype=float)
    relative_gap = (total_costs - baseline_total_costs) / baseline_total_costs

    print(f"\n{name}")
    print(
        f"  mean total cost     : {total_costs.mean():.4f} +/- {total_costs.std():.4f}"
    )
    print(f"  mean tracking cost  : {state_costs.mean():.4f}")
    print(f"  mean control cost   : {control_costs.mean():.4f}")
    print(f"  mean terminal ||e|| : {terminal_norms.mean():.4f}")
    print(
        f"  mean wall time      : {wall_times.mean():.4f}s total "
        f"({(wall_times / CFG.T).mean():.4f}s/step, full closed-loop sim)"
    )
    if name != "MPC-known":
        print(
            f"  relative cost gap   : {relative_gap.mean() * 100:.2f}% +/- {relative_gap.std() * 100:.2f}%"
        )


def export_results_mat(
    results: dict[str, list[dict]],
    path: Path,
    cfg: ExperimentCfg,
    a,
    b,
    q,
    r,
    x_ref,
) -> None:
    """Save all trial trajectories and scalars to a MATLAB-readable .mat file.

    Costs use tracking error e = x - x_ref with x_ref = 1.
    """
    md: dict[str, object] = {
        "meta_n": np.int32(cfg.n),
        "meta_kappa": np.float64(cfg.kappa),
        "meta_dt": np.float64(cfg.dt),
        "meta_sigma": np.float64(cfg.sigma),
        "meta_T": np.int32(cfg.T),
        "meta_H": np.int32(cfg.H),
        "meta_S": np.int32(cfg.S),
        "meta_m": np.int32(cfg.m),
        "meta_u_bound": np.float64(cfg.u_bound),
        "meta_num_trials": np.int32(cfg.num_trials),
        "meta_seed": np.int32(cfg.seed),
        "meta_x_ref_value": np.float64(cfg.x_ref_value),
        "meta_x0_noise_std": np.float64(cfg.x0_noise_std),
        "A_sys": np.asarray(a, dtype=np.float64),
        "B_sys": np.asarray(b, dtype=np.float64),
        "Q_cost": np.asarray(q, dtype=np.float64),
        "R_cost": np.asarray(r, dtype=np.float64),
        "x_ref": np.asarray(x_ref, dtype=np.float64),
    }
    q_np = np.asarray(q, dtype=np.float64)
    r_np = np.asarray(r, dtype=np.float64)
    x_ref_np = np.asarray(x_ref, dtype=np.float64)
    for name, trials in results.items():
        prefix = name.replace("-", "_")
        x_stack = np.stack([t["x_traj"] for t in trials], axis=0)
        u_stack = np.stack([t["u_traj"] for t in trials], axis=0)
        err_stack = x_stack[:, :-1, :] - x_ref_np
        state_per_step = np.sum((err_stack @ q_np) * err_stack, axis=2)
        ctrl_per_step = np.sum((u_stack @ r_np) * u_stack, axis=2)
        cum_total_cost = np.cumsum(state_per_step + ctrl_per_step, axis=1)

        md[f"{prefix}_x_traj"] = x_stack
        md[f"{prefix}_u_traj"] = u_stack
        md[f"{prefix}_cum_total_cost"] = cum_total_cost
        md[f"{prefix}_total_cost"] = np.array(
            [t["total_cost"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_state_cost"] = np.array(
            [t["state_cost"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_control_cost"] = np.array(
            [t["control_cost"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_terminal_norm"] = np.array(
            [t["terminal_norm"] for t in trials], dtype=np.float64
        )
        md[f"{prefix}_wall_time"] = np.array(
            [t["wall_time"] for t in trials], dtype=np.float64
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), md, do_compression=True)


def main():
    cfg = CFG
    a, b = build_system_matrices(n=cfg.n, kappa=cfg.kappa, dt=cfg.dt)
    q = jnp.eye(cfg.n)
    r = 0.01 * jnp.eye(cfg.m)
    x_ref = jnp.linspace(-0, 0, cfg.n)
    u_ff = equilibrium_feedforward(a, b, x_ref)
    u_eff_ref = boundary_input(x_ref, u_ff)
    eq_resid = float(jnp.linalg.norm(x_ref - a @ x_ref - b @ u_eff_ref))

    print("Boundary control: -kappa x_xi(0)=x(0)u_L, kappa x_xi(1)=x(1)u_R")
    print(f"Tracking target: x_ref(xi) = {cfg.x_ref_value:g} (constant)")
    print(f"  ||x_ref|| = {float(jnp.linalg.norm(x_ref)):.4f}")
    print(f"  x0 = x_ref + N(0, {cfg.x0_noise_std:g}^2 I)")
    print(f"  u_ff (lstsq equilibrium) = {np.asarray(u_ff)}")
    print(f"  ||x_ref - A x_ref - B (x_bdry ⊙ u_ff)|| = {eq_resid:.3e}")

    known_controller = make_known_mpc(
        a, b, cfg.H, cfg.S, q, cfg.sigma, cfg.n, cfg.u_bound, x_ref, m=cfg.m
    )
    unpruned_controller = make_unpruned_surrogate_mpc(
        a, b, cfg.H, cfg.S, q, cfg.sigma, cfg.n, cfg.u_bound, x_ref, m=cfg.m
    )
    pruned_controller = make_pruned_surrogate_mpc(
        a, b, cfg.H, cfg.S, q, cfg.sigma, cfg.n, cfg.u_bound, x_ref, m=cfg.m
    )

    controller_map = {
        "MPC-known": known_controller,
        "MPC-surrogate-unpruned": unpruned_controller,
        "MPC-surrogate-pruned": pruned_controller,
    }
    results = {name: [] for name in controller_map}
    summary_order = list(controller_map.keys())
    print(f"seed: {cfg.seed}")

    master_key = jr.PRNGKey(cfg.seed)
    for trial_idx in range(cfg.num_trials):
        master_key, trial_key = jr.split(master_key)
        x0_key, dyn_key, plan_known_key, plan_unpruned_key, plan_pruned_key = jr.split(
            trial_key, 5
        )
        x0 = 100 + cfg.x0_noise_std * jr.normal(x0_key, shape=(cfg.n,))
        if trial_idx == 0:
            # Compile all controllers on this trial's shapes before any timed run.
            warmup_controllers(
                controller_map,
                a,
                b,
                cfg.sigma,
                x0,
                cfg.H,
                cfg.m,
                u_ff,
                seed=cfg.seed + 1,
            )
        controller_keys = {
            "MPC-known": (dyn_key, plan_known_key),
            "MPC-surrogate-unpruned": (dyn_key, plan_unpruned_key),
            "MPC-surrogate-pruned": (dyn_key, plan_pruned_key),
        }

        for name, controller in controller_map.items():
            key_dynamics, key_mpc = controller_keys[name]
            t0 = time.perf_counter()
            x_traj, u_traj = simulate_mpc(
                controller,
                a,
                b,
                cfg.sigma,
                x0,
                cfg.T,
                cfg.H,
                cfg.m,
                key_dynamics,
                key_mpc,
                u_ff=u_ff,
            )
            # Ensure async device work finishes before stopping the clock.
            jax.block_until_ready((x_traj, u_traj))
            wall_time = time.perf_counter() - t0
            total_cost, state_cost, control_cost = compute_cost(
                x_traj, u_traj, q, r, x_ref
            )
            results[name].append(
                {
                    "x_traj": np.asarray(x_traj),
                    "u_traj": np.asarray(u_traj),
                    "total_cost": total_cost,
                    "state_cost": state_cost,
                    "control_cost": control_cost,
                    "terminal_norm": float(jnp.linalg.norm(x_traj[-1] - x_ref)),
                    "wall_time": wall_time,
                }
            )
            print(
                f"trial={trial_idx + 1:02d}/{cfg.num_trials:02d} "
                f"{name:24s} total_cost={total_cost:.4f} wall_time={wall_time:.2f}s", flush=True,
            )
        print("------------------------------------------------------")
        print("------------------------------------------------------")

    baseline_total_costs = np.array(
        [r["total_cost"] for r in results["MPC-known"]], dtype=float
    )
    for name in summary_order:
        summarize(name, results[name], baseline_total_costs)

    out_mat = (CONTROL_DIR / cfg.results_mat_path).resolve()
    export_results_mat(results, out_mat, cfg, a, b, q, r, x_ref)

    x_ref_np = np.asarray(x_ref)
    q_np = np.asarray(q)
    r_np = np.asarray(r)
    t_grid = np.arange(cfg.T + 1)
    t_u = np.arange(cfg.T)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    style_map = {
        "MPC-known": ("Known", "--", "b"),
        "MPC-surrogate-unpruned": ("Surrogate-unpruned", "--", "r"),
        "MPC-surrogate-pruned": ("Surrogate-pruned", ":", "g"),
    }

    for name, (label, linestyle, color) in style_map.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        u_stack = np.stack([r["u_traj"] for r in results[name]], axis=0)
        err_stack = x_stack - x_ref_np
        mean_err_norm = np.linalg.norm(err_stack, axis=2).mean(axis=0)
        mean_u = u_stack.mean(axis=0)
        err_steps = err_stack[:, :-1, :]
        mean_cum_track_cost = np.mean(
            np.cumsum(np.sum((err_steps @ q_np) * err_steps, axis=2), axis=1),
            axis=0,
        )

        axes[0].semilogy(
            t_grid,
            mean_err_norm,
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )
        axes[1].plot(
            t_u,
            mean_cum_track_cost,
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )
        axes[2].plot(
            t_u,
            mean_u[:, 0],
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )
        axes[3].plot(
            t_u,
            mean_u[:, 1],
            label=label,
            linewidth=2,
            linestyle=linestyle,
            color=color,
        )

    axes[0].set_title(r"Mean tracking error $\|x - x^\star\|$")
    axes[1].set_title("Mean cumulative tracking cost")
    axes[2].set_title(r"Mean control $u_L$")
    axes[3].set_title(r"Mean control $u_R$")
    for ax in axes:
        ax.legend()

    plt.tight_layout()
    out = (CONTROL_DIR / cfg.plot_path).resolve()
    plt.savefig(out, dpi=150)
    plt.close()

    fig_all, axes_all = plt.subplots(1, 3, figsize=(18, 4))
    color_lookup = {name: f"C{idx}" for idx, name in enumerate(style_map)}
    for name, (label, linestyle, color) in style_map.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        u_stack = np.stack([r["u_traj"] for r in results[name]], axis=0)
        err_norms = np.linalg.norm(x_stack - x_ref_np, axis=2)

        for idx in range(err_norms.shape[0]):
            axes_all[0].semilogy(
                t_grid,
                err_norms[idx],
                color=color_lookup[name],
                alpha=0.18,
                linestyle=linestyle,
                linewidth=1,
            )
            axes_all[1].plot(
                t_u,
                u_stack[idx, :, 0],
                color=color_lookup[name],
                alpha=0.18,
                linestyle=linestyle,
                linewidth=1,
            )
            axes_all[2].plot(
                t_u,
                u_stack[idx, :, 1],
                color=color_lookup[name],
                alpha=0.18,
                linestyle=linestyle,
                linewidth=1,
            )

        axes_all[0].semilogy(
            t_grid,
            err_norms.mean(axis=0),
            color=color_lookup[name],
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
        )
        axes_all[1].plot(
            t_u,
            u_stack[:, :, 0].mean(axis=0),
            color=color_lookup[name],
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
        )
        axes_all[2].plot(
            t_u,
            u_stack[:, :, 1].mean(axis=0),
            color=color_lookup[name],
            linewidth=2.5,
            linestyle=linestyle,
            label=label,
        )

    axes_all[0].set_title(r"All tracking-error trajectories $\|x-x^\star\|$")
    axes_all[1].set_title(r"All $u_L$ trajectories")
    axes_all[2].set_title(r"All $u_R$ trajectories")
    for ax in axes_all:
        ax.legend()

    plt.tight_layout()
    out_all = (CONTROL_DIR / cfg.all_traj_plot_path).resolve()
    plt.savefig(out_all, dpi=150)
    plt.close()

    cum_cost_style = {
        "MPC-known": ("Known dynamics", "--", "r"),
        "MPC-surrogate-unpruned": ("Surrogate (unpruned)", "-.", "g"),
        "MPC-surrogate-pruned": ("Surrogate (pruned)", "--", "m"),
    }
    fig_mpc, ax_mpc = plt.subplots()
    for name, (label, linestyle, color) in cum_cost_style.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        u_stack = np.stack([r["u_traj"] for r in results[name]], axis=0)
        err_steps = x_stack[:, :-1, :] - x_ref_np
        state_per_step = np.sum((err_steps @ q_np) * err_steps, axis=2)
        ctrl_per_step = np.sum((u_stack @ r_np) * u_stack, axis=2)
        mean_cum_total = np.mean(
            np.cumsum(state_per_step + ctrl_per_step, axis=1), axis=0
        )
        ax_mpc.plot(
            np.arange(cfg.T),
            mean_cum_total,
            label=label,
            linewidth=2.5,
            linestyle=linestyle,
            color=color,
        )
    ax_mpc.set_xlabel("time step")
    ax_mpc.set_ylabel("mean cumulative cost")
    ax_mpc.set_title(rf"Mean cumulative tracking cost, $x^\star={cfg.x_ref_value:g}$")
    ax_mpc.legend()
    ax_mpc.grid(True, alpha=0.3)
    plt.tight_layout()
    out_mpc_cum = (CONTROL_DIR / cfg.mpc_mean_cum_total_cost_path).resolve()
    plt.savefig(out_mpc_cum, dpi=150)
    plt.close()

    xi = np.linspace(0.0, 1.0, cfg.n)
    fig_end, ax_end = plt.subplots(figsize=(8, 4.5))
    x0_stack = np.stack([r["x_traj"][0] for r in results["MPC-known"]], axis=0)
    if x0_stack.shape[0] > 1:
        for idx in range(x0_stack.shape[0]):
            ax_end.plot(xi, x0_stack[idx], color="0.55", alpha=0.25, linewidth=1.0)
        ax_end.plot(
            xi,
            x0_stack.mean(axis=0),
            color="0.35",
            linestyle="-.",
            linewidth=2.2,
            label=r"$x(\xi, 0)$ (mean)",
        )
    else:
        ax_end.plot(
            xi,
            x0_stack[0],
            color="0.35",
            linestyle="-.",
            linewidth=2.2,
            label=r"$x(\xi, 0)$",
        )
    ax_end.plot(
        xi,
        x_ref_np,
        color="k",
        linestyle="-",
        linewidth=2.0,
        label=rf"$x^\star={cfg.x_ref_value:g}$",
    )
    for name, (label, linestyle, color) in style_map.items():
        x_stack = np.stack([r["x_traj"] for r in results[name]], axis=0)
        end_profiles = x_stack[:, -1, :]
        if end_profiles.shape[0] > 1:
            for idx in range(end_profiles.shape[0]):
                ax_end.plot(
                    xi,
                    end_profiles[idx],
                    color=color,
                    alpha=0.22,
                    linestyle=linestyle,
                    linewidth=1.0,
                )
            ax_end.plot(
                xi,
                end_profiles.mean(axis=0),
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                label=f"{label} (mean)",
            )
        else:
            ax_end.plot(
                xi,
                end_profiles[0],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
    ax_end.set_xlabel(r"$\xi$")
    ax_end.set_ylabel(r"$x(\xi)$")
    ax_end.set_title("Initial and terminal spatial profiles")
    ax_end.legend()
    ax_end.grid(True, alpha=0.3)
    plt.tight_layout()
    out_end = (CONTROL_DIR / cfg.end_profile_plot_path).resolve()
    plt.savefig(out_end, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
