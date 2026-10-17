"""
Load a SAC policy trained with pendulum_sac_missing.py and evaluate on the
same true stochastic pendulum (partial obs: angle window + past actions).

Default checkpoint: W=3, reward B, sigma=0.3 → logs/partial_obs_true/W3_RB_s0.3/
"""

import argparse
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

from case1 import PartialObsStochasticPendulum

# Matches pendulum_sac_missing.py __main__ defaults for the W3 RB s=0.3 run
DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logs",
    "partial_obs_true",
    "W3_RB_s0.3",
)


def make_eval_env(
    window_size: int = 3,
    reward_variant: str = "B",
    sigma: float = 0.3,
    max_torque: float = 2.0,
    max_episode_steps: int = 200,
):
    env = PartialObsStochasticPendulum(
        window_size=window_size,
        reward_variant=reward_variant,
        sigma=sigma,
        max_torque=max_torque,
    )
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)


def _env_dt(env: gym.Env) -> float:
    inner = env.unwrapped
    return float(getattr(inner, "dt", 0.05))


def rollout_episode(model, env: gym.Env, seed: int, deterministic: bool) -> dict:
    """One episode; returns theta, theta_dot, action, reward (same as pendulum_sac_missing.rollout)."""
    obs, _ = env.reset(seed=seed)
    dt = _env_dt(env)
    thetas, theta_dots, actions, rewards = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        thetas.append(info["theta"])
        theta_dots.append(info["true_theta_dot"])
        a = np.asarray(action).reshape(-1)
        actions.append(float(a[0]))
        rewards.append(float(reward))
        done = terminated or truncated
    return {
        "dt": dt,
        "theta": np.array(thetas),
        "theta_dot": np.array(theta_dots),
        "action": np.array(actions),
        "reward": np.array(rewards),
    }


def plot_trajectories(
    trajectories: List[dict],
    title: str,
    save_path: Optional[str],
    show: bool,
):
    """Stacked time series (θ, θ̇, u, reward) with one curve per episode."""
    if not trajectories:
        return

    dt = trajectories[0]["dt"]
    n = len(trajectories)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n, 2)))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, squeeze=False)
    axr = axes.ravel()
    fig.suptitle(title, fontsize=12)

    row_labels = [
        "θ (degrees)",
        "θ̇ (rad/s)  [true, not observed]",
        "Torque u (N·m)",
        "Reward",
    ]

    for i, traj in enumerate(trajectories):
        T = len(traj["theta"])
        t = np.arange(T) * dt
        c = colors[i % len(colors)]
        theta_wrap = ((traj["theta"] + np.pi) % (2 * np.pi)) - np.pi
        axr[0].plot(t, np.degrees(theta_wrap), color=c, alpha=0.85, label=f"ep {i + 1}")
        axr[1].plot(t, traj["theta_dot"], color=c, alpha=0.85)
        axr[2].plot(t, traj["action"], color=c, alpha=0.85)
        axr[3].plot(t, traj["reward"], color=c, alpha=0.85)

    axr[0].axhline(0, color="k", linestyle="--", lw=0.8)
    axr[0].axhline(180, color="gray", linestyle=":", lw=0.6)
    axr[0].axhline(-180, color="gray", linestyle=":", lw=0.6)
    axr[2].axhline(2.0, color="gray", linestyle=":", lw=0.8)
    axr[2].axhline(-2.0, color="gray", linestyle=":", lw=0.8)

    for row in range(4):
        axr[row].set_ylabel(row_labels[row])
        axr[row].grid(True, alpha=0.3)
    axr[-1].set_xlabel("Time (s)")
    if n <= 12:
        axr[0].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_return_bars(
    returns: List[float],
    labels: List[str],
    title: str,
    save_path: Optional[str],
    show: bool,
):
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(returns)), 4))
    x = np.arange(len(returns))
    ax.bar(x, returns, color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Episode return")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Roll out trained SAC on true partial-obs pendulum"
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_DIR,
        help="Directory containing best_model.zip / final_model.zip",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        choices=("best", "final"),
        default="best",
        help="Use best_model.zip (EvalCallback) or final_model.zip",
    )
    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--reward-variant", type=str, default="B")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--max-torque", type=float, default=2.0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument(
        "--seed-start", type=int, default=0, help="Seeds: seed_start, seed_start+1, ..."
    )
    p.add_argument(
        "--stochastic-policy",
        action="store_true",
        help="Sample actions (default: deterministic)",
    )
    p.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="Save figures as {prefix}_trajectories.png and {prefix}_returns.png (default: <model-dir>/eval)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (only save files)",
    )
    args = p.parse_args()

    path = os.path.join(args.model_dir, f"{args.checkpoint}_model.zip")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing model file: {path}")

    env = make_eval_env(
        window_size=args.window_size,
        reward_variant=args.reward_variant,
        sigma=args.sigma,
        max_torque=args.max_torque,
    )

    model = SAC.load(path, env=env)
    det = not args.stochastic_policy

    plot_prefix = args.plot_prefix
    if plot_prefix is None:
        plot_prefix = os.path.join(args.model_dir, "eval")

    show_plots = not args.no_show

    returns = []
    trajectories = []
    tag = f"W{args.window_size}_R{args.reward_variant}_s{args.sigma}"

    for ep in range(args.episodes):
        seed = args.seed_start + ep
        traj = rollout_episode(model, env, seed=seed, deterministic=det)
        trajectories.append(traj)
        ep_ret = float(traj["reward"].sum())
        theta_final = float(((traj["theta"][-1] + np.pi) % (2 * np.pi)) - np.pi)
        returns.append(ep_ret)
        print(
            f"episode {ep + 1:2d}  seed={seed:3d}  "
            f"return={ep_ret:8.2f}  final θ={np.degrees(theta_final):7.2f}°"
        )

    env.close()
    print()
    print(f"mean return = {np.mean(returns):.2f} ± {np.std(returns):.2f}")

    traj_title = (
        f"True env — partial obs SAC  ({tag})  [{args.checkpoint}_model]\n"
        f"mean return = {np.mean(returns):.1f} ± {np.std(returns):.1f}"
    )
    plot_trajectories(
        trajectories,
        title=traj_title,
        save_path=f"{plot_prefix}_trajectories.png",
        show=show_plots,
    )
    plot_return_bars(
        returns,
        labels=[f"ep{i + 1}" for i in range(len(returns))],
        title=f"Episode returns — {tag} ({args.checkpoint}_model)",
        save_path=f"{plot_prefix}_returns.png",
        show=show_plots,
    )


if __name__ == "__main__":
    main()
