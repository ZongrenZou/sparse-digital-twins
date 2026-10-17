"""
Compare SAC policies on the *same* true stochastic pendulum (W=3, reward B):

  1) Trained directly in the true env (pendulum_sac_missing.py)
     default: logs/partial_obs_true/W3_RB_s0.3/best_model.zip

  2) Trained in FM surrogate with W=5 dynamics (pendulum_fm_surrogate_15.py)
     default: logs/fm_surrogate_w5/best_model.zip

  3) Trained in FM surrogate v2 with skip-lag / 6-dim dynamics (pendulum_fm_surrogate_6_v2.py)
     default: logs/fm_surrogate_skip/best_model.zip

  4) Trained in FM surrogate with current-state-only / 3-dim dynamics (pendulum_fm_surrogate_3.py)
     default: logs/fm_surrogate_w1/best_model.zip

All policies share the same 9-dim observation and reward B. Evaluation uses
the same true partial-observation environment for every agent, with the common
reset protocol from case1: random latent (theta, theta_dot), then 10
zero-control warm-up steps before control begins.

Requirements: gymnasium, stable-baselines3, matplotlib, numpy
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from test_partial_obs_sac_true_env import make_eval_env, rollout_episode

ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TRUE_MODEL = os.path.join(
    ROOT, "logs", "partial_obs_true", "W3_RB_s0.3", "best_model.zip"
)
DEFAULT_FM_W5_MODEL = os.path.join(ROOT, "logs", "fm_surrogate_w10", "best_model.zip")
DEFAULT_FM_V2_1_MODEL = os.path.join(
    ROOT, "logs", "fm_surrogate_skip_1", "best_model.zip"
)
DEFAULT_FM_V2_2_MODEL = os.path.join(
    ROOT, "logs", "fm_surrogate_skip_2", "best_model.zip"
)
DEFAULT_FM_W1_MODEL = os.path.join(ROOT, "logs", "fm_surrogate_w1", "best_model.zip")


def evaluate_agent(
    model_path: str,
    env,
    episodes: int,
    seed_start: int,
    deterministic: bool,
    label: str,
) -> Tuple[List[dict], np.ndarray]:
    model = SAC.load(model_path, env=env)
    trajs: List[dict] = []
    returns: List[float] = []
    for ep in range(episodes):
        seed = seed_start + ep
        traj = rollout_episode(model, env, seed=seed, deterministic=deterministic)
        trajs.append(traj)
        returns.append(float(traj["reward"].sum()))
    del model
    print(f"\n{label}")
    print("-" * 52)
    for ep, g in enumerate(returns):
        th = trajs[ep]
        tf = float(((th["theta"][-1] + np.pi) % (2 * np.pi)) - np.pi)
        print(
            f"  ep {ep + 1:2d}  seed={seed_start + ep:3d}  "
            f"return={g:9.2f}  final θ={np.degrees(tf):7.2f}°"
        )
    arr = np.array(returns, dtype=np.float64)
    print(f"  mean return = {arr.mean():.2f} ± {arr.std():.2f}")
    return trajs, arr


def plot_comparison(
    bundles: Sequence[Tuple[np.ndarray, str, str]],
    seed_start: int,
    title_suffix: str,
    save_path: str,
    show: bool,
):
    """
    bundles: list of (returns_per_seed, legend_label, color).
    """
    n = len(bundles[0][0])
    x = np.arange(n)
    n_b = len(bundles)
    # Bar widths so n_b bars fit in each group (~width 0.8 total)
    total_w = 0.75
    w = total_w / n_b
    centers = (np.arange(n_b) - (n_b - 1) / 2.0) * w

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    for i, (ret, lab, col) in enumerate(bundles):
        ax.bar(x + centers[i], ret, width=w * 0.92, label=lab, color=col, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"seed {seed_start + i}" for i in range(n)],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Episode return (sum of rewards)")
    ax.set_title("Per-seed returns (same true env)")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    means = [b[0].mean() for b in bundles]
    stds = [b[0].std() for b in bundles]
    names = [b[1].replace(" ", "\n") for b in bundles]
    colors = [b[2] for b in bundles]
    xb = np.arange(n_b)
    bars = ax2.bar(xb, means, yerr=stds, capsize=5, color=colors, alpha=0.9, width=0.55)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Mean ± std over seeds")
    ax2.set_title("Aggregate")
    ax2.grid(True, axis="y", alpha=0.3)
    ymax = max(m + s for m, s in zip(means, stds))
    for bar, m in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.05 + 2,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(
        "True stochastic pendulum — partial obs W=3, reward B\n" + title_suffix,
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_multi_trajectories(
    trajs: Sequence[dict],
    labels: Sequence[str],
    colors: Sequence[str],
    dt: float,
    save_path: str,
    show: bool,
):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for traj, lab, col in zip(trajs, labels, colors):
        tt = np.arange(len(traj["theta"])) * dt
        th = ((traj["theta"] + np.pi) % (2 * np.pi)) - np.pi
        axes[0].plot(tt, np.degrees(th), color=col, lw=1.4, label=lab, alpha=0.9)
        axes[1].plot(tt, traj["theta_dot"], color=col, lw=1.4, alpha=0.9)
        axes[2].plot(tt, traj["action"], color=col, lw=1.4, alpha=0.9)
        axes[3].plot(tt, traj["reward"], color=col, lw=1.4, alpha=0.9)

    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("θ (deg)")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("θ̇ (rad/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(2.0, color="gray", ls=":", lw=0.8)
    axes[2].axhline(-2.0, color="gray", ls=":", lw=0.8)
    axes[2].set_ylabel("u (N·m)")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Single-episode overlay (same seed, true env dynamics)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Compare true-trained vs FM-surrogate SAC policies on the true pendulum"
    )
    p.add_argument(
        "--true-model",
        type=str,
        default=DEFAULT_TRUE_MODEL,
        help="Path to true-env SAC .zip",
    )
    p.add_argument(
        "--fm-w5-model",
        type=str,
        default=DEFAULT_FM_W5_MODEL,
        help="Path to SAC trained in pendulum_fm_surrogate_15 (fm_surrogate_w5)",
    )
    p.add_argument(
        "--fm-v2-1-model",
        type=str,
        default=DEFAULT_FM_V2_1_MODEL,
        help="Path to SAC trained in pendulum_fm_surrogate_6_v2 (fm_surrogate_skip_1)",
    )
    p.add_argument(
        "--fm-v2-2-model",
        type=str,
        default=DEFAULT_FM_V2_2_MODEL,
        help="Path to SAC trained in pendulum_fm_surrogate_6_v2 (fm_surrogate_skip_2)",
    )
    p.add_argument(
        "--fm-w1-model",
        type=str,
        default=DEFAULT_FM_W1_MODEL,
        help="Path to SAC trained in pendulum_fm_surrogate_3 (fm_surrogate_w1)",
    )
    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--max-torque", type=float, default=2.0)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed-start", type=int, default=23333)
    p.add_argument("--stochastic-policy", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(ROOT, "logs", "compare_true_vs_fm_sac"),
        help="Directory for comparison plots",
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--trajectory-seed",
        type=int,
        default=0,
        help="Env reset seed for the trajectory overlay (must be in evaluated range)",
    )
    args = p.parse_args()

    for path, name in [
        (args.true_model, "true-env"),
        (args.fm_w5_model, "FM W5 (surrogate_15)"),
        (args.fm_v2_1_model, "FM v2 skip (surrogate_6_v2_1)"),
        (args.fm_v2_2_model, "FM v2 skip (surrogate_6_v2_2)"),
        (args.fm_w1_model, "FM W1 (surrogate_3)"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name} model: {path}")

    env = make_eval_env(
        window_size=args.window_size,
        reward_variant="B",
        sigma=args.sigma,
        max_torque=args.max_torque,
    )
    det = not args.stochastic_policy
    show = not args.no_show

    print(
        "\nEvaluation environment: true stochastic pendulum with a common "
        "10-step zero-control warm-up before control starts."
    )

    tr_true, ret_true = evaluate_agent(
        args.true_model,
        env,
        episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=det,
        label="(1) TRUE physics training — pendulum_sac_missing",
    )
    tr_fm5, ret_fm5 = evaluate_agent(
        args.fm_w5_model,
        env,
        episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=det,
        label="(2) FM surrogate — pendulum_fm_surrogate_15 (W=5 dynamics, logs/fm_surrogate_w5)",
    )
    tr_v2_1, ret_v2_1 = evaluate_agent(
        args.fm_v2_1_model,
        env,
        episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=det,
        label="(3) FM surrogate v2 — pendulum_fm_surrogate_6_v2 (skip-lag, logs/fm_surrogate_skip_1)",
    )
    tr_v2_2, ret_v2_2 = evaluate_agent(
        args.fm_v2_2_model,
        env,
        episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=det,
        label="(4) FM surrogate v2 — pendulum_fm_surrogate_6_v2 (skip-lag, logs/fm_surrogate_skip_2)",
    )
    tr_w1, ret_w1 = evaluate_agent(
        args.fm_w1_model,
        env,
        episodes=args.episodes,
        seed_start=args.seed_start,
        deterministic=det,
        label="(5) FM surrogate — pendulum_fm_surrogate_3 (W=1 dynamics, logs/fm_surrogate_w1)",
    )

    print(
        "\nPaired return differences (same seed, higher = better for the second named policy):"
    )
    print("  FM W5 − true:")
    for i in range(args.episodes):
        print(f"    seed {args.seed_start + i}: {ret_fm5[i] - ret_true[i]:+.2f}")
    print(
        f"    mean Δ = {(ret_fm5 - ret_true).mean():+.2f} ± {(ret_fm5 - ret_true).std():.2f}"
    )

    print("  FM v2_1 − true:")
    for i in range(args.episodes):
        print(f"    seed {args.seed_start + i}: {ret_v2_1[i] - ret_true[i]:+.2f}")
    print(
        f"    mean Δ = {(ret_v2_1 - ret_true).mean():+.2f} ± {(ret_v2_1 - ret_true).std():.2f}"
    )

    print("  FM v2_1 − FM W5:")
    for i in range(args.episodes):
        print(f"    seed {args.seed_start + i}: {ret_v2_1[i] - ret_fm5[i]:+.2f}")
    print(
        f"    mean Δ = {(ret_v2_1 - ret_fm5).mean():+.2f} ± {(ret_v2_1 - ret_fm5).std():.2f}"
    )

    print("  FM W1 − true:")
    for i in range(args.episodes):
        print(f"    seed {args.seed_start + i}: {ret_w1[i] - ret_true[i]:+.2f}")
    print(
        f"    mean Δ = {(ret_w1 - ret_true).mean():+.2f} ± {(ret_w1 - ret_true).std():.2f}"
    )

    print("  FM W1 − FM W5:")
    for i in range(args.episodes):
        print(f"    seed {args.seed_start + i}: {ret_w1[i] - ret_fm5[i]:+.2f}")
    print(
        f"    mean Δ = {(ret_w1 - ret_fm5).mean():+.2f} ± {(ret_w1 - ret_fm5).std():.2f}"
    )

    print("  FM W1 − FM v2_1:")
    for i in range(args.episodes):
        print(f"    seed {args.seed_start + i}: {ret_w1[i] - ret_v2_1[i]:+.2f}")
    print(
        f"    mean Δ = {(ret_w1 - ret_v2_1).mean():+.2f} ± {(ret_w1 - ret_v2_1).std():.2f}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    plot_comparison(
        [
            (ret_true, "True physics", "steelblue"),
            (ret_fm5, "FM (surrogate_30)", "darkorange"),
            (ret_v2_1, "FM skip (6_v2_1)", "seagreen"),
            (ret_v2_2, "FM skip (6_v2_2)", "seagreen"),
            (ret_w1, "FM W1 (surrogate_3)", "firebrick"),
        ],
        seed_start=args.seed_start,
        title_suffix=f"σ={args.sigma}, W={args.window_size}, reward B",
        save_path=os.path.join(args.out_dir, "compare_returns.png"),
        show=show,
    )
    import scipy.io as sio

    sio.savemat(
        "./logs/returns.mat",
        {
            "ret_true": ret_true,
            "ret_fm5": ret_fm5,
            "ret_v2_1": ret_v2_1,
            "ret_v2_2": ret_v2_2,
            "ret_w1": ret_w1,
        },
    )
    trajs_true = np.stack(
        [tr_true[idx]["theta"] for idx in range(args.episodes)], axis=0
    )
    trajs_fm5 = np.stack([tr_fm5[idx]["theta"] for idx in range(args.episodes)], axis=0)
    trajs_v2_1 = np.stack(
        [tr_v2_1[idx]["theta"] for idx in range(args.episodes)], axis=0
    )
    trajs_v2_2 = np.stack(
        [tr_v2_2[idx]["theta"] for idx in range(args.episodes)], axis=0
    )
    trajs_w1 = np.stack([tr_w1[idx]["theta"] for idx in range(args.episodes)], axis=0)

    sio.savemat(
        "./logs/trajs.mat",
        {
            "trajs_true": trajs_true,
            "trajs_fm5": trajs_fm5,
            "trajs_v2_1": trajs_v2_1,
            "trajs_v2_2": trajs_v2_2,
            "trajs_w1": trajs_w1,
        },
    )

    seed_lo = args.seed_start
    seed_hi = args.seed_start + args.episodes
    ts = args.trajectory_seed
    if not (seed_lo <= ts < seed_hi):
        ts = args.seed_start
        print(f"\nNote: --trajectory-seed out of range; using seed {ts} for overlay.")
    idx = ts - args.seed_start
    dt = tr_true[idx]["dt"]
    plot_multi_trajectories(
        [tr_true[idx], tr_fm5[idx], tr_v2_1[idx], tr_v2_2[idx], tr_w1[idx]],
        labels=[
            "True physics",
            "FM surrogate_30",
            "FM surrogate_6_v2_1",
            "FM surrogate_6_v2_2",
            "FM surrogate_3",
        ],
        colors=["steelblue", "darkorange", "seagreen", "seagreen", "firebrick"],
        dt=dt,
        save_path=os.path.join(
            args.out_dir,
            f"compare_trajectories_seed{args.seed_start + idx}.png",
        ),
        show=show,
    )

    env.close()

    print("End main.")


if __name__ == "__main__":
    main()
