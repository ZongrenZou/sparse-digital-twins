#!/usr/bin/env python3
"""
Load three SAC policies and evaluate them on the true LunarLander-v3 environment:

  1) lunar_fm (full FM surrogate training) — e.g. logs/fm_impulse/final_model.zip
  2) lunar_fm_pruned_case_b — auto-detected or explicit path
  3) Reference — trained only on the true env — e.g. lunar_true_env_sac.py output

Produces:
  - One 2D flight-path figure per agent (N episodes overlaid).
  - A bar plot comparing mean episode return ± std across those episodes.

Edit the CONFIG block below, then run:  python compare_agents_true_lunar.py
"""
import glob
import os
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

_ROOT = Path(__file__).resolve().parent



MODEL_FULL = str(_ROOT / "logs_seed_1" / "fm_full" / "best_model.zip")
# Set to "" to auto-pick logs/fm_impulse_pruned_*/final_model.zip (newest file if several exist).
MODEL_PRUNED = str(_ROOT / "logs_seed_1" / "fm_pruned" / "best_model.zip")
# Reference policy: trained on true env only (e.g. lunar_true_env_sac.py).
MODEL_REFERENCE = str(_ROOT / "logs_seed_1" / "sac_true_lunar" / "best_model.zip")
OUT_DIR = str(_ROOT / "compare_outputs")
N_EPISODES = 1_000
BASE_SEED = 6666
print(f"Using seed: {BASE_SEED}")
USE_STOCHASTIC_POLICY = False
# Gymnasium considers LunarLander "solved" when the average return is ≥ 200; per-episode, return ≥ 200 is a common success indicator.
SOLVED_THRESHOLD = 200.0


def rollout_episode(model: SAC, env: gym.Env, *, deterministic: bool, seed: int | None):
    """One episode on `env`; returns normalized x/y trajectory and total reward."""
    obs, _ = env.reset(seed=seed)
    xs: list[float] = []
    ys: list[float] = []
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

    return {
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
        "total_reward": total_reward,
    }


def evaluate_policy(
    model_path: str,
    env: gym.Env,
    *,
    n_episodes: int,
    base_seed: int,
    deterministic: bool,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = SAC.load(model_path, env=env)
    trajectories: list[dict[str, np.ndarray]] = []
    rewards: list[float] = []

    for ep in range(n_episodes):
        traj = rollout_episode(
            model,
            env,
            deterministic=deterministic,
            seed=base_seed + ep,
        )
        trajectories.append({"x": traj["x"], "y": traj["y"]})
        rewards.append(traj["total_reward"])

    return {
        "rewards": np.asarray(rewards, dtype=np.float64),
        "trajectories": trajectories,
    }


def print_per_episode_returns(title: str, rewards: np.ndarray, threshold: float) -> None:
    """Print each episode's total return and count how many meet the success threshold."""
    n = len(rewards)
    n_solved = int(np.sum(rewards >= threshold))
    print(f"\n{title}")
    print(f"  Per-episode return (≥ {threshold:g} counts as success):")
    # for i in range(n):
    #     ret = float(rewards[i])
    #     tag = "  (success)" if ret >= threshold else ""
    #     print(f"    episode {i + 1:3d}: {ret:+9.2f}{tag}")
    print(f"  Success (≥ {threshold:g}): {n_solved}/{n}")


def plot_flight_paths(
    trajectories: list[dict[str, np.ndarray]],
    *,
    title: str,
    out_path: str,
    n_episodes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    assert len(trajectories) == n_episodes
    n = len(trajectories)
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(n, 2)))

    for i, traj in enumerate(trajectories):
        color = cmap[i % len(cmap)]
        ax.plot(traj["x"], traj["y"], color=color, alpha=0.75, lw=1.1)
        ax.scatter(traj["x"][0], traj["y"][0], color=color, marker="o", s=45, zorder=5)
        ax.scatter(traj["x"][-1], traj["y"][-1], color=color, marker="*", s=90, zorder=5)

    ax.axvline(0.0, color="k", linestyle="--", lw=0.8, label="Pad (x=0)")
    ax.axhline(0.0, color="gray", linestyle=":", lw=0.8, label="Ground (y=0)")
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reward_comparison(
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    *,
    n_episodes: int,
    success_counts: Sequence[int],
    solved_threshold: float,
    out_path: str,
) -> None:
    k = len(labels)
    if not (len(means) == len(stds) == len(success_counts) == k):
        raise ValueError("labels, means, stds, and success_counts must have the same length")

    fig, ax = plt.subplots(figsize=(max(7.0, 2.2 * k), 5.8))
    x = np.arange(k)
    colors = plt.cm.tab10(np.linspace(0, 0.85, k))
    means_f = [float(m) for m in means]
    stds_f = [float(s) for s in stds]
    n_ok_list = [int(c) for c in success_counts]
    bars = ax.bar(
        x,
        means_f,
        yerr=stds_f,
        capsize=8,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    # ax.set_xticks(x)
    # ax.set_xticklabels(list(labels))
    # ax.set_title(
    #     f"Return (sum of rewards) across {n_episodes} episodes\n"
    #     f"(success = episode return ≥ {solved_threshold:g})"
    # )
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3)

    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin if ymax > ymin else 1.0
    label_pad = 0.02 * y_span

    for rect, m, s, n_ok in zip(bars, means_f, stds_f, n_ok_list):
        y_top = m + s
        rate_pct = 100.0 * n_ok / n_episodes if n_episodes > 0 else 0.0
        x_c = rect.get_x() + rect.get_width() / 2.0
        y_anchor = y_top + label_pad
        # ax.text(
        #     x_c,
        #     y_anchor,
        #     f"{m:.2f}\n{n_ok}/{n_episodes} ({rate_pct:.0f}%)",
        #     ha="center",
        #     va="bottom",
        #     fontsize=10,
        #     linespacing=1.15,
        # )

    # Room for bar labels (two lines) above error bars — keep fontsize unchanged
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin, ymax + 0.1 * y_span)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_pruned_path(user_path: str) -> str:
    if user_path.strip():
        return user_path

    pattern = str(_ROOT / "logs" / "fm_impulse_pruned_*" / "final_model.zip")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        print(
            "Could not auto-detect MODEL_PRUNED: no logs/fm_impulse_pruned_*/final_model.zip found.\n"
            "Set MODEL_PRUNED in compare_agents_true_lunar.py to your final_model.zip path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(matches) > 1:
        newest = max(matches, key=lambda p: os.path.getmtime(p))
        print(
            f"MODEL_PRUNED auto-detect: {len(matches)} checkpoints found; "
            f"using newest by file time:\n  {newest}",
        )
        return newest

    return matches[0]


def main() -> None:
    deterministic = not USE_STOCHASTIC_POLICY

    model_full = MODEL_FULL
    model_pruned = resolve_pruned_path(MODEL_PRUNED)
    model_reference = MODEL_REFERENCE

    if not os.path.isfile(model_reference):
        print(
            f"Reference model not found: {model_reference}\n"
            "Set MODEL_REFERENCE in compare_agents_true_lunar.py "
            "(e.g. output of lunar_true_env_sac.py or lunar_surrogate_vs_true_experiment.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    n = N_EPISODES
    # bar_labels = (
    #     "lunar_fm\n(full FM)",
    #     "pruned\ncase_b",
    #     "reference\n(true train)",
    # )
    bar_labels = (
        "Using all variables",
        "Using the discovered graph",
        "Reference\n(trained with the true env)",
    )

    print("True LunarLander-v3 evaluation (three agents)")
    print(f"  Episodes per agent: {n}")
    print(f"  Policy: {'deterministic' if deterministic else 'stochastic'}")
    print(f"  Full FM model:     {model_full}")
    print(f"  Pruned FM model:      {model_pruned}")
    print(f"  Reference model:   {model_reference}")
    print(f"  Output directory: {OUT_DIR}")

    env = gym.make("LunarLander-v3", continuous=True)
    try:
        print("\nLoading & evaluating agent 1 (lunar_fm, full FM) ...")
        r1 = evaluate_policy(
            model_full,
            env,
            n_episodes=n,
            base_seed=BASE_SEED,
            deterministic=deterministic,
        )
        print_per_episode_returns(
            "Agent 1 — lunar_fm (full FM cond.)",
            r1["rewards"],
            SOLVED_THRESHOLD,
        )

        print("\nLoading & evaluating agent 2 (lunar_fm_pruned_case_b) ...")
        r2 = evaluate_policy(
            model_pruned,
            env,
            n_episodes=n,
            base_seed=BASE_SEED,
            deterministic=deterministic,
        )
        print_per_episode_returns(
            "Agent 2 — lunar_fm_pruned_case_b",
            r2["rewards"],
            SOLVED_THRESHOLD,
        )

        print("\nLoading & evaluating agent 3 (reference, trained on true env) ...")
        r3 = evaluate_policy(
            model_reference,
            env,
            n_episodes=n,
            base_seed=BASE_SEED,
            deterministic=deterministic,
        )
        print_per_episode_returns(
            "Agent 3 — reference (SAC on true env only)",
            r3["rewards"],
            SOLVED_THRESHOLD,
        )
    finally:
        env.close()

    m1, s1 = float(r1["rewards"].mean()), float(r1["rewards"].std(ddof=0))
    m2, s2 = float(r2["rewards"].mean()), float(r2["rewards"].std(ddof=0))
    m3, s3 = float(r3["rewards"].mean()), float(r3["rewards"].std(ddof=0))

    print(f"\nSummary (mean ± std over {n} episodes):")
    print(f"  Agent 1 (lunar_fm):           {m1:+.2f} ± {s1:.2f}")
    print(f"  Agent 2 (pruned case_b):      {m2:+.2f} ± {s2:.2f}")
    print(f"  Agent 3 (reference, true):  {m3:+.2f} ± {s3:.2f}")

    path1 = os.path.join(OUT_DIR, "flight_paths_full.png")
    path2 = os.path.join(OUT_DIR, "flight_paths_pruned.png")
    path3 = os.path.join(OUT_DIR, "flight_paths_reference_true_train.png")
    path_bar = os.path.join(OUT_DIR, "reward_comparison_bar.png")

    plot_flight_paths(
        r1["trajectories"],
        title=f"Flight paths — lunar_fm (true env, {n} episodes)",
        out_path=path1,
        n_episodes=n,
    )
    plot_flight_paths(
        r2["trajectories"],
        title=f"Flight paths ({n} episodes)",
        out_path=path2,
        n_episodes=n,
    )
    plot_flight_paths(
        r3["trajectories"],
        title=f"Flight paths — reference SAC, trained on true env (true env, {n} episodes)",
        out_path=path3,
        n_episodes=n,
    )
    n_ok1 = int(np.sum(r1["rewards"] >= SOLVED_THRESHOLD))
    n_ok2 = int(np.sum(r2["rewards"] >= SOLVED_THRESHOLD))
    n_ok3 = int(np.sum(r3["rewards"] >= SOLVED_THRESHOLD))

    plot_reward_comparison(
        bar_labels,
        (m1, m2, m3),
        (s1, s2, s3),
        n_episodes=n,
        success_counts=(n_ok1, n_ok2, n_ok3),
        solved_threshold=SOLVED_THRESHOLD,
        out_path=path_bar,
    )

    print(f"\nSaved:\n  {path1}\n  {path2}\n  {path3}\n  {path_bar}")


if __name__ == "__main__":
    main()
