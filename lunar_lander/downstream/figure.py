#!/usr/bin/env python3
"""
Evaluate two SAC policies in the SAME true environment from
lunar_true_custom.py (8-d lander + N_EXTRA spectator extras + fuel):

  1) Exact-dynamics agent — trained in LunarLanderTrueEnv
     (lunar_true_custom.py). Uses the full observation.
  2) FM-pruned agent — trained in the 8-d conditional-FM surrogate
     (lunar_fm_pruned.py). At eval time it is given only obs[:8]
     (the lander state) and ignores extras / fuel.

Physics, reward, spectators, and fuel are identical for both. The
policies differ in training dynamics and in what they observe.

Produces:
    - One 2D flight-path figure per agent (N episodes overlaid).
    - A bar plot comparing mean episode return ± std across those episodes.

Edit the CONFIG block below, then run:  python compare_agents_true_custom.py
"""
import os
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from lunar_true_custom import N_EXTRA, N_TOTAL_EXTRA, SAVE_DIR as TRUE_CUSTOM_SAVE_DIR, make_true_env

_ROOT = Path(__file__).resolve().parent

N_LANDER_OBS = 8
# SAC trained on lunar_true_custom.LunarLanderTrueEnv (exact dynamics, full obs).
MODEL_TRUE_EXACT = str(_ROOT / "logs_seed_1" / "sac_true_custom" / "best_model.zip")
# SAC trained on lunar_fm_pruned.LunarLanderFMImpulseEnv (8-d FM surrogate).
MODEL_FM_PRUNED = str(_ROOT / "logs_seed_1" / "fm_pruned" / "best_model.zip")
OUT_DIR = str(_ROOT / "compare_outputs_true_custom")
N_EPISODES = 1_000
BASE_SEED = 6666
print(f"Using seed: {BASE_SEED}")
USE_STOCHASTIC_POLICY = False
SOLVED_THRESHOLD = 200.0


def _obs_dim(model: SAC) -> int:
    return int(np.prod(model.observation_space.shape))


def obs_for_policy(model: SAC, obs: np.ndarray) -> np.ndarray:
    """Feed each policy the observation width it was trained on."""
    obs = np.asarray(obs)
    n_in = _obs_dim(model)
    n_env = int(obs.shape[-1])
    if n_in == n_env:
        return obs
    if n_in == N_LANDER_OBS and n_env > N_LANDER_OBS:
        return obs[..., :N_LANDER_OBS]
    raise ValueError(
        f"Policy expects obs dim {n_in}, env returned {n_env}. "
        f"True env is 8 + N_TOTAL_EXTRA = {N_LANDER_OBS + N_TOTAL_EXTRA} "
        f"(N_EXTRA={N_EXTRA} spectators + 1 fuel). "
        f"The exact-dynamics policy should match that width; "
        f"the FM-pruned policy should be {N_LANDER_OBS}-d."
    )


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
        action, _ = model.predict(obs_for_policy(model, obs), deterministic=deterministic)
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
    expected_obs_dim: int | None = None,
):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Do not pass env= here: the 8-d FM policy's space does not match
    # the concatenated true env, and SAC.set_env would reject it.
    model = SAC.load(model_path)
    n_in = _obs_dim(model)
    if expected_obs_dim is not None and n_in != expected_obs_dim:
        raise ValueError(
            f"{model_path} has policy obs dim {n_in}, expected {expected_obs_dim}."
        )
    print(f"  Loaded {model_path} (policy obs dim = {n_in})")
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
    # colors = plt.cm.tab10(np.linspace(0, 0.85, k))
    colors = ["black", "grey"]
    means_f = [float(m) for m in means]
    stds_f = [float(s) for s in stds]
    n_ok_list = [int(c) for c in success_counts]
    # means_f.reverse()
    # stds_f.reverse()
    # n_ok_list.reverse()
    bars = ax.bar(
        x,
        means_f,
        yerr=stds_f,
        capsize=8,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_xticks([])
    # ax.set_xticks(x)
    # ax.set_xticklabels(list(labels), fontsize=9)
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

    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin, ymax + 0.1 * y_span)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    deterministic = not USE_STOCHASTIC_POLICY

    model_true_exact = MODEL_TRUE_EXACT
    model_fm_pruned = MODEL_FM_PRUNED

    if not os.path.isfile(model_true_exact):
        print(
            f"Exact-dynamics model not found: {model_true_exact}\n"
            "Train it with lunar_true_custom.py, or set MODEL_TRUE_EXACT.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(model_fm_pruned):
        print(
            f"FM-pruned model not found: {model_fm_pruned}\n"
            "Train it with lunar_fm_pruned.py, or set MODEL_FM_PRUNED.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    n = N_EPISODES
    env_obs_dim = N_LANDER_OBS + N_TOTAL_EXTRA
    bar_labels = (
        "Exact dynamics\n(all variables)",
        "FM pruned surrogate\n(8-d lander only)",
    )

    print("True-env evaluation (lunar_true_custom.LunarLanderTrueEnv)")
    print(
        f"  Env obs dim: {env_obs_dim} "
        f"(8 lander + {N_EXTRA} spectators + 1 fuel)"
    )
    print(f"  Episodes per agent: {n}")
    print(f"  Policy: {'deterministic' if deterministic else 'stochastic'}")
    print(f"  Exact-dynamics model: {model_true_exact}")
    print(f"  FM-pruned model:      {model_fm_pruned}")
    print(f"  Output directory: {OUT_DIR}")

    env = make_true_env()
    try:
        print("\nLoading & evaluating agent 1 (exact dynamics, full observation) ...")
        r1 = evaluate_policy(
            model_true_exact,
            env,
            n_episodes=n,
            base_seed=BASE_SEED,
            deterministic=deterministic,
            expected_obs_dim=env_obs_dim,
        )
        print_per_episode_returns(
            "Agent 1 — SAC trained on exact LunarLanderTrueEnv (full obs)",
            r1["rewards"],
            SOLVED_THRESHOLD,
        )

        print("\nLoading & evaluating agent 2 (FM pruned surrogate, lander state only) ...")
        r2 = evaluate_policy(
            model_fm_pruned,
            env,
            n_episodes=n,
            base_seed=BASE_SEED,
            deterministic=deterministic,
            expected_obs_dim=N_LANDER_OBS,
        )
        print_per_episode_returns(
            "Agent 2 — SAC trained on FM-pruned surrogate (eval uses obs[:8])",
            r2["rewards"],
            SOLVED_THRESHOLD,
        )
    finally:
        env.close()

    m1, s1 = float(r1["rewards"].mean()), float(r1["rewards"].std(ddof=0))
    m2, s2 = float(r2["rewards"].mean()), float(r2["rewards"].std(ddof=0))

    print(f"\nSummary (mean ± std over {n} episodes, same true env):")
    print(f"  Agent 1 (exact, all variables):  {m1:+.2f} ± {s1:.2f}")
    print(f"  Agent 2 (FM pruned, 8-d):        {m2:+.2f} ± {s2:.2f}")

    path1 = os.path.join(OUT_DIR, "flight_paths_exact_all_vars.png")
    path2 = os.path.join(OUT_DIR, "flight_paths_fm_pruned_8d.png")
    path_bar = os.path.join(OUT_DIR, "reward_comparison_bar.png")

    plot_flight_paths(
        r1["trajectories"],
        title=f"Flight paths — exact dynamics, all variables ({n} episodes)",
        out_path=path1,
        n_episodes=n,
    )
    plot_flight_paths(
        r2["trajectories"],
        title=f"Flight paths — FM pruned, 8-d lander only ({n} episodes)",
        out_path=path2,
        n_episodes=n,
    )
    n_ok1 = int(np.sum(r1["rewards"] >= SOLVED_THRESHOLD))
    n_ok2 = int(np.sum(r2["rewards"] >= SOLVED_THRESHOLD))

    plot_reward_comparison(
        bar_labels,
        (m1, m2),
        (s1, s2),
        n_episodes=n,
        success_counts=(n_ok1, n_ok2),
        solved_threshold=SOLVED_THRESHOLD,
        out_path=path_bar,
    )

    print(f"\nSaved:\n  {path1}\n  {path2}\n  {path_bar}")


if __name__ == "__main__":
    main()
