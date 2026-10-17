#!/usr/bin/env python3
"""
SAC on true LunarLander-v3 (continuous): train, save, evaluate.

Mirrors the procedure in lunar_fm.py (same SAC hyperparameters, EvalCallback,
Monitor, vec env, save path pattern), but both training and evaluation use the
standard Gymnasium simulator — no flow-matching surrogate.

Edit CONFIG below, then run:  python lunar_true_env_sac.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

_ROOT = Path(__file__).resolve().parent

# ── CONFIG (aligned with lunar_fm.py) ────────────────────────────────────────

TOTAL_TIMESTEPS = 500_000
N_EVAL_EPISODES = 50
SEED = 12345
# SEED = 23451
# SEED = 34512
print(f"SEED: {SEED}")
SAVE_DIR = str(_ROOT / "logs" / "sac_true_lunar")


# ── Training (same structure as lunar_fm.train, true env for train + eval) ───

def train(save_dir: str, seed: int = 42, total_timesteps: int = 500_000) -> SAC:
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        return Monitor(gym.make("LunarLander-v3", continuous=True))

    train_env = make_vec_env(make_env, n_envs=1, seed=seed)
    eval_env = Monitor(gym.make("LunarLander-v3", continuous=True))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
        verbose=1,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=7.3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[400, 300]),
        verbose=1,
        seed=seed,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
    )

    print("\nTraining SAC on TRUE LunarLander-v3 ...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.time() - t0
    model.save(os.path.join(save_dir, "final_model"))
    print(f"  Done in {elapsed:.0f}s. Saved to {save_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model


# ── Rollout (same as lunar_fm.rollout) ───────────────────────────────────────

def rollout(model, env, deterministic=True, seed=None):
    obs, _ = env.reset(seed=seed)
    done = False
    states, actions, rewards = [], [], []

    while not done:
        states.append(obs.copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(action.copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    return {
        "x": np.array([s[0] for s in states]),
        "y": np.array([s[1] for s in states]),
        "vx": np.array([s[2] for s in states]),
        "vy": np.array([s[3] for s in states]),
        "angle": np.array([s[4] for s in states]),
        "angular_vel": np.array([s[5] for s in states]),
        "main_engine": np.array([a[0] for a in actions]),
        "lateral_engine": np.array([a[1] for a in actions]),
        "reward": np.array(rewards),
    }


# ── Evaluation on true env (same plots as lunar_fm.evaluate, updated titles) ─

def evaluate(model, n_episodes=20, seed=42, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make("LunarLander-v3", continuous=True)
    trajectories, total_rewards = [], []

    print(f"\nEvaluating on TRUE environment ({n_episodes} episodes) ...")
    for ep in range(n_episodes):
        traj = rollout(model, env, deterministic=True, seed=seed + ep)
        total_r = float(traj["reward"].sum())
        trajectories.append(traj)
        total_rewards.append(total_r)
        solved = total_r >= 200
        print(
            f"  Episode {ep + 1:2d}: total reward = {total_r:+7.1f}"
            f"  {'Solved' if solved else ''}"
        )

    env.close()

    mean_r = np.mean(total_rewards)
    std_r = np.std(total_rewards)
    n_solved = np.sum(np.array(total_rewards) >= 200)
    print(f"\n  Mean reward  : {mean_r:+.2f} ± {std_r:.2f}")
    print(f"  Solved (≥200): {n_solved}/{n_episodes}")

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_episodes))
    fps = 50.0

    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(
        "LunarLander: SAC trained on TRUE environment\n"
        "Evaluated on TRUE environment",
        fontsize=12,
    )

    for traj, color in zip(trajectories, colors):
        t_axis = np.arange(len(traj["x"])) / fps
        axes[0].plot(t_axis, traj["x"], color=color, alpha=0.8)
        axes[0].plot(t_axis, traj["y"], color=color, alpha=0.8, linestyle="--")
        axes[1].plot(t_axis, traj["vx"], color=color, alpha=0.8)
        axes[1].plot(t_axis, traj["vy"], color=color, alpha=0.8, linestyle="--")
        axes[2].plot(t_axis, np.degrees(traj["angle"]), color=color, alpha=0.8)
        axes[3].plot(t_axis, traj["main_engine"], color=color, alpha=0.8)
        axes[4].plot(t_axis, traj["lateral_engine"], color=color, alpha=0.8)
        axes[5].plot(t_axis, traj["reward"], color=color, alpha=0.8)

    axes[0].axhline(0, color="k", linestyle="--", lw=0.8, label="Landing pad (x=0, y=0)")
    axes[2].axhline(0, color="k", linestyle="--", lw=0.8, label="Level (0°)")

    axes[0].set_ylabel("Position")
    axes[0].set_title("Position  [solid=x, dashed=y]")
    axes[1].set_ylabel("Velocity")
    axes[1].set_title("Velocity  [solid=vx, dashed=vy]")
    axes[2].set_ylabel("Angle (deg)")
    axes[2].set_title("Lander Angle")
    axes[3].set_ylabel("Main engine")
    axes[3].set_title("Main Engine Throttle")
    axes[4].set_ylabel("Lateral")
    axes[4].set_title("Lateral Engine")
    axes[5].set_ylabel("Reward")
    axes[5].set_title("Instantaneous Reward")
    axes[5].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path1 = os.path.join(save_dir, "eval_trajectories.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  Trajectory plot saved to {path1}")
    plt.close()

    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        "2D Flight Path: x vs y\n"
        "SAC trained on TRUE env — evaluated on TRUE env",
        fontsize=10,
    )

    for traj, color in zip(trajectories, colors):
        ax.plot(traj["x"], traj["y"], color=color, alpha=0.8)
        ax.scatter(traj["x"][0], traj["y"][0], color=color, marker="o", s=60, zorder=5)
        ax.scatter(traj["x"][-1], traj["y"][-1], color=color, marker="*", s=120, zorder=5)

    ax.axvline(0, color="k", linestyle="--", lw=0.8, label="Landing pad")
    ax.axhline(0, color="gray", linestyle=":", lw=0.8, label="Ground (y=0)")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "eval_flight_paths.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  Flight path plot saved to {path2}")
    plt.close()

    return trajectories, np.array(total_rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"SEED: {SEED}")
    model = train(
        save_dir=SAVE_DIR,
        seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS,
    )
    evaluate(
        model,
        n_episodes=N_EVAL_EPISODES,
        seed=SEED,
        save_dir=SAVE_DIR,
    )
    print("Done.")
