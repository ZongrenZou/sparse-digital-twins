"""
SAC on the TRUE stochastic pendulum with partial observation.

The agent never sees theta_dot. It observes a window of past angles
and past actions, and interacts directly with the true environment.

Observation (window size W):
    [cos(theta_t),   sin(theta_t),   u_{t-1},
     cos(theta_{t-1}), sin(theta_{t-1}), u_{t-2},
     ...
     cos(theta_{t-W+1}), sin(theta_{t-W+1}), u_{t-W}]
    dimension: 3*W

Action: u_t in [-max_torque, max_torque]

Two reward variants compared:
    Variant A (standard): -(theta_norm^2 + 0.1*theta_dot^2 + 0.001*u^2)
        Uses the TRUE theta_dot internally — agent gets this reward
        but cannot observe theta_dot directly. The reward signal
        contains information the agent cannot act on.

    Variant B (angle-only): -(theta_norm^2 + 0.001*u^2)
        No theta_dot penalty. Purely angle-based. The agent can
        fully observe everything that drives its reward.

    Variant C (approx velocity): -(theta_norm^2 + 0.1*v_approx^2 + 0.001*u^2)
        Uses finite-difference velocity: v_approx = (theta_t - theta_{t-1}) / dt
        The agent's reward is computable from its own observation.
        Only valid for W >= 2.

Stochastic dynamics:
    w_t ~ N(0, sigma^2)
    alpha = 3g/(2l)*(1+w_t)*sin(theta) + 3/(ml^2)*u
    theta_dot_new = clip(theta_dot + alpha*dt, -max_speed, max_speed)
    theta_new     = theta + theta_dot_new * dt
"""

import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

TRUE_WARMUP_STEPS = 10


def _generate_true_zero_control_history(
    rng: np.random.Generator,
    *,
    g: float,
    m: float,
    l: float,
    max_speed: float,
    dt: float,
    sigma: float,
    steps: int = TRUE_WARMUP_STEPS,
):
    """Sample a latent state, then roll true dynamics forward with u=0."""
    theta = rng.uniform(-np.pi, np.pi)
    theta_dot = rng.uniform(-1.0, 1.0)
    frames = []

    for _ in range(steps):
        w_t = rng.normal(0.0, sigma)
        alpha = (3.0 * g) / (2.0 * l) * (1.0 + w_t) * np.sin(theta)
        theta_dot = np.clip(theta_dot + alpha * dt, -max_speed, max_speed)
        theta = theta + theta_dot * dt
        frames.append(np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32))

    return theta, theta_dot, frames


# ── 1. Partial observation wrapper ────────────────────────────────────────────


class PartialObsStochasticPendulum(gym.Env):
    """
    True stochastic pendulum — agent sees only angle history, not theta_dot.

    Internal state: (theta, theta_dot) — full physics, always correct.
    Observation:    window of (cos theta, sin theta, u) for W past steps.
    Reward:         one of three variants (see module docstring).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        window_size: int = 3,
        reward_variant: str = "A",  # "A", "B", or "C"
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        max_torque: float = 2.0,
        max_speed: float = 8.0,
        dt: float = 0.05,
        sigma: float = 0.3,
    ):
        super().__init__()
        assert reward_variant in (
            "A",
            "B",
            "C",
        ), "reward_variant must be 'A', 'B', or 'C'"
        if reward_variant == "C":
            assert (
                window_size >= 2
            ), "Variant C requires window_size >= 2 (needs theta_{t-1})"

        self.W = window_size
        self.reward_variant = reward_variant
        self.g = g
        self.m = m
        self.l = l
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.dt = dt
        self.sigma = sigma

        # Observation: W frames of (cos_theta, sin_theta, u_{t-1})
        # obs_dim = 3 * window_size
        # self.observation_space = spaces.Box(
        #     low  = -np.ones(obs_dim, dtype=np.float32),
        #     high =  np.ones(obs_dim, dtype=np.float32),
        # )
        frame_low = np.array([-1.0, -1.0, -max_torque], dtype=np.float32)
        frame_high = np.array([1.0, 1.0, max_torque], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.tile(frame_low, window_size),
            high=np.tile(frame_high, window_size),
        )
        self.action_space = spaces.Box(
            low=np.array([-max_torque], dtype=np.float32),
            high=np.array([max_torque], dtype=np.float32),
        )

        # Internal state
        self.theta = None
        self.theta_dot = None
        self._buffer = None  # deque of (cos, sin, last_u) frames
        self._last_u = 0.0
        self.np_random = np.random.default_rng()

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        warmup_steps = max(TRUE_WARMUP_STEPS, self.W)
        self.theta, self.theta_dot, frames = _generate_true_zero_control_history(
            self.np_random,
            g=self.g,
            m=self.m,
            l=self.l,
            max_speed=self.max_speed,
            dt=self.dt,
            sigma=self.sigma,
            steps=warmup_steps,
        )
        self._last_u = 0.0
        self._buffer = deque(frames[-self.W :], maxlen=self.W)

        return self._get_obs(), {}

    def step(self, action):
        u = float(np.clip(action, -self.max_torque, self.max_torque))

        # ── True physics ──────────────────────────────────────────────────────
        w_t = self.np_random.normal(0.0, self.sigma)
        alpha = (3.0 * self.g) / (2.0 * self.l) * (1.0 + w_t) * np.sin(self.theta) + (
            3.0 / (self.m * self.l**2)
        ) * u
        theta_dot_new = np.clip(
            self.theta_dot + alpha * self.dt, -self.max_speed, self.max_speed
        )
        theta_new = self.theta + theta_dot_new * self.dt

        # ── Reward ────────────────────────────────────────────────────────────
        theta_norm = ((theta_new + np.pi) % (2 * np.pi)) - np.pi

        if self.reward_variant == "A":
            # Standard Pendulum-v1 — uses TRUE theta_dot (agent cannot observe)
            reward = -(theta_norm**2 + 0.1 * theta_dot_new**2 + 0.001 * u**2)

        elif self.reward_variant == "B":
            # Angle-only — fully observable by agent
            reward = -(theta_norm**2 + 0.001 * u**2)

        else:  # "C"
            # Approx velocity via finite difference — observable by agent
            theta_prev = np.arctan2(self._buffer[-1][1], self._buffer[-1][0])
            v_approx = (theta_new - theta_prev) / self.dt
            reward = -(theta_norm**2 + 0.1 * v_approx**2 + 0.001 * u**2)

        # ── Update state and buffer ───────────────────────────────────────────
        self.theta = theta_new
        self.theta_dot = theta_dot_new

        # Append new frame: (cos_new, sin_new, u_just_applied)
        self._buffer.append(
            np.array([np.cos(theta_new), np.sin(theta_new), u], dtype=np.float32)
        )
        self._last_u = u

        # Store true theta_dot for external logging
        info = {"true_theta_dot": theta_dot_new, "theta": theta_new}

        return self._get_obs(), float(reward), False, False, info

    def _get_obs(self):
        # Most recent frame first
        frames = list(reversed(self._buffer))
        return np.concatenate(frames).astype(np.float32)

    def render(self):
        pass

    def print_params(self):
        print("=" * 60)
        print("  PartialObsStochasticPendulum")
        print("=" * 60)
        print(f"  window_size    = {self.W}  (obs dim = {3*self.W})")
        print(f"  reward_variant = {self.reward_variant}")
        print(f"  sigma          = {self.sigma}")
        print(f"  max_torque     = {self.max_torque}")
        print(f"  g, m, l        = {self.g}, {self.m}, {self.l}")
        print(f"  dt             = {self.dt}")
        print()
        if self.reward_variant == "A":
            print("  Reward A: -(theta_norm² + 0.1·theta_dot² + 0.001·u²)")
            print("  WARNING: theta_dot in reward is NOT observable by agent")
        elif self.reward_variant == "B":
            print("  Reward B: -(theta_norm² + 0.001·u²)")
            print("  Fully observable by agent")
        else:
            print("  Reward C: -(theta_norm² + 0.1·v_approx² + 0.001·u²)")
            print("  v_approx = (theta_t - theta_{t-1}) / dt")
            print("  Fully observable by agent (requires W >= 2)")
        print("=" * 60)


# ── 2. Training ───────────────────────────────────────────────────────────────


def make_env_fn(
    window_size: int, reward_variant: str, sigma: float, max_torque: float = 2.0
):
    def _make():
        env = PartialObsStochasticPendulum(
            window_size=window_size,
            reward_variant=reward_variant,
            sigma=sigma,
            max_torque=max_torque,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
        env = Monitor(env)
        return env

    return _make


def train(
    window_size: int = 3,
    reward_variant: str = "A",
    sigma: float = 0.3,
    max_torque: float = 2.0,
    total_timesteps: int = 200_000,
    seed: int = 42,
    save_dir: str = None,
):
    tag = f"W{window_size}_R{reward_variant}_s{sigma}"
    if save_dir is None:
        save_dir = f"./logs/partial_obs_true/{tag}"
    os.makedirs(save_dir, exist_ok=True)

    env_preview = PartialObsStochasticPendulum(
        window_size=window_size, reward_variant=reward_variant, sigma=sigma
    )
    env_preview.print_params()

    train_env = make_vec_env(
        make_env_fn(window_size, reward_variant, sigma, max_torque),
        n_envs=4,
        seed=seed,
    )
    eval_env = Monitor(
        gym.wrappers.TimeLimit(
            PartialObsStochasticPendulum(
                window_size=window_size,
                reward_variant=reward_variant,
                sigma=sigma,
                max_torque=max_torque,
            ),
            max_episode_steps=200,
        )
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        seed=seed,
        tensorboard_log=f"./logs/tensorboard/{tag}",
    )

    print(f"\nTraining  W={window_size}  reward={reward_variant}  sigma={sigma} ...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_dir}/final_model")
    print(f"Saved: {save_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model, tag


# ── 3. Rollout ────────────────────────────────────────────────────────────────


def rollout(
    model,
    window_size: int,
    reward_variant: str,
    sigma: float = 0.3,
    max_torque: float = 2.0,
    deterministic: bool = True,
    seed: int = None,
):
    env = gym.wrappers.TimeLimit(
        PartialObsStochasticPendulum(
            window_size=window_size,
            reward_variant=reward_variant,
            sigma=sigma,
            max_torque=max_torque,
        ),
        max_episode_steps=200,
    )
    obs, _ = env.reset(seed=seed)
    done = False

    thetas, theta_dots, actions, rewards = [], [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        thetas.append(info["theta"])
        theta_dots.append(info["true_theta_dot"])
        actions.append(float(action))
        rewards.append(reward)
        done = terminated or truncated

    env.close()
    return {
        "theta": np.array(thetas),
        "theta_dot": np.array(theta_dots),
        "action": np.array(actions),
        "reward": np.array(rewards),
    }


# ── 4. Plotting ───────────────────────────────────────────────────────────────


def plot_comparison(
    results: dict, save_path: str = "./logs/partial_obs_true/comparison.png"
):
    """
    Compare trajectories across (window_size, reward_variant) combinations.

    results: {label_string: list_of_trajectory_dicts}
    """
    labels = list(results.keys())
    n_cases = len(labels)
    colors = plt.cm.tab10(np.linspace(0, 0.9, 5))

    # fig, axes = plt.subplots(4, n_cases, figsize=(6 * n_cases, 13), squeeze=False)
    fig, axes = plt.subplots(4, n_cases, figsize=(10, 8), squeeze=False)
    fig.suptitle(
        "Partial observation pendulum — true env, angle+memory only\n"
        "Agent never sees theta_dot",
        fontsize=12,
    )

    row_labels = [
        "θ (degrees)",
        "θ̇ (rad/s)  [true, not observed]",
        "Torque u (N·m)",
        "Reward",
    ]

    for col, label in enumerate(labels):
        trajs = results[label]
        mean_r = np.mean([t["reward"].sum() for t in trajs])
        axes[0, col].set_title(f"{label}\nmean reward={mean_r:.1f}", fontsize=10)

        for i, traj in enumerate(trajs):
            T = len(traj["theta"])
            t = np.arange(T) * 0.05
            color = colors[i % len(colors)]

            theta_wrap = ((traj["theta"] + np.pi) % (2 * np.pi)) - np.pi
            axes[0, col].plot(t, np.degrees(theta_wrap), color=color, alpha=0.8)
            axes[1, col].plot(t, traj["theta_dot"], color=color, alpha=0.8)
            axes[2, col].plot(t, traj["action"], color=color, alpha=0.8)
            axes[3, col].plot(t, traj["reward"], color=color, alpha=0.8)

        axes[0, col].axhline(0, color="k", linestyle="--", lw=0.8, label="Upright (0°)")
        axes[0, col].axhline(180, color="gray", linestyle=":", lw=0.6)
        axes[0, col].axhline(-180, color="gray", linestyle=":", lw=0.6)
        axes[2, col].axhline(2.0, color="gray", linestyle=":", lw=0.8)
        axes[2, col].axhline(-2.0, color="gray", linestyle=":", lw=0.8)

        for row in range(4):
            axes[row, col].set_ylabel(row_labels[row] if col == 0 else "")
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel("Time (s)" if row == 3 else "")

        axes[0, col].legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_reward_summary(
    results: dict, save_path: str = "./logs/partial_obs_true/reward_summary.png"
):
    labels = list(results.keys())
    means = [np.mean([t["reward"].sum() for t in results[l]]) for l in labels]
    stds = [np.std([t["reward"].sum() for t in results[l]]) for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 4))
    bars = ax.bar(
        range(len(labels)), means, yerr=stds, capsize=5, color="steelblue", alpha=0.8
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Control performance — partial observation (angle only)")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    SIGMA = 0.3
    TIMESTEPS = 200_000
    MAX_TORQUE = 2.0
    SEED = 42
    N_EVAL = 10

    # Experiments: (window_size, reward_variant)
    # W=1, reward A: no memory, standard reward (baseline partial obs)
    # W=3, reward A: memory, standard reward (agent can approx theta_dot)
    # W=3, reward B: memory, angle-only reward (fully observable reward)
    # W=3, reward C: memory, approx-velocity reward (observable, richer signal)
    # experiments = [
    #     (1, "A"),   # no memory, standard reward
    #     (3, "A"),   # memory,    standard reward
    #     (3, "B"),   # memory,    angle-only reward
    #     (3, "C"),   # memory,    approx-velocity reward
    # ]
    experiments = [
        (3, "B"),  # memory,    angle-only reward
    ]

    results = {}

    for W, R in experiments:
        label = f"W={W} R={R}"
        print(f"\n{'='*60}")
        print(f"  Experiment: window={W}  reward={R}")
        print(f"{'='*60}")

        model, tag = train(
            window_size=W,
            reward_variant=R,
            sigma=SIGMA,
            total_timesteps=TIMESTEPS,
            seed=SEED,
            max_torque=MAX_TORQUE,
        )

        print(f"\nEvaluating {label} ...")
        trajs = []
        for ep in range(N_EVAL):
            traj = rollout(
                model,
                window_size=W,
                reward_variant=R,
                max_torque=MAX_TORQUE,
                sigma=SIGMA,
                deterministic=True,
                seed=ep,
            )
            trajs.append(traj)
            theta_final = np.degrees(
                ((traj["theta"][-1] + np.pi) % (2 * np.pi)) - np.pi
            )
            print(
                f"  ep {ep+1}: reward={traj['reward'].sum():.1f}  "
                f"final θ={theta_final:.1f}°"
            )

        results[label] = trajs

    # Plots
    os.makedirs("./logs/partial_obs_true", exist_ok=True)
    plot_comparison(results, save_path="./logs/partial_obs_true/comparison.png")
    plot_reward_summary(results, save_path="./logs/partial_obs_true/reward_summary.png")

    # Summary table
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    for label in results:
        rewards = [t["reward"].sum() for t in results[label]]
        print(
            f"  {label:15s}: mean={np.mean(rewards):7.1f}  " f"± {np.std(rewards):.1f}"
        )
    print("=" * 55)
