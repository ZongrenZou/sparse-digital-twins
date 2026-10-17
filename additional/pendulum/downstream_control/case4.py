"""
Flow matching surrogate environment for the stochastic pendulum with missing theta_dot.

The surrogate environment replaces the true physics with the trained flow
matching model directly — no GP surrogate layer on top.

At each step:
    1. Construct x = [sin(theta_t), sin(theta_{t-1}),
                       cos(theta_t), cos(theta_{t-1}),
                       u_t,          u_{t-1}]   shape (6,)
    2. Sample z0 ~ N(0, 1)
    3. Integrate ODE: dz/dt = v_theta(t, z, x)  from t=0 to t=1
    4. delta_theta = z1 * y_sd + y_mu
    5. theta_{t+1} = theta_t + delta_theta

Agent observation (W=3, 9-dim):
    [cos(theta_t), sin(theta_t), u_{t-1},
     cos(theta_{t-1}), sin(theta_{t-1}), u_{t-2},
     cos(theta_{t-2}), sin(theta_{t-2}), u_{t-3}]

Reward B: -(theta_norm^2 + 0.001 * u^2)

Requirements:
    pip install gymnasium stable-baselines3 numpy scipy matplotlib jax diffrax flax
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

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
from ckpt_io import load_model
import models as fm_models

TRUE_WARMUP_STEPS = 10


def _generate_true_zero_control_history(
    rng: np.random.Generator,
    *,
    g: float = 9.81,
    m: float = 1.0,
    l: float = 1.0,
    max_speed: float = 8.0,
    dt: float = 0.05,
    sigma: float = 0.3,
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


# ── 1. Flow matching surrogate ────────────────────────────────────────────────


class FMSurrogate:
    """
    Wraps the trained flow matching model for use as a simulator.

    Sampling: given conditioning vector x (normalized),
        1. draw z0 ~ N(0, 1)
        2. integrate v_theta(t, z, x) from t=0 to t=1 via Tsit5
        3. return z1 * y_sd + y_mu  as delta_theta

    All inference is JIT-compiled via JAX/diffrax.
    """

    def __init__(
        self,
        ckpt_dir: str,
        key_seed: int = 0,
    ):
        # Load params, model architecture, and normalisation stats
        self.params, self.model, extras = load_model(ckpt_dir, fm_models.VelocityMLP)
        self.x_mu = extras["x_mu"]
        self.x_sd = extras["x_sd"]
        self.y_mu = float(extras["y_mu"].squeeze())
        self.y_sd = float(extras["y_sd"].squeeze())

        # Build ODE term once
        term = dfx.ODETerm(lambda t, y, args: self._velocity(t, y, args))

        # JIT-compile the single-trajectory solver
        @jax.jit
        def _solve(z0, x_norm):
            sol = dfx.diffeqsolve(
                term,
                dfx.Tsit5(),
                t0=0.0,
                t1=1.0,
                dt0=None,
                y0=z0,
                args=x_norm,
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
                max_steps=1_000_000,
            )
            return sol.ys[0]  # shape (1,) -> delta_theta (normalised)

        self._solve = _solve

        # Warm up JIT
        print("FMSurrogate: warming up JIT ...")
        dummy_z = jnp.zeros((1,))
        dummy_x = jnp.zeros((3,))
        _ = self._solve(dummy_z, dummy_x)
        print("FMSurrogate: JIT ready.")

        # JAX RNG key
        self.jax_key = jr.PRNGKey(key_seed)

    def _velocity(self, t, z, x_norm):
        """Velocity field: calls the flow matching MLP."""
        out = self.model.apply({"params": self.params}, t, z, x_norm)
        return out.reshape(-1)  # always (d,) = (1,)

    def sample(self, x: np.ndarray) -> float:
        """
        Sample one delta_theta given raw (unnormalized) input x of shape (6,).

        Returns delta_theta as a scalar float.
        """
        # Normalize input
        x_norm = jnp.array((x - self.x_mu) / self.x_sd)  # (6,)

        # Sample z0 ~ N(0, 1)
        self.jax_key, subkey = jr.split(self.jax_key)
        z0 = jr.normal(subkey, shape=(1,))  # (1,)

        # Integrate ODE
        z1 = self._solve(z0, x_norm)  # (1,)

        # Denormalize
        delta_theta = float(z1[0]) * self.y_sd + self.y_mu
        return delta_theta

    def copy_with_seed(self, key_seed: int | None):
        """Create an independent surrogate instance with its own RNG state."""
        copied = object.__new__(FMSurrogate)
        copied.params = self.params
        copied.model = self.model
        copied.x_mu = self.x_mu
        copied.x_sd = self.x_sd
        copied.y_mu = self.y_mu
        copied.y_sd = self.y_sd
        copied._solve = self._solve
        copied.jax_key = jr.PRNGKey(0 if key_seed is None else key_seed)
        return copied

    def reset_rng(self, key_seed: int | None):
        self.jax_key = jr.PRNGKey(0 if key_seed is None else key_seed)


def load_fm_surrogate(ckpt_dir: str) -> FMSurrogate:
    """Load flow matching surrogate from checkpoint directory."""
    surrogate = FMSurrogate(ckpt_dir=ckpt_dir)
    print(f"FMSurrogate loaded from {ckpt_dir}")
    print(f"  x_mu shape : {surrogate.x_mu.shape}")
    print(f"  y_mu, y_sd : {surrogate.y_mu:.4f}, {surrogate.y_sd:.4f}")
    return surrogate


# ── 2. FM surrogate environment  (W=3, Reward B) ─────────────────────────────


class FMSurrogateEnv(gym.Env):
    """
    Flow matching surrogate environment with W=3 observation window and reward B.

    Internal state: rolling buffer of last 3 (cos, sin, u) frames.
    Observation (W=3, 9-dim):
        [cos(theta_t),   sin(theta_t),   u_{t-1},
         cos(theta_{t-1}), sin(theta_{t-1}), u_{t-2},
         cos(theta_{t-2}), sin(theta_{t-2}), u_{t-3}]

    Dynamics input to GP surrogate (6-dim, W=2):
        [sin(theta_t), sin(theta_{t-1}),
         cos(theta_t), cos(theta_{t-1}),
         u_t,          u_{t-1}]

    Reward B (angle-only, fully observable):
        -(theta_norm^2 + 0.001 * u^2)

    Note: the dynamics model uses W=2 memory while the agent policy
    uses W=3 — these are independent design choices.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        surrogate: FMSurrogate,
        max_torque: float = 2.0,
        seed: int = None,
    ):
        super().__init__()
        self.surrogate = surrogate.copy_with_seed(seed)
        self.max_torque = max_torque
        self.rng = np.random.default_rng(seed)

        # W=3 observation: 3 frames × (cos, sin, u) = 9-dim
        self.W = 3
        frame_low = np.array([-1.0, -1.0, -max_torque], dtype=np.float32)
        frame_high = np.array([1.0, 1.0, max_torque], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.tile(frame_low, self.W),
            high=np.tile(frame_high, self.W),
        )
        self.action_space = spaces.Box(
            low=np.array([-max_torque], dtype=np.float32),
            high=np.array([max_torque], dtype=np.float32),
        )

        self._buffer = None  # deque of (cos, sin, last_u) frames, maxlen=W (agent obs)
        self._dbuffer = (
            None  # deque of (cos, sin, last_u) frames, maxlen=1 (dynamics input)
        )

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.surrogate.reset_rng(seed)

        dyn_len = 1
        warmup_steps = max(TRUE_WARMUP_STEPS, dyn_len, self.W)
        _, _, frames = _generate_true_zero_control_history(
            self.rng,
            steps=warmup_steps,
        )
        self._buffer = deque(frames[-self.W :], maxlen=self.W)
        self._dbuffer = deque(frames[-dyn_len:], maxlen=dyn_len)

        return self._get_obs(), {}

    def step(self, action):
        u_t = float(np.clip(action, -self.max_torque, self.max_torque))

        # W=1 dynamics input from _dbuffer
        cos_t, sin_t, _ = self._dbuffer[-1]

        # FM surrogate input (W=1, 3-dim): [sin_t, cos_t, u_t]
        x = np.array(
            [
                sin_t,
                cos_t,
                u_t,
            ],
            dtype=np.float64,
        )

        # Sample delta_theta from surrogate
        delta_theta = self.surrogate.sample(x)

        # Update theta
        theta_t = np.arctan2(sin_t, cos_t)
        theta_new = theta_t + delta_theta

        new_frame = np.array(
            [np.cos(theta_new), np.sin(theta_new), u_t], dtype=np.float32
        )

        # Update both buffers
        self._buffer.append(new_frame.copy())  # W=3 agent obs buffer
        self._dbuffer.append(new_frame.copy())  # W=1 dynamics buffer

        # Reward B: angle-only, fully observable
        theta_norm = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
        reward = -(theta_norm**2 + 0.001 * u_t**2)

        return self._get_obs(), float(reward), False, False, {}

    def _get_obs(self):
        # Most recent frame first
        frames = list(reversed(self._buffer))
        return np.concatenate(frames).astype(np.float32)

    def render(self):
        pass


# ── 3. True environment for evaluation  (W=3, Reward B) ──────────────────────


class TruePartialObsPendulum(gym.Env):
    """
    True stochastic pendulum with W=3 partial observation and reward B.

    Used for evaluation — same observation space as FMSurrogateEnv
    so the trained policy transfers directly.

    Observation (W=3, 9-dim):
        [cos(theta_t),   sin(theta_t),   u_{t-1},
         cos(theta_{t-1}), sin(theta_{t-1}), u_{t-2},
         cos(theta_{t-2}), sin(theta_{t-2}), u_{t-3}]

    Reward B: -(theta_norm^2 + 0.001 * u^2)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        g: float = 9.81,
        m: float = 1.0,
        l: float = 1.0,
        max_torque: float = 2.0,
        max_speed: float = 8.0,
        dt: float = 0.05,
        sigma: float = 0.3,
    ):
        super().__init__()
        self.g = g
        self.m = m
        self.l = l
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.dt = dt
        self.sigma = sigma

        self.W = 3
        frame_low = np.array([-1.0, -1.0, -max_torque], dtype=np.float32)
        frame_high = np.array([1.0, 1.0, max_torque], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.tile(frame_low, self.W),
            high=np.tile(frame_high, self.W),
        )
        self.action_space = spaces.Box(
            low=np.array([-max_torque], dtype=np.float32),
            high=np.array([max_torque], dtype=np.float32),
        )

        self.theta = None
        self.theta_dot = None
        self._buffer = None
        self.rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        warmup_steps = max(TRUE_WARMUP_STEPS, self.W)
        self.theta, self.theta_dot, frames = _generate_true_zero_control_history(
            self.rng,
            g=self.g,
            m=self.m,
            l=self.l,
            max_speed=self.max_speed,
            dt=self.dt,
            sigma=self.sigma,
            steps=warmup_steps,
        )
        self._buffer = deque(frames[-self.W :], maxlen=self.W)
        return self._get_obs(), {}

    def step(self, action):
        u = float(np.clip(action, -self.max_torque, self.max_torque))
        w_t = self.rng.normal(0.0, self.sigma)

        alpha = (3.0 * self.g) / (2.0 * self.l) * (1.0 + w_t) * np.sin(self.theta) + (
            3.0 / (self.m * self.l**2)
        ) * u
        theta_dot_new = np.clip(
            self.theta_dot + alpha * self.dt, -self.max_speed, self.max_speed
        )
        theta_new = self.theta + theta_dot_new * self.dt

        self.theta = theta_new
        self.theta_dot = theta_dot_new

        self._buffer.append(
            np.array([np.cos(theta_new), np.sin(theta_new), u], dtype=np.float32)
        )

        theta_norm = ((theta_new + np.pi) % (2 * np.pi)) - np.pi
        reward = -(theta_norm**2 + 0.001 * u**2)

        return self._get_obs(), float(reward), False, False, {}

    def _get_obs(self):
        frames = list(reversed(self._buffer))
        return np.concatenate(frames).astype(np.float32)

    def render(self):
        pass


# ── 4. Training ───────────────────────────────────────────────────────────────


def train(
    surrogate: FMSurrogate,
    max_torque: float = 2.0,
    total_timesteps: int = 200_000,
    seed: int = 0,
    save_dir: str = "./logs/gp_surrogate",
):
    os.makedirs(save_dir, exist_ok=True)

    def make_surrogate_env():
        env = FMSurrogateEnv(surrogate=surrogate, max_torque=max_torque)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_surrogate_env, n_envs=4, seed=seed)

    eval_env = Monitor(
        gym.wrappers.TimeLimit(
            FMSurrogateEnv(surrogate=surrogate, max_torque=max_torque),
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
    )

    print(f"\nTraining SAC in FM surrogate environment ...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(f"{save_dir}/final_model")
    print(f"Model saved to {save_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model


# ── 5. Rollout ────────────────────────────────────────────────────────────────


def rollout(model, env, deterministic=True):
    """Run one episode and record trajectory."""
    obs, _ = env.reset()
    done = False

    thetas, actions, rewards = [], [], []

    while not done:
        # Recover theta from cos/sin
        thetas.append(np.arctan2(obs[1], obs[0]))
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(float(action))
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    return {
        "theta": np.array(thetas),
        "action": np.array(actions),
        "reward": np.array(rewards),
    }


# ── 6. Evaluation ─────────────────────────────────────────────────────────────


def evaluate(
    model,
    sigma: float = 0.3,
    max_torque: float = 2.0,
    n_episodes: int = 10,
    save_dir: str = "./logs/gp_surrogate",
):
    """Evaluate the policy on the TRUE stochastic environment."""
    os.makedirs(save_dir, exist_ok=True)

    true_env = gym.wrappers.TimeLimit(
        TruePartialObsPendulum(sigma=sigma, max_torque=max_torque),
        max_episode_steps=200,
    )

    trajectories, total_rewards = [], []

    print("\nEvaluating on TRUE environment ...")
    for ep in range(n_episodes):
        traj = rollout(model, true_env, deterministic=True)
        total_r = traj["reward"].sum()
        total_rewards.append(total_r)
        trajectories.append(traj)
        print(f"  Episode {ep+1:2d}: total reward = {total_r:.2f}")

    true_env.close()
    print(f"\n  Mean: {np.mean(total_rewards):.2f} " f"± {np.std(total_rewards):.2f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"Policy evaluation on TRUE environment\n"
        f"(trained in FM surrogate, W=3, reward B, σ={sigma})",
        fontsize=12,
    )
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_episodes))

    for traj, color in zip(trajectories, colors):
        T = len(traj["theta"])
        t = np.arange(T) * 0.05
        axes[0].plot(t, np.degrees(traj["theta"]), color=color, alpha=0.8)
        axes[1].plot(t, traj["action"], color=color, alpha=0.8)
        axes[2].plot(t, traj["reward"], color=color, alpha=0.8)

    axes[0].axhline(0, color="k", linestyle="--", lw=0.8, label="Upright (0°)")
    axes[1].axhline(max_torque, color="gray", linestyle=":", lw=0.8)
    axes[1].axhline(-max_torque, color="gray", linestyle=":", lw=0.8)

    axes[0].set_ylabel("θ (degrees)")
    axes[0].set_title("Angle  [0° = upright]")
    axes[1].set_ylabel("Torque (N·m)")
    axes[1].set_title("Control input")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Instantaneous reward")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/eval_true_env.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_dir}/eval_true_env.png")
    plt.show()

    return trajectories


# ── 7. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    surrogate = load_fm_surrogate(
        ckpt_dir="./checkpoints/cfm_pruned_3",
    )

    # Train SAC in FM surrogate environment (W=3 agent obs, W=1 dynamics)
    model = train(
        surrogate=surrogate,
        max_torque=2.0,
        total_timesteps=200_000,
        seed=42,
        save_dir="./logs/fm_surrogate_w1",
    )

    # Evaluate on true environment
    evaluate(
        model=model,
        sigma=0.3,
        max_torque=2.0,
        n_episodes=10,
        save_dir="./logs/fm_surrogate_w1",
    )

    print("End main.")
