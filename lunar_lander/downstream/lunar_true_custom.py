import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    FPS,
    SCALE,
    VIEWPORT_W,
    VIEWPORT_H,
    LEG_DOWN,
    MAIN_ENGINE_POWER,
    SIDE_ENGINE_POWER,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_AWAY,
    MAIN_ENGINE_Y_LOCATION,
)

_ROOT = Path(__file__).resolve().parent

# ── config ────────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 500_000
N_EVAL_EPISODES = 50
SEED = 12345
# SEED = 23451
# SEED = 34512
# SEED = 45123
# SEED = 51234
# SEED = 93891
SAVE_DIR = str(_ROOT / "logs_6" / "sac_true_custom")
MAX_EPISODE_STEPS = 1000
N_EXTRA = 100  # spectator extra state (does not affect Box2D)
N_TOTAL_EXTRA = N_EXTRA + 1  # N_EXTRA spectators + 1 fuel coordinate
FUEL_IDX = N_EXTRA  # extra_state[FUEL_IDX] is remaining fuel in [0, 1]
DT = 1.0 / FPS
# Mean-reversion time (seconds) and process noise for the spectator extras.
# Smaller EXTRA_TAU → faster decay; EXTRA_SIGMA=0 → deterministic extra dynamics.
EXTRA_TAU = 1.0
EXTRA_SIGMA = 0.05
# Weak fuel coupling: empty tank still has (1 - FUEL_THRUST_GAIN) of stock
# main-engine power. Gain 0.15 → thrust in [0.85, 1.0] × MAIN_ENGINE_POWER.
FUEL_THRUST_GAIN = 0.15
# Seconds of continuous m_power=1.0 that empty a full tank.
FUEL_FULL_THROTTLE_SECONDS = 12.0


# ── True environment with an editable step() ─────────────────────────────────

class LunarLanderTrueEnv(LunarLander):
    """
    Continuous LunarLander whose step() lives in this file.

    Observation is [8-d lander, N_EXTRA spectator extras, 1 fuel].
    All N_EXTRA + 1 extras are true state (stored and evolved in step).
    Only fuel weakly scales MAIN_ENGINE_POWER; the spectators do not
    affect Box2D. Reward / termination still use the 8-d lander.
    """

    def __init__(self, render_mode=None):
        super().__init__(continuous=True, render_mode=render_mode)
        # Parent space is Box(8,). SAC sizes the policy from this, so it must
        # match the concatenated [lander, N_EXTRA spectators, fuel] vector.
        extra_low = np.full((N_TOTAL_EXTRA,), -np.inf, dtype=np.float32)
        extra_high = np.full((N_TOTAL_EXTRA,), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([self.observation_space.low, extra_low]),
            high=np.concatenate([self.observation_space.high, extra_high]),
            dtype=np.float32,
        )
        self.extra_state = None  # set in reset(), updated in step()

    def reset(self, *, seed=None, options=None):
        # Parent reset() ends by calling step([0, 0]). Clear extra_state first
        # so that dummy step initializes z_0 rather than evolving a stale z.
        self.extra_state = None
        return super().reset(seed=seed, options=options)

    def _obs(self, lander_state):
        return np.concatenate(
            [
                np.asarray(lander_state, dtype=np.float32),
                np.asarray(self.extra_state, dtype=np.float32),
            ]
        )

    def _init_extra_state(self):
        """Sample z_0: N_EXTRA spectator extras ~ N(0, 1), fuel = 1."""
        z = np.empty(N_TOTAL_EXTRA, dtype=np.float32)
        z[:N_EXTRA] = self.np_random.normal(0.0, 1.0, size=N_EXTRA)
        z[FUEL_IDX] = 1.0
        return z

    def _fuel_scale(self):
        """Weak main-engine multiplier in [1 - GAIN, 1] from remaining fuel."""
        fuel = float(np.clip(self.extra_state[FUEL_IDX], 0.0, 1.0))
        return 1.0 - FUEL_THRUST_GAIN * (1.0 - fuel)

    def _update_extra_state(self, lander_state, action, m_power):
        """
        Extra-state transition for the N_EXTRA spectators and fuel.

        Spectators (z[:N_EXTRA]) are independent Ornstein–Uhlenbeck
        filters of the 8-d lander plus the 2-d action. They do not
        affect Box2D:

            dz = (-z / tau + drive) dt + sigma dW

        Fuel (z[FUEL_IDX]) is remaining main-engine propellant:

            fuel' = clip(fuel - m_power * dt / T_full, 0, 1)
        """
        z = self.extra_state.copy()
        if N_EXTRA > 0:
            rest = z[:N_EXTRA]
            drive = np.concatenate(
                [
                    np.asarray(lander_state, dtype=np.float32),
                    np.asarray(action, dtype=np.float32),
                ]
            )
            n_drive = drive.shape[0]  # 10 = 8 lander + 2 action
            drive_i = drive[np.arange(N_EXTRA) % n_drive]
            noise = self.np_random.normal(0.0, 1.0, size=N_EXTRA).astype(np.float32)
            z[:N_EXTRA] = (
                rest
                + DT * (-rest / EXTRA_TAU + drive_i)
                + EXTRA_SIGMA * np.sqrt(DT) * noise
            )
        z[FUEL_IDX] = np.clip(
            z[FUEL_IDX] - float(m_power) * DT / FUEL_FULL_THROTTLE_SECONDS,
            0.0,
            1.0,
        )
        self.extra_state = z.astype(np.float32)

    def step(self, action):
        """True-dynamics step. Edit this method to customize physics."""
        assert self.lander is not None, "You forgot to call reset()"

        # Wind (disabled by default)
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + math.sin(math.pi * 0.01 * self.wind_idx)
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + math.sin(math.pi * 0.01 * self.torque_idx)
                )
                * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)

        # Fuel is needed before the main-engine impulse. Gymnasium reset()
        # ends in a dummy step([0, 0]); that call only samples z_0.
        extra_just_inited = self.extra_state is None
        if extra_just_inited:
            self.extra_state = self._init_extra_state()
        fuel_scale = self._fuel_scale()

        # ── Engine impulses (this is the block the FM files replace) ──────
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if self.continuous and action[0] > 0.0:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
            main_power = MAIN_ENGINE_POWER * fuel_scale
            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse(
                    (ox * main_power * m_power, oy * main_power * m_power),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * main_power * m_power, -oy * main_power * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if self.continuous and np.abs(action[1]) > 0.5:
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                    impulse_pos,
                    True,
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        # ── Box2D physics step ────────────────────────────────────────────
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # ── State extraction ──────────────────────────────────────────────
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        # ── Extra state ───────────────────────────────────────────────────
        # Dummy reset step already sampled z_0 above. Later steps evolve
        # the spectators and drain fuel from the delivered m_power.
        if not extra_just_inited:
            self._update_extra_state(state, action, m_power)

        # ── Reward ────────────────────────────────────────────────────────
        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03

        # ── Termination ───────────────────────────────────────────────────
        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()

        # Reward / termination still use the 8-d lander. The policy sees
        # the concatenated 8 + N_EXTRA + 1 observation (fuel last).
        return self._obs(state), reward, terminated, False, {}


def make_true_env(render_mode=None):
    env = LunarLanderTrueEnv(render_mode=render_mode)
    return TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)


# ── Training ──────────────────────────────────────────────────────────────────

def train(save_dir: str, seed: int = 42, total_timesteps: int = 500_000) -> SAC:
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        return make_true_env()

    train_env = make_vec_env(make_env, n_envs=1, seed=seed)
    eval_env = Monitor(make_true_env())

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

    print("\nTraining SAC on TRUE LunarLander (customizable step) ...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    elapsed = time.time() - t0
    model.save(os.path.join(save_dir, "final_model"))
    print(f"  Done in {elapsed:.0f}s. Saved to {save_dir}/final_model.zip")

    train_env.close()
    eval_env.close()
    return model


# ── Rollout ───────────────────────────────────────────────────────────────────

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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, n_episodes=20, seed=42, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    env = make_true_env()
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
        "LunarLander: SAC trained on TRUE environment (custom step)\n"
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
        "SAC trained on TRUE env (custom step) — evaluated on TRUE env",
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
    print("Number of extra variables: ", N_TOTAL_EXTRA)
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
