import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
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

from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    FPS, SCALE, VIEWPORT_W, VIEWPORT_H, LEG_DOWN,
)

# ── config ────────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 500_000
# TOTAL_TIMESTEPS = 200_000
N_EVAL_EPISODES = 50
SEED = 12345
# SEED = 23451
# SEED = 34512
SAVE_DIR        = f"./logs/fm_pruned"
print(f"SEED: {SEED}")

CKPT = "./checkpoints/cfm_pruned"

# ── normalization constants ───────────────────────────────────────────────────
_W_HALF = (VIEWPORT_W / SCALE) / 2   # = 10.0
_H_HALF = (VIEWPORT_H / SCALE) / 2   # = 6.667


# ── FM surrogate wrapper ──────────────────────────────────────────────────────

class FMSurrogate:
    def __init__(self, ckpt_dir: str, name: str = ""):
        self.name = name
        self.params, self.model, extras = load_model(ckpt_dir, fm_models.VelocityMLP)
        self.x_mu = extras["x_mu"]
        self.x_sd = extras["x_sd"]
        self.y_mu = extras["y_mu"]
        self.y_sd = extras["y_sd"]

        term = dfx.ODETerm(lambda t, y, args: self._velocity(t, y, args))

        @jax.jit
        def _solve(z0, x_norm):
            sol = dfx.diffeqsolve(
                term, dfx.Tsit5(),
                t0=0.0, t1=1.0, dt0=None,
                y0=z0,
                args=x_norm,
                saveat=dfx.SaveAt(t1=True),
                stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
                max_steps=1_000_000,
            )
            return sol.ys[0]

        self._solve = _solve
        print(f"FMSurrogate [{name}]: warming up JIT ...")
        _ = self._solve(jnp.zeros((3,)), jnp.zeros(self.x_mu.shape))
        print(f"FMSurrogate [{name}]: ready.")
        self.jax_key = jr.PRNGKey(0)

    def _velocity(self, t, z, x_norm):
        return self.model.apply({"params": self.params}, t, z, x_norm).reshape(-1)

    def sample(self, x: np.ndarray) -> float:
        x_norm = jnp.array((x - self.x_mu) / self.x_sd)
        self.jax_key, subkey = jr.split(self.jax_key)
        z0 = jr.normal(subkey, shape=(3,))
        z1 = self._solve(z0, x_norm)
        return z1 * self.y_sd + self.y_mu


# ── FM Impulse Surrogate Environment ─────────────────────────────────────────

class LunarLanderFMImpulseEnv(LunarLander):
    """
    Surrogate environment for LunarLanderContinuous-v3.

    The only change from the true env: instead of calling ApplyLinearImpulse
    with true stochastic physics, we directly set the Box2D body velocities
    using FM-predicted deltas.

    Everything else is IDENTICAL to the true env:
        - world.Step() runs normally
        - gravity, contact detection, leg contact flags
        - sleep detection (not lander.awake → +100)
        - game_over detection (body contact → -100)
        - out of bounds termination
        - reward shaping formula
        - state normalization

    FM models take the full 10-dim input matching training data:
        [x, y, vx, vy, angle, ang_vel, leg0, leg1, a1, a2]

    After KMD identifies sparse ancestors, subclass and override
    _cond_vx/_cond_vy/_cond_av to pass only the pruned inputs.
    """

    def __init__(self, fm, render_mode=None):
        super().__init__(continuous=True, render_mode=render_mode)
        self.fm = fm

    def _cond(self, obs_norm, a1, a2):
        return np.concatenate([obs_norm[4:5], [a1, a2]]).astype(np.float64)

    def _get_obs_norm(self):
        """Read current Box2D state and return normalized 8-dim observation."""
        pos = self.lander.position
        vel = self.lander.linearVelocity
        return np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float64)

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        assert self.lander is not None

        # Wind (disabled by default)
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + math.sin(math.pi * 0.01 * self.wind_idx)
                ) * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter((wind_mag, 0.0), True)

            torque_mag = (
                math.tanh(
                    math.sin(0.02 * self.torque_idx)
                    + math.sin(math.pi * 0.01 * self.torque_idx)
                ) * self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(torque_mag, True)

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)

        a1 = float(action[0])
        a2 = float(action[1])

        # ── Read current normalized state x_t for FM conditioning ────────
        obs_norm = self._get_obs_norm()

        # ── FM: sample velocity deltas ────────────────────────────────────
        # Outputs are in normalized observation units:
        # delta_vx_norm = delta_vx_physical * (W/2) / FPS
        # delta_vy_norm = delta_vy_physical * (H/2) / FPS
        # delta_av_norm = delta_av_physical * 20   / FPS
        delta_norm = self.fm.sample(self._cond(obs_norm, a1, a2))
        delta_vx_norm, delta_vy_norm, delta_av_norm = delta_norm

        # ── Convert normalized deltas back to physical units ──────────────
        delta_vx_phys = delta_vx_norm * FPS / _W_HALF
        delta_vy_phys = delta_vy_norm * FPS / _H_HALF
        delta_av_phys = delta_av_norm * FPS / 20.0

        # ── Inject velocity deltas into Box2D body ────────────────────────
        # Read current velocities, add FM-predicted deltas, write back.
        # This replaces ApplyLinearImpulse entirely.
        vx_new = self.lander.linearVelocity.x + delta_vx_phys
        vy_new = self.lander.linearVelocity.y + delta_vy_phys
        av_new = self.lander.angularVelocity   + delta_av_phys

        self.lander.linearVelocity  = (float(vx_new), float(vy_new))
        self.lander.angularVelocity = float(av_new)
        # self.lander.awake           = True

        # ── For reward computation: replicate m_power/s_power ────────────
        # These are needed for fuel cost penalty — same logic as true env.
        m_power = 0.0
        if a1 > 0.0:
            m_power = (np.clip(a1, 0.0, 1.0) + 1.0) * 0.5

        s_power = 0.0
        if np.abs(a2) > 0.5:
            s_power = np.clip(np.abs(a2), 0.5, 1.0)

        # ── Box2D physics step — IDENTICAL to true env ────────────────────
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # ── State extraction — IDENTICAL to true env ──────────────────────
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

        # ── Reward — IDENTICAL to true env ───────────────────────────────
        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10  * state[6]
            + 10  * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03

        # ── Termination — IDENTICAL to true env ──────────────────────────
        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()

        return np.array(state, dtype=np.float32), reward, terminated, False, {}


# ── Training ──────────────────────────────────────────────────────────────────

def train(fm, save_dir, seed=42, total_timesteps=500_000):
    os.makedirs(save_dir, exist_ok=True)

    def make_env():
        env = LunarLanderFMImpulseEnv(fm=fm)
        return TimeLimit(env, max_episode_steps=1000)

    train_env = make_vec_env(make_env, n_envs=1, seed=seed)
    # Evaluate on SURROGATE env during training (for best-model selection)
    eval_env = Monitor(
        TimeLimit(
            LunarLanderFMImpulseEnv(fm=fm),
            max_episode_steps=1000,
        )
    )
    

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

    print("\nTraining SAC in FM Impulse Surrogate environment ...")
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
    done   = False
    states, actions, rewards = [], [], []

    while not done:
        states.append(obs.copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(action.copy())
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    return {
        "x":              np.array([s[0] for s in states]),
        "y":              np.array([s[1] for s in states]),
        "vx":             np.array([s[2] for s in states]),
        "vy":             np.array([s[3] for s in states]),
        "angle":          np.array([s[4] for s in states]),
        "angular_vel":    np.array([s[5] for s in states]),
        "main_engine":    np.array([a[0] for a in actions]),
        "lateral_engine": np.array([a[1] for a in actions]),
        "reward":         np.array(rewards),
    }


# ── Evaluation on TRUE environment ────────────────────────────────────────────

def evaluate(model, n_episodes=20, seed=42, save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make("LunarLander-v3", continuous=True)
    trajectories, total_rewards = [], []

    print(f"\nEvaluating on TRUE environment ({n_episodes} episodes) ...")
    for ep in range(n_episodes):
        traj    = rollout(model, env, deterministic=True, seed=seed + ep)
        total_r = traj["reward"].sum()
        trajectories.append(traj)
        total_rewards.append(total_r)
        solved = total_r >= 200
        print(f"  Episode {ep+1:2d}: total reward = {total_r:+7.1f}"
              f"  {'✓ Solved' if solved else ''}")

    env.close()

    mean_r   = np.mean(total_rewards)
    std_r    = np.std(total_rewards)
    n_solved = np.sum(np.array(total_rewards) >= 200)
    print(f"\n  Mean reward  : {mean_r:+.2f} ± {std_r:.2f}")
    print(f"  Solved (≥200): {n_solved}/{n_episodes}")

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_episodes))

    # ── 6-panel trajectory plot ───────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
    fig.suptitle(
        "LunarLander: SAC trained in FM Impulse Surrogate\n"
        "Evaluated on TRUE environment",
        fontsize=12
    )

    for traj, color in zip(trajectories, colors):
        T = len(traj["x"])
        t = np.arange(T) / 50
        axes[0].plot(t, traj["x"],                 color=color, alpha=0.8)
        axes[0].plot(t, traj["y"],                 color=color, alpha=0.8,
                     linestyle="--")
        axes[1].plot(t, traj["vx"],                color=color, alpha=0.8)
        axes[1].plot(t, traj["vy"],                color=color, alpha=0.8,
                     linestyle="--")
        axes[2].plot(t, np.degrees(traj["angle"]), color=color, alpha=0.8)
        axes[3].plot(t, traj["main_engine"],       color=color, alpha=0.8)
        axes[4].plot(t, traj["lateral_engine"],    color=color, alpha=0.8)
        axes[5].plot(t, traj["reward"],            color=color, alpha=0.8)

    axes[0].axhline(0, color="k", linestyle="--", lw=0.8,
                    label="Landing pad (x=0, y=0)")
    axes[2].axhline(0, color="k", linestyle="--", lw=0.8, label="Level (0°)")

    axes[0].set_ylabel("Position");    axes[0].set_title("Position  [solid=x, dashed=y]")
    axes[1].set_ylabel("Velocity");    axes[1].set_title("Velocity  [solid=vx, dashed=vy]")
    axes[2].set_ylabel("Angle (deg)"); axes[2].set_title("Lander Angle")
    axes[3].set_ylabel("Main engine"); axes[3].set_title("Main Engine Throttle")
    axes[4].set_ylabel("Lateral");     axes[4].set_title("Lateral Engine")
    axes[5].set_ylabel("Reward");      axes[5].set_title("Instantaneous Reward")
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

    # ── 2D flight path plot ───────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(
        "2D Flight Path: x vs y\n"
        "SAC trained in FM Impulse Surrogate — evaluated on TRUE env",
        fontsize=10
    )

    for traj, color in zip(trajectories, colors):
        ax.plot(traj["x"], traj["y"], color=color, alpha=0.8)
        ax.scatter(traj["x"][0],  traj["y"][0],  color=color,
                   marker="o", s=60, zorder=5)
        ax.scatter(traj["x"][-1], traj["y"][-1], color=color,
                   marker="*", s=120, zorder=5)

    ax.axvline(0, color="k",    linestyle="--", lw=0.8, label="Landing pad")
    ax.axhline(0, color="gray", linestyle=":",  lw=0.8, label="Ground (y=0)")
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

    print("Loading FM impulse surrogates ...")
    fm = FMSurrogate(ckpt_dir=CKPT, name="delta_all")

    model = train(
        fm              = fm,
        save_dir        = SAVE_DIR,
        seed            = SEED,
        total_timesteps = TOTAL_TIMESTEPS,
    )

    evaluate(
        model,
        n_episodes = N_EVAL_EPISODES,
        seed       = SEED,
        save_dir   = SAVE_DIR,
    )

    print("Done.")
