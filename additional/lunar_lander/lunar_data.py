"""
Data generation for LunarLanderContinuous-v3 — VELOCITY IMPULSE LEARNING.

Instead of learning the full step transition x_t -> x_{t+1}, we learn only
the velocity changes caused by engine impulses, before world.Step() runs.

This cleanly isolates the stochastic engine physics from the deterministic
kinematics (position/angle integration) and contact physics (Box2D).

Input  (10-dim):
    [x, y, vx, vy, angle, ang_vel, left_leg, right_leg, a1, a2]
    = full state x_t (8-dim) + action a_t (2-dim)
    Leg variables are included so KMD can prune them if they are not ancestors.

Output (3-dim):
    [delta_vx, delta_vy, delta_ang_vel]
    = velocity changes due to engine impulses ONLY, before world.Step()

How it works:
    Inside a subclassed LunarLander.step(), we read linearVelocity and
    angularVelocity immediately before and after ApplyLinearImpulse calls,
    but before world.Step(). The difference is the pure impulse effect.

Output (.mat file):
    inputs       : (N, 10)  [x, y, vx, vy, angle, ang_vel, leg0, leg1, a1, a2]
    delta_vx     : (N,)     vx change from engine impulse
    delta_vy     : (N,)     vy change from engine impulse
    delta_angvel : (N,)     ang_vel change from engine impulse
    states       : (N, 8)   x_t  (for reference)
    actions      : (N, 2)   a_t  (for reference)
    next_states  : (N, 8)   x_{t+1} (for reference)

Requirements:
    pip install gymnasium[box2d] numpy scipy stable-baselines3
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    FPS, SCALE, VIEWPORT_W, VIEWPORT_H,
    MAIN_ENGINE_POWER, SIDE_ENGINE_POWER,
    MAIN_ENGINE_Y_LOCATION, SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY,
)

from Box2D import b2Vec2

# ── config ────────────────────────────────────────────────────────────────────

N_RANDOM  = 50    # trajectories from random policy
MAX_STEPS = 1000
SEED      = 42
SAVE_PATH = "./data/lunar_lander_impulse_data.mat"


# ── Instrumented LunarLander ──────────────────────────────────────────────────

class LunarLanderInstrumented(LunarLander):
    """
    Subclass of LunarLander that records the intermediate velocity changes
    caused by engine impulses, before world.Step() is called.

    After each step(), the following attributes are populated:
        _delta_vx     : float  — vx change from engine impulse
        _delta_vy     : float  — vy change from engine impulse
        _delta_angvel : float  — ang_vel change from engine impulse
        _impulse_recorded : bool — True if impulse was recorded this step
    """

    def __init__(self, render_mode=None):
        super().__init__(continuous=True, render_mode=render_mode)
        self._delta_vx        = 0.0
        self._delta_vy        = 0.0
        self._delta_angvel    = 0.0
        self._impulse_recorded = False

    def step(self, action):
        assert self.lander is not None

        # Wind (disabled by default, but handle anyway)
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

        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE
                      for _ in range(2)]

        # ── Record velocity BEFORE impulse application ────────────────────
        vx_before  = self.lander.linearVelocity.x
        vy_before  = self.lander.linearVelocity.y
        av_before  = self.lander.angularVelocity

        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print("position before: ", self.lander.position)
        # print("angle before: ", self.lander.angle)

        m_power = 0.0
        if self.continuous and action[0] > 0.0:
            # self.lander.position = 3 * self.lander.position
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5

            ox = (
                tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                + side[0] * dispersion[1]
            )
            oy = (
                -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
                - side[1] * dispersion[1]
            )
            impulse_pos = (
                self.lander.position[0] + ox,
                self.lander.position[1] + oy
            )
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0],
                                          impulse_pos[1], m_power)
                p.ApplyLinearImpulse(
                    (ox * MAIN_ENGINE_POWER * m_power,
                     oy * MAIN_ENGINE_POWER * m_power),
                    impulse_pos, True
                )

            print("@@@@@@@@@@@@@@@@@@@@@@@@ before @@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(self.lander.linearVelocity)
            print(self.lander.angularVelocity)
            print(self.lander.position)
            print(self.lander.angle)
            

            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power,
                 -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos, True
            )
            print("@@@@@@@@@@@@@@@@@@@@@@@@ after @@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(self.lander.linearVelocity)
            print(self.lander.angularVelocity)
            print(self.lander.position)
            print(self.lander.angle)

            print("Hey")

            ##### ZZ edits #####
            # _pos = self.lander.position
            # _angle = self.lander.angle
            # _linear_velocity = self.lander.linearVelocity.copy()
            # _angular_velocity = self.lander.angularVelocity
            # # print("postion before: ", _pos)
            # # print("angle before: ", _angle)
            # # print("linear velocity before: ", _linear_velocity)
            # #print("angular velocity before: ", _angular_velocity)

            # self.lander.ApplyLinearImpulse(
            #     (-ox * MAIN_ENGINE_POWER * m_power,
            #      -oy * MAIN_ENGINE_POWER * m_power),
            #     impulse_pos, True
            # )

            # # print("postion after: ", self.lander.position)
            # # print("angle after: ", self.lander.angle)
            # # print("linear velocity after: ", self.lander.linearVelocity)
            # # print("angular velocity after: ", self.lander.angularVelocity)

            # zz_linear_velocity = self.lander.linearVelocity.copy()
            # zz_angular_velocity = self.lander.angularVelocity

            # ##### reset #####
            # # print("Resetting to original values...")
            # # print(_linear_velocity)
            # # print(_angular_velocity)
            # # self.lander.position = _pos
            # # self.lander.angle = _angle
            # self.lander.linearVelocity = _linear_velocity.copy()
            # self.lander.angularVelocity = _angular_velocity

            # ##### we change the impulse position #####
            # # print("Changing impulse position...")
            # # self.lander.position = 3 * self.lander.position
            # _x1 = np.random.normal()
            # _x2 = np.random.normal()
            # print(self.lander.position)
            # self.lander.position = b2Vec2(_x1, _x2)
            # print(self.lander.position)
            # _impulse_pos = (
            #     self.lander.position[0] + ox,
            #     self.lander.position[1] + oy
            # )

            # self.lander.ApplyLinearImpulse(
            #     (-ox * MAIN_ENGINE_POWER * m_power,
            #      -oy * MAIN_ENGINE_POWER * m_power),
            #     _impulse_pos, True
            # )

            # # print("postion after: ", self.lander.position)
            # # print("angle after: ", self.lander.angle)
            # # print("linear velocity after: ", self.lander.linearVelocity)
            # # print("angular velocity after: ", self.lander.angularVelocity)
            # print(self.lander.linearVelocity == zz_linear_velocity)
            # print(self.lander.angularVelocity == zz_angular_velocity)
            # print(zz_angular_velocity)
            # # if self.lander.angularVelocity == zz_angular_velocity:
            # #     print("Trouble")
            # print("difference in linear velocity: ", self.lander.linearVelocity - zz_linear_velocity)
            # print("difference in angular velocity: ", self.lander.angularVelocity - zz_angular_velocity)

            # # print("end edit.")

        s_power = 0.0
        if self.continuous and np.abs(action[1]) > 0.5:
            direction = np.sign(action[1])
            s_power   = np.clip(np.abs(action[1]), 0.5, 1.0)

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
                p = self._create_particle(0.7, impulse_pos[0],
                                          impulse_pos[1], s_power)
                p.ApplyLinearImpulse(
                    (ox * SIDE_ENGINE_POWER * s_power,
                     oy * SIDE_ENGINE_POWER * s_power),
                    impulse_pos, True
                )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power,
                 -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos, True
            )

        # ── Record velocity AFTER impulse, BEFORE world.Step() ───────────
        vx_after = self.lander.linearVelocity.x
        vy_after = self.lander.linearVelocity.y
        av_after = self.lander.angularVelocity

        # print("position after: ", self.lander.position)
        # print("angle after: ", self.lander.angle)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

        # ── Compute impulse deltas ────────────────────────────────────────
        # These are in physical (un-normalized) units.
        # We convert to normalized units to match the observation space.
        self._delta_vx     = (vx_after - vx_before) * _W_HALF / FPS
        self._delta_vy     = (vy_after - vy_before) * _H_HALF / FPS
        self._delta_angvel = (av_after - av_before) * 20.0 / FPS
        self._impulse_recorded = True

        # ── Now run Box2D physics ─────────────────────────────────────────
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # ── State extraction (identical to original) ──────────────────────
        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + 18 / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]

        reward = 0.0
        shaping = (
            -100 * np.sqrt(state[0]**2 + state[1]**2)
            - 100 * np.sqrt(state[2]**2 + state[3]**2)
            - 100 * abs(state[4])
            + 10  * state[6]
            + 10  * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03

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


# ── Normalization constants ───────────────────────────────────────────────────
# Used to convert physical velocities to normalized observation units.
_W_HALF = (VIEWPORT_W / SCALE) / 2   # = 10.0
_H_HALF = (VIEWPORT_H / SCALE) / 2   # = 6.667


# ── Trajectory collector ──────────────────────────────────────────────────────

def collect_trajectories(n_trajectories, max_steps, rng, policy=None,
                         label=""):
    """
    Collect impulse velocity data from n_trajectories episodes.

    Args:
        policy : callable (obs, deterministic) -> (action, _), or None for random
    """
    env = LunarLanderInstrumented()

    all_inputs      = []   # (10,) = [state(8), action(2)]
    all_delta_vx    = []
    all_delta_vy    = []
    all_delta_angvel= []
    all_states      = []   # x_t  for reference
    all_actions     = []   # a_t  for reference
    all_next_states = []   # x_{t+1} for reference

    total_steps    = 0
    total_crashes  = 0
    total_landings = 0

    action_label = "trained SAC" if policy is not None else "random"
    print(f"\nCollecting {n_trajectories} trajectories [{label}]")
    print(f"  Action source: {action_label}")

    for traj_i in range(n_trajectories):
        ep_seed = int(rng.integers(0, int(1e6)))
        obs, _  = env.reset(seed=ep_seed)
        done    = False
        steps   = 0

        t_rewards = []
        t_dones   = []

        while not done and steps < max_steps:
            if policy is None:
                action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            else:
                action, _ = policy(obs, deterministic=False)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

            obs_before = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            # done = terminated or truncated
            done = terminated or truncated or np.abs(obs[0]) >= 0.2 or np.abs(obs[1]) >= 0.1

            # Build 10-dim input: [state(8), action(2)]
            inp = np.concatenate([obs_before, action]).astype(np.float32)

            all_inputs.append(inp)
            all_delta_vx.append(env._delta_vx)
            all_delta_vy.append(env._delta_vy)
            all_delta_angvel.append(env._delta_angvel)
            all_states.append(obs_before)
            all_actions.append(action)
            all_next_states.append(obs.copy())

            t_rewards.append(reward)
            t_dones.append(float(terminated))
            steps += 1

        total_steps += steps
        if len(t_dones) > 0 and t_dones[-1] == 1.0:
            if t_rewards[-1] > 0:
                total_landings += 1
            else:
                total_crashes += 1

        if (traj_i + 1) % 500 == 0:
            print(f"  [{traj_i+1:>5}/{n_trajectories}]  "
                  f"transitions: {total_steps:>8,}  "
                  f"mean ep len: {total_steps/(traj_i+1):.1f}")

    env.close()

    print(f"  Total transitions  : {total_steps:,}")
    print(f"  Mean episode length: {total_steps/n_trajectories:.1f}")
    print(f"  Crashes / Landings : {total_crashes} / {total_landings}")

    return {
        "inputs":       np.array(all_inputs,       dtype=np.float32),
        "delta_vx":     np.array(all_delta_vx,     dtype=np.float32),
        "delta_vy":     np.array(all_delta_vy,     dtype=np.float32),
        "delta_angvel": np.array(all_delta_angvel,  dtype=np.float32),
        "states":       np.array(all_states,        dtype=np.float32),
        "actions":      np.array(all_actions,       dtype=np.float32),
        "next_states":  np.array(all_next_states,   dtype=np.float32),
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def diagnose(inputs, delta_vx, delta_vy, delta_angvel, save_path=None):
    """Print coverage stats and plot distributions."""
    N = len(inputs)
    labels_in = ["x", "y", "vx", "vy", "angle", "ang_vel",
                 "left_leg", "right_leg", "a1", "a2"]

    print("\n" + "=" * 60)
    print("  Dataset summary")
    print("=" * 60)
    print(f"  Total transitions : {N:,}")
    print()
    print(f"  {'Variable':<14} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*52}")
    for i, label in enumerate(labels_in):
        col = inputs[:, i]
        print(f"  {label:<14} {col.min():>8.3f} {col.max():>8.3f} "
              f"{col.mean():>8.3f} {col.std():>8.3f}")
    for label, col in [("delta_vx", delta_vx),
                        ("delta_vy", delta_vy),
                        ("delta_av", delta_angvel)]:
        print(f"  {label:<14} {col.min():>8.4f} {col.max():>8.4f} "
              f"{col.mean():>8.4f} {col.std():>8.4f}")
    print("=" * 60)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("LunarLander Impulse Data Diagnostics", fontsize=14)
    axes = axes.flatten()

    for i, label in enumerate(labels_in):
        axes[i].hist(inputs[:, i], bins=60, color="steelblue",
                     alpha=0.8, edgecolor="none")
        axes[i].set_title(f"Input: {label}")
        axes[i].grid(True, alpha=0.3)

    for i, (label, col) in enumerate([("delta_vx",  delta_vx),
                                       ("delta_vy",  delta_vy),
                                       ("delta_av",  delta_angvel)]):
        axes[10 + i].hist(col, bins=60, color="tomato",
                          alpha=0.8, edgecolor="none")
        axes[10 + i].set_title(f"Output: {label}")
        axes[10 + i].grid(True, alpha=0.3)

    # Coverage: x vs y
    subsample = np.random.default_rng(0).choice(N, size=min(10000, N),
                                                 replace=False)
    axes[13].scatter(inputs[subsample, 0], inputs[subsample, 1],
                     s=0.5, alpha=0.2, color="purple")
    axes[13].axvline(0, color="k", linestyle="--", lw=0.8)
    axes[13].axhline(0, color="gray", linestyle=":", lw=0.8)
    axes[13].set_xlim([-1, 1])
    axes[13].set_ylim([-0.2, 1.5])
    axes[13].set_title("Coverage: x vs y")
    axes[13].set_xlabel("x"); axes[13].set_ylabel("y")
    axes[13].grid(True, alpha=0.3)

    # delta_vx vs angle (should show clear dependence)
    axes[14].scatter(inputs[subsample, 4],
                     delta_vx[subsample],
                     s=0.5, alpha=0.2, color="seagreen")
    axes[14].set_title("delta_vx vs angle")
    axes[14].set_xlabel("angle"); axes[14].set_ylabel("delta_vx")
    axes[14].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Diagnostics plot saved to {save_path}")
    # plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # Collect random-policy data only
    data = collect_trajectories(
        n_trajectories = N_RANDOM,
        max_steps      = MAX_STEPS,
        rng            = rng,
        policy         = None,
        label          = "random policy",
    )

    inputs       = data["inputs"]
    delta_vx     = data["delta_vx"]
    delta_vy     = data["delta_vy"]
    delta_angvel = data["delta_angvel"]
    states       = data["states"]
    actions      = data["actions"]
    next_states  = data["next_states"]

    N = len(inputs)
    print(f"  Total transitions: {N:,}")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)
    savemat(SAVE_PATH, {
        # Primary learning targets
        "inputs":       inputs,        # (N, 10) [state(8), action(2)]
        "delta_vx":     delta_vx,      # (N,)
        "delta_vy":     delta_vy,      # (N,)
        "delta_angvel": delta_angvel,  # (N,)
        # Reference columns
        "states":       states,        # (N, 8)  x_t
        "actions":      actions,       # (N, 2)  a_t
        "next_states":  next_states,   # (N, 8)  x_{t+1}
        # Metadata
        "n_trajectories": np.array([[N_RANDOM]]),
        "total":          np.array([[N]]),
        "seed":           np.array([[SEED]]),
        # Column labels (for reference)
        "input_labels":  np.array(["x", "y", "vx", "vy", "angle",
                                   "ang_vel", "leg0", "leg1", "a1", "a2"]),
        "output_labels": np.array(["delta_vx", "delta_vy", "delta_angvel"]),
    })
    print(f"Data saved to {SAVE_PATH}")

    # Diagnostics
    diag_path = SAVE_PATH.replace(".mat", "_diagnostics.png")
    diagnose(inputs, delta_vx, delta_vy, delta_angvel, save_path=diag_path)

    print("Done.")
