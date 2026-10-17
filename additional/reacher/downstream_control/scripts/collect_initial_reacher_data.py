#!/usr/bin/env python3
"""
Collect an initial random-interaction dataset for MBRLReacher3D-v0.

This script does not train a model or run MPC. It only rolls out the true environment
with random actions and saves the resulting transitions/trajectories.

Example:
  python scripts/collect_initial_reacher_data.py --num-rollouts 20 --horizon 150 --out data/reacher_init.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rollouts", type=int, default=1, help="Number of episodes to collect")
    parser.add_argument("--horizon", type=int, default=150, help="Maximum steps per rollout")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output .npz path. Default: data/reacher_init_<timestamp>.npz",
    )
    args = parser.parse_args()

    import gymnasium as gym
    import numpy as np

    import dmbrl.env  # noqa: F401

    timestamp = time.strftime("%Y-%m-%d--%H:%M:%S", time.localtime())
    out_path = args.out or os.path.join("data", "reacher_init_%s.npz" % timestamp)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    env = gym.make("MBRLReacher3D-v0")

    traj_obs = []
    traj_actions = []
    traj_next_obs = []
    traj_rewards = []
    traj_goals = []

    flat_obs = []
    flat_actions = []
    flat_next_obs = []
    flat_rewards = []
    flat_goals = []
    rollout_lengths = []

    returns = []

    for rollout_idx in range(args.num_rollouts):
        obs, _info = env.reset(seed=args.seed + rollout_idx)
        env.action_space.seed(args.seed + rollout_idx)

        obs_list = []
        ac_list = []
        next_obs_list = []
        reward_list = []
        goal_list = []
        ret = 0.0

        for _ in range(args.horizon):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _info = env.step(action)
            goal = np.asarray(env.unwrapped.goal, dtype=np.float64).copy()

            obs_arr = np.asarray(obs, dtype=np.float64).copy()
            action_arr = np.asarray(action, dtype=np.float64).copy()
            next_obs_arr = np.asarray(next_obs, dtype=np.float64).copy()
            reward_val = float(reward)

            obs_list.append(obs_arr)
            ac_list.append(action_arr)
            next_obs_list.append(next_obs_arr)
            reward_list.append(reward_val)
            goal_list.append(goal)

            flat_obs.append(obs_arr)
            flat_actions.append(action_arr)
            flat_next_obs.append(next_obs_arr)
            flat_rewards.append(reward_val)
            flat_goals.append(goal)

            ret += reward_val
            obs = next_obs

            if terminated or truncated:
                break

        traj_obs.append(np.asarray(obs_list, dtype=np.float64))
        traj_actions.append(np.asarray(ac_list, dtype=np.float64))
        traj_next_obs.append(np.asarray(next_obs_list, dtype=np.float64))
        traj_rewards.append(np.asarray(reward_list, dtype=np.float64))
        traj_goals.append(np.asarray(goal_list, dtype=np.float64))
        rollout_lengths.append(len(ac_list))
        returns.append(ret)

        print(
            "rollout %d/%d: steps=%d return=%.3f"
            % (rollout_idx + 1, args.num_rollouts, len(ac_list), ret)
        )

    np.savez_compressed(
        out_path,
        obs=np.asarray(flat_obs, dtype=np.float64),
        actions=np.asarray(flat_actions, dtype=np.float64),
        next_obs=np.asarray(flat_next_obs, dtype=np.float64),
        rewards=np.asarray(flat_rewards, dtype=np.float64),
        goals=np.asarray(flat_goals, dtype=np.float64),
        deltas=np.asarray(flat_next_obs, dtype=np.float64) - np.asarray(flat_obs, dtype=np.float64),
        traj_obs=np.asarray(traj_obs, dtype=object),
        traj_actions=np.asarray(traj_actions, dtype=object),
        traj_next_obs=np.asarray(traj_next_obs, dtype=object),
        traj_rewards=np.asarray(traj_rewards, dtype=object),
        traj_goals=np.asarray(traj_goals, dtype=object),
        rollout_lengths=np.asarray(rollout_lengths, dtype=np.int64),
        returns=np.asarray(returns, dtype=np.float64),
        metadata_json=json.dumps(
            {
                "env_name": "MBRLReacher3D-v0",
                "num_rollouts": args.num_rollouts,
                "horizon": args.horizon,
                "seed": args.seed,
                "obs_dim": int(env.observation_space.shape[0]),
                "action_dim": int(env.action_space.shape[0]),
            }
        ),
    )

    env.close()

    print("\nSaved dataset to %s" % out_path)
    print("Transitions: %d" % len(flat_actions))
    print("Observation dim: %d" % env.observation_space.shape[0])
    print("Action dim: %d" % env.action_space.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
