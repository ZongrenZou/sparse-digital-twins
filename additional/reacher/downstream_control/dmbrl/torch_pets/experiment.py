"""Model-based RL experiment loop (Gymnasium) for PyTorch PETS."""

from __future__ import annotations

import os
from time import localtime, strftime

import numpy as np
from scipy.io import savemat


def run_mb_experiment(
    env,
    policy,
    *,
    task_horizon: int,
    ntrain_iters: int,
    nrollouts_per_iter: int,
    ninit_rollouts: int,
    logdir: str,
    nrecord: int = 0,
    neval: int = 1,
) -> None:
    """Same structure as dmbrl.misc.MBExperiment.run_experiment (simplified)."""
    logdir = os.path.join(
        logdir,
        strftime("%Y-%m-%d--%H:%M:%S", localtime()),
    )
    os.makedirs(logdir, exist_ok=True)

    traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

    samples = []
    for _ in range(ninit_rollouts):
        samples.append(_rollout(env, task_horizon, policy))
        traj_obs.append(samples[-1]["obs"])
        traj_acs.append(samples[-1]["ac"])
        traj_rews.append(samples[-1]["rewards"])

    if ninit_rollouts > 0:
        policy.train(
            [s["obs"] for s in samples],
            [s["ac"] for s in samples],
            [s["rewards"] for s in samples],
        )

    for i in range(ntrain_iters):
        print("####################################################################")
        print("Starting training iteration %d." % (i + 1))

        iter_dir = os.path.join(logdir, "train_iter%d" % (i + 1))
        os.makedirs(iter_dir, exist_ok=True)

        samples = []
        for _ in range(nrecord):
            samples.append(_rollout(env, task_horizon, policy))

        for _ in range(max(neval, nrollouts_per_iter) - nrecord):
            samples.append(_rollout(env, task_horizon, policy))

        print("Rewards obtained:", [s["reward_sum"] for s in samples[:neval]])
        traj_obs.extend([s["obs"] for s in samples[:nrollouts_per_iter]])
        traj_acs.extend([s["ac"] for s in samples[:nrollouts_per_iter]])
        traj_rets.extend([s["reward_sum"] for s in samples[:neval]])
        traj_rews.extend([s["rewards"] for s in samples[:nrollouts_per_iter]])
        samples = samples[:nrollouts_per_iter]

        policy.dump_logs(logdir, iter_dir)
        savemat(
            os.path.join(logdir, "logs.mat"),
            {
                "observations": np.asarray(traj_obs, dtype=object),
                "actions": np.asarray(traj_acs, dtype=object),
                "returns": np.array([traj_rets], dtype=np.float64),
                "rewards": np.asarray(traj_rews, dtype=object),
            },
        )

        if len(os.listdir(iter_dir)) == 0:
            try:
                os.rmdir(iter_dir)
            except OSError:
                pass

        if i < ntrain_iters - 1:
            policy.train(
                [s["obs"] for s in samples],
                [s["ac"] for s in samples],
                [s["rewards"] for s in samples],
            )

    print("Experiment data saved to", logdir)


def _rollout(env, horizon: int, policy):
    obs, _ = env.reset()
    policy.reset()

    obs_list = [obs]
    ac_list = []
    rewards = []
    ret = 0.0
    terminated = truncated = False

    for t in range(horizon):
        a = policy.act(obs, t)
        a = np.asarray(a, dtype=np.float32).reshape(env.action_space.shape)
        obs, r, terminated, truncated, _ = env.step(a)
        obs_list.append(obs)
        ac_list.append(a)
        rewards.append(r)
        ret += float(r)
        if terminated or truncated:
            break

    return {
        "obs": np.array(obs_list),
        "ac": np.array(ac_list),
        "reward_sum": ret,
        "rewards": np.array(rewards),
    }
