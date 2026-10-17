#!/usr/bin/env python3
"""
Run PETS on MBRLReacher3D-v0 (7-DoF reacher) with PyTorch + Gymnasium.

Example:
  python scripts/run_pets_reacher_torch.py --logdir log --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=str, default="log", help="Parent log directory")
    parser.add_argument("--ntrain-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--popsize",
        type=int,
        default=400,
        help="CEM population (paper-style default 400)",
    )
    parser.add_argument(
        "--cem-max-iters",
        type=int,
        default=5,
        help="CEM outer iterations per replan",
    )
    args = parser.parse_args()

    import gymnasium as gym
    import numpy as np
    import torch

    import dmbrl.env as env
    from dmbrl.torch_pets.ensemble import ProbabilisticEnsemble
    from dmbrl.torch_pets.experiment import run_mb_experiment
    from dmbrl.torch_pets.mpc import TorchMPC
    from dmbrl.torch_pets.reacher_setup import (
        HIDDEN_DIMS,
        LEARNING_RATE,
        MODEL_IN,
        MODEL_OUT,
        WEIGHT_DECAYS,
        ac_cost_torch,
        build_model_input,
        make_obs_cost_torch,
        obs_preproc_numpy,
        targ_proc_numpy,
    )

    device = torch.device(args.device)

    env = gym.make("MBRLReacher3D-v0")
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    plan_hor = 25
    npart = 20
    num_nets = 5

    ensemble = ProbabilisticEnsemble(
        MODEL_IN,
        MODEL_OUT,
        num_networks=num_nets,
        device=device,
        learning_rate=LEARNING_RATE,
        hidden_dims=HIDDEN_DIMS,
        weight_decays=WEIGHT_DECAYS,
    )

    obs_cost_torch = make_obs_cost_torch(env.unwrapped)

    def obs_postproc(o, d):
        return o + d

    def obs_postproc2(o):
        return o

    mpc = TorchMPC(
        env,
        ensemble,
        plan_hor=plan_hor,
        per=1,
        npart=npart,
        prop_mode="TSinf",
        build_model_input=build_model_input,
        obs_postproc=obs_postproc,
        obs_postproc2=obs_postproc2,
        obs_cost_torch=obs_cost_torch,
        ac_cost_torch=ac_cost_torch,
        obs_preproc_numpy=obs_preproc_numpy,
        targ_proc_numpy=targ_proc_numpy,
        cem_cfg={
            "popsize": args.popsize,
            "num_elites": max(2, args.popsize // 10),
            "max_iters": args.cem_max_iters,
            "alpha": 0.1,
            "epsilon": 0.001,
        },
        device=device,
    )

    class Policy:
        def train(self, obs_trajs, acs_trajs, rews_trajs):
            mpc.train(obs_trajs, acs_trajs, rews_trajs)

        def reset(self):
            mpc.reset()

        def act(self, obs, t):
            return mpc.act(np.asarray(obs), t)

        def dump_logs(self, primary_logdir, iter_logdir):
            mpc.dump_logs(primary_logdir, iter_logdir)

    run_mb_experiment(
        env,
        Policy(),
        task_horizon=150,
        ntrain_iters=args.ntrain_iters,
        nrollouts_per_iter=1,
        ninit_rollouts=1,
        logdir=args.logdir,
        nrecord=0,
        neval=1,
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
