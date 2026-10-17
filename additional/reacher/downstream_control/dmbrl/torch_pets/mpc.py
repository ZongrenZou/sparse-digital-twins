"""Model predictive control with PE dynamics and TS∞ propagation (PyTorch)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from dmbrl.torch_pets.cem import CEMOptimizer
from dmbrl.torch_pets.ensemble import ProbabilisticEnsemble

# Match dmbrl `targ_proc` default used when predicting state deltas (e.g. cartpole, pusher).
def default_targ_proc_numpy(obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    """Training targets for each transition: ``next_obs - obs`` (state residual)."""
    return next_obs - obs


class TorchMPC:
    """PETS-style MPC: CEM over action sequences, TS∞ particle rollout, Gaussian dynamics."""

    def __init__(
        self,
        env,
        ensemble: ProbabilisticEnsemble,
        *,
        plan_hor: int,
        per: int,
        npart: int,
        prop_mode: str,
        build_model_input: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        obs_postproc: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        obs_postproc2: Callable[[torch.Tensor], torch.Tensor],
        obs_cost_torch: Callable[[torch.Tensor], torch.Tensor],
        ac_cost_torch: Callable[[torch.Tensor], torch.Tensor],
        obs_preproc_numpy: Callable[[np.ndarray], np.ndarray],
        targ_proc_numpy: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cem_cfg: dict,
        device: torch.device | None = None,
    ):
        """Build MPC.

        ``targ_proc_numpy(obs, next_obs)`` returns one training target row per transition
        (same leading shape as ``obs`` / ``next_obs``). It must stay consistent with
        ``obs_postproc`` / rollout dynamics (e.g. delta targets with ``obs + pred``).
        Use :func:`default_targ_proc_numpy` for the usual ``next_obs - obs`` residual.
        """
        self.env = env
        self.model = ensemble
        self.dO = int(env.observation_space.shape[0])
        self.dU = int(env.action_space.shape[0])
        self.plan_hor = plan_hor
        self.per = per
        self.npart = npart
        self.prop_mode = prop_mode
        self.build_model_input = build_model_input
        self.obs_postproc = obs_postproc
        self.obs_postproc2 = obs_postproc2
        self.obs_cost_torch = obs_cost_torch
        self.ac_cost_torch = ac_cost_torch
        self._obs_preproc_np = obs_preproc_numpy
        self._targ_proc_np = targ_proc_numpy
        self.device = device or torch.device("cpu")

        self.ac_ub = torch.as_tensor(
            env.action_space.high, dtype=torch.float32, device=self.device
        )
        self.ac_lb = torch.as_tensor(
            env.action_space.low, dtype=torch.float32, device=self.device
        )

        if prop_mode != "TSinf":
            raise NotImplementedError("PyTorch MPC supports prop_mode=TSinf only.")

        num_nets = self.model.num_networks
        if npart % num_nets != 0:
            raise ValueError("npart must be divisible by ensemble size.")
        self._group = npart // num_nets

        sol_dim = plan_hor * self.dU
        opt = cem_cfg
        self.optimizer = CEMOptimizer(
            sol_dim=sol_dim,
            max_iters=opt["max_iters"],
            popsize=opt["popsize"],
            num_elites=opt["num_elites"],
            upper_bound=np.tile(self.ac_ub.cpu().numpy(), plan_hor),
            lower_bound=np.tile(self.ac_lb.cpu().numpy(), plan_hor),
            epsilon=opt.get("epsilon", 0.001),
            alpha=opt.get("alpha", 0.1),
        )

        self.has_been_trained = False
        self.ac_buf = torch.zeros(0, self.dU, device=self.device)
        mid = (self.ac_lb + self.ac_ub) / 2
        self.prev_sol = torch.cat([mid] * plan_hor)
        span = (self.ac_ub - self.ac_lb) ** 2 / 16
        self.init_var = torch.cat([span] * plan_hor)

        self.train_in = np.zeros((0, ensemble.in_dim), dtype=np.float64)
        self.train_targs = np.zeros((0, ensemble.out_dim), dtype=np.float64)

    def train(self, obs_trajs, acs_trajs, rews_trajs) -> None:
        del rews_trajs
        new_in, new_targ = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            o = np.asarray(obs, dtype=np.float64)
            a = np.asarray(acs, dtype=np.float64)
            new_in.append(np.concatenate([self._obs_preproc_np(o[:-1]), a], axis=-1))
            new_targ.append(self._targ_proc_np(o[:-1], o[1:]))
        self.train_in = np.concatenate([self.train_in] + new_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_targ, axis=0)
        self.model.train_from_data(
            self.train_in,
            self.train_targs,
            batch_size=32,
            epochs=5,
            holdout_ratio=0.0,
            hide_progress=True,
        )
        self.has_been_trained = True

    def reset(self) -> None:
        mid = (self.ac_lb + self.ac_ub) / 2
        self.prev_sol = torch.cat([mid] * self.plan_hor)
        self.optimizer.reset()

    @torch.no_grad()
    def act(self, obs: np.ndarray, t: int, get_pred_cost: bool = False):
        del t
        obs = np.asarray(obs, dtype=np.float32)
        if not self.has_been_trained:
            return np.random.uniform(
                self.env.action_space.low,
                self.env.action_space.high,
                size=self.env.action_space.shape,
            ).astype(np.float32)

        if self.ac_buf.shape[0] > 0:
            a = self.ac_buf[0].cpu().numpy()
            self.ac_buf = self.ac_buf[1:]
            return a

        cur = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        def cem_cost(samples: np.ndarray) -> np.ndarray:
            sol = torch.as_tensor(samples, device=self.device, dtype=torch.float32)
            return self._rollout_cost_batch(cur, sol).cpu().numpy()

        self.optimizer.setup(cem_cost)
        sol_t = self.optimizer.obtain_solution_torch(
            self.prev_sol,
            self.init_var,
            lambda sols: self._rollout_cost_batch(cur, sols),
        )
        self.prev_sol = torch.cat(
            [
                sol_t[self.per * self.dU :],
                torch.zeros(self.per * self.dU, device=self.device),
            ]
        )
        self.ac_buf = sol_t[: self.per * self.dU].reshape(-1, self.dU)

        if get_pred_cost:
            c = self._rollout_cost_batch(cur, sol_t.unsqueeze(0))[0]
            a = self.ac_buf[0].cpu().numpy()
            self.ac_buf = self.ac_buf[1:]
            return a, float(c.item())

        a = self.ac_buf[0].cpu().numpy()
        self.ac_buf = self.ac_buf[1:]
        return a

    @torch.no_grad()
    def _rollout_cost_batch(
        self, cur_obs: torch.Tensor, sol_batch: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized imagined rollouts for the full CEM population.

        cur_obs: (dO,) current state. sol_batch: (pop, plan_hor * dU).
        Returns cost per candidate (scalar), shape (pop,), mean over particles then sum over time.
        """
        pop = sol_batch.shape[0]
        obs = cur_obs.unsqueeze(0).unsqueeze(0).expand(pop, self.npart, -1).clone()
        total = torch.zeros(pop, self.npart, device=self.device, dtype=torch.float32)
        ac_all = sol_batch.reshape(pop, self.plan_hor, self.dU)

        for t in range(self.plan_hor):
            ac = ac_all[:, t, :].unsqueeze(1).expand(-1, self.npart, -1)
            next_obs = self._predict_next_batched(obs, ac)
            next_obs = self.obs_postproc2(next_obs)
            c_obs = self.obs_cost_torch(next_obs.reshape(-1, self.dO)).view(pop, self.npart)
            c_ac = self.ac_cost_torch(ac.reshape(-1, self.dU)).view(pop, self.npart)
            step = c_obs + c_ac
            step = torch.nan_to_num(step, nan=1e6, posinf=1e6, neginf=1e6)
            total = total + step
            obs = next_obs

        return total.mean(dim=1)

    @torch.no_grad()
    def _predict_next_batched(
        self, obs: torch.Tensor, ac: torch.Tensor
    ) -> torch.Tensor:
        """TS∞ for many parallel imagined trajectories.

        obs: (pop, npart, dO), ac: (pop, npart, dU) -> next_obs (pop, npart, dO).
        """
        pop, P, _d = obs.shape
        del _d
        B = self.model.num_networks
        g = self._group
        assert P == self.npart and ac.shape == (pop, self.npart, self.dU)

        x_in = self.build_model_input(obs, ac)
        o_ts = x_in.reshape(pop, B, g, -1).permute(1, 0, 2, 3).reshape(B, pop * g, -1)
        mean, var = self.model.forward_all_nets(o_ts)
        delta = mean + torch.randn_like(mean) * torch.sqrt(var)
        delta = delta.reshape(B, pop, g, -1).permute(1, 0, 2, 3).reshape(pop, self.npart, -1)
        return self.obs_postproc(obs, delta)

    @torch.no_grad()
    def _predict_next(self, obs: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        """TS∞: obs (P, dO), ac (P, dU) — single candidate (used if ever needed)."""
        return self._predict_next_batched(obs.unsqueeze(0), ac.unsqueeze(0))[0]

    def dump_logs(self, primary_logdir: str, iter_logdir: str) -> None:
        del iter_logdir
        import os

        path = os.path.join(primary_logdir, "dynamics.pt")
        torch.save(self.model.state_dict(), path)
