"""Cross-entropy method (numpy only; matches dmbrl.misc.optimizers.cem non-TF path)."""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
import torch


class CEMOptimizer:
    def __init__(
        self,
        sol_dim: int,
        max_iters: int,
        popsize: int,
        num_elites: int,
        upper_bound: np.ndarray,
        lower_bound: np.ndarray,
        epsilon: float = 0.001,
        alpha: float = 0.25,
    ):
        if num_elites > popsize:
            raise ValueError("num_elites must be at most popsize.")
        self.sol_dim = sol_dim
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites
        self.ub = np.asarray(upper_bound, dtype=np.float64)
        self.lb = np.asarray(lower_bound, dtype=np.float64)
        self.epsilon = epsilon
        self.alpha = alpha
        self.cost_function = None

    def setup(self, cost_function):
        self.cost_function = cost_function

    def reset(self) -> None:
        pass

    def obtain_solution(self, init_mean: np.ndarray, init_var: np.ndarray) -> np.ndarray:
        mean = np.asarray(init_mean, dtype=np.float64)
        var = np.asarray(init_var, dtype=np.float64)
        t = 0
        x = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var
            )

            samples = x.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            costs = self.cost_function(samples)
            elites = samples[np.argsort(costs)[: self.num_elites]]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1

        return mean

    def obtain_solution_torch(
        self,
        init_mean: torch.Tensor,
        init_var: torch.Tensor,
        cost_function,
    ) -> torch.Tensor:
        """Torch-native CEM path to avoid NumPy/SciPy/device transfer overhead."""
        mean = init_mean.clone()
        var = init_var.clone()
        lb = torch.as_tensor(self.lb, dtype=mean.dtype, device=mean.device)
        ub = torch.as_tensor(self.ub, dtype=mean.dtype, device=mean.device)

        t = 0
        while (t < self.max_iters) and torch.max(var).item() > self.epsilon:
            lb_dist, ub_dist = mean - lb, ub - mean
            constrained_var = torch.minimum(
                torch.minimum((lb_dist / 2).square(), (ub_dist / 2).square()),
                var,
            )

            noise = torch.empty(
                self.popsize, self.sol_dim, dtype=mean.dtype, device=mean.device
            )
            torch.nn.init.trunc_normal_(noise, mean=0.0, std=1.0, a=-2.0, b=2.0)
            samples = noise * torch.sqrt(constrained_var) + mean
            costs = cost_function(samples)
            elite_idx = torch.topk(costs, k=self.num_elites, largest=False).indices
            elites = samples.index_select(0, elite_idx)

            new_mean = elites.mean(dim=0)
            new_var = elites.var(dim=0, unbiased=False)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1

        return mean
