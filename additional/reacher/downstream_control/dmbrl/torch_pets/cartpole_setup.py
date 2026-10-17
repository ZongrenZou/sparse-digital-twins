"""Cartpole-specific preprocessing and costs (torch + numpy) for PETS."""

from __future__ import annotations

import numpy as np
import torch

PENDULUM_LENGTH = 0.6


def obs_preproc_numpy(obs: np.ndarray) -> np.ndarray:
    """obs: (..., 4) -> (..., 5) sin/cos + cart + velocities."""
    return np.concatenate(
        [
            np.sin(obs[..., 1:2]),
            np.cos(obs[..., 1:2]),
            obs[..., :1],
            obs[..., 2:],
        ],
        axis=-1,
    )


def obs_preproc_torch(obs: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.sin(obs[..., 1:2]),
            torch.cos(obs[..., 1:2]),
            obs[..., :1],
            obs[..., 2:],
        ],
        dim=-1,
    )


def build_model_input(obs: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
    """obs (..., 4), ac (..., 1) -> (..., 6)."""
    return torch.cat([obs_preproc_torch(obs), ac], dim=-1)


def targ_proc_numpy(obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    return next_obs - obs


def obs_cost_torch(obs: torch.Tensor) -> torch.Tensor:
    """Negative shaped reward (minimize). obs: (N, 4)."""
    l = PENDULUM_LENGTH
    ee = torch.stack(
        [obs[:, 0] - l * torch.sin(obs[:, 1]), -l * torch.cos(obs[:, 1])], dim=1
    )
    target = obs.new_tensor([0.0, l])
    return -torch.exp(-torch.sum((ee - target) ** 2, dim=1) / (l**2))


def ac_cost_torch(acs: torch.Tensor) -> torch.Tensor:
    """acs: (N, 1)."""
    return 0.01 * torch.sum(acs**2, dim=1)
