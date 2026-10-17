"""7-DoF Reacher — preprocessing, kinematics, and costs (torch + numpy).

This variant supports a pruned input graph for the dynamics model. The model still
predicts the full 17-D observation delta, but it only receives a selected subset of
observation coordinates plus all 7 control coordinates as input.
"""

from __future__ import annotations

import numpy as np
import torch

OBS_DIM = 17
ACTION_DIM = 7
# PRUNED_OBS_COLS_1BASED = (3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16)

# ## Case 1:
# print("########################################")
# print("Case 1")
# PRUNED_OBS_COLS_1BASED = (
#     3, 4, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17
# )
# print(PRUNED_OBS_COLS_1BASED)
# print("########################################")

# # Case 2:
# print("########################################")
# print("Case 2")
# PRUNED_OBS_COLS_1BASED = (
#     3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
# )
# print(PRUNED_OBS_COLS_1BASED)
# print("########################################")

# ## Case 3:
# print("########################################")
# print("Case 3")
# PRUNED_OBS_COLS_1BASED = (
#     3, 4, 6, 8, 9, 10, 13, 14, 15, 16, 17
# )
# print(PRUNED_OBS_COLS_1BASED)
# print("########################################")

# # Case 4:
# print("########################################")
# print("Case 4")
# PRUNED_OBS_COLS_1BASED = (
#     3, 4, 6, 8, 9, 10, 13, 14, 15, 16
# )
# print(PRUNED_OBS_COLS_1BASED)
# print("########################################")

# ## Case 5:
# print("########################################")
# print("Case 5")
# PRUNED_OBS_COLS_1BASED = (
#     3, 4, 6, 8, 9, 10, 12, 13, 14, 15, 16
# )
# print(PRUNED_OBS_COLS_1BASED)
# print("########################################")

## Case 0:
print("########################################")
print("Case 0")
PRUNED_OBS_COLS_1BASED = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
)
print(PRUNED_OBS_COLS_1BASED)
print("########################################")

PRUNED_OBS_COLS = tuple(i - 1 for i in PRUNED_OBS_COLS_1BASED)
MODEL_IN = len(PRUNED_OBS_COLS) + ACTION_DIM
MODEL_OUT = 17


def obs_preproc_numpy(obs: np.ndarray) -> np.ndarray:
    """Keep only the selected observation coordinates for the pruned graph."""
    return obs[..., PRUNED_OBS_COLS]


def obs_preproc_torch(obs: torch.Tensor) -> torch.Tensor:
    idx = torch.as_tensor(PRUNED_OBS_COLS, device=obs.device, dtype=torch.long)
    return obs.index_select(dim=-1, index=idx)


def build_model_input(obs: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
    """obs (..., 17), ac (..., 7) -> (..., 16) after observation pruning."""
    return torch.cat([obs_preproc_torch(obs), ac], dim=-1)


def targ_proc_numpy(obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    return next_obs - obs


def get_ee_pos_torch(states: torch.Tensor) -> torch.Tensor:
    """End-effector position from first 7 joint angles (batch)."""
    theta1 = states[:, :1]
    theta2 = states[:, 1:2]
    theta3 = states[:, 2:3]
    theta4 = states[:, 3:4]
    theta5 = states[:, 4:5]
    theta6 = states[:, 5:6]

    rot_axis = torch.cat(
        [
            torch.cos(theta2) * torch.cos(theta1),
            torch.cos(theta2) * torch.sin(theta1),
            -torch.sin(theta2),
        ],
        dim=1,
    )
    rot_perp_axis = torch.cat(
        [-torch.sin(theta1), torch.cos(theta1), torch.zeros_like(theta1)], dim=1
    )
    cur_end = torch.cat(
        [
            0.1 * torch.cos(theta1) + 0.4 * torch.cos(theta1) * torch.cos(theta2),
            0.1 * torch.sin(theta1) + 0.4 * torch.sin(theta1) * torch.cos(theta2) - 0.188,
            -0.4 * torch.sin(theta2),
        ],
        dim=1,
    )

    for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
        perp_all_axis = torch.cross(rot_axis, rot_perp_axis, dim=1)
        x = torch.cos(hinge) * rot_axis
        y = torch.sin(hinge) * torch.sin(roll) * rot_perp_axis
        z = -torch.sin(hinge) * torch.cos(roll) * perp_all_axis
        new_rot_axis = x + y + z
        new_rot_perp_axis = torch.cross(new_rot_axis, rot_axis, dim=1)
        nrm = torch.linalg.norm(new_rot_perp_axis, dim=1, keepdim=True)
        small = nrm < 1e-30
        new_rot_perp_axis = torch.where(
            small.expand_as(new_rot_perp_axis), rot_perp_axis, new_rot_perp_axis
        )
        new_rot_perp_axis = new_rot_perp_axis / (
            torch.linalg.norm(new_rot_perp_axis, dim=1, keepdim=True) + 1e-30
        )
        rot_axis, rot_perp_axis, cur_end = (
            new_rot_axis,
            new_rot_perp_axis,
            cur_end + length * new_rot_axis,
        )

    return cur_end


def make_obs_cost_torch(env):
    """MPC cost (positive): squared EE distance to goal + action cost applied separately.

    Pass the **base** env (e.g. ``env.unwrapped``) so ``.goal`` is available; wrappers hide it.
    """

    def obs_cost_torch(obs: torch.Tensor) -> torch.Tensor:
        g = torch.as_tensor(env.goal, dtype=obs.dtype, device=obs.device).view(1, 3)
        ee = get_ee_pos_torch(obs[:, :7])
        return torch.sum((ee - g) ** 2, dim=1)

    return obs_cost_torch


def ac_cost_torch(acs: torch.Tensor) -> torch.Tensor:
    return 0.01 * torch.sum(acs**2, dim=1)


# Reacher NN hyperparameters (dmbrl/config/reacher.py)
HIDDEN_DIMS = (200, 200, 200, 200)
WEIGHT_DECAYS = (0.00025, 0.0005, 0.0005, 0.0005, 0.00075)
LEARNING_RATE = 0.00075
