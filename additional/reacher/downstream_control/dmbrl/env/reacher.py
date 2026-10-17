"""7-DoF reacher (MuJoCo), Gymnasium API — matches original MBRL Reacher3D task."""

from __future__ import annotations

import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 2.5,
    "elevation": -30,
    "azimuth": 270,
}


class Reacher3DEnv(MujocoEnv, utils.EzPickle):
    """3D reacher arm with random goal in qpos (same as legacy env)."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str | None = None,
        frame_skip: int = 2,
        default_camera_config: dict[str, float | int] | None = None,
        **kwargs,
    ):
        if xml_file is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            xml_file = os.path.join(dir_path, "assets", "reacher3d.xml")
        if default_camera_config is None:
            default_camera_config = dict(DEFAULT_CAMERA_CONFIG)

        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            **kwargs,
        )

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
        )

        self.goal = np.zeros(3, dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            **self.metadata,
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel_partial": self.data.qvel.size - 3,
        }

    def step(self, action):
        self.do_simulation(np.asarray(action, dtype=np.float64), self.frame_skip)
        ob = self._get_obs()

        reward = float(
            -np.sum(np.square(self.get_EE_pos(ob[None]) - self.goal))
            - 0.01 * np.sum(np.square(np.asarray(action, dtype=np.float64)))
        )

        terminated = False
        truncated = False

        if self.render_mode == "human":
            self.render()

        return ob, reward, terminated, truncated, {}

    def reset_model(self):
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel)
        qpos[-3:] += self.np_random.normal(loc=0, scale=0.1, size=[3])
        qvel[-3:] = 0
        self.goal = qpos[-3:].astype(np.float64)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos.ravel(), self.data.qvel.ravel()[:-3]]
        )

    def get_EE_pos(self, states):
        """End-effector position from joint angles (states include first 7 qpos)."""
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = (
            states[:, :1],
            states[:, 1:2],
            states[:, 2:3],
            states[:, 3:4],
            states[:, 4:5],
            states[:, 5:6],
            states[:, 6:],
        )
        rot_axis = np.concatenate(
            [
                np.cos(theta2) * np.cos(theta1),
                np.cos(theta2) * np.sin(theta1),
                -np.sin(theta2),
            ],
            axis=1,
        )
        rot_perp_axis = np.concatenate(
            [-np.sin(theta1), np.cos(theta1), np.zeros(theta1.shape)], axis=1
        )
        cur_end = np.concatenate(
            [
                0.1 * np.cos(theta1) + 0.4 * np.cos(theta1) * np.cos(theta2),
                0.1 * np.sin(theta1) + 0.4 * np.sin(theta1) * np.cos(theta2) - 0.188,
                -0.4 * np.sin(theta2),
            ],
            axis=1,
        )

        for length, hinge, roll in [(0.321, theta4, theta3), (0.16828, theta6, theta5)]:
            perp_all_axis = np.cross(rot_axis, rot_perp_axis)
            x = np.cos(hinge) * rot_axis
            y = np.sin(hinge) * np.sin(roll) * rot_perp_axis
            z = -np.sin(hinge) * np.cos(roll) * perp_all_axis
            new_rot_axis = x + y + z
            new_rot_perp_axis = np.cross(new_rot_axis, rot_axis)
            new_rot_perp_axis[
                np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30
            ] = rot_perp_axis[np.linalg.norm(new_rot_perp_axis, axis=1) < 1e-30]
            new_rot_perp_axis /= np.linalg.norm(new_rot_perp_axis, axis=1, keepdims=True)
            rot_axis, rot_perp_axis, cur_end = (
                new_rot_axis,
                new_rot_perp_axis,
                cur_end + length * new_rot_axis,
            )

        return cur_end
