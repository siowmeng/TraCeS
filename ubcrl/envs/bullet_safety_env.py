from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch

import gymnasium
import bullet_safety_gym

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box

from ubcrl.classify.classifier import mujoco_safety_gymnasium_dict

@env_register
class BulletSafetyEnv(CMDP):

    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _support_envs: ClassVar[list[str]] = ['SafetyAntRun-v0', 'SafetyBallRun-v0', 'SafetyCarRun-v0', 'SafetyDroneRun-v0',
                                          'SafetyAntCircle-v0', 'SafetyBallCircle-v0', 'SafetyCarCircle-v0', 'SafetyDroneCircle-v0']
    # _action_space: OmnisafeSpace
    # _observation_space: OmnisafeSpace
    # metadata: ClassVar[dict[str, int]] = {}
    # env_spec_log: dict[str, Any]
    # _num_envs = 1

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        # self._count = 0
        # self._observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
        # self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # self._max_episode_steps = 10
        # self.env_spec_log = {}
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if num_envs > 1:
            self._env = gymnasium.vector.make(id=env_id, num_envs=num_envs, **kwargs)
            assert isinstance(self._env.single_action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.single_observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True
            self._env = gymnasium.make(id=env_id, autoreset=True, **kwargs)
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata
        self._max_episode_steps = mujoco_safety_gymnasium_dict[env_id]['horizon']

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        # print("Obs", obs)
        # print("Reward", reward)
        # print("Terminated", terminated)
        # print("Truncated", truncated)
        # print("Info", info)
        assert 'cost' in info or 'final_info' in info
        if 'cost' in info:
            cost = info['cost']
        else:
            if isinstance(info['final_info'], dict):
                cost = info['final_info']['cost']
            else:
                cost = np.array([info_entry['cost'] for info_entry in info['final_info']])

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the environment.

        Args:
            seed (int): The random seed.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Any: An array representing the rendered environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
