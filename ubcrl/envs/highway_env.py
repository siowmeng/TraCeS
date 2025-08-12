from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch

import gymnasium
import highway_env

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box

from ubcrl.classify.classifier import mujoco_safety_gymnasium_dict
from ubcrl.envs.wrapper import FlattenObservation, DictObsToBoxWrapper

@env_register
class HighWayEnv(CMDP):

    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _support_envs: ClassVar[list[str]] = ['highway-v0', 'highway-fast-v0', 'parking-v0']

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if num_envs > 1:

            # Wrapper function
            def apply_wrappers(env):
                if env_id == 'parking-v0':
                    env = DictObsToBoxWrapper(env)
                else:
                    env = FlattenObservation(env)
                return env

            self._env = gymnasium.vector.make(id=env_id, num_envs=num_envs, wrappers=apply_wrappers,
                                              config={"action": {"type": "ContinuousAction"}}, **kwargs)
            assert isinstance(self._env.single_action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.single_observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
            # print("Obs Space Shape", self._observation_space.shape)
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True
            self._env = gymnasium.make(id=env_id, autoreset=True,
                                       config={"action": {"type": "ContinuousAction"}}, **kwargs)
            if env_id == 'parking-v0':
                self._env = DictObsToBoxWrapper(self._env)
            else:
                self._env = FlattenObservation(self._env)
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space

        self._metadata = self._env.metadata
        # print("Metadata", self._env.metadata)
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
        cost = [0.0] * self._num_envs if self._num_envs > 1 else 0.0

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if 'final_observation' in info:
            if self._num_envs > 1:
                info['final_observation'] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info['final_observation']
                    ],
                )
            else:
                info['final_observation'] = info['final_observation'].reshape(1, -1) if info['final_observation'] is not None else np.zeros(obs.shape[-1])
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
