from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch
from gymnasium import spaces, ObservationWrapper

from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, Wrapper
from omnisafe.envs.wrapper import ActionScale, Unsqueeze, RewardNormalize

from ubcrl.classify.classifier import PtEstGRU, DistributionGRU
from ubcrl.common import MeanNormalizer

class HiddenObsNormalize(Wrapper):

    def __init__(self, env: CMDP, device: torch.device, hidden_dim: int,  normalize: bool = True, norm: Normalizer | None = None) -> None:

        super().__init__(env=env, device=device)
        assert isinstance(self.observation_space, spaces.Box), 'Observation space must be Box'
        self._hidden_obs_normalizer: Normalizer

        if normalize:
            if norm is not None:
                self._hidden_obs_normalizer = norm.to(self._device)
            else:
                self._hidden_obs_normalizer = Normalizer((hidden_dim, ), clip=5).to(self._device)
        else:
            self._hidden_obs_normalizer = None

    def step(
        self,
        action: torch.Tensor,
        orig_obs: torch.Tensor = None,
        full_hidden_obs: torch.Tensor = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        assert orig_obs is not None, "Original Obs must not be None"
        assert full_hidden_obs is not None, "Full Hidden Obs must not be None"
        assert classifier is not None, "Classifier must not be None"

        obs_action = torch.concat((orig_obs, action), dim=-1)

        # print("Obs Action", obs_action.unsqueeze(dim=1))
        # print("One", torch.FloatTensor([1] * obs_action.shape[0]).to(self._device))
        # print("Full H", full_hidden_obs)

        prob_feasible, dict_logscores_t, next_hidden_obs_t, next_full_hidden_obs = classifier(
            obs_action.unsqueeze(dim=1),
            torch.FloatTensor([1] * obs_action.shape[0]).to(self._device),
            init_h=full_hidden_obs
        )

        logscores_t, mean_logscores_t, var_logscores_t = (
            dict_logscores_t['log_scores'], dict_logscores_t['mean'], dict_logscores_t['var']
        )

        # Get logC and H at the curr timestep h
        logscores, next_hidden_obs, mean_logscores, var_logscores = (
            logscores_t[-1], next_hidden_obs_t[-1], mean_logscores_t[-1], var_logscores_t[-1]
        )
        # Classifier log score output is clipped at classifier.py MIN_LOGSCORE = -7
        logscores = logscores.squeeze(axis=-1)
        mean_logscores, var_logscores = mean_logscores.squeeze(axis=-1), var_logscores.squeeze(axis=-1)

        if action.shape[0] == 1:
            action = action.squeeze(dim=0)

        # print("Wrapper Action", action)
        obs, reward, cost, terminated, truncated, info = super().step(action)

        if 'final_observation' in info:
            final_obs_slice = info['_final_observation'] if self.num_envs > 1 else slice(None)
            info['final_hidden_obs'] = next_hidden_obs
            info['original_final_hidden_obs'] = info['final_hidden_obs']
            if self._hidden_obs_normalizer is not None:
                info['final_hidden_obs'][final_obs_slice] = self._hidden_obs_normalizer.normalize(
                    next_hidden_obs[final_obs_slice],
                )
            num_final_obs = final_obs_slice.sum() if self.num_envs > 1 else 1
            next_hidden_obs[final_obs_slice] = torch.zeros(num_final_obs, classifier.nb_gru_units).to(self._device)
            next_full_hidden_obs[:, final_obs_slice] = torch.zeros((classifier.gru_layers, num_final_obs,
                                                                    classifier.nb_gru_units)).to(self._device)

        # for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
        #     if done or time_out:
        #         next_hidden_obs[idx] = torch.zeros(classifier.nb_gru_units)
        #         next_full_hidden_obs[:, idx] = torch.zeros((classifier.gru_layers, classifier.nb_gru_units))

        info['original_hidden_obs'] = next_hidden_obs
        if self._hidden_obs_normalizer is not None:
            next_hidden_obs = self._hidden_obs_normalizer.normalize(next_hidden_obs)
        info['next_hidden_obs'] = next_hidden_obs
        info['next_full_hidden_obs'] = next_full_hidden_obs
        info['logscores'] = logscores
        info['mean_logscores'] = mean_logscores
        info['var_logscores'] = var_logscores

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        assert classifier is not None, "Classifier must not be None"
        obs, info = super().reset(seed=seed, options=options)

        if len(obs.shape) <= 1:
            hidden_obs = torch.zeros((classifier.nb_gru_units)).to(self._device)
            full_hidden_obs = torch.zeros((classifier.gru_layers, 1,
                                           classifier.nb_gru_units)).to(self._device)
        else:
            hidden_obs = torch.zeros((obs.shape[0], classifier.nb_gru_units)).to(self._device)
            full_hidden_obs = torch.zeros((classifier.gru_layers, obs.shape[0],
                                           classifier.nb_gru_units)).to(self._device)

        info['original_hidden_obs'] = hidden_obs
        if self._hidden_obs_normalizer is not None:
            hidden_obs = self._hidden_obs_normalizer.normalize(hidden_obs)
        info['hidden_obs'] = hidden_obs
        info['full_hidden_obs'] = full_hidden_obs
        return obs, info

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the observation normalizer.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize. When evaluating the saved model, the normalizer
            should be loaded.

        Returns:
            The saved components, that is the observation normalizer.
        """
        saved = super().save()
        saved['hidden_obs_normalizer'] = self._hidden_obs_normalizer
        return saved

class UBCRLRewardNormalize(RewardNormalize):
    def __init__(self, env: CMDP, device: torch.device, norm: Normalizer | None = None) -> None:
        """Initialize an instance of :class:`RewardNormalize`."""
        super(RewardNormalize, self).__init__(env=env, device=device)
        self._reward_normalizer: MeanNormalizer

        if norm is not None:
            self._reward_normalizer = norm.to(self._device)
        else:
            self._reward_normalizer = MeanNormalizer((), clip=5).to(self._device)

class UBCRLActionScale(ActionScale):

    _env: HiddenObsNormalize

    def step(
        self,
        action: torch.Tensor,
        orig_obs: torch.Tensor = None,
        full_hidden_obs: torch.Tensor = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action = self._old_min_action + (self._old_max_action - self._old_min_action) * (
            action - self._min_action
        ) / (self._max_action - self._min_action)

        if (orig_obs is None) and (full_hidden_obs is None) and (classifier is None):
            return self._env.step(action)
        else:
            return self._env.step(action, orig_obs, full_hidden_obs, classifier)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if classifier is None:
            return self._env.reset(seed=seed, options=options)
        else:
            return self._env.reset(seed=seed, options=options, classifier=classifier)

class UBCRLUnsqueeze(Unsqueeze):

    _env: HiddenObsNormalize

    def step(
        self,
        action: torch.Tensor,
        orig_obs: torch.Tensor = None,
        full_hidden_obs: torch.Tensor = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        # action = action.squeeze(0)
        obs, reward, cost, terminated, truncated, info = self._env.step(action, orig_obs, full_hidden_obs, classifier)
        obs, reward, cost, terminated, truncated = (
            x.unsqueeze(0) for x in (obs, reward, cost, terminated, truncated)
        )
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        classifier: Union[PtEstGRU, DistributionGRU] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed, options=options, classifier=classifier)
        obs = obs.unsqueeze(0)
        for k, v in info.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.unsqueeze(0)

        return obs, info


class FlattenObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(np.prod(original_shape),),
            dtype=np.float32
        )

    def observation(self, obs):
        return obs.flatten()


def get_wrapper_by_type(env, wrapper_type):
    current = env
    while True:
        if isinstance(current, wrapper_type):
            return current
        elif hasattr(current, '_env'):
            current = current._env
        else:
            raise ValueError(f"Wrapper {wrapper_type} not found in the stack.")


class DictObsToBoxWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Get sizes of all dict parts
        keys = ['achieved_goal', 'desired_goal', 'observation']
        self._obs_keys = keys
        size = sum(self.env.observation_space[k].shape[0] for k in keys)
        # New flat Box space
        low = np.concatenate([self.env.observation_space[k].low for k in keys])
        high = np.concatenate([self.env.observation_space[k].high for k in keys])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        # Concatenate the three parts in fixed order
        flat = np.concatenate([obs[k] for k in self._obs_keys]).astype(np.float32)
        return flat
