from __future__ import annotations

import torch
from torch import optim

from ubcrl.models.actor_critic.actor_critic import ActorCriticH
from ubcrl.models.base import CriticH
from ubcrl.models.critic.critic_builder import CriticBuilderH
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig


class ConstraintActorCriticH(ActorCriticH):

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
        layer_norm: bool = True,
        log_std_reduce: float = 0.,
    ) -> None:
        """Initialize an instance of :class:`ConstraintActorCritic`."""
        super().__init__(obs_space, hidden_obs_size, act_space, model_cfgs, epochs, log_std_reduce=log_std_reduce)
        self.cost_critic: CriticH = CriticBuilderH(
            obs_space=obs_space,
            hidden_obs_size=hidden_obs_size,
            act_space=act_space,
            obs_encoder_sizes=model_cfgs.cost_critic.obs_encoder,
            hidden_sizes=model_cfgs.cost_critic.hidden_sizes,
            h_encoder_sizes=model_cfgs.cost_critic.h_encoder,
            activation=model_cfgs.cost_critic.activation,
            obs_encoder_activation=model_cfgs.cost_critic.obs_encoder_activation,
            h_encoder_activation=model_cfgs.cost_critic.h_encoder_activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
            # output_activation='softplus',
            # output_activation='relu',
            output_activation=model_cfgs.cost_critic.out_activation,
            layer_norm=layer_norm,
        ).build_critic('v')
        self.add_module('cost_critic', self.cost_critic)

        if model_cfgs.cost_critic.lr is not None:
            self.cost_critic_optimizer: optim.Optimizer
            self.cost_critic_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=model_cfgs.cost_critic.lr,
            )

    def step(
        self,
        obs: torch.Tensor,
        hidden_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            hidden_obs (torch.tensor): The hidden obs from classifier.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        assert hidden_obs is not None, "Hidden Obs must not be None"
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            # value_r = self.reward_critic(obs, hidden_obs)
            value_c = self.cost_critic(obs, hidden_obs)

            action = self.actor.predict(obs, hidden_obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(action)

        return action, value_r[0], value_c[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        hidden_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose action based on observation.

        Args:
            obs (torch.Tensor): Observation from environments.
            hidden_obs (torch.tensor): The hidden obs from classifier.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            value_c: The cost value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, hidden_obs, deterministic=deterministic)
