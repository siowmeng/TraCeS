from __future__ import annotations

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR

from ubcrl.models.actor import GaussianLearningActorH
from ubcrl.models.actor.actor_builder import ActorBuilderH
from ubcrl.models.base import ActorH, CriticH
from ubcrl.models.critic.critic_builder import CriticBuilderH
from omnisafe.models.base import Critic
from omnisafe.models.critic.critic_builder import CriticBuilder
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import ModelConfig
from omnisafe.utils.schedule import PiecewiseSchedule, Schedule


class ActorCriticH(nn.Module):

    std_schedule: Schedule

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        model_cfgs: ModelConfig,
        epochs: int,
        log_std_reduce: float = 0.,
    ) -> None:
        """Initialize an instance of :class:`ActorCritic`."""
        super().__init__()

        self.actor: ActorH = ActorBuilderH(
            obs_space=obs_space,
            hidden_obs_size=hidden_obs_size,
            act_space=act_space,
            obs_encoder_sizes=model_cfgs.actor.obs_encoder,
            hidden_sizes=model_cfgs.actor.hidden_sizes,
            h_encoder_sizes=model_cfgs.actor.h_encoder,
            activation=model_cfgs.actor.activation,
            output_activation=model_cfgs.actor.out_activation,
            obs_encoder_activation=model_cfgs.actor.obs_encoder_activation,
            h_encoder_activation=model_cfgs.actor.h_encoder_activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            layer_norm=True,
            log_std_reduce=log_std_reduce,
            last_layer_init_weight=model_cfgs.actor.last_layer_init_weight,
        ).build_actor(
            actor_type=model_cfgs.actor_type,
        )
        # self.reward_critic: Critic = CriticBuilder(
        #     obs_space=obs_space,
        #     # hidden_obs_size=hidden_obs_size,
        #     act_space=act_space,
        #     hidden_sizes=model_cfgs.critic.hidden_sizes,
        #     activation=model_cfgs.critic.activation,
        #     weight_initialization_mode=model_cfgs.weight_initialization_mode,
        #     num_critics=1,
        #     use_obs_encoder=False,
        # ).build_critic(critic_type='v')
        self.reward_critic: CriticH = CriticBuilderH(
            obs_space=obs_space,
            hidden_obs_size=0,
            # hidden_obs_size=hidden_obs_size,
            act_space=act_space,
            obs_encoder_sizes=model_cfgs.critic.obs_encoder,
            hidden_sizes=model_cfgs.critic.hidden_sizes,
            activation=model_cfgs.critic.activation,
            obs_encoder_activation=model_cfgs.critic.obs_encoder_activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            num_critics=1,
            use_obs_encoder=False,
            output_activation=model_cfgs.critic.out_activation,
            layer_norm=True,
        ).build_critic(critic_type='v')
        self.add_module('actor', self.actor)
        self.add_module('reward_critic', self.reward_critic)

        if model_cfgs.actor.lr is not None:
            self.actor_optimizer: optim.Optimizer
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=model_cfgs.actor.lr)
        if model_cfgs.critic.lr is not None:
            self.reward_critic_optimizer: optim.Optimizer = optim.Adam(
                self.reward_critic.parameters(),
                lr=model_cfgs.critic.lr,
            )
        if model_cfgs.actor.lr is not None:
            self.actor_scheduler: LinearLR | ConstantLR
            if model_cfgs.linear_lr_decay:
                self.actor_scheduler = LinearLR(
                    self.actor_optimizer,
                    start_factor=1.0,
                    end_factor=0.0,
                    total_iters=epochs,
                )
            else:
                self.actor_scheduler = ConstantLR(
                    self.actor_optimizer,
                    factor=1.0,
                    total_iters=epochs,
                )

    def step(
        self,
        obs: torch.Tensor,
        hidden_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            hidden_obs (torch.tensor): The hidden obs from classifier.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            log_prob: The log probability of the action.
        """
        with torch.no_grad():
            value_r = self.reward_critic(obs)
            # value_r = self.reward_critic(obs, hidden_obs)
            act = self.actor.predict(obs, hidden_obs, deterministic=deterministic)
            log_prob = self.actor.log_prob(act)
        return act, value_r[0], log_prob

    def forward(
        self,
        obs: torch.Tensor,
        hidden_obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Choose the action based on the observation. used in training with gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            hidden_obs (torch.tensor): The hidden obs from classifier.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            action: The deterministic action if ``deterministic`` is True, otherwise the action with
                Gaussian noise.
            value_r: The reward value of the observation.
            log_prob: The log probability of the action.
        """
        return self.step(obs, hidden_obs, deterministic=deterministic)

    def set_annealing(self, epochs: list[int], std: list[float]) -> None:
        """Set the annealing mode for the actor.

        Args:
            epochs (list of int): The list of epochs.
            std (list of float): The list of standard deviation.
        """
        assert isinstance(
            self.actor,
            GaussianLearningActorH,
        ), 'Only GaussianLearningActorH support annealing.'
        self.std_schedule = PiecewiseSchedule(
            endpoints=list(zip(epochs, std)),
            outside_value=std[-1],
        )

    def annealing(self, epoch: int) -> None:
        """Set the annealing mode for the actor.

        Args:
            epoch (int): The current epoch.
        """
        assert isinstance(
            self.actor,
            GaussianLearningActorH,
        ), 'Only GaussianLearningActorH support annealing.'
        self.actor.std = self.std_schedule.value(epoch)
