from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace

from ubcrl.models.actor.gaussian_actor import GaussianActorH
from ubcrl.utils.model import build_encoder_network, build_mlp_network


# pylint: disable-next=too-many-instance-attributes
class GaussianLearningActorH(GaussianActorH):
    """Implementation of GaussianLearningActor.

    GaussianLearningActor is a Gaussian actor with a learnable standard deviation. It is used in
    on-policy algorithms such as ``PPO``, ``TRPO`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    _current_dist: Normal

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        obs_encoder_sizes: list[int],
        h_encoder_sizes: list[int],
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        output_activation: Activation = 'identity',
        obs_encoder_activation: Activation = 'identity',
        h_encoder_activation: Activation = 'identity',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        layer_norm: bool = False,
        log_std_reduce: float = 0.,
        last_layer_init_weight: float = None,
    ) -> None:
        """Initialize an instance of :class:`GaussianLearningActor`."""
        super().__init__(obs_space, hidden_obs_size, act_space, obs_encoder_sizes, h_encoder_sizes, hidden_sizes, activation,
                         obs_encoder_activation, h_encoder_activation, weight_initialization_mode)

        encoder_sizes = [self._obs_dim, *self._obs_encoder_sizes]

        self.obs_encoder: nn.Module = build_encoder_network(
                sizes=encoder_sizes,
                activation=self._obs_encoder_activation,
                weight_initialization_mode=self._weight_initialization_mode,
                layer_norm=layer_norm,
        )

        h_encoder_sizes = [self._hidden_obs_dim, *self._h_encoder_sizes]

        self.h_encoder = build_encoder_network(
            sizes=h_encoder_sizes,
            activation=self._h_encoder_activation,
            weight_initialization_mode=self._weight_initialization_mode,
            layer_norm=layer_norm,
        )

        # self.combined_layernorm = nn.LayerNorm(encoder_sizes[-1] + h_encoder_sizes[-1])

        self._output_activation = output_activation
        self.mean: nn.Module = build_mlp_network(
            # sizes=[self._obs_dim + self._hidden_obs_dim, *self._hidden_sizes, self._act_dim],
            # sizes=[encoder_sizes[-1]+ self._hidden_obs_dim, *self._hidden_sizes, self._act_dim],
            sizes=[encoder_sizes[-1] + h_encoder_sizes[-1], *self._hidden_sizes, self._act_dim],
            activation=self._activation,
            output_activation=self._output_activation,
            weight_initialization_mode=weight_initialization_mode,
            layer_norm=layer_norm,
            last_layer_init_weight=last_layer_init_weight,
        )
        self.log_std: nn.Parameter = nn.Parameter(torch.zeros(self._act_dim) - log_std_reduce, requires_grad=True)

    def _distribution(self, obs: torch.Tensor, hidden_obs: torch.Tensor) -> Normal:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        Args:
            obs (torch.Tensor): Observation from environments.
            hidden_obs (torch.Tensor): Hidden obs from classifier.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        encoded_obs = self.obs_encoder(obs)
        encoded_h = self.h_encoder(hidden_obs)

        combined_input = torch.cat((encoded_obs, encoded_h), dim=-1)
        # normalized_combined_input = self.combined_layernorm(combined_input)
        normalized_combined_input = combined_input

        mean = self.mean(normalized_combined_input)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def predict(self, obs: torch.Tensor, hidden_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            hidden_obs (torch.Tensor): Hidden obs from classifier.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs, hidden_obs)
        self._after_inference = True
        if deterministic:
            return self._current_dist.mean
        return self._current_dist.rsample()

    def forward(self, obs: torch.Tensor, hidden_obs: torch.Tensor) -> Distribution:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.
            hidden_obs (torch.Tensor): Hidden obs from classifier.

        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs, hidden_obs)
        self._after_inference = True
        return self._current_dist

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward` .

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False
        # TODO report omnisafe bug
        # return self._current_dist.log_prob(act).sum(axis=-1)
        return self._current_dist.log_prob(act).sum(dim=-1)

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return torch.exp(self.log_std).mean().item()

    @std.setter
    def std(self, std: float) -> None:
        device = self.log_std.device
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=device)))
