from __future__ import annotations

import torch
import torch.nn as nn

from ubcrl.models.base import CriticH
from omnisafe.utils.model import initialize_layer
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from ubcrl.utils.model import build_mlp_network, build_encoder_network


class VCriticH(CriticH):
    """Implementation of VCritic.

    A V-function approximator that uses a multi-layer perceptron (MLP) to map observations to V-values.
    This class is an inherit class of :class:`Critic`.
    You can design your own V-function approximator by inheriting this class or :class:`Critic`.

    Args:
        obs_dim (int): Observation dimension.
        hidden_obs_dim (int): Hidden Obs dimension.
        act_dim (int): Action dimension.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        obs_encoder_sizes: list[int],
        h_encoder_sizes: list[int],
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        obs_encoder_activation: Activation = 'identity',
        h_encoder_activation: Activation = 'identity',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        output_activation: Activation = 'identity',
        layer_norm: bool = False,
    ) -> None:
        """Initialize an instance of :class:`VCritic`."""
        super().__init__(
            obs_space,
            hidden_obs_size,
            act_space,
            obs_encoder_sizes,
            h_encoder_sizes,
            hidden_sizes,
            activation,
            obs_encoder_activation,
            h_encoder_activation,
            weight_initialization_mode,
            num_critics,
            use_obs_encoder=False,
        )
        # self.net_lst: list[nn.Module]
        # self.net_lst = []
        #
        # for idx in range(self._num_critics):
        #     net = build_mlp_network(
        #         sizes=[self._obs_dim + self._hidden_obs_dim, *self._hidden_sizes, 1],
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #         output_activation=output_activation,
        #         layer_norm=layer_norm,
        #     )
        #     self.net_lst.append(net)
        #     self.add_module(f'critic_{idx}', net)
        self.net_lst: list[list[nn.Module]]
        self.net_lst = []

        for idx in range(self._num_critics):

            encoder_sizes = [self._obs_dim, *self._obs_encoder_sizes]

            obs_encoder = build_encoder_network(
                # sizes=[self._obs_dim, *self._obs_encoder_sizes],
                sizes=encoder_sizes,
                activation=self._obs_encoder_activation,
                weight_initialization_mode=self._weight_initialization_mode,
                layer_norm=layer_norm,
            )

            h_encoder_sizes = [self._hidden_obs_dim, *self._h_encoder_sizes]

            h_encoder = build_encoder_network(
                # sizes=[self._obs_dim, *self._obs_encoder_sizes],
                sizes=h_encoder_sizes,
                activation=self._h_encoder_activation,
                weight_initialization_mode=self._weight_initialization_mode,
                layer_norm=layer_norm,
            )

            obs_h_encoder_sizes = [encoder_sizes[-1] + self._hidden_obs_dim, self._hidden_obs_dim]

            obs_h_encoder = build_encoder_network(
                # sizes=[self._obs_dim, *self._obs_encoder_sizes],
                sizes=obs_h_encoder_sizes,
                activation=self._obs_encoder_activation,
                weight_initialization_mode=self._weight_initialization_mode,
                layer_norm=layer_norm,
            )

            net = build_mlp_network(
                # sizes=[self._obs_encoder_sizes[-1] + self._hidden_obs_dim, *self._hidden_sizes, 1],
                # sizes=[encoder_sizes[-1] + self._hidden_obs_dim, *self._hidden_sizes, 1],
                sizes=[encoder_sizes[-1] + h_encoder_sizes[-1], *self._hidden_sizes, 1],
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
                output_activation=output_activation,
                layer_norm=layer_norm,
            )
            self.net_lst.append([obs_encoder, h_encoder, obs_h_encoder, net])
            self.add_module(f'critic_obs_{idx}', obs_encoder)
            self.add_module(f'critic_h_{idx}', h_encoder)
            self.add_module(f'critic_obs_h_{idx}', obs_h_encoder)
            self.add_module(f'critic_{idx}', net)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_obs: torch.Tensor = None
    ) -> list[torch.Tensor]:
        """Forward function.

        Specifically, V function approximator maps observations to V-values.

        Args:
            obs (torch.Tensor): Observations from environments.
            hidden_obs (torch.Tensor): Hidden obs from classifier.

        Returns:
            The V critic value of observation.
        """
        # res = []
        # for critic in self.net_lst:
        #     res.append(torch.squeeze(critic(torch.cat((obs, hidden_obs), dim=-1)), -1))
        #     # critic_out = torch.squeeze(critic(torch.cat((obs, hidden_obs), dim=-1)), -1)
        #     # res.append(torch.exp(critic_out))
        # return res

        res = []
        for obs_encoder, h_encoder, obs_h_encoder, critic in self.net_lst:
            encoded_obs = obs_encoder(obs)
            if hidden_obs is None:
                concat_obs = encoded_obs
            else:
                encoded_h = h_encoder(hidden_obs)
                concat_obs_h = torch.cat((encoded_obs, encoded_h), dim=-1)
                encoded_obs_h = obs_h_encoder(concat_obs_h)
                concat_obs = torch.cat((encoded_h, encoded_obs_h), dim=-1)
            # concat_obs = encoded_obs if hidden_obs is None else torch.cat((encoded_obs, hidden_obs), dim=-1)
            res.append(torch.squeeze(critic(concat_obs), -1))
        return res
