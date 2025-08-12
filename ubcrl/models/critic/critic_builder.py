from __future__ import annotations

from ubcrl.models.base import CriticH
from ubcrl.models.critic.v_critic import VCriticH
from omnisafe.typing import Activation, CriticType, InitFunction, OmnisafeSpace


# pylint: disable-next=too-few-public-methods
class CriticBuilderH:
    """Implementation of CriticBuilder.

    .. note::
        A :class:`CriticBuilder` is a class for building a critic network. In OmniSafe, instead of
        building the critic network directly, we build it by integrating various types of critic
        networks into the :class:`CriticBuilder`. The advantage of this is that each type of critic
        has a uniform way of passing parameters. This makes it easy for users to use existing
        critics, and also facilitates the extension of new critic types.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        hidden_obs_size (int): Hidden obs size.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        num_critics (int, optional): Number of critics. Defaults to 1.
        use_obs_encoder (bool, optional): Whether to use observation encoder, only used in q critic.
            Defaults to False.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        obs_encoder_sizes: list[int],
        hidden_sizes: list[int],
        h_encoder_sizes: list[int] = None,
        activation: Activation = 'relu',
        obs_encoder_activation: Activation = 'identity',
        h_encoder_activation: Activation = 'identity',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        num_critics: int = 1,
        use_obs_encoder: bool = False,
        output_activation: Activation = 'identity',
        layer_norm: bool = False,
    ) -> None:
        """Initialize an instance of :class:`CriticBuilder`."""
        if h_encoder_sizes is None:
            h_encoder_sizes = []
        self._obs_space: OmnisafeSpace = obs_space
        self._hidden_obs_dim = hidden_obs_size
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._obs_encoder_activation: Activation = obs_encoder_activation
        self._h_encoder_activation: Activation = h_encoder_activation
        self._obs_encoder_sizes: list[int] = obs_encoder_sizes
        self._hidden_sizes: list[int] = hidden_sizes
        self._h_encoder_sizes: list[int] = h_encoder_sizes
        self._num_critics: int = num_critics
        self._use_obs_encoder: bool = use_obs_encoder
        self._output_activation: Activation = output_activation
        self._layer_norm: bool = layer_norm

    def build_critic(
        self,
        critic_type: CriticType,
    ) -> CriticH:
        """Build critic.

        Currently, we support two types of critics: ``q`` and ``v``.
        If you want to add a new critic type, you can simply add it here.

        Args:
            critic_type (str): Critic type.

        Returns:
            An instance of V-Critic or Q-Critic

        Raises:
            NotImplementedError: If the critic type is not ``q`` or ``v``.
        """
        # if critic_type == 'q':
        #     return QCritic(
        #         obs_space=self._obs_space,
        #         act_space=self._act_space,
        #         hidden_sizes=self._hidden_sizes,
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #         num_critics=self._num_critics,
        #         use_obs_encoder=self._use_obs_encoder,
        #     )
        if critic_type == 'v':
            return VCriticH(
                obs_space=self._obs_space,
                hidden_obs_size=self._hidden_obs_dim,
                act_space=self._act_space,
                obs_encoder_sizes=self._obs_encoder_sizes,
                h_encoder_sizes=self._h_encoder_sizes,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                obs_encoder_activation=self._obs_encoder_activation,
                h_encoder_activation=self._h_encoder_activation,
                weight_initialization_mode=self._weight_initialization_mode,
                num_critics=self._num_critics,
                output_activation=self._output_activation,
                layer_norm=self._layer_norm,
            )

        raise NotImplementedError(
            f'critic_type "{critic_type}" is not implemented.'
            # 'Available critic types are: "q", "v".',
            'Available critic types are: "v".',
        )
