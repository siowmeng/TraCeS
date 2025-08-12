from __future__ import annotations

from ubcrl.models.actor.gaussian_learning_actor import GaussianLearningActorH
# from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
# from omnisafe.models.actor.mlp_actor import MLPActor
# from omnisafe.models.actor.perturbation_actor import PerturbationActor
# from omnisafe.models.actor.vae_actor import VAE
from ubcrl.models.base import ActorH
from omnisafe.typing import Activation, ActorType, InitFunction, OmnisafeSpace


# pylint: disable-next=too-few-public-methods
class ActorBuilderH:
    """Class for building actor networks.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        hidden_obs_size (int): Hidden Obs size.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        hidden_obs_size: int,
        act_space: OmnisafeSpace,
        obs_encoder_sizes: list[int],
        hidden_sizes: list[int],
        h_encoder_sizes: list[int] = None,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity',
        obs_encoder_activation: Activation = 'identity',
        h_encoder_activation: Activation = 'identity',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        layer_norm: bool = False,
        log_std_reduce: float = 0.,
        last_layer_init_weight: float = None,
    ) -> None:
        """Initialize an instance of :class:`ActorBuilder`."""
        if h_encoder_sizes is None:
            h_encoder_sizes = []
        self._obs_space: OmnisafeSpace = obs_space
        self._hidden_obs_dim: int = hidden_obs_size
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._output_activation = output_activation
        self._obs_encoder_activation: Activation = obs_encoder_activation
        self._h_encoder_activation: Activation = h_encoder_activation
        self._obs_encoder_sizes: list[int] = obs_encoder_sizes
        self._hidden_sizes: list[int] = hidden_sizes
        self._h_encoder_sizes: list[int] = h_encoder_sizes
        self._layer_norm: bool = layer_norm
        self._log_std_reduce: float = log_std_reduce
        self._last_layer_init_weight: float = last_layer_init_weight

    # pylint: disable-next=too-many-return-statements
    def build_actor(
        self,
        actor_type: ActorType,
    ) -> ActorH:
        """Build actor network.

        Currently, we support the following actor types:
            - ``gaussian_learning``: Gaussian actor with learnable standard deviation parameters.
            - ``gaussian_sac``: Gaussian actor with learnable standard deviation network.
            - ``mlp``: Multi-layer perceptron actor, used in ``DDPG`` and ``TD3``.

        Args:
            actor_type (ActorType): Type of actor network, e.g. ``gaussian_learning``.

        Returns:
            Actor network, ranging from GaussianLearningActor, GaussianSACActor to MLPActor.

        Raises:
            NotImplementedError: If the actor type is not implemented.
        """
        if actor_type == 'gaussian_learning':
            return GaussianLearningActorH(
                self._obs_space,
                self._hidden_obs_dim,
                self._act_space,
                self._obs_encoder_sizes,
                self._h_encoder_sizes,
                self._hidden_sizes,
                activation=self._activation,
                output_activation=self._output_activation,
                obs_encoder_activation=self._obs_encoder_activation,
                h_encoder_activation=self._h_encoder_activation,
                weight_initialization_mode=self._weight_initialization_mode,
                layer_norm=self._layer_norm,
                log_std_reduce=self._log_std_reduce,
                last_layer_init_weight=self._last_layer_init_weight,
            )
        # if actor_type == 'gaussian_sac':
        #     return GaussianSACActor(
        #         self._obs_space,
        #         self._act_space,
        #         self._hidden_sizes,
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #     )
        # if actor_type == 'mlp':
        #     return MLPActor(
        #         self._obs_space,
        #         self._act_space,
        #         self._hidden_sizes,
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #     )
        # if actor_type == 'vae':
        #     return VAE(
        #         self._obs_space,
        #         self._act_space,
        #         self._hidden_sizes,
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #     )
        # if actor_type == 'perturbation':
        #     return PerturbationActor(
        #         self._obs_space,
        #         self._act_space,
        #         self._hidden_sizes,
        #         activation=self._activation,
        #         weight_initialization_mode=self._weight_initialization_mode,
        #     )
        raise NotImplementedError(
            f'Actor type {actor_type} is not implemented! '
            # f'Available actor types are: gaussian_learning, gaussian_sac, mlp, vae, perturbation.',
            f'Available actor types are: gaussian_learning.',
        )
