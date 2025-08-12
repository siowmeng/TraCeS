import torch as th
from torch import nn

from omnisafe.typing import Activation, InitFunction
from omnisafe.utils.model import initialize_layer

class ExponentialLayer(nn.Module):
    def forward(self, input: th.Tensor) -> th.Tensor:
        return th.exp(input)

def get_activation(
    activation: Activation,
) -> type[nn.Identity | nn.ReLU | nn.Sigmoid | nn.Softplus | nn.Tanh]:
    activations = {
        'identity': nn.Identity,
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'tanh': nn.Tanh,
        'exp': ExponentialLayer,
    }
    assert activation in activations
    return activations[activation]

def build_mlp_network(
    sizes: list[int],
    activation: Activation,
    output_activation: Activation = 'identity',
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
    layer_norm: bool = False,
    last_layer_init_weight: float = None,
) -> nn.Module:
    """Build the MLP network.

    Examples:
        >>> build_mlp_network([64, 64, 64], 'relu', 'tanh')
        Sequential(
            (0): Linear(in_features=64, out_features=64, bias=True)
            (1): ReLU()
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ReLU()
            (4): Linear(in_features=64, out_features=64, bias=True)
            (5): Tanh()
        )

    Args:
        sizes (list of int): The sizes of the layers.
        activation (Activation): The activation function.
        output_activation (Activation, optional): The output activation function. Defaults to
            ``identity``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        layer_norm (boolean): Whether to use layernorm
        last_layer_init_weight: To initialize small weight to last affine layer

    Returns:
        The MLP network.
    """
    activation_fn = get_activation(activation)
    output_activation_fn = get_activation(output_activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn if j < len(sizes) - 2 else output_activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        if (j == len(sizes) - 2) and (last_layer_init_weight is not None):
            nn.init.normal_(affine_layer.weight, mean=0.0, std=last_layer_init_weight)
            nn.init.constant_(affine_layer.bias, 0.0)
        layers += [affine_layer, act_fn(), nn.LayerNorm(sizes[j + 1]) if j < len(sizes) - 2 and layer_norm else nn.Identity()]
    return nn.Sequential(*layers)

def build_encoder_network(
    sizes: list[int],
    activation: Activation,
    weight_initialization_mode: InitFunction = 'kaiming_uniform',
    layer_norm: bool = False,
) -> nn.Module:

    activation_fn = get_activation(activation)
    layers = []
    for j in range(len(sizes) - 1):
        act_fn = activation_fn
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization_mode, affine_layer)
        layers += [affine_layer, act_fn(), nn.LayerNorm(sizes[j + 1]) if layer_norm else nn.Identity()]
    return nn.Sequential(*layers)

