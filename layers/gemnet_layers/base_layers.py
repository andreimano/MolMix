import torch
from torch.nn.utils import spectral_norm


def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

# class Dense(torch.nn.Module):
#     """
#     Combines dense layer and scaling for swish activation.

#     Parameters
#     ----------
#         units: int
#             Output embedding size.
#         activation: str
#             Name of the activation function to use.
#         bias: bool
#             True if use bias.
#     """

#     def __init__(
#         self, in_features, out_features, bias=False, activation=None, name=None
#     ):
#         super().__init__()

#         self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
#         self.reset_parameters()
#         self.weight = self.linear.weight
#         self.bias = self.linear.bias

#         if isinstance(activation, str):
#             activation = activation.lower()
#         if activation in ["swish", "silu"]:
#             self._activation = ScaledSiLU()
#         elif activation is None:
#             self._activation = torch.nn.Identity()
#         else:
#             raise NotImplementedError(
#                 "Activation function not implemented for GemNet (yet)."
#             )

#     def reset_parameters(self):
#         he_orthogonal_init(self.linear.weight)
#         if self.linear.bias is not None:
#             self.linear.bias.data.fill_(0)

#     def forward(self, x):
#         x = self.linear(x)
#         x = self._activation(x)
#         return x

class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)

class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(self, in_features, out_features, bias=False, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        elif activation == "gelu":
            self._activation = torch.nn.GELU()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self, initializer=he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x

class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
    """

    def __init__(self, units: int, nLayers: int = 2, activation=None, name=None):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                Dense(units, units, activation=activation, bias=False)
                for i in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / (2.0 ** 0.5)

    def forward(self, inputs):
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x
