import math

import numpy as np
import torch
from torch import nn as nn
from typing import Optional, List
import torch.nn.functional as F


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()

        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>

    ## Learning-rate Equalized Linear Layer

    This uses [learning-rate equalized weights](#equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class StyleBlock(nn.Module):

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ResidualBlock(nn.Module):
    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()

        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        x = self.style_block1(x, w, None)
        x = self.style_block2(x, w, None)

        return x


class Control_Module(nn.Module):
    def __init__(self, w_dim, feature_dim):
        super(Control_Module, self).__init__()

        self.w_dim = w_dim
        self.feature_dim = feature_dim

        self.l_styleblock = ResidualBlock(self.w_dim, self.feature_dim, self.feature_dim)
        self.a_styleblock = ResidualBlock(self.w_dim, self.feature_dim, self.feature_dim)
        self.b_styleblock = ResidualBlock(self.w_dim, self.feature_dim, self.feature_dim)

        self.G_weight = nn.Sequential(
            nn.Conv2d(self.feature_dim * 3, 32, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x, sl, sa, sb):
        f_l = self.l_styleblock(x, sl)
        f_a = self.a_styleblock(x, sa)
        f_b = self.b_styleblock(x, sb)

        f_lab = torch.cat((f_l, f_a, f_b), dim=1)
        f_weight = self.G_weight(f_lab)
        f_weight = nn.functional.softmax(f_weight, dim=1)

        weight_l = f_weight[:, 0, :, :].unsqueeze(1)
        weight_a = f_weight[:, 1, :, :].unsqueeze(1)
        weight_b = f_weight[:, 2, :, :].unsqueeze(1)

        out = weight_l * f_l + weight_a * f_a + weight_b * f_b

        return out


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,
            norm_layer=nn.BatchNorm2d, activation=nn.ELU,
            bias=True,
    ):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            norm_layer(out_channels) if norm_layer is not None else nn.Identity(),
            activation(),
        )

    def forward(self, x):
        return self.block(x)
