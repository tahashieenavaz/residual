import torch
from typing import Type


class ResidualNetworkEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        block: Type[torch.nn.Module],
        alpha_blocks: int,
        beta_blocks: int,
        gamma_blocks: int,
        delta_blocks: int,
    ):
        super().__init__()
        self.alpha = self.__make_layer(block, 64, alpha_blocks, stride=1)
        self.beta = self.__make_layer(block, 128, beta_blocks, stride=2)
        self.gamma = self.__make_layer(block, 256, gamma_blocks, stride=2)
        self.delta = self.__make_layer(block, 512, delta_blocks, stride=2)

    def __make_layer(
        self,
        module: Type[torch.nn.Module],
        out_channels: int,
        num_blocks: int,
        stride: int,
        expansion: int,
    ) -> torch.nn.Module:
        # Only the first block in a layer might have a stride of 2; the rest have a stride of 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for _stride in strides:
            layers.append(module(self.in_channels, out_channels, _stride))
            self.in_channels = out_channels * expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.alpha(x)
        x = self.beta(x)
        x = self.gamma(x)
        x = self.delta(x)
        return x
