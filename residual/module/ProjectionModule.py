import torch
from typing import Type


class ProjectionModule(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        *,
        kernel: int = 7,
        stride: int = 2,
        padding: int = 3,
        pool_kernel: int = 3,
        pool_stride: int = 2,
        pool_padding: int = 1,
        use_bias: bool = False,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
        normalization: Type[torch.nn.Module] = torch.nn.BatchNorm2d,
    ):
        super().__init__()
        self.convolutional = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        self.normalization = normalization(out_channels)
        self.activation = activation()
        self.pool = torch.nn.MaxPool2d(
            kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional(x)
        x = self.normalization(x)
        x = self.activation(x)
        x = self.pool(x)
        return x
