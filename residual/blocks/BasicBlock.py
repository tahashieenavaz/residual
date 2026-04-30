import torch
from typing import Tuple
from typing import Type


class BasicBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        expansion: int = 1,
        kernel_size: int | Tuple[int] = 3,
        chi: Type[torch.nn.Module] = torch.nn.ReLU,
        psi: Type[torch.nn.Module] = torch.nn.ReLU,
        normalization: Type[torch.nn.Module] = torch.nn.BatchNorm2d,
    ):
        super().__init__()
        self.psi = chi()
        self.chi = psi()
        self.first_normalization = normalization(out_channels)
        self.second_normalization = normalization(out_channels)
        self.shortcut = torch.nn.Sequential()

        self.first_convolutional = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.second_convolutional = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_channels != out_channels * expansion:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels * expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                normalization(out_channels * expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.first_convolutional(x)
        x = self.first_normalization(x)
        x = self.chi(x)

        x = self.second_convolutional(x)
        x = self.second_convolutional(x)

        x += identity
        x = self.psi(x)

        return x


if __name__ == "__main__":
    block = BasicBlock(64, 128)
    feature_maps = torch.randn(1, 64, 20, 20)
    after_block = block(feature_maps)
    assert after_block.size(1) == 128
