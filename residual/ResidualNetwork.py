import torch
from typing import Type
from .blocks import ProjectionBlock
from .ResidualNetworkEncoder import ResidualNetworkEncoder


class ResidualNetwork(torch.nn.Module):
    def __init__(
        self,
        block: Type[torch.nn.Module],
        num_blocks: int,
        in_channels: int,
        projection_channels: int,
        expansion: int,
        num_classes: int,
    ):
        super().__init__()
        self.projection = ProjectionBlock(
            in_channels=in_channels, out_channels=projection_channels
        )
        self.encoder = ResidualNetworkEncoder(
            block=block,
            projection_channels=projection_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.head = torch.nn.Linear(512 * expansion, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.pool(features).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
