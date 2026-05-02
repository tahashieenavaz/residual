import torch
from typing import Type
from .blocks import ProjectionBlock


class ResidualNetworkEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        block: Type[torch.nn.Module],
        num_blocks: int,
        in_channels: int,
        projection_channels: int,
    ):
        super().__init__()
        self.in_channels = 64
        self.projection = ProjectionBlock(
            in_channels=in_channels, out_channels=projection_channels
        )

        self.alpha = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.beta = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.gamma = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.theta = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # Only the first block in a layer might have a stride of 2; the rest have a stride of 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
