"""ResNet-18 backbone with selectable head: linear ``fc`` (CUB 对齐 R50) 或原 MLP 头（非 CUB）。"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as models

try:
    from torchvision.models import ResNet18_Weights
except ImportError:
    ResNet18_Weights = None


class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MLPModuleResnet(nn.Module):
    """历史实现：512→512→ReLU→(Dropout)→output。"""

    def __init__(self, output_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


HeadMode = Literal["linear", "mlp"]


class Resnet18MLP(nn.Module):
    """ResNet-18：``head_mode=linear`` 时与 ``resnet50_mlp`` 同范式（单层 fc + Xavier）；``mlp`` 时为 Identity + ``MLPModuleResnet``。"""

    def __init__(
        self,
        output_size: int,
        in_channels: int = 3,
        dropout: float = 0.0,
        small_input: bool = False,
        head_style: HeadMode = "linear",
    ) -> None:
        super().__init__()
        self.head_mode: HeadMode = head_style
        use_pretrained = (in_channels == 3) and (not small_input)
        if use_pretrained:
            if ResNet18_Weights is not None:
                self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.resnet18 = models.resnet18(pretrained=True)
        else:
            self.resnet18 = models.resnet18(weights=None)
        if small_input:
            self.resnet18.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.resnet18.maxpool = nn.Identity()
        else:
            self.resnet18.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        in_features = int(self.resnet18.fc.in_features)
        if self.head_mode == "linear":
            self.resnet18.fc = nn.Linear(in_features, int(output_size))
            nn.init.xavier_uniform_(self.resnet18.fc.weight)
            if self.resnet18.fc.bias is not None:
                nn.init.zeros_(self.resnet18.fc.bias)
            self.mlp = None
        else:
            self.resnet18.fc = Identity()
            self.mlp = MLPModuleResnet(int(output_size), dropout=dropout)

        self.backbone_drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, epoch_id=None, **kwargs) -> torch.Tensor:
        del epoch_id, kwargs
        if self.head_mode == "mlp":
            features = self.resnet18(x)
            features = self.backbone_drop(features)
            assert self.mlp is not None
            return self.mlp(features)

        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone_drop(x)
        return self.resnet18.fc(x)
