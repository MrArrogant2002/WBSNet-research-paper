from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("bn1", nn.BatchNorm2d(64)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stem(x)
        x = self.maxpool(stem)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return stem, layer1, layer2, layer3, layer4

    def load_checkpoint(self, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for key, value in state.items():
            if key.startswith("encoder."):
                cleaned[key.replace("encoder.", "", 1)] = value
            elif key in self.state_dict():
                cleaned[key] = value
        self.load_state_dict(cleaned, strict=False)

    def load_imagenet_pretrained(self) -> None:
        try:
            from torchvision.models import ResNet34_Weights, resnet34

            state = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        except Exception as exc:
            raise RuntimeError(
                "Loading ImageNet pretrained ResNet-34 requires torchvision and either internet access or cached weights."
            ) from exc

        current = self.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        self.load_state_dict(compatible, strict=False)
