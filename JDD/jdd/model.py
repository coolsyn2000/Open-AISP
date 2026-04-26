from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channels // 2, dw_channels // 2, 1))
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv2(y)
        x = x + self.dropout1(y) * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg(y)
        y = self.conv4(y)
        return x + self.dropout2(y) * self.gamma


class JointDenoiseDemosaicNAFNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cfa_scale: int,
        width: int = 32,
        middle_blocks: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfa_scale = int(cfa_scale)
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width, dropout=dropout) for _ in range(int(middle_blocks))])
        self.outro = nn.Conv2d(width, 3 * self.cfa_scale * self.cfa_scale, 3, padding=1)
        self.upsample = nn.PixelShuffle(self.cfa_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.intro(x)
        y = self.body(y)
        y = self.outro(y)
        y = self.upsample(y)
        return y.clamp(0.0, 1.0)


def build_model(config: dict[str, Any], raw_channels: int, scale: int) -> JointDenoiseDemosaicNAFNet:
    model_cfg = config.get("model", {})
    frames = int(config.get("data", {}).get("frames", 3))
    in_channels = frames * int(raw_channels) + 1
    return JointDenoiseDemosaicNAFNet(
        in_channels=in_channels,
        cfa_scale=scale,
        width=int(model_cfg.get("width", 32)),
        middle_blocks=int(model_cfg.get("middle_blocks", 8)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def infer_raw_channels(camera: dict[str, Any]) -> int:
    cfa_type = camera["cfa"].get("type", "bayer").lower().replace("_", "")
    return 16 if cfa_type == "quadbayer" else 4
