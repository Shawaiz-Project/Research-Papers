from __future__ import annotations

import torch
from torch import nn


def valid_groups(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(valid_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(valid_groups(in_channels), in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.GroupNorm(valid_groups(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(valid_groups(out_channels), out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.activation(self.depthwise_norm(self.depthwise(x)))
        x = self.dropout(self.pointwise_norm(self.pointwise(x)))
        return self.activation(x + residual)


class CoordinateAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(valid_groups(hidden), hidden)
        self.act = nn.SiLU(inplace=True)
        self.expand_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.expand_w = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        pooled_h = x.mean(dim=3, keepdim=True)
        pooled_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        merged = torch.cat([pooled_h, pooled_w], dim=2)
        merged = self.act(self.norm(self.reduce(merged)))
        attention_h, attention_w = torch.split(merged, [height, width], dim=2)
        attention_w = attention_w.transpose(2, 3)
        attention_h = torch.sigmoid(self.expand_h(attention_h))
        attention_w = torch.sigmoid(self.expand_w(attention_w))
        return x * attention_h * attention_w


class MLP(nn.Module):
    def __init__(self, dimensions: list[int], dropout: float = 0.0, final_activation: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(len(dimensions) - 1):
            input_dim, output_dim = dimensions[index], dimensions[index + 1]
            layers.append(nn.Linear(input_dim, output_dim))
            is_last = index == len(dimensions) - 2
            if not is_last or final_activation:
                layers.extend([nn.LayerNorm(output_dim), nn.SiLU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
