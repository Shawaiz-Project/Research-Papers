from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, CoordinateAttention, DepthwiseSeparableResidual, valid_groups


class LightweightVisualEncoder(nn.Module):
    """Paper-aligned lightweight branch: 24→32→48→96→160 with CA after stages 2 and 3."""

    def __init__(
        self,
        input_channels: int,
        channels: list[int],
        embedding_dim: int,
        dropout: float,
        coordinate_attention: bool = True,
        attention_stage_indices: tuple[int, ...] = (1, 2),
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError("TopoLite visual_channels must contain exactly five values: stem plus four stages.")
        self.stem = ConvNormAct(input_channels, channels[0], 3, stride=2)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    DepthwiseSeparableResidual(channels[index - 1], channels[index], stride=2, dropout=dropout * 0.25),
                    DepthwiseSeparableResidual(channels[index], channels[index], stride=1, dropout=dropout * 0.25),
                )
                for index in range(1, len(channels))
            ]
        )
        self.attention_stage_indices = tuple(int(index) for index in attention_stage_indices)
        self.attentions = nn.ModuleDict(
            {
                str(index): CoordinateAttention(channels[index + 1]) if coordinate_attention else nn.Identity()
                for index in self.attention_stage_indices
            }
        )
        hidden = max(embedding_dim * 3, 192)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def _apply_stage_attention(self, x: torch.Tensor, stage_index: int) -> torch.Tensor:
        key = str(stage_index)
        return self.attentions[key](x) if key in self.attentions else x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage_index, stage in enumerate(self.stages):
            x = self._apply_stage_attention(stage(x), stage_index)
            features.append(x)
        embedding = self.projection(self.pool(x))
        return {"embedding": embedding, "feature_map": x, "stage_features": features}


class StandardCNNEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, embedding_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        channels = [16, 32, 64, 128]
        blocks: list[nn.Module] = []
        current = input_channels
        for output in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(current, output, 3, padding=1, bias=False),
                    nn.GroupNorm(valid_groups(output), output),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(output, output, 3, padding=1, bias=False),
                    nn.GroupNorm(valid_groups(output), output),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            current = output
        self.stages = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(channels[-1], embedding_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return {"embedding": self.projection(self.pool(x)), "feature_map": x, "stage_features": features}


class TopologyConditionedVisualEncoder(LightweightVisualEncoder):
    def __init__(self, *args, topology_dim: int, **kwargs):
        super().__init__(*args, **kwargs)
        channels = kwargs["channels"] if "channels" in kwargs else args[1]
        conditioned_channels = channels[-3:]
        self.condition_start = len(self.stages) - 3
        self.film_layers = nn.ModuleList([nn.Linear(topology_dim, 2 * channel) for channel in conditioned_channels])

    def forward(self, x: torch.Tensor, topology_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        features = []
        film_index = 0
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            if stage_index >= self.condition_start:
                gamma, beta = self.film_layers[film_index](topology_embedding).chunk(2, dim=1)
                x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
                film_index += 1
            x = self._apply_stage_attention(x, stage_index)
            features.append(x)
        embedding = self.projection(self.pool(x))
        return {"embedding": embedding, "feature_map": x, "stage_features": features}
