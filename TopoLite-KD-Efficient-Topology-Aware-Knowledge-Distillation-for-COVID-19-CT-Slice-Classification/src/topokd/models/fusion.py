from __future__ import annotations

import torch
from torch import nn

from .blocks import MLP


class GatedFusion(nn.Module):
    """Reliability-aware gate with an explicit visual–topology interaction term."""

    def __init__(self, visual_dim: int, topology_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, output_dim)
        self.topology_projection = nn.Linear(topology_dim, output_dim)
        gate_hidden = max(output_dim + 16, output_dim)
        self.gate = MLP([visual_dim + topology_dim, gate_hidden, output_dim], dropout=dropout)
        self.interaction_projection = MLP([output_dim * 2, output_dim], dropout=dropout)

    def forward(self, visual: torch.Tensor, topology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.gate(torch.cat([visual, topology], dim=1)))
        visual_projected = self.visual_projection(visual)
        topology_projected = self.topology_projection(topology)
        blend = gate * visual_projected + (1.0 - gate) * topology_projected
        interaction = visual_projected * topology_projected
        fused = self.interaction_projection(torch.cat([blend, interaction], dim=1))
        return fused, gate


class FixedFusion(nn.Module):
    def __init__(self, visual_dim: int, topology_dim: int, output_dim: int):
        super().__init__()
        self.visual_projection = nn.Linear(visual_dim, output_dim)
        self.topology_projection = nn.Linear(topology_dim, output_dim)
        self.interaction_projection = nn.Linear(output_dim * 2, output_dim)

    def forward(self, visual: torch.Tensor, topology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual_projected = self.visual_projection(visual)
        topology_projected = self.topology_projection(topology)
        blend = 0.5 * visual_projected + 0.5 * topology_projected
        fused = self.interaction_projection(torch.cat([blend, visual_projected * topology_projected], dim=1))
        gate = torch.full_like(blend, 0.5)
        return fused, gate


class ConcatenationFusion(nn.Module):
    def __init__(self, visual_dim: int, topology_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.network = MLP([visual_dim + topology_dim, output_dim * 2, output_dim], dropout=dropout)

    def forward(self, visual: torch.Tensor, topology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.network(torch.cat([visual, topology], dim=1))
        return fused, torch.empty(0, device=fused.device)


class RoutedFusion(nn.Module):
    def __init__(self, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.gated = GatedFusion(dimension, dimension, dimension, dropout)
        self.concat = ConcatenationFusion(dimension, dimension, dimension, dropout)
        self.interaction = MLP([dimension * 3, dimension * 2, dimension], dropout=dropout)
        self.router = MLP([dimension * 2, dimension, 3], dropout=dropout)

    def forward(self, visual: torch.Tensor, topology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated, _ = self.gated(visual, topology)
        concatenated, _ = self.concat(visual, topology)
        interaction = self.interaction(torch.cat([visual, topology, visual * topology], dim=1))
        expert_stack = torch.stack([gated, concatenated, interaction], dim=1)
        weights = torch.softmax(self.router(torch.cat([visual, topology], dim=1)), dim=1)
        fused = (expert_stack * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights
