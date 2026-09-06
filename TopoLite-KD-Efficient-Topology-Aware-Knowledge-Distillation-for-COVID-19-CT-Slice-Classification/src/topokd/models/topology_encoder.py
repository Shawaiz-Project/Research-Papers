from __future__ import annotations

import torch
from torch import nn

from .blocks import MLP


class DescriptorTopologyEncoder(nn.Module):
    def __init__(self, input_dim: int = 134, embedding_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = MLP([input_dim, 128, embedding_dim], dropout=dropout)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.encoder(descriptor)


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (sequence * weights).sum(dim=1) / denominator


class MultiScaleTopologyEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 192,
        maximum_groups: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_projection = nn.Sequential(nn.Linear(4, embedding_dim), nn.LayerNorm(embedding_dim), nn.SiLU())
        self.dimension_embedding = nn.Embedding(2, embedding_dim)
        self.group_embedding = nn.Embedding(maximum_groups, embedding_dim)
        encoder_layer_h0 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer_h1 = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.h0_encoder = nn.TransformerEncoder(encoder_layer_h0, num_layers=layers, enable_nested_tensor=False)
        self.h1_encoder = nn.TransformerEncoder(encoder_layer_h1, num_layers=layers, enable_nested_tensor=False)
        self.h0_to_h1 = nn.MultiheadAttention(embedding_dim, heads, dropout=dropout, batch_first=True)
        self.h1_to_h0 = nn.MultiheadAttention(embedding_dim, heads, dropout=dropout, batch_first=True)
        self.output = MLP([embedding_dim * 2, embedding_dim * 2, embedding_dim], dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        dimension_ids: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = self.token_projection(tokens)
        embedded = embedded + self.dimension_embedding(dimension_ids) + self.group_embedding(group_ids)
        h0_selector = dimension_ids[0] == 0
        h1_selector = dimension_ids[0] == 1
        h0, h1 = embedded[:, h0_selector], embedded[:, h1_selector]
        mask_h0, mask_h1 = mask[:, h0_selector].clone(), mask[:, h1_selector].clone()
        empty_h0 = ~mask_h0.any(dim=1)
        empty_h1 = ~mask_h1.any(dim=1)
        if empty_h0.any():
            mask_h0[empty_h0, 0] = True
        if empty_h1.any():
            mask_h1[empty_h1, 0] = True
        encoded_h0 = self.h0_encoder(h0, src_key_padding_mask=~mask_h0)
        encoded_h1 = self.h1_encoder(h1, src_key_padding_mask=~mask_h1)
        h0_context, _ = self.h1_to_h0(encoded_h0, encoded_h1, encoded_h1, key_padding_mask=~mask_h1)
        h1_context, _ = self.h0_to_h1(encoded_h1, encoded_h0, encoded_h0, key_padding_mask=~mask_h0)
        pooled_h0 = masked_mean(encoded_h0 + h0_context, mask_h0)
        pooled_h1 = masked_mean(encoded_h1 + h1_context, mask_h1)
        embedding = self.output(torch.cat([pooled_h0, pooled_h1], dim=1))
        return {
            "embedding": embedding,
            "h0_embedding": pooled_h0,
            "h1_embedding": pooled_h1,
            "h0_tokens": encoded_h0 + h0_context,
            "h1_tokens": encoded_h1 + h1_context,
            "h0_mask": mask_h0,
            "h1_mask": mask_h1,
        }
