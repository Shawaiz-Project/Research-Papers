from __future__ import annotations

import torch
from torch import nn

from .fusion import RoutedFusion
from .topology_encoder import MultiScaleTopologyEncoder
from .visual_encoder import TopologyConditionedVisualEncoder


class TopoLiteMSFKD(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        msf_cfg = model_cfg["msf"]
        dimension = int(msf_cfg["embedding_dim"])
        dropout = float(model_cfg["dropout"])
        self.topology_encoder = MultiScaleTopologyEncoder(
            embedding_dim=dimension,
            heads=int(msf_cfg["transformer_heads"]),
            layers=int(msf_cfg["transformer_layers"]),
            feedforward_dim=int(msf_cfg["transformer_ffn_dim"]),
            dropout=dropout,
        )
        self.visual_encoder = TopologyConditionedVisualEncoder(
            input_channels=int(model_cfg["input_channels"]),
            channels=[int(value) for value in model_cfg["visual_channels"]],
            embedding_dim=dimension,
            dropout=dropout,
            coordinate_attention=bool(model_cfg.get("coordinate_attention", True)),
            topology_dim=dimension,
        )
        self.fusion = RoutedFusion(dimension, dropout)
        self.fused_head = nn.Linear(dimension, 1)
        self.visual_head = nn.Linear(dimension, 1)
        self.topology_head = nn.Linear(dimension, 1)
        self.student_to_teacher = (
            nn.Linear(dimension, int(model_cfg.get("teacher_feature_dim", 1280)))
            if float(config["loss"].get("feature_kd_weight", 0.0)) > 0
            else None
        )

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        dimension_ids: torch.Tensor,
        group_ids: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        topology_payload = self.topology_encoder(tokens, mask, dimension_ids, group_ids)
        topology = topology_payload["embedding"]
        visual_payload = self.visual_encoder(image, topology)
        visual = visual_payload["embedding"]
        fused, router_weights = self.fusion(visual, topology)
        output = {
            "logits": self.fused_head(fused).squeeze(1),
            "visual_logits": self.visual_head(visual).squeeze(1),
            "topology_logits": self.topology_head(topology).squeeze(1),
            "visual_embedding": visual,
            "topology_embedding": topology,
            "fused_embedding": fused,
            "feature_map": visual_payload["feature_map"],
            "stage_features": visual_payload["stage_features"],
            "router_weights": router_weights,
            "h0_embedding": topology_payload["h0_embedding"],
            "h1_embedding": topology_payload["h1_embedding"],
        }
        if self.student_to_teacher is not None:
            output["projected_student_feature"] = self.student_to_teacher(visual)
        return output
