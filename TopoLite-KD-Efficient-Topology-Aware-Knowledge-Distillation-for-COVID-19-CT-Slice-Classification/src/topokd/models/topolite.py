from __future__ import annotations

import torch
from torch import nn

from .fusion import ConcatenationFusion, FixedFusion, GatedFusion
from .topology_encoder import DescriptorTopologyEncoder
from .visual_encoder import LightweightVisualEncoder, StandardCNNEncoder


class TopoLiteModel(nn.Module):
    def __init__(self, config: dict, baseline: bool = False):
        super().__init__()
        model_cfg = config["model"]
        topology_cfg = config["topology"]
        dropout = float(model_cfg["dropout"])
        self.variant = model_cfg["name"]
        self.visual_only = self.variant == "visual_only"
        self.topology_only = self.variant == "tda_only"
        visual_dim = int(model_cfg["visual_embedding_dim"])
        topology_dim = int(model_cfg["topology_embedding_dim"])
        fusion_dim = int(model_cfg["fusion_dim"])
        if baseline:
            visual_dim = max(128, visual_dim)
            self.visual_encoder = StandardCNNEncoder(int(model_cfg["input_channels"]), visual_dim, dropout)
        else:
            self.visual_encoder = LightweightVisualEncoder(
                input_channels=int(model_cfg["input_channels"]),
                channels=[int(value) for value in model_cfg["visual_channels"]],
                embedding_dim=visual_dim,
                dropout=dropout,
                coordinate_attention=bool(model_cfg.get("coordinate_attention", True)),
            )
        self.topology_encoder = DescriptorTopologyEncoder(
            input_dim=int(topology_cfg["descriptor_dim"]),
            embedding_dim=topology_dim,
            dropout=dropout,
        )
        fusion_name = model_cfg.get("fusion", "gated")
        if fusion_name == "gated":
            self.fusion = GatedFusion(visual_dim, topology_dim, fusion_dim, dropout)
        elif fusion_name == "fixed":
            self.fusion = FixedFusion(visual_dim, topology_dim, fusion_dim)
        elif fusion_name == "concat":
            self.fusion = ConcatenationFusion(visual_dim, topology_dim, fusion_dim, dropout)
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_name}")
        self.visual_head = nn.Linear(visual_dim, 1)
        self.topology_head = nn.Linear(topology_dim, 1)
        self.fused_head = nn.Linear(fusion_dim, 1)
        self.visual_only_head = nn.Linear(visual_dim, 1)
        self.topology_only_head = nn.Linear(topology_dim, 1)
        teacher_feature_dim = int(model_cfg.get("teacher_feature_dim", 1280))
        self.student_to_teacher = (
            nn.Linear(visual_dim, teacher_feature_dim)
            if float(config["loss"].get("feature_kd_weight", 0.0)) > 0
            else None
        )

    def forward(self, image: torch.Tensor, descriptor: torch.Tensor | None = None, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        visual_payload = self.visual_encoder(image)
        visual = visual_payload["embedding"]
        output = {
            "visual_embedding": visual,
            "feature_map": visual_payload["feature_map"],
            "stage_features": visual_payload["stage_features"],
            "visual_logits": self.visual_head(visual).squeeze(1),
        }
        if self.student_to_teacher is not None:
            output["projected_student_feature"] = self.student_to_teacher(visual)
        if self.visual_only:
            output["logits"] = self.visual_only_head(visual).squeeze(1)
            output["fused_embedding"] = visual
            return output
        if descriptor is None:
            raise ValueError("A topology descriptor is required for this model variant.")
        topology = self.topology_encoder(descriptor)
        output.update(
            {
                "topology_embedding": topology,
                "topology_logits": self.topology_head(topology).squeeze(1),
            }
        )
        if self.topology_only:
            output["logits"] = self.topology_only_head(topology).squeeze(1)
            output["fused_embedding"] = topology
            return output
        fused, gate = self.fusion(visual, topology)
        output.update(
            {
                "logits": self.fused_head(fused).squeeze(1),
                "fused_embedding": fused,
                "fusion_weights": gate,
            }
        )
        return output


class VisualOnlyModel(nn.Module):
    """Visual-only ablation with no instantiated topology or fusion parameters."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        visual_dim = int(model_cfg["visual_embedding_dim"])
        self.visual_encoder = LightweightVisualEncoder(
            input_channels=int(model_cfg["input_channels"]),
            channels=[int(value) for value in model_cfg["visual_channels"]],
            embedding_dim=visual_dim,
            dropout=float(model_cfg["dropout"]),
            coordinate_attention=bool(model_cfg.get("coordinate_attention", True)),
        )
        self.classifier = nn.Linear(visual_dim, 1)
        self.student_to_teacher = (
            nn.Linear(visual_dim, int(model_cfg.get("teacher_feature_dim", 1280)))
            if float(config["loss"].get("feature_kd_weight", 0.0)) > 0
            else None
        )

    def forward(self, image: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        payload = self.visual_encoder(image)
        embedding = payload["embedding"]
        output = {
            "logits": self.classifier(embedding).squeeze(1),
            "visual_logits": self.classifier(embedding).squeeze(1),
            "visual_embedding": embedding,
            "fused_embedding": embedding,
            "feature_map": payload["feature_map"],
            "stage_features": payload["stage_features"],
        }
        if self.student_to_teacher is not None:
            output["projected_student_feature"] = self.student_to_teacher(embedding)
        return output


class TopologyOnlyModel(nn.Module):
    """Topology-only ablation with no instantiated visual parameters."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.topology_encoder = DescriptorTopologyEncoder(
            input_dim=int(config["topology"]["descriptor_dim"]),
            embedding_dim=int(model_cfg["topology_embedding_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
        self.classifier = nn.Linear(int(model_cfg["topology_embedding_dim"]), 1)

    def forward(self, descriptor: torch.Tensor | None = None, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        if descriptor is None:
            raise ValueError("TopologyOnlyModel requires a descriptor tensor.")
        embedding = self.topology_encoder(descriptor)
        logits = self.classifier(embedding).squeeze(1)
        return {
            "logits": logits,
            "topology_logits": logits,
            "topology_embedding": embedding,
            "fused_embedding": embedding,
        }
