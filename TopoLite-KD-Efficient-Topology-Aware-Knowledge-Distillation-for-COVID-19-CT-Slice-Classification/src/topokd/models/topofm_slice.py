from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .blocks import MLP
from .topology_encoder import MultiScaleTopologyEncoder, masked_mean


class DINOv2Backbone(nn.Module):
    """DINOv2 adapter supporting the official torch.hub entrypoint or a local clone.

    A local repository/checkpoint is recommended for an offline Kaggle run. The
    adapter intentionally fails with an actionable error instead of replacing the
    foundation model with a dummy network.
    """

    _FEATURE_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, config: dict):
        super().__init__()
        architecture = str(config.get("architecture", "dinov2_vits14"))
        if architecture not in self._FEATURE_DIMS:
            raise ValueError(f"Unsupported DINOv2 architecture: {architecture}")
        repository_path = config.get("repository_path")
        checkpoint = config.get("checkpoint")
        pretrained = bool(config.get("pretrained", True)) and not checkpoint
        try:
            if repository_path:
                repository = Path(repository_path).expanduser()
                if not repository.exists():
                    raise FileNotFoundError(f"Local DINOv2 repository does not exist: {repository}")
                backbone = torch.hub.load(str(repository), architecture, source="local", pretrained=pretrained)
            else:
                backbone = torch.hub.load("facebookresearch/dinov2", architecture, pretrained=pretrained)
        except Exception as exc:
            raise RuntimeError(
                "DINOv2 could not be constructed. For offline execution, set "
                "model.dinov2.repository_path to a local facebookresearch/dinov2 clone and optionally "
                "model.dinov2.checkpoint to a local state_dict."
            ) from exc
        if checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint_path}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"DINOv2 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.backbone = backbone
        self.feature_dim = self._FEATURE_DIMS[architecture]
        self.freeze = bool(config.get("freeze", True))
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        payload = self.backbone.forward_features(image)
        if not isinstance(payload, dict) or "x_norm_patchtokens" not in payload:
            raise RuntimeError("Loaded DINOv2 model does not expose x_norm_patchtokens through forward_features().")
        patches = payload["x_norm_patchtokens"]
        cls = payload.get("x_norm_clstoken", patches.mean(dim=1))
        return {"patch_tokens": patches, "cls_token": cls}


class TopoFMSliceV1(nn.Module):
    """Faithful component-level reconstruction of the completed TopoFM-Slice-v1 run.

    Included components: DINOv2 visual features, learnable multi-scale topology
    tokens, bidirectional cross-attention, supervised contrastive learning support,
    and a calibrated binary classifier. Unrun patient-level/DG extensions are not
    silently enabled here.
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        fm_cfg = model_cfg["topofm"]
        topology_cfg = fm_cfg["topology_encoder"]
        dimension = int(fm_cfg.get("fusion_dim", 192))
        heads = int(fm_cfg.get("attention_heads", 6))
        dropout = float(model_cfg.get("dropout", 0.1))
        self.visual_backbone = DINOv2Backbone(model_cfg["dinov2"])
        self.topology_encoder = MultiScaleTopologyEncoder(
            embedding_dim=int(topology_cfg.get("embedding_dim", dimension)),
            heads=int(topology_cfg.get("heads", heads)),
            layers=int(topology_cfg.get("layers", 2)),
            feedforward_dim=int(topology_cfg.get("feedforward_dim", dimension * 4)),
            dropout=dropout,
        )
        topology_dim = int(topology_cfg.get("embedding_dim", dimension))
        self.visual_projection = nn.Sequential(nn.Linear(self.visual_backbone.feature_dim, dimension), nn.LayerNorm(dimension))
        self.topology_projection = nn.Sequential(nn.Linear(topology_dim, dimension), nn.LayerNorm(dimension))
        self.topology_token_projection = nn.Sequential(nn.Linear(topology_dim, dimension), nn.LayerNorm(dimension))
        self.topology_to_visual = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.visual_to_topology = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.fusion = MLP([dimension * 3, dimension * 2, dimension], dropout=dropout)
        self.classifier = nn.Linear(dimension, 1)
        self.visual_head = nn.Linear(dimension, 1)
        self.topology_head = nn.Linear(dimension, 1)

    def forward(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        dimension_ids: torch.Tensor,
        group_ids: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual_payload = self.visual_backbone(image)
        visual_patches = self.visual_projection(visual_payload["patch_tokens"])
        visual_cls = self.visual_projection(visual_payload["cls_token"])

        topology_payload = self.topology_encoder(tokens, mask, dimension_ids, group_ids)
        topology_embedding = self.topology_projection(topology_payload["embedding"])
        topology_tokens = torch.cat([topology_payload["h0_tokens"], topology_payload["h1_tokens"]], dim=1)
        topology_mask = torch.cat([topology_payload["h0_mask"], topology_payload["h1_mask"]], dim=1)
        topology_tokens = self.topology_token_projection(topology_tokens)

        topology_context, _ = self.topology_to_visual(
            topology_tokens,
            visual_patches,
            visual_patches,
            need_weights=False,
        )
        visual_context, _ = self.visual_to_topology(
            visual_cls.unsqueeze(1),
            topology_tokens,
            topology_tokens,
            key_padding_mask=~topology_mask,
            need_weights=False,
        )
        topology_context = masked_mean(topology_context, topology_mask)
        visual_context = visual_context.squeeze(1)
        fused = self.fusion(torch.cat([visual_cls, topology_context, visual_context], dim=1))
        return {
            "logits": self.classifier(fused).squeeze(1),
            "visual_logits": self.visual_head(visual_cls).squeeze(1),
            "topology_logits": self.topology_head(topology_embedding).squeeze(1),
            "visual_embedding": visual_cls,
            "topology_embedding": topology_embedding,
            "fused_embedding": fused,
            "visual_patch_tokens": visual_patches,
            "topology_tokens": topology_tokens,
        }
