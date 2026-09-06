from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetTeacher(nn.Module):
    feature_dim = 1280

    def __init__(
        self,
        pretrained: bool = True,
        checkpoint: str | None = None,
        freeze: bool = True,
        backbone_checkpoint: str | None = None,
    ):
        super().__init__()
        checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
        backbone_path = Path(backbone_checkpoint).expanduser() if backbone_checkpoint else None
        if checkpoint_path is not None and not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {checkpoint_path}. Run scripts/train_teacher.py or attach the archived teacher_best.pt."
            )
        if backbone_path is not None and not backbone_path.exists():
            raise FileNotFoundError(f"EfficientNet-B0 backbone checkpoint not found: {backbone_path}")

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained and checkpoint_path is None and backbone_path is None else None
        try:
            model = efficientnet_b0(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "EfficientNet-B0 pretrained weights could not be loaded. Supply model.teacher.backbone_checkpoint, "
                "attach a trained teacher checkpoint, enable internet/cache access, or set model.teacher.pretrained=false."
            ) from exc

        if backbone_path is not None and checkpoint_path is None:
            payload = torch.load(backbone_path, map_location="cpu", weights_only=False)
            state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
            model.load_state_dict(state, strict=True)

        original = model.features[0][0]
        grayscale = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        with torch.no_grad():
            grayscale.weight.copy_(original.weight.mean(dim=1, keepdim=True))
        model.features[0][0] = grayscale
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        self.model = model

        if checkpoint_path is not None:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = payload.get("model_state", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
            missing, unexpected = self.load_state_dict(state, strict=False)
            if missing or unexpected:
                warnings.warn(f"Teacher checkpoint load: missing={missing}, unexpected={unexpected}")
        if freeze:
            self.requires_grad_(False)
            self.eval()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.model.features(image)
        pooled = self.model.avgpool(features).flatten(1)
        logits = self.model.classifier(pooled).squeeze(1)
        return {"logits": logits, "features": pooled, "feature_map": features}

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.parameters()):
            super().train(False)
        return self
