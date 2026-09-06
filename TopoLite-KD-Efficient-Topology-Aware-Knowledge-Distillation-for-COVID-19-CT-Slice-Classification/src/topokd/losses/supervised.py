from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .distillation import SupConLoss, binary_response_distillation, normalized_feature_distillation


class ResearchObjective(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super().__init__()
        loss_cfg = config["loss"]
        pos_weight = loss_cfg.get("pos_weight")
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), device=device) if pos_weight is not None else torch.tensor(1.0, device=device),
        )
        self.use_pos_weight = pos_weight is not None
        self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        self.supervised_weight = float(loss_cfg.get("supervised_weight", 1.0))
        self.response_weight = float(loss_cfg.get("response_kd_weight", 0.0))
        self.temperature = float(loss_cfg.get("response_temperature", 4.0))
        self.feature_weight = float(loss_cfg.get("feature_kd_weight", 0.0))
        self.visual_aux_weight = float(loss_cfg.get("visual_aux_weight", 0.0))
        self.topology_aux_weight = float(loss_cfg.get("topology_aux_weight", 0.0))
        self.supcon_weight = float(loss_cfg.get("supervised_contrastive_weight", 0.0))
        self.supcon = SupConLoss()

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0:
            return labels
        return labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def _bce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            self._targets(labels),
            pos_weight=self.pos_weight if self.use_pos_weight else None,
        )

    def forward(
        self,
        student_output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        teacher_output: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        supervised = self._bce(student_output["logits"], labels)
        total = self.supervised_weight * supervised
        parts = {"supervised": float(supervised.detach()), "supervised_weighted": float((self.supervised_weight * supervised).detach())}
        if self.visual_aux_weight > 0 and "visual_logits" in student_output:
            visual_aux = self._bce(student_output["visual_logits"], labels)
            total = total + self.visual_aux_weight * visual_aux
            parts["visual_aux"] = float(visual_aux.detach())
        if self.topology_aux_weight > 0 and "topology_logits" in student_output:
            topology_aux = self._bce(student_output["topology_logits"], labels)
            total = total + self.topology_aux_weight * topology_aux
            parts["topology_aux"] = float(topology_aux.detach())
        if self.response_weight > 0:
            if teacher_output is None:
                raise RuntimeError("Response KD is enabled but teacher output is missing.")
            response_kd = binary_response_distillation(
                student_output["logits"], teacher_output["logits"], self.temperature
            )
            total = total + self.response_weight * response_kd
            parts["response_kd"] = float(response_kd.detach())
        if self.feature_weight > 0:
            if teacher_output is None:
                raise RuntimeError("Feature KD is enabled but teacher output is missing.")
            feature_kd = normalized_feature_distillation(
                student_output["projected_student_feature"], teacher_output["features"]
            )
            total = total + self.feature_weight * feature_kd
            parts["feature_kd"] = float(feature_kd.detach())
        if self.supcon_weight > 0:
            contrastive = self.supcon(student_output["fused_embedding"], labels.long())
            total = total + self.supcon_weight * contrastive
            parts["supcon"] = float(contrastive.detach())
        parts["total"] = float(total.detach())
        return total, parts
