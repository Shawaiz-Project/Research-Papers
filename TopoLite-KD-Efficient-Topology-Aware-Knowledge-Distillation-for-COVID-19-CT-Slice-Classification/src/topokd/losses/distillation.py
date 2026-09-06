from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def binary_response_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    teacher_targets = torch.sigmoid(teacher_logits.detach() / temperature)
    return F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_targets) * (temperature**2)


def normalized_feature_distillation(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    student = F.normalize(student_features, dim=1)
    teacher = F.normalize(teacher_features.detach(), dim=1)
    return F.mse_loss(student, teacher)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        similarities = features @ features.T / self.temperature
        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.T)
        identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive_mask = positive_mask & ~identity
        logits_mask = ~identity
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(similarities) * logits_mask
        log_prob = similarities - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not valid.any():
            return features.new_zeros(())
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
        return -mean_log_prob_pos[valid].mean()
