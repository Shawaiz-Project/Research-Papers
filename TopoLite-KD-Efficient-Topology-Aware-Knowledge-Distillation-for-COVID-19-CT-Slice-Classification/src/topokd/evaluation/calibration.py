from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


class TemperatureScaler:
    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = float(initial_temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        log_temperature = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(0.05, 20.0)
            loss = F.binary_cross_entropy_with_logits(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.exp().detach().clamp(0.05, 20.0))
        return self.temperature

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits, dtype=np.float64) / self.temperature

    def transform_probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = self.transform_logits(logits)
        return 1.0 / (1.0 + np.exp(-scaled))
