from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import binary_metrics


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(labels)
    indices = []
    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        indices.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    result = np.concatenate(indices)
    rng.shuffle(result)
    return result


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    samples: int = 2000,
    seed: int = 2026,
    confidence_level: float = 0.95,
    ece_bins: int = 15,
) -> dict[str, dict[str, float]]:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(samples):
        indices = stratified_bootstrap_indices(labels, rng)
        metrics = binary_metrics(labels[indices], probabilities[indices], threshold, ece_bins)
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and key not in {"n", "tn", "fp", "fn", "tp", "threshold"}:
                values.setdefault(key, []).append(float(value))
    alpha = 1.0 - confidence_level
    output = {}
    point = binary_metrics(labels, probabilities, threshold, ece_bins)
    for key, estimates in values.items():
        clean = np.asarray(estimates, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        output[key] = {
            "point": float(point[key]),
            "lower": float(np.quantile(clean, alpha / 2.0)),
            "upper": float(np.quantile(clean, 1.0 - alpha / 2.0)),
        }
    return output
