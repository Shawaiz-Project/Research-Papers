from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(labels: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels)
    predictions_a = np.asarray(predictions_a)
    predictions_b = np.asarray(predictions_b)
    correct_a = predictions_a == labels
    correct_b = predictions_b == labels
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    discordant = b + c
    p_value = float(binomtest(min(b, c), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "discordant": discordant, "p_value": p_value}


def paired_bootstrap_metric_difference(
    labels: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
    metric_fn,
    samples: int = 5000,
    seed: int = 2026,
) -> dict[str, float]:
    labels = np.asarray(labels)
    probabilities_a = np.asarray(probabilities_a)
    probabilities_b = np.asarray(probabilities_b)
    rng = np.random.default_rng(seed)
    differences = []
    n = len(labels)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        differences.append(metric_fn(labels[indices], probabilities_a[indices]) - metric_fn(labels[indices], probabilities_b[indices]))
    differences = np.asarray(differences, dtype=np.float64)
    return {
        "difference": float(metric_fn(labels, probabilities_a) - metric_fn(labels, probabilities_b)),
        "lower": float(np.quantile(differences, 0.025)),
        "upper": float(np.quantile(differences, 0.975)),
        "p_two_sided": float(min(1.0, 2.0 * min(np.mean(differences <= 0), np.mean(differences >= 0)))),
    }
