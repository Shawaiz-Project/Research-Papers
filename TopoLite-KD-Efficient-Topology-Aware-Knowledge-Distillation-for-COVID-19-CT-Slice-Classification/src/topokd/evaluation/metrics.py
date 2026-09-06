from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selector = (probabilities > lower) & (probabilities <= upper)
        if not selector.any():
            continue
        confidence = probabilities[selector].mean()
        accuracy = labels[selector].mean()
        ece += selector.mean() * abs(accuracy - confidence)
    return float(ece)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    ece_bins: int = 15,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "n": int(len(labels)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "cohen_kappa": float(cohen_kappa_score(labels, predictions)),
        "nll": float(log_loss(labels, np.clip(probabilities, 1e-7, 1.0 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities, ece_bins),
    }
    metrics["auroc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else math.nan
    metrics["auprc"] = (
        float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) > 1 else math.nan
    )
    return metrics


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, strategy: str = "youden") -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int64)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if strategy == "youden":
            score = sensitivity + specificity - 1.0
        elif strategy == "f1":
            score = f1_score(labels, predictions, zero_division=0)
        elif strategy == "mcc":
            score = matthews_corrcoef(labels, predictions)
        else:
            raise ValueError(f"Unsupported threshold strategy: {strategy}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
