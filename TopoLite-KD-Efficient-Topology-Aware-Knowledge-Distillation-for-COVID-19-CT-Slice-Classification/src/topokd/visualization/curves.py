from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, roc_curve


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _curve_data_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "curve_data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_history(history_csv: str | Path, output_dir: str | Path) -> None:
    frame = pd.read_csv(history_csv)
    output_dir = Path(output_dir)
    if {"epoch", "train_loss", "val_loss"}.issubset(frame.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(frame["epoch"], frame["train_loss"], label="Train")
        ax.plot(frame["epoch"], frame["val_loss"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, output_dir / "loss_curves.png")
    metric_columns = [column for column in ("val_accuracy", "val_f1", "val_mcc", "val_auroc", "val_auprc") if column in frame]
    if metric_columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for column in metric_columns:
            ax.plot(frame["epoch"], frame[column], label=column.removeprefix("val_").upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.02)
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(alpha=0.25)
        _save(fig, output_dir / "validation_metrics.png")
    if "learning_rate" in frame:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(frame["epoch"], frame["learning_rate"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning-Rate Schedule")
        ax.grid(alpha=0.25)
        _save(fig, output_dir / "learning_rate.png")


def plot_confusion(labels: np.ndarray, probabilities: np.ndarray, threshold: float, path: str | Path) -> None:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(matrix, display_labels=["Non-COVID", "COVID"])
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})")
    _save(fig, path)


def plot_roc_pr(labels: np.ndarray, probabilities: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    data_dir = _curve_data_dir(output_dir)
    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_thresholds}).to_csv(data_dir / "roc_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "roc_curve.png")

    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall, precision)
    padded_thresholds = np.concatenate([pr_thresholds, [np.nan]])
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded_thresholds}).to_csv(data_dir / "precision_recall_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AUPRC = {pr_auc:.4f}")
    ax.axhline(labels.mean(), linestyle="--", label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "precision_recall_curve.png")


def plot_calibration(labels: np.ndarray, probabilities: np.ndarray, path: str | Path, bins: int = 15) -> None:
    observed, predicted = calibration_curve(labels, probabilities, n_bins=bins, strategy="uniform")
    path = Path(path)
    data_dir = _curve_data_dir(path.parent)
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_positive_fraction": observed}).to_csv(data_dir / "calibration_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_probability_histogram(labels: np.ndarray, probabilities: np.ndarray, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(probabilities[labels == 0], bins=20, alpha=0.65, label="Non-COVID")
    ax.hist(probabilities[labels == 1], bins=20, alpha=0.65, label="COVID")
    ax.set_xlabel("Predicted COVID probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Probability Distribution")
    ax.legend()
    _save(fig, path)


def plot_threshold_analysis(labels: np.ndarray, probabilities: np.ndarray, selected_threshold: float, output_dir: str | Path) -> None:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1_score(labels, predictions, zero_division=0),
                "mcc": matthews_corrcoef(labels, predictions),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    frame.to_csv(_curve_data_dir(output_dir) / "threshold_analysis.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for column in ("sensitivity", "specificity", "f1", "mcc"):
        ax.plot(frame["threshold"], frame[column], label=column.replace("_", " ").title())
    ax.axvline(selected_threshold, linestyle="--", label=f"Selected = {selected_threshold:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Threshold Sensitivity Analysis")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "threshold_analysis.png")


def plot_fusion_diagnostics(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    if "fusion_weight_mean" in predictions:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(predictions["fusion_weight_mean"], bins=20)
        ax.set_xlabel("Mean visual gate weight")
        ax.set_ylabel("Samples")
        ax.set_title("Reliability Gate Distribution")
        ax.grid(alpha=0.25)
        _save(fig, output_dir / "fusion_gate_distribution.png")
    router_columns = [column for column in predictions.columns if column.startswith("router_expert_")]
    if router_columns:
        means = predictions[router_columns].mean(axis=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar([column.replace("_weight", "") for column in router_columns], means.values)
        ax.set_ylabel("Mean routing probability")
        ax.set_title("Fusion Expert Utilization")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        _save(fig, output_dir / "router_expert_utilization.png")
