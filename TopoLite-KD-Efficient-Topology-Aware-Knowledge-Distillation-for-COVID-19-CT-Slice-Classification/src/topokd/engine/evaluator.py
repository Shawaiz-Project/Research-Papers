from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topokd.evaluation import TemperatureScaler, binary_metrics, bootstrap_confidence_intervals, choose_threshold
from topokd.utils.io import atomic_json_dump, ensure_dir
from topokd.visualization import (
    plot_calibration,
    plot_confusion,
    plot_fusion_diagnostics,
    plot_probability_histogram,
    plot_roc_pr,
    plot_threshold_analysis,
)

from .inference import predict_loader


def evaluate_splits(
    model: torch.nn.Module,
    val_loader,
    test_loader,
    config: dict,
    device: torch.device,
    output_dir: str | Path,
) -> dict:
    output_dir = ensure_dir(output_dir)
    metrics_dir = ensure_dir(output_dir / "metrics")
    figures_dir = ensure_dir(output_dir / "figures")
    predictions_dir = ensure_dir(output_dir / "predictions")
    val = predict_loader(model, val_loader, device, "Validation inference")
    scaler = TemperatureScaler()
    if bool(config["evaluation"].get("temperature_scaling", True)):
        temperature = scaler.fit(val["logits"], val["labels"])
    else:
        temperature = 1.0
    val_probabilities = scaler.transform_probabilities(val["logits"])
    strategy = config["evaluation"].get("threshold_strategy", "youden")
    if strategy == "fixed":
        threshold = float(config["evaluation"].get("fixed_threshold", 0.5))
    else:
        threshold = choose_threshold(val["labels"], val_probabilities, strategy)
    test = predict_loader(model, test_loader, device, "Test inference")
    test_probabilities = scaler.transform_probabilities(test["logits"])
    ece_bins = int(config["evaluation"].get("ece_bins", 15))
    val_metrics = binary_metrics(val["labels"], val_probabilities, threshold, ece_bins)
    test_metrics = binary_metrics(test["labels"], test_probabilities, threshold, ece_bins)
    intervals = bootstrap_confidence_intervals(
        test["labels"],
        test_probabilities,
        threshold,
        samples=int(config["evaluation"].get("bootstrap_samples", 2000)),
        seed=int(config["evaluation"].get("bootstrap_seed", 2026)),
        confidence_level=float(config["evaluation"].get("confidence_level", 0.95)),
        ece_bins=ece_bins,
    )
    calibration = {"temperature": temperature, "threshold": threshold, "threshold_strategy": strategy}
    atomic_json_dump(calibration, metrics_dir / "calibration.json")
    atomic_json_dump(val_metrics, metrics_dir / "validation_metrics.json")
    atomic_json_dump(test_metrics, metrics_dir / "test_metrics.json")
    atomic_json_dump(intervals, metrics_dir / "test_bootstrap_ci.json")
    _save_predictions(val, val_probabilities, threshold, predictions_dir / "validation_predictions.csv")
    test_predictions_path = predictions_dir / "test_predictions.csv"
    _save_predictions(test, test_probabilities, threshold, test_predictions_path)
    plot_confusion(test["labels"], test_probabilities, threshold, figures_dir / "confusion_matrix.png")
    plot_roc_pr(test["labels"], test_probabilities, figures_dir)
    plot_calibration(test["labels"], test_probabilities, figures_dir / "calibration_curve.png", ece_bins)
    plot_probability_histogram(test["labels"], test_probabilities, figures_dir / "probability_histogram.png")
    plot_threshold_analysis(test["labels"], test_probabilities, threshold, figures_dir)
    plot_fusion_diagnostics(pd.read_csv(test_predictions_path), figures_dir)
    return {
        "calibration": calibration,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_bootstrap_ci": intervals,
    }


def _save_predictions(payload: dict, probabilities: np.ndarray, threshold: float, path: Path) -> None:
    frame = pd.DataFrame(
        {
            "path": payload["paths"],
            "sha256": payload["sha256"],
            "label": payload["labels"].astype(int),
            "logit": payload["logits"],
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
            "threshold": np.full(len(probabilities), threshold, dtype=np.float64),
        }
    )
    for optional in ("visual_logits", "topology_logits"):
        if optional in payload:
            frame[optional] = payload[optional]
    if "fusion_weights" in payload and payload["fusion_weights"].ndim == 2:
        frame["fusion_weight_mean"] = payload["fusion_weights"].mean(axis=1)
        frame["fusion_weight_min"] = payload["fusion_weights"].min(axis=1)
        frame["fusion_weight_max"] = payload["fusion_weights"].max(axis=1)
    if "router_weights" in payload and payload["router_weights"].ndim == 2:
        for index in range(payload["router_weights"].shape[1]):
            frame[f"router_expert_{index}_weight"] = payload["router_weights"][:, index]
    frame.to_csv(path, index=False)
