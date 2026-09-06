from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "resolved_config.yaml",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "logs/run.log",
    "logs/history.csv",
    "logs/environment.json",
    "logs/pip_freeze.txt",
    "metrics/calibration.json",
    "metrics/validation_metrics.json",
    "metrics/test_metrics.json",
    "metrics/test_bootstrap_ci.json",
    "predictions/validation_predictions.csv",
    "predictions/test_predictions.csv",
    "figures/confusion_matrix.png",
    "figures/roc_curve.png",
    "figures/precision_recall_curve.png",
    "figures/calibration_curve.png",
    "figures/threshold_analysis.png",
    "figures/curve_data/roc_curve.csv",
    "figures/curve_data/precision_recall_curve.csv",
    "figures/curve_data/calibration_curve.csv",
    "figures/curve_data/threshold_analysis.csv",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a completed run contains the required paper artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--require-gradcam", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    missing = [relative for relative in REQUIRED_FILES if not (run_dir / relative).is_file()]
    if args.require_gradcam and not any((run_dir / "gradcam").glob("*.png")):
        missing.append("gradcam/*.png")
    metrics_path = run_dir / "metrics" / "test_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        required_metrics = {"accuracy", "sensitivity", "specificity", "f1", "mcc", "auroc", "auprc", "brier", "nll", "ece"}
        absent_metrics = sorted(required_metrics - set(metrics))
        missing.extend([f"test_metrics.json::{metric}" for metric in absent_metrics])
    if missing:
        print("Run audit failed. Missing artifacts:")
        for item in missing:
            print(f"  - {item}")
        raise SystemExit(1)
    print(f"Run audit passed: {run_dir}")


if __name__ == "__main__":
    main()
