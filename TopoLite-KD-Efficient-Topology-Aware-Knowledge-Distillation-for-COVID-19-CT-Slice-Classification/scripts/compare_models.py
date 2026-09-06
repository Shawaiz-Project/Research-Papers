from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.evaluation.metrics import binary_metrics
from topokd.evaluation.statistical_tests import mcnemar_exact, paired_bootstrap_metric_difference


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two models on exactly aligned frozen-test predictions.")
    parser.add_argument("--model-a", required=True, help="test_predictions.csv for model A")
    parser.add_argument("--model-b", required=True, help="test_predictions.csv for model B")
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--output", default="results/model_comparison.json")
    args = parser.parse_args()
    a = pd.read_csv(args.model_a)
    b = pd.read_csv(args.model_b)
    merged = a.merge(b, on="sha256", suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Prediction sets do not contain exactly the same hashes.")
    if not np.array_equal(merged["label_a"].to_numpy(), merged["label_b"].to_numpy()):
        raise ValueError("Aligned rows contain conflicting labels.")
    labels = merged["label_a"].to_numpy()
    probabilities_a = merged["probability_a"].to_numpy()
    probabilities_b = merged["probability_b"].to_numpy()
    predictions_a = merged["prediction_a"].to_numpy()
    predictions_b = merged["prediction_b"].to_numpy()
    threshold_a = float(merged["threshold_a"].iloc[0]) if "threshold_a" in merged else 0.5
    threshold_b = float(merged["threshold_b"].iloc[0]) if "threshold_b" in merged else 0.5
    result = {
        args.name_a: binary_metrics(labels, probabilities_a, threshold_a),
        args.name_b: binary_metrics(labels, probabilities_b, threshold_b),
        "mcnemar_exact": mcnemar_exact(labels, predictions_a, predictions_b),
        "paired_bootstrap_auroc_difference_a_minus_b": paired_bootstrap_metric_difference(
            labels, probabilities_a, probabilities_b, roc_auc_score
        ),
        "aligned_samples": int(len(merged)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
