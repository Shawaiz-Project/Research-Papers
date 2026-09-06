import numpy as np

from topokd.evaluation.metrics import binary_metrics, choose_threshold


def test_perfect_binary_metrics():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.90, 0.99])
    metrics = binary_metrics(labels, probabilities, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["auroc"] == 1.0


def test_threshold_selection_returns_valid_probability():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.10, 0.40, 0.60, 0.90])
    threshold = choose_threshold(labels, probabilities, strategy="youden")
    assert 0.0 <= threshold <= 1.0
