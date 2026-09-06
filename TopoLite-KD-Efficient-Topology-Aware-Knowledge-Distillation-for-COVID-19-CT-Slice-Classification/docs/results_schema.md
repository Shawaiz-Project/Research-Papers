# Result Schema

## `test_metrics.json`

Scalar fields include:

- confusion counts: `tn`, `fp`, `fn`, `tp`, `n`;
- threshold metrics: `accuracy`, `balanced_accuracy`, `sensitivity`, `specificity`, `precision`, `negative_predictive_value`, `false_positive_rate`, `false_negative_rate`, `f1`, `mcc`, and `cohen_kappa`;
- ranking metrics: `auroc`, `auprc`;
- calibration/probabilistic metrics: `brier`, `nll`, and `ece`;
- the fixed validation-selected `threshold`.

## `test_bootstrap_ci.json`

Each bootstrapped metric contains `point`, `lower`, and `upper` under the configured confidence level.

## Prediction CSV

Every row contains `path`, `sha256`, `label`, `logit`, `probability`, `prediction`, and `threshold`. Optional fields include branch logits, visual-gate summaries, and individual routed-expert probabilities. SHA-256 values enable strict paired alignment for McNemar and bootstrap analysis.

## Curve data

`figures/curve_data/` stores the numeric data behind ROC, precision-recall, calibration, and threshold-analysis plots. These files support exact paper-figure regeneration without rerunning inference.
