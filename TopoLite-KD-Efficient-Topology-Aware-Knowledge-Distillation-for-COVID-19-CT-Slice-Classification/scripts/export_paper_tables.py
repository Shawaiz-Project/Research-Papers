from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready CSV and LaTeX tables from completed runs.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/paper_tables")
    args = parser.parse_args()
    rows = []
    for metrics_path in Path(args.results_root).glob("*/seed_*/metrics/test_metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "experiment": metrics_path.parents[2].name,
                "seed": int(metrics_path.parents[1].name.removeprefix("seed_")),
                **metrics,
            }
        )
    if not rows:
        raise FileNotFoundError("No completed run metrics found.")
    frame = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "all_runs.csv", index=False)
    metrics = ["accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "auroc", "auprc", "brier", "ece"]
    summary = frame.groupby("experiment")[metrics].agg(["mean", "std"])
    summary.to_csv(output_dir / "mean_std.csv")
    (output_dir / "mean_std.tex").write_text(summary.to_latex(float_format="%.4f"), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
