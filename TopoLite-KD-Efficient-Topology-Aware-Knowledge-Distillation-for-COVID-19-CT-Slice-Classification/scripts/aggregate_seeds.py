from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate test metrics across seeds and experiments.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", default="results/aggregate_seed_metrics.csv")
    args = parser.parse_args()
    rows = []
    for path in Path(args.results_root).glob("*/seed_*/metrics/test_metrics.json"):
        metrics = json.loads(path.read_text())
        seed = int(path.parents[1].name.removeprefix("seed_"))
        experiment = path.parents[2].name
        rows.append({"experiment": experiment, "seed": seed, **metrics})
    if not rows:
        raise FileNotFoundError("No test_metrics.json files were found.")
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    numeric = [column for column in frame.select_dtypes("number").columns if column not in {"seed", "n", "tn", "fp", "fn", "tp"}]
    summary = frame.groupby("experiment")[numeric].agg(["mean", "std", "min", "max"])
    summary.to_csv(output.with_name(output.stem + "_summary.csv"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
