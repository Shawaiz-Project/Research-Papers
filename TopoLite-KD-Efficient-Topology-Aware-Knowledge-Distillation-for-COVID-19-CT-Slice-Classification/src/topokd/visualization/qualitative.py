from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topokd.engine.inference import model_inputs

from .gradcam import GradCAM, save_gradcam_figure


def select_qualitative_cases(predictions: pd.DataFrame, per_class: int = 12) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correct"] = frame["label"] == frame["prediction"]
    frame["confidence"] = np.where(frame["prediction"] == 1, frame["probability"], 1.0 - frame["probability"])
    class_selections = []
    for label in (0, 1):
        subset = frame[frame["label"] == label].copy()
        if subset.empty:
            continue
        correct_budget = max(1, per_class // 2) if per_class > 1 else 1
        correct = subset[subset["correct"]].sort_values("confidence", ascending=False).head(correct_budget)
        selected_hashes = set(correct["sha256"])
        remaining = max(0, per_class - len(correct))
        errors = (
            subset[(~subset["correct"]) & (~subset["sha256"].isin(selected_hashes))]
            .sort_values("confidence", ascending=False)
            .head(remaining)
        )
        selected_hashes.update(errors["sha256"])
        remaining = max(0, per_class - len(correct) - len(errors))
        uncertain = (
            subset[~subset["sha256"].isin(selected_hashes)]
            .assign(distance_to_boundary=lambda item: (item["probability"] - 0.5).abs())
            .sort_values("distance_to_boundary")
            .head(remaining)
            .drop(columns=["distance_to_boundary"])
        )
        selection = pd.concat([correct, errors, uncertain], ignore_index=True).drop_duplicates(subset=["sha256"]).head(per_class)
        class_selections.append(selection)
    if not class_selections:
        return frame.iloc[0:0].copy()
    return pd.concat(class_selections, ignore_index=True)


def generate_gradcams(
    model: torch.nn.Module,
    dataset,
    predictions_csv: str | Path,
    output_dir: str | Path,
    target_layer: str,
    threshold: float,
    per_class: int,
    device: torch.device,
) -> None:
    frame = pd.read_csv(predictions_csv)
    frame["prediction"] = (frame["probability"] >= threshold).astype(int)
    selected = select_qualitative_cases(frame, per_class)
    lookup = {row["sha256"]: index for index, row in dataset.frame.iterrows()}
    gradcam = GradCAM(model, target_layer)
    model.eval()
    try:
        for _, row in selected.iterrows():
            index = lookup.get(row["sha256"])
            if index is None:
                continue
            sample = dataset[index]
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0)
                else:
                    batch[key] = [value]
            kwargs = model_inputs(batch, device)
            heatmap, probability = gradcam(kwargs, target_class=int(row["prediction"]))
            status = "correct" if int(row["label"]) == int(row["prediction"]) else "error"
            filename = f"{status}_y{int(row['label'])}_p{probability:.3f}_{row['sha256'][:10]}.png"
            save_gradcam_figure(row["path"], heatmap, probability, int(row["label"]), Path(output_dir) / filename)
    finally:
        gradcam.close()
