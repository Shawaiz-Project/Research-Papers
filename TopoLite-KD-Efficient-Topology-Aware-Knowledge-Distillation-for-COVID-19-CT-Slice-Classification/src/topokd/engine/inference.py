from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm


def model_inputs(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    inputs = {"image": batch["image"].to(device, non_blocking=True)}
    for key in ("descriptor", "tokens", "mask", "dimension_ids", "group_ids"):
        if key in batch:
            inputs[key] = batch[key].to(device, non_blocking=True)
    return inputs


@torch.inference_mode()
def predict_loader(model: torch.nn.Module, loader, device: torch.device, description: str = "Predict") -> dict:
    model.eval()
    arrays: dict[str, list] = defaultdict(list)
    paths: list[str] = []
    hashes: list[str] = []
    for batch in tqdm(loader, desc=description, leave=False):
        output = model(**model_inputs(batch, device))
        logits = output["logits"]
        arrays["logits"].append(logits.detach().cpu().numpy())
        arrays["labels"].append(batch["label"].numpy())
        for optional in ("visual_logits", "topology_logits", "fusion_weights", "router_weights"):
            if optional in output and output[optional].numel() > 0:
                arrays[optional].append(output[optional].detach().cpu().numpy())
        paths.extend(list(batch["path"]))
        hashes.extend(list(batch["sha256"]))
    result = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    result["probabilities"] = 1.0 / (1.0 + np.exp(-result["logits"]))
    result["paths"] = paths
    result["sha256"] = hashes
    return result
