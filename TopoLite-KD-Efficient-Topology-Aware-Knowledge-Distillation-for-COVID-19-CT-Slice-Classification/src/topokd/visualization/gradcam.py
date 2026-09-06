from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def resolve_module(root: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    current = root
    for part in dotted_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = resolve_module(model, target_layer)
        self.forward_handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient

    def __call__(self, model_kwargs: dict[str, torch.Tensor], target_class: int = 1) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        output = self.model(**model_kwargs)
        logit = output["logits"][0]
        score = logit if int(target_class) == 1 else -logit
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=model_kwargs["image"].shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), float(torch.sigmoid(logit).detach())

    def close(self):
        self.forward_handle.remove()


def save_gradcam_figure(
    image_path: str,
    heatmap: np.ndarray,
    probability: float,
    label: int,
    output_path: str | Path,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("L").resize((heatmap.shape[1], heatmap.shape[0]), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(array, cmap="gray")
    axes[0].set_title("CT Slice")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(array, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title(f"Overlay | y={label} | p={probability:.3f}")
    for axis in axes:
        axis.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
