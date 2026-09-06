from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .io import torch_save_atomic


def save_checkpoint(path: str | Path, **payload: Any) -> None:
    torch_save_atomic(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary payload.")
    return checkpoint
