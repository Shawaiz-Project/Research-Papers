from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import torch

from .io import atomic_json_dump


def capture_environment(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count(),
    }
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        payload["git_commit"] = None
    atomic_json_dump(payload, output_dir / "environment.json")
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output_dir / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    except Exception:
        pass
