from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.models import build_student
from scripts.profile_model import build_profile_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate configuration and execute a CPU forward pass.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = build_student(config).eval()
    inputs = build_profile_inputs(config, torch.device("cpu"), batch_size=1)
    with torch.inference_mode():
        output = model(**inputs)
    assert output["logits"].shape == (1,)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"OK: {config['model']['name']} | logits={tuple(output['logits'].shape)} | trainable_parameters={trainable:,}")


if __name__ == "__main__":
    main()
