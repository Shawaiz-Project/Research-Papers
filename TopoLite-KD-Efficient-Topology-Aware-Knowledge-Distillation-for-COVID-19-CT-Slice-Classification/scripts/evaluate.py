from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.data import build_dataloaders
from topokd.engine import evaluate_splits
from topokd.models import build_student
from topokd.utils import resolve_device, seed_everything
from topokd.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained student with validation-only calibration and thresholding.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    seed_everything(int(config["training"]["seed"]), bool(config["training"].get("deterministic", True)))
    device = resolve_device(args.device)
    model = build_student(config).to(device)
    payload = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(payload["model_state"])
    loaders = build_dataloaders(config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parents[1]
    results = evaluate_splits(model, loaders["val"], loaders["test"], config, device, output_dir)
    print(json.dumps(results["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
