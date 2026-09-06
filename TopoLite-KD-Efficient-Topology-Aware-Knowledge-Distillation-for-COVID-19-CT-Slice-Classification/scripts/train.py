from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.data import build_dataloaders
from topokd.engine import Trainer
from topokd.models import build_student, build_teacher
from topokd.utils import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TopoKD research model.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    seed_everything(int(config["training"]["seed"]), bool(config["training"].get("deterministic", True)))
    device = resolve_device(args.device)
    loaders = build_dataloaders(config)
    student = build_student(config)
    teacher = build_teacher(config, require_checkpoint=True)
    trainer = Trainer(config, student, teacher, loaders, device)
    results = trainer.fit()
    print(json.dumps(results["test_metrics"], indent=2))
    print(f"Run directory: {trainer.run_dir.resolve()}")


if __name__ == "__main__":
    main()
