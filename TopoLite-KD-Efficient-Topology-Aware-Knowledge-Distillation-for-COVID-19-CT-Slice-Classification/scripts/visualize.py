from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.data import build_dataloaders
from topokd.models import build_student
from topokd.utils import resolve_device
from topokd.utils.checkpoint import load_checkpoint
from topokd.visualization import generate_gradcams, plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training plots and Grad-CAM qualitative results.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.checkpoint).resolve().parents[1]
    history = run_dir / "logs" / "history.csv"
    if history.exists():
        plot_training_history(history, run_dir / "figures")
    calibration_path = run_dir / "metrics" / "calibration.json"
    predictions_path = run_dir / "predictions" / "test_predictions.csv"
    if not calibration_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Run evaluation first so calibration.json and test_predictions.csv exist.")
    calibration = json.loads(calibration_path.read_text())
    device = resolve_device(args.device)
    model = build_student(config).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
    loaders = build_dataloaders(config)
    generate_gradcams(
        model=model,
        dataset=loaders["test"].dataset,
        predictions_csv=predictions_path,
        output_dir=run_dir / "gradcam",
        target_layer=config["evaluation"]["target_layer"],
        threshold=float(calibration["threshold"]),
        per_class=int(config["evaluation"]["gradcam_samples_per_class"]),
        device=device,
    )
    print(f"Visual outputs saved under: {(run_dir / 'figures').resolve()} and {(run_dir / 'gradcam').resolve()}")


if __name__ == "__main__":
    main()
