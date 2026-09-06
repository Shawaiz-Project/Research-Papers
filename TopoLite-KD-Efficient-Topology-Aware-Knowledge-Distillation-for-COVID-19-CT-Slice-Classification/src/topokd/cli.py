from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from topokd.config import apply_overrides, load_config
from topokd.data import build_dataloaders
from topokd.data.transforms import build_transforms
from topokd.engine import Trainer, evaluate_splits
from topokd.models import build_student, build_teacher
from topokd.utils import resolve_device, seed_everything
from topokd.utils.checkpoint import load_checkpoint


def train_main() -> None:
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


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TopoKD student.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    device = resolve_device(args.device)
    model = build_student(config).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
    loaders = build_dataloaders(config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parents[1]
    results = evaluate_splits(model, loaders["val"], loaders["test"], config, device, output_dir)
    print(json.dumps(results["test_metrics"], indent=2))


def infer_main() -> None:
    parser = argparse.ArgumentParser(description="Run TopoKD inference on one CT image.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    device = resolve_device(args.device)
    model = build_student(config).to(device)
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model_state"])
    model.eval()
    with Image.open(args.image) as image:
        image_tensor = build_transforms(config, train=False)(image.convert("L")).unsqueeze(0).to(device)
    kwargs: dict[str, torch.Tensor] = {"image": image_tensor}
    if config["topology"].get("enabled", True):
        from topokd.topology import extract_descriptor, extract_multiscale_tokens
        from topokd.topology.cache import TopologyCache
        if config["topology"]["mode"] == "descriptor":
            descriptor = extract_descriptor(args.image, config["topology"])
            cache = TopologyCache(config)
            if not cache.standardizer_path.exists():
                raise FileNotFoundError(f"TDA standardizer not found: {cache.standardizer_path}")
            with np.load(cache.standardizer_path, allow_pickle=False) as standardizer:
                descriptor = (descriptor - standardizer["mean"]) / standardizer["std"]
            kwargs["descriptor"] = torch.from_numpy(descriptor.astype(np.float32)).unsqueeze(0).to(device)
        else:
            payload = extract_multiscale_tokens(args.image, config["topology"])
            for key, value in payload.items():
                kwargs[key] = torch.from_numpy(value).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = float(model(**kwargs)["logits"][0])
    probability = float(torch.sigmoid(torch.tensor(logit / args.temperature)))
    print(
        json.dumps(
            {
                "image": str(Path(args.image).resolve()),
                "logit": logit,
                "probability_covid": probability,
                "threshold": args.threshold,
                "prediction": "COVID" if probability >= args.threshold else "Non-COVID",
            },
            indent=2,
        )
    )
