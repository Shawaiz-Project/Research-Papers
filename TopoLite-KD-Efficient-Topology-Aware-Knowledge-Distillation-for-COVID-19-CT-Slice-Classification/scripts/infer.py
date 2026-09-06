from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.data.transforms import build_transforms
from topokd.models import build_student
from topokd.topology import extract_descriptor, extract_multiscale_tokens
from topokd.topology.cache import TopologyCache
from topokd.utils import resolve_device
from topokd.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one CT image.")
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
    payload = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    with Image.open(args.image) as image:
        image_tensor = build_transforms(config, train=False)(image.convert("L")).unsqueeze(0).to(device)
    kwargs = {"image": image_tensor}
    if config["topology"].get("enabled", True):
        if config["topology"]["mode"] == "descriptor":
            descriptor = extract_descriptor(args.image, config["topology"])
            cache = TopologyCache(config)
            if not cache.standardizer_path.exists():
                raise FileNotFoundError(f"TDA standardizer not found: {cache.standardizer_path}")
            with np.load(cache.standardizer_path, allow_pickle=False) as standardizer:
                descriptor = (descriptor - standardizer["mean"]) / standardizer["std"]
            kwargs["descriptor"] = torch.from_numpy(descriptor.astype(np.float32)).unsqueeze(0).to(device)
        else:
            payload_topology = extract_multiscale_tokens(args.image, config["topology"])
            for key, value in payload_topology.items():
                kwargs[key] = torch.from_numpy(value).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = float(model(**kwargs)["logits"][0])
    probability = float(torch.sigmoid(torch.tensor(logit / args.temperature)))
    result = {
        "image": str(Path(args.image).resolve()),
        "logit": logit,
        "probability_covid": probability,
        "threshold": args.threshold,
        "prediction": "COVID" if probability >= args.threshold else "Non-COVID",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
