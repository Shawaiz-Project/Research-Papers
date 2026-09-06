from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.models import build_student
from topokd.utils import resolve_device


def build_profile_inputs(config: dict, device: torch.device, batch_size: int = 1) -> dict:
    size = int(config["data"]["image_size"])
    payload = {"image": torch.randn(batch_size, int(config["model"]["input_channels"]), size, size, device=device)}
    if config["topology"].get("enabled", True):
        if config["topology"]["mode"] == "descriptor":
            payload["descriptor"] = torch.randn(batch_size, int(config["topology"]["descriptor_dim"]), device=device)
        else:
            count = (
                len(config["topology"]["multiscale_sizes"])
                * len(config["topology"]["multiscale_filtrations"])
                * len(config["topology"]["homology_dims"])
                * int(config["topology"]["pairs_per_group"])
            )
            payload["tokens"] = torch.rand(batch_size, count, 4, device=device)
            payload["mask"] = torch.ones(batch_size, count, dtype=torch.bool, device=device)
            dimension_pattern = []
            group_pattern = []
            group = 0
            for _ in config["topology"]["multiscale_sizes"]:
                for _ in config["topology"]["multiscale_filtrations"]:
                    for dimension in config["topology"]["homology_dims"]:
                        dimension_pattern.extend([dimension] * int(config["topology"]["pairs_per_group"]))
                        group_pattern.extend([group] * int(config["topology"]["pairs_per_group"]))
                        group += 1
            payload["dimension_ids"] = torch.tensor(dimension_pattern, device=device).unsqueeze(0).repeat(batch_size, 1)
            payload["group_ids"] = torch.tensor(group_pattern, device=device).unsqueeze(0).repeat(batch_size, 1)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model parameters, serialized size, latency, and CUDA memory.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    device = resolve_device(args.device)
    model = build_student(config).to(device).eval()
    inputs = build_profile_inputs(config, device, args.batch_size)
    warmup = int(config["evaluation"].get("profile_warmup", 20))
    iterations = int(config["evaluation"].get("profile_iterations", 100))
    with torch.inference_mode():
        for _ in range(warmup):
            model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timings = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(**inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "model": config["model"]["name"],
        "device": str(device),
        "batch_size": args.batch_size,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_size_mb_fp32": total * 4 / (1024**2),
        "latency_ms_mean": statistics.mean(timings),
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p95": sorted(timings)[int(0.95 * (len(timings) - 1))],
        "throughput_images_per_second": args.batch_size * 1000.0 / statistics.mean(timings),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else None,
    }
    output = Path(args.output) if args.output else Path(config["project"]["output_root"]) / config["project"]["experiment_name"] / "profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
