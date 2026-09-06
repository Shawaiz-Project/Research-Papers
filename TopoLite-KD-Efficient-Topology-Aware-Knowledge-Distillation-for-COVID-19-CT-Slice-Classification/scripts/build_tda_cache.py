from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.topology import TopologyCache


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute persistent-homology features for every manifest sample.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    manifest_path = Path(config["data"]["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run prepare_manifest.py first.")
    frame = pd.read_csv(manifest_path)
    cache = TopologyCache(config)
    failures = []
    train_descriptors = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Building topology cache"):
        try:
            payload = cache.load_or_compute(row.path, row.sha256)
            if cache.mode == "descriptor" and row.split == "train":
                train_descriptors.append(payload["descriptor"])
        except Exception as exc:
            failures.append((row.path, repr(exc)))
    if failures:
        failure_path = manifest_path.parent / "tda_cache_failures.csv"
        pd.DataFrame(failures, columns=["path", "error"]).to_csv(failure_path, index=False)
        raise RuntimeError(f"Topology extraction failed for {len(failures)} images. See {failure_path}")
    if cache.mode == "descriptor":
        if not train_descriptors:
            raise RuntimeError("No training descriptors were available to fit the topology standardizer.")
        matrix = __import__("numpy").stack(train_descriptors).astype("float32")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = __import__("numpy").where(std < 1e-6, 1.0, std).astype("float32")
        cache.standardizer_path.parent.mkdir(parents=True, exist_ok=True)
        __import__("numpy").savez_compressed(cache.standardizer_path, mean=mean.astype("float32"), std=std)
        print(f"Fitted train-only TDA standardizer: {cache.standardizer_path.resolve()}")
    print(f"Topology cache complete: {cache.root.resolve()} | mode={cache.mode} | signature={cache.signature}")


if __name__ == "__main__":
    main()
