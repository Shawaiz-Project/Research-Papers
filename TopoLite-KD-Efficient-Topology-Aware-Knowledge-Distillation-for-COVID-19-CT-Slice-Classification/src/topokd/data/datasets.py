from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms


class CTTopologyDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, config: dict, split: str):
        self.frame = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Manifest contains no samples for split '{split}'")
        self.config = config
        self.split = split
        self.transform = build_transforms(config, train=split == "train")
        self.topology_enabled = bool(config["topology"].get("enabled", True))
        if self.topology_enabled:
            from topokd.topology.cache import TopologyCache
            self.topology_cache = TopologyCache(config)
        else:
            self.topology_cache = None
        self.standardizer = None
        if self.topology_enabled and self.topology_cache.mode == "descriptor":
            standardizer_path = self.topology_cache.standardizer_path
            if not standardizer_path.exists():
                raise FileNotFoundError(
                    f"TDA standardizer not found: {standardizer_path}. Run scripts/build_tda_cache.py before training."
                )
            with np.load(standardizer_path, allow_pickle=False) as payload:
                self.standardizer = (payload["mean"].astype(np.float32), payload["std"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image)
        sample = {
            "image": tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        if self.topology_enabled:
            payload = self.topology_cache.load_or_compute(row["path"], row["sha256"])
            if "descriptor" in payload and self.standardizer is not None:
                mean, std = self.standardizer
                payload["descriptor"] = (payload["descriptor"] - mean) / std
            for key, value in payload.items():
                if value.dtype == np.bool_:
                    sample[key] = torch.from_numpy(value.astype(np.bool_))
                elif np.issubdtype(value.dtype, np.integer):
                    sample[key] = torch.from_numpy(value.astype(np.int64))
                else:
                    sample[key] = torch.from_numpy(value.astype(np.float32))
        return sample


def load_manifest(config: dict) -> pd.DataFrame:
    path = Path(config["data"]["manifest"])
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run scripts/prepare_manifest.py first.")
    frame = pd.read_csv(path)
    required = {"path", "label", "split", "sha256", "class_name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if "relative_path" in frame.columns:
        root = Path(config["data"]["root"]).expanduser()
        missing_paths = ~frame["path"].map(lambda value: Path(value).exists())
        if missing_paths.any():
            rebased = frame.loc[missing_paths, "relative_path"].map(lambda value: str((root / value).resolve()))
            frame.loc[missing_paths, "path"] = rebased
    unresolved = frame.loc[~frame["path"].map(lambda value: Path(value).exists()), "path"]
    if len(unresolved):
        raise FileNotFoundError(f"Manifest contains {len(unresolved)} unresolved image paths; first: {unresolved.iloc[0]}")
    if frame["sha256"].duplicated().any():
        raise ValueError("Frozen manifest contains duplicate SHA-256 values.")
    if not set(frame["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError("Manifest split column contains unsupported values.")
    if set(frame["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Manifest must contain train, val, and test samples.")
    if "patient_id" in frame.columns and frame["patient_id"].notna().all():
        leakage = frame.groupby("patient_id")["split"].nunique()
        if (leakage > 1).any():
            raise ValueError("Patient-level leakage detected across manifest splits.")
    return frame


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    frame = load_manifest(config)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = CTTopologyDataset(frame, config, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=split == "train",
            num_workers=int(config["data"].get("num_workers", 0)),
            pin_memory=bool(config["data"].get("pin_memory", True)),
            persistent_workers=bool(config["data"].get("persistent_workers", True)) and int(config["data"].get("num_workers", 0)) > 0,
            drop_last=split == "train",
        )
    return loaders
