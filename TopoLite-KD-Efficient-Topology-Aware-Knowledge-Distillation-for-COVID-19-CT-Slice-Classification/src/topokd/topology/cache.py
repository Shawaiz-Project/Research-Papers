from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from topokd.utils.hashing import stable_text_hash

from .multiscale_tokens import extract_multiscale_tokens
from .persistent_features import extract_descriptor


class TopologyCache:
    def __init__(self, config: dict):
        self.config = config
        topology_cfg = config["topology"]
        self.root = Path(topology_cfg["cache_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.mode = topology_cfg["mode"]
        relevant = json.dumps(topology_cfg, sort_keys=True)
        self.signature = stable_text_hash(relevant)[:16]
        configured_standardizer = Path(topology_cfg.get("standardization_path", "artifacts/tda_standardizer.npz"))
        self.standardizer_path = configured_standardizer.with_name(
            f"{configured_standardizer.stem}_{self.signature}{configured_standardizer.suffix}"
        )

    def cache_path(self, image_hash: str) -> Path:
        return self.root / self.signature / image_hash[:2] / f"{image_hash}.npz"

    def load_or_compute(self, image_path: str, image_hash: str) -> dict[str, np.ndarray]:
        path = self.cache_path(image_hash)
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                return {key: payload[key] for key in payload.files}
        if self.mode == "descriptor":
            payload = {"descriptor": extract_descriptor(image_path, self.config["topology"])}
        elif self.mode == "multiscale_tokens":
            payload = extract_multiscale_tokens(image_path, self.config["topology"])
        else:
            raise ValueError(f"Unsupported topology mode: {self.mode}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(temp_name, **payload)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return payload
