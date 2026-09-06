from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from topokd.utils.hashing import sha256_file


def discover_images(config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    extensions = {suffix.lower() for suffix in data_cfg["extensions"]}
    class_names = list(data_cfg["class_names"])
    positive_class = data_cfg["positive_class"]
    regex = data_cfg.get("patient_id_regex")
    compiled = re.compile(regex) if regex else None
    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        parts_lower = {part.lower(): part for part in path.parts}
        detected = None
        for class_name in class_names:
            if class_name.lower() in parts_lower:
                detected = class_name
                break
        if detected is None:
            parent_text = path.parent.name.lower().replace("_", "-")
            for class_name in class_names:
                if class_name.lower().replace("_", "-") == parent_text:
                    detected = class_name
                    break
        if detected is None:
            continue
        patient_id = None
        if compiled:
            match = compiled.search(path.name)
            patient_id = match.group(1) if match else None
        records.append(
            {
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "class_name": detected,
                "label": int(detected == positive_class),
                "patient_id": patient_id,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not records:
        raise RuntimeError(
            "No class-labelled images were found. Expected class directory names matching: "
            + ", ".join(class_names)
        )
    return pd.DataFrame.from_records(records)


def remove_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["sha256", "path"]).reset_index(drop=True)
    duplicate_mask = ordered.duplicated(subset=["sha256"], keep="first")
    duplicates = ordered.loc[duplicate_mask].copy()
    clean = ordered.loc[~duplicate_mask].copy().reset_index(drop=True)
    cross_class = ordered.groupby("sha256")["label"].nunique()
    conflicts = cross_class[cross_class > 1]
    if len(conflicts):
        raise ValueError(f"Found {len(conflicts)} exact duplicate hashes with conflicting labels.")
    return clean, duplicates
