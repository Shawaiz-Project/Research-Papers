from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config
from topokd.data.manifest import discover_images, remove_exact_duplicates
from topokd.data.splitting import stratified_or_group_split
from topokd.utils.io import atomic_json_dump, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover, audit, deduplicate, and split the CT dataset.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override config values, e.g. data.root=/path")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing frozen manifest.")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    manifest_path = Path(config["data"]["manifest"])
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Frozen manifest already exists: {manifest_path}. Refusing to regenerate it without --force."
        )
    discovered = discover_images(config)
    if config["data"].get("remove_exact_duplicates", True):
        clean, duplicates = remove_exact_duplicates(discovered)
    else:
        clean, duplicates = discovered.copy(), discovered.iloc[0:0].copy()
    manifest = stratified_or_group_split(clean, config)
    ensure_dir(manifest_path.parent)
    manifest.to_csv(manifest_path, index=False)
    duplicates.to_csv(manifest_path.parent / "exact_duplicates_removed.csv", index=False)
    summary = {
        "discovered": int(len(discovered)),
        "retained": int(len(manifest)),
        "duplicates_removed": int(len(duplicates)),
        "class_counts": {str(k): int(v) for k, v in manifest["class_name"].value_counts().items()},
        "split_counts": {str(k): int(v) for k, v in manifest["split"].value_counts().items()},
        "split_class_counts": {
            f"{split}:{class_name}": int(count)
            for (split, class_name), count in manifest.groupby(["split", "class_name"]).size().items()
        },
        "patient_grouped": bool(manifest["patient_id"].notna().all() and manifest["patient_id"].nunique() < len(manifest)),
        "split_seed": int(config["data"]["split_seed"]),
    }
    atomic_json_dump(summary, manifest_path.parent / "manifest_summary.json")
    print(json.dumps(summary, indent=2))
    print(f"Saved frozen manifest to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
