from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def stratified_or_group_split(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    seed = int(data_cfg["split_seed"])
    train_fraction = float(data_cfg["train_fraction"])
    val_fraction = float(data_cfg["val_fraction"])
    test_fraction = float(data_cfg["test_fraction"])
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")
    result = frame.copy()
    result["split"] = ""
    has_groups = result["patient_id"].notna().all() and result["patient_id"].nunique() < len(result)
    if has_groups:
        outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_val_idx, test_idx = next(outer.split(result, result["label"], groups=result["patient_id"]))
        train_val = result.iloc[train_val_idx]
        relative_val = val_fraction / (train_fraction + val_fraction)
        inner = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
        train_local, val_local = next(inner.split(train_val, train_val["label"], groups=train_val["patient_id"]))
        train_idx = train_val.index[train_local]
        val_idx = train_val.index[val_local]
    else:
        train_val_idx, test_idx = train_test_split(
            result.index,
            test_size=test_fraction,
            random_state=seed,
            stratify=result["label"],
        )
        relative_val = val_fraction / (train_fraction + val_fraction)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            random_state=seed + 1,
            stratify=result.loc[train_val_idx, "label"],
        )
    result.loc[train_idx, "split"] = "train"
    result.loc[val_idx, "split"] = "val"
    result.loc[test_idx, "split"] = "test"
    if (result["split"] == "").any():
        raise RuntimeError("Split assignment failed for one or more samples.")
    return result.sort_values(["split", "class_name", "path"]).reset_index(drop=True)
