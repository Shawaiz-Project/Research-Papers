from __future__ import annotations

import cv2
import numpy as np

from .persistent_features import load_grayscale, persistence_intervals


def _field(image: np.ndarray, filtration: str) -> np.ndarray:
    if filtration == "sublevel":
        return image
    if filtration == "superlevel":
        return 1.0 - image
    if filtration == "gradient":
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx * gx + gy * gy)
        maximum = float(magnitude.max())
        return magnitude / maximum if maximum > 0 else magnitude
    raise ValueError(f"Unsupported filtration: {filtration}")


def extract_multiscale_tokens(path: str, topology_config: dict) -> dict[str, np.ndarray]:
    sizes = [int(value) for value in topology_config["multiscale_sizes"]]
    filtrations = list(topology_config["multiscale_filtrations"])
    dimensions = [int(value) for value in topology_config["homology_dims"]]
    pairs_per_group = int(topology_config["pairs_per_group"])
    tokens: list[list[float]] = []
    masks: list[bool] = []
    dimension_ids: list[int] = []
    group_ids: list[int] = []
    group_index = 0
    for size_index, size in enumerate(sizes):
        image = load_grayscale(path, size)
        for filtration_index, filtration in enumerate(filtrations):
            diagrams = persistence_intervals(_field(image, filtration), tuple(dimensions))
            for dimension in dimensions:
                intervals = diagrams.get(dimension, np.zeros((0, 2), dtype=np.float64))
                if len(intervals):
                    lifetimes = np.maximum(intervals[:, 1] - intervals[:, 0], 0.0)
                    order = np.argsort(-lifetimes)
                    intervals = intervals[order]
                for rank in range(pairs_per_group):
                    if rank < len(intervals):
                        birth, death = intervals[rank]
                        lifetime = max(float(death - birth), 0.0)
                        tokens.append([float(birth), float(death), lifetime, rank / max(pairs_per_group - 1, 1)])
                        masks.append(True)
                    else:
                        tokens.append([0.0, 0.0, 0.0, rank / max(pairs_per_group - 1, 1)])
                        masks.append(False)
                    dimension_ids.append(dimension)
                    group_ids.append(group_index)
                group_index += 1
    token_array = np.asarray(tokens, dtype=np.float32)
    expected = len(sizes) * len(filtrations) * len(dimensions) * pairs_per_group
    if token_array.shape != (expected, 4):
        raise RuntimeError(f"Token tensor has shape {token_array.shape}; expected {(expected, 4)}")
    return {
        "tokens": token_array,
        "mask": np.asarray(masks, dtype=np.bool_),
        "dimension_ids": np.asarray(dimension_ids, dtype=np.int64),
        "group_ids": np.asarray(group_ids, dtype=np.int64),
    }
