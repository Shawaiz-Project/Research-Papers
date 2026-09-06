from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import gudhi as gd
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DescriptorSettings:
    """Configuration for the fixed-width TopoLite persistent-homology descriptor.

    The descriptor is always 134-dimensional:
      * 4 filtration/dimension groups (sublevel/superlevel × H0/H1)
      * 30 values per group (10 lifetime statistics + 4 persistence counts
        + a 16-point Betti curve) = 120
      * 14 thresholded component/hole counts = 134

    Disabled filtrations or homology dimensions are represented by zero blocks. This
    keeps the downstream MLP identical across ablations and avoids parameter-count
    confounding.
    """

    resize: int = 64
    persistence_min: float = 0.0
    filtrations: tuple[str, ...] = ("sublevel", "superlevel")
    homology_dims: tuple[int, ...] = (0, 1)
    persistence_thresholds: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    betti_points: int = 16
    binary_thresholds: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


CANONICAL_FILTRATIONS = ("sublevel", "superlevel")
CANONICAL_DIMENSIONS = (0, 1)
GROUP_WIDTH = 30
BINARY_WIDTH = 14
DESCRIPTOR_WIDTH = 134


def load_grayscale(path: str, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def persistence_intervals(field: np.ndarray, dimensions: tuple[int, ...] = (0, 1)) -> dict[int, np.ndarray]:
    complex_ = gd.CubicalComplex(top_dimensional_cells=np.asarray(field, dtype=np.float64))
    complex_.persistence(homology_coeff_field=2, min_persistence=0.0)
    maximum = float(np.max(field))
    output: dict[int, np.ndarray] = {}
    for dimension in dimensions:
        intervals = np.asarray(complex_.persistence_intervals_in_dimension(dimension), dtype=np.float64)
        if intervals.size == 0:
            output[dimension] = np.zeros((0, 2), dtype=np.float64)
            continue
        intervals = intervals.reshape(-1, 2)
        intervals[:, 1] = np.where(np.isfinite(intervals[:, 1]), intervals[:, 1], maximum)
        intervals[:, 0] = np.clip(intervals[:, 0], 0.0, 1.0)
        intervals[:, 1] = np.clip(intervals[:, 1], 0.0, 1.0)
        output[dimension] = intervals
    return output


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    total = float(lifetimes.sum())
    if total <= 0:
        return 0.0
    probabilities = lifetimes / total
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    maximum = math.log(max(len(probabilities), 2))
    return entropy / maximum


def _lifetime_statistics(lifetimes: np.ndarray) -> np.ndarray:
    """Ten bounded/scale-stable statistics for one persistence diagram."""
    if lifetimes.size == 0:
        return np.zeros(10, dtype=np.float32)
    total = float(lifetimes.sum())
    maximum = float(lifetimes.max())
    return np.asarray(
        [
            min(float(lifetimes.size) / 256.0, 1.0),
            float(lifetimes.mean()),
            float(lifetimes.std()),
            float(lifetimes.min()),
            maximum,
            float(np.median(lifetimes)),
            float(np.quantile(lifetimes, 0.25)),
            float(np.quantile(lifetimes, 0.75)),
            min(total / 64.0, 1.0),
            _persistence_entropy(lifetimes),
        ],
        dtype=np.float32,
    )


def _threshold_counts(lifetimes: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    if lifetimes.size == 0:
        return np.zeros(len(thresholds), dtype=np.float32)
    return np.asarray([min(float(np.count_nonzero(lifetimes >= threshold)) / 256.0, 1.0) for threshold in thresholds], dtype=np.float32)


def _betti_curve(intervals: np.ndarray, points: int) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros(points, dtype=np.float32)
    thresholds = np.linspace(0.0, 1.0, points, dtype=np.float64)
    alive = ((intervals[:, 0, None] <= thresholds[None, :]) & (thresholds[None, :] < intervals[:, 1, None])).sum(axis=0)
    return np.clip(alive.astype(np.float32) / 256.0, 0.0, 1.0)


def _diagram_block(intervals: np.ndarray, settings: DescriptorSettings) -> np.ndarray:
    if intervals.size:
        lifetimes = np.maximum(intervals[:, 1] - intervals[:, 0], 0.0)
        keep = lifetimes >= settings.persistence_min
        intervals = intervals[keep]
        lifetimes = lifetimes[keep]
    else:
        intervals = np.zeros((0, 2), dtype=np.float64)
        lifetimes = np.zeros(0, dtype=np.float64)
    block = np.concatenate(
        [
            _lifetime_statistics(lifetimes),
            _threshold_counts(lifetimes, settings.persistence_thresholds),
            _betti_curve(intervals, settings.betti_points),
        ]
    ).astype(np.float32)
    if block.shape != (GROUP_WIDTH,):
        raise RuntimeError(f"Persistence block has shape {block.shape}; expected {(GROUP_WIDTH,)}")
    return block


def _component_and_hole_counts(image: np.ndarray, settings: DescriptorSettings) -> np.ndarray:
    """Seven component counts and seven hole counts from binary intensity cuts."""
    components: list[float] = []
    holes: list[float] = []
    for threshold in settings.binary_thresholds:
        mask = (image >= threshold).astype(np.uint8)
        component_count, _ = cv2.connectedComponents(mask, connectivity=8)
        components.append(min(max(component_count - 1, 0) / 256.0, 1.0))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        del contours
        hole_count = 0 if hierarchy is None else int(np.count_nonzero(hierarchy[0, :, 3] >= 0))
        holes.append(min(hole_count / 256.0, 1.0))
    if 0 not in settings.homology_dims:
        components = [0.0] * len(components)
    if 1 not in settings.homology_dims:
        holes = [0.0] * len(holes)
    block = np.asarray(components + holes, dtype=np.float32)
    if block.shape != (BINARY_WIDTH,):
        raise RuntimeError(f"Binary topology block has shape {block.shape}; expected {(BINARY_WIDTH,)}")
    return block


def extract_descriptor(path: str, topology_config: dict) -> np.ndarray:
    settings = DescriptorSettings(
        resize=int(topology_config.get("resize", 64)),
        persistence_min=float(topology_config.get("persistence_min", 0.0)),
        filtrations=tuple(topology_config.get("filtrations", CANONICAL_FILTRATIONS)),
        homology_dims=tuple(int(value) for value in topology_config.get("homology_dims", CANONICAL_DIMENSIONS)),
        persistence_thresholds=tuple(float(value) for value in topology_config.get("persistence_thresholds", [0.01, 0.03, 0.05, 0.10])),
        betti_points=int(topology_config.get("betti_points", 16)),
        binary_thresholds=tuple(float(value) for value in topology_config.get("binary_thresholds", [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])),
    )
    if settings.betti_points != 16 or len(settings.persistence_thresholds) != 4 or len(settings.binary_thresholds) != 7:
        raise ValueError("The paper-aligned descriptor requires 16 Betti points, 4 persistence thresholds, and 7 binary thresholds.")
    unsupported = set(settings.filtrations) - set(CANONICAL_FILTRATIONS)
    if unsupported:
        raise ValueError(f"Unsupported descriptor filtrations: {sorted(unsupported)}")
    unsupported_dims = set(settings.homology_dims) - set(CANONICAL_DIMENSIONS)
    if unsupported_dims:
        raise ValueError(f"Unsupported homology dimensions: {sorted(unsupported_dims)}")

    image = load_grayscale(path, settings.resize)
    diagrams_by_filtration: dict[str, dict[int, np.ndarray]] = {}
    for filtration in settings.filtrations:
        field = image if filtration == "sublevel" else 1.0 - image
        diagrams_by_filtration[filtration] = persistence_intervals(field, CANONICAL_DIMENSIONS)

    features: list[np.ndarray] = []
    for filtration in CANONICAL_FILTRATIONS:
        for dimension in CANONICAL_DIMENSIONS:
            if filtration in settings.filtrations and dimension in settings.homology_dims:
                intervals = diagrams_by_filtration[filtration][dimension]
                features.append(_diagram_block(intervals, settings))
            else:
                features.append(np.zeros(GROUP_WIDTH, dtype=np.float32))
    features.append(_component_and_hole_counts(image, settings))
    descriptor = np.concatenate(features).astype(np.float32)
    configured = int(topology_config.get("descriptor_dim", DESCRIPTOR_WIDTH))
    if configured != DESCRIPTOR_WIDTH:
        raise ValueError(f"descriptor_dim must remain {DESCRIPTOR_WIDTH} for architecture-matched ablations; received {configured}")
    if descriptor.shape != (DESCRIPTOR_WIDTH,):
        raise RuntimeError(f"Descriptor shape {descriptor.shape} does not match {(DESCRIPTOR_WIDTH,)}")
    if not np.isfinite(descriptor).all():
        raise RuntimeError("Descriptor contains non-finite values.")
    return descriptor
