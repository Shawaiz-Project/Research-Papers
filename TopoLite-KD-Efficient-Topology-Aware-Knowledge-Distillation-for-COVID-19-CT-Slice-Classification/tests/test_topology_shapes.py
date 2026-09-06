from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("gudhi")

from topokd.config import load_config
from topokd.topology.multiscale_tokens import extract_multiscale_tokens
from topokd.topology.persistent_features import extract_descriptor


def test_topology_output_shapes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    image_path = tmp_path / "slice.png"
    array = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    Image.fromarray(array).save(image_path)
    config = load_config(root / "configs" / "topolite_kd.yaml")
    descriptor = extract_descriptor(str(image_path), config["topology"])
    assert descriptor.shape == (134,)
    msf = load_config(root / "configs" / "topolite_msf_kd.yaml")
    payload = extract_multiscale_tokens(str(image_path), msf["topology"])
    assert payload["tokens"].shape == (144, 4)
    assert payload["mask"].shape == (144,)
