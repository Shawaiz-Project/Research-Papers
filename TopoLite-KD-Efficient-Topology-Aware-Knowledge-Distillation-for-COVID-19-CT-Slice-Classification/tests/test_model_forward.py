from pathlib import Path

import torch

from topokd.config import load_config
from topokd.models import build_student


def test_topolite_forward_shapes():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "topolite_kd.yaml")
    config["data"]["image_size"] = 32
    model = build_student(config).eval()
    with torch.inference_mode():
        output = model(image=torch.randn(2, 1, 32, 32), descriptor=torch.randn(2, 134))
    assert output["logits"].shape == (2,)
    assert output["visual_embedding"].shape == (2, 64)
    assert output["topology_embedding"].shape == (2, 64)
    assert output["fusion_weights"].shape == (2, 64)


def test_topolite_default_parameter_budget_is_approximately_197k():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "topolite_kd.yaml")
    model = build_student(config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert 190_000 <= trainable <= 205_000
