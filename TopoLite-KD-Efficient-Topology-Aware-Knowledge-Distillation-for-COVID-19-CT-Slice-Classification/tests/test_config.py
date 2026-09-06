from pathlib import Path

from topokd.config import load_config


def test_config_inheritance():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "topolite_kd.yaml")
    assert config["model"]["name"] == "topolite_kd"
    assert config["topology"]["descriptor_dim"] == 134
    assert config["data"]["image_size"] == 224
