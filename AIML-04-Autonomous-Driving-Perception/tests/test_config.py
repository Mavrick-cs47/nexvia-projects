from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


def test_data_yaml_is_consistent():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["nc"] == len(cfg["names"]) == 10
    assert cfg["names"][2] == "car"
    assert {"path", "train", "val"} <= cfg.keys()
