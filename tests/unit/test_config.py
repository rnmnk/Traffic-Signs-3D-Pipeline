from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ts3d.config.loader import load_experiment_config


def test_load_pointnext_baseline_config():
    root = Path(__file__).resolve().parents[2]
    cfg = load_experiment_config(root / "configs" / "pointnext-b.yaml")
    assert cfg.model.name == "pointnext"
    assert cfg.data.num_points > 0


def test_unknown_model_name_rejected(tmp_path: Path):
    bad = {
        "experiment_name": "bad",
        "data": {"root": "/tmp/x"},
        "model": {"name": "sparse_resunet"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_experiment_config(p)


def test_extra_data_field_rejected(tmp_path: Path):
    """`feature_channels` is no longer a valid field; ensure schemas reject extras."""
    bad = {
        "experiment_name": "bad",
        "data": {"root": "/tmp/x", "feature_channels": ["intensity"]},
        "model": {"name": "pointnext"},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        load_experiment_config(p)
