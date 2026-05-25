from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from ts3d.data.metadata import Metadata, MetadataRow


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "a",
                "split": "train",
                "label": 1,
                "bbox3d_center_xyz": [0.0, 1.0, 2.0],
                "bbox3d_size_xyz": [0.5, 0.1, 0.8],
                "bbox3d_yaw": 0.0,
            },
            {
                "candidate_id": "b",
                "split": "val",
                "label": 0,
                "bbox3d_center_xyz": [5.0, 5.0, 2.0],
                "bbox3d_size_xyz": [0.5, 0.1, 0.8],
                "bbox3d_yaw": 0.3,
            },
        ]
    )


def test_metadata_roundtrip(tmp_path: Path):
    df = _sample_df()
    m = Metadata(df)
    out = tmp_path / "m.json"
    m.to_json(out)
    reloaded = Metadata.from_file(out)
    assert len(reloaded) == 2
    assert reloaded.row(0).candidate_id == "a"


def test_metadata_filter_split():
    m = Metadata(_sample_df())
    train = m.filter_split("train")
    val = m.filter_split("val")
    assert len(train) == 1
    assert len(val) == 1
    assert train.row(0).candidate_id == "a"


def test_metadata_row_validation():
    row = MetadataRow.model_validate(
        {
            "candidate_id": "x",
            "split": "train",
            "label": 1,
            "bbox3d_center_xyz": [0, 0, 0],
            "bbox3d_size_xyz": [1, 1, 1],
        }
    )
    assert row.bbox3d_center_xyz == (0.0, 0.0, 0.0)
    assert row.label == 1

    with pytest.raises(ValidationError):
        MetadataRow.model_validate(
            {
                "candidate_id": "x",
                "split": "train",
                "label": 5,  # invalid
                "bbox3d_center_xyz": [0, 0, 0],
                "bbox3d_size_xyz": [1, 1, 1],
            }
        )


def test_metadata_missing_required_columns_raises():
    df = pd.DataFrame([{"candidate_id": "a", "split": "train"}])
    with pytest.raises(ValueError):
        Metadata(df)


def test_class_counts():
    m = Metadata(_sample_df())
    counts = m.class_counts()
    assert counts == {0: 1, 1: 1}
