from __future__ import annotations

from pathlib import Path

import numpy as np

from ts3d.data.collate import PointNeXtCollator
from ts3d.data.dataset import CandidateClusterDataset
from ts3d.data.transforms import build_transform_pipeline


def test_dataset_round_trip(synthetic_dataset: Path):
    tfm = build_transform_pipeline(
        num_points=128, augment=False, rotate_z=False, jitter_sigma=0.0, seed=0
    )
    ds = CandidateClusterDataset(
        synthetic_dataset,
        split="train",
        transform=tfm,
    )
    assert len(ds) == 2

    sample = ds[0]
    assert sample.xyz.shape == (128, 3)
    assert sample.label in (0, 1)

    batch = PointNeXtCollator()([ds[0], ds[1]])
    assert batch.xyz.shape == (2, 128, 3)
    assert {int(label) for label in batch.labels.tolist()} == {0, 1}
    # xyz is recentered around origin after transforms
    assert float(np.abs(batch.xyz.mean(dim=1)).max()) < 1.0
