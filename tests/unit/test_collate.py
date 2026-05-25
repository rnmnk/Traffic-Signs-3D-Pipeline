from __future__ import annotations

import numpy as np

from ts3d.data.collate import PointNeXtCollator, build_collator
from ts3d.data.dataset import ClusterSample


def _sample(label: int = 1) -> ClusterSample:
    rng = np.random.default_rng(0)
    return ClusterSample(
        candidate_id=f"x_{label}",
        xyz=(rng.random((64, 3)).astype(np.float32) - 0.5),
        bbox_center=np.zeros(3, dtype=np.float32),
        bbox_size=np.ones(3, dtype=np.float32),
        label=label,
    )


def test_pointnext_collator_shapes():
    samples = [_sample(0), _sample(1)]
    batch = PointNeXtCollator()(samples)
    assert batch.xyz.shape == (2, 64, 3)
    assert batch.labels.tolist() == [0.0, 1.0]


def test_build_collator_returns_pointnext():
    assert isinstance(build_collator("pointnext"), PointNeXtCollator)
