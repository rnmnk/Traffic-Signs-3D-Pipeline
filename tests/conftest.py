from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import pytest


def _make_laz(path: Path, xyz: np.ndarray, intensity: np.ndarray | None = None) -> None:
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header=header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    if intensity is not None:
        las.intensity = intensity.astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Build a tiny standardized dataset layout with both splits populated."""
    root = tmp_path / "dataset"
    rows = []
    rng = np.random.default_rng(0)
    for i, (split, label) in enumerate(
        [("train", 1), ("train", 0), ("val", 1), ("val", 0), ("test", 1), ("test", 0)]
    ):
        cid = f"cand_{i:04d}"
        n = 600
        # Cluster shaped as a rectangle roughly around center.
        center = np.array([float(i), 0.0, 2.0])
        size = np.array([0.6, 0.05, 0.8])
        pts = (rng.random((n, 3)) - 0.5) * size + center
        intensity = rng.integers(0, 65535, size=n, dtype=np.uint32).astype(np.uint16)
        _make_laz(root / "clusters" / split / f"{cid}.laz", pts.astype(np.float32), intensity)
        rows.append(
            {
                "candidate_id": cid,
                "split": split,
                "label": label,
                "match_iou_3d": 0.5 if label else 0.0,
                "matched_gt_id": f"gt_{i}" if label else None,
                "bbox3d_center_xyz": [float(c) for c in center],
                "bbox3d_size_xyz": [float(s) for s in size],
                "bbox3d_yaw": 0.0,
                "source_dataset_id": "ds0",
                "session_id": "sess0",
                "image_id": f"img_{i}",
                "n_points": n,
            }
        )
    df = pd.DataFrame(rows)
    df.to_json(root / "metadata.json", orient="records", indent=2)
    return root
