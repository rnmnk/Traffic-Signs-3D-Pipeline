from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np


@dataclass(frozen=True)
class PointCloud:
    """A point cloud for a single candidate cluster."""

    xyz: np.ndarray

    def __post_init__(self) -> None:
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N, 3); got {self.xyz.shape}.")

    @property
    def num_points(self) -> int:
        return int(self.xyz.shape[0])


class LazReader:
    """Reads per-candidate LAZ files."""

    def read(self, path: str | Path) -> PointCloud:
        las_path = Path(path)
        if not las_path.is_file():
            raise FileNotFoundError(f"LAZ file not found: {las_path}")

        with laspy.open(str(las_path)) as reader:
            las = reader.read()

        xyz = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64, copy=False)
        return PointCloud(xyz=xyz)
