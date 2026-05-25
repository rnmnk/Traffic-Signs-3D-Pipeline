from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from ts3d.data.io import LazReader, PointCloud
from ts3d.data.metadata import Metadata, MetadataRow, SplitName


@dataclass
class ClusterSample:
    """One sample handed to the model pipeline after I/O and transforms."""

    candidate_id: str
    xyz: np.ndarray  # (N, 3) float32
    bbox_center: np.ndarray  # (3,) float32
    bbox_size: np.ndarray  # (3,) float32
    label: int


SampleTransform = Callable[[ClusterSample], ClusterSample]


class CandidateClusterDataset(Dataset[ClusterSample]):
    """Iterates candidate clusters for a given split."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: SplitName,
        *,
        metadata_filename: str = "metadata.json",
        clusters_subdir: str = "clusters",
        transform: SampleTransform | None = None,
    ) -> None:
        self._root = Path(dataset_root)
        self._split: SplitName = split
        self._clusters_dir = self._root / clusters_subdir / split
        self._metadata = Metadata.from_file(self._root / metadata_filename).filter_split(split)
        self._reader = LazReader()
        self._transform = transform

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def _resolve_laz_path(self, candidate_id: str) -> Path:
        return self._clusters_dir / f"{candidate_id}.laz"

    def _load(self, row: MetadataRow) -> PointCloud:
        return self._reader.read(self._resolve_laz_path(row.candidate_id))

    def __getitem__(self, index: int) -> ClusterSample:
        row = self._metadata.row(index)
        pc = self._load(row)
        sample = ClusterSample(
            candidate_id=row.candidate_id,
            xyz=pc.xyz,
            bbox_center=np.asarray(row.bbox3d_center_xyz, dtype=np.float64),
            bbox_size=np.asarray(row.bbox3d_size_xyz, dtype=np.float32),
            label=row.label,
        )
        if self._transform is not None:
            sample = self._transform(sample)
        return sample
