from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ts3d.data.dataset import ClusterSample


@dataclass
class DenseBatch:
    """Dense ``(B, N, 3)`` xyz batch with per-sample labels."""

    xyz: torch.Tensor
    labels: torch.Tensor
    candidate_ids: list[str]

    def to(self, device: torch.device | str) -> DenseBatch:
        return DenseBatch(
            xyz=self.xyz.to(device, non_blocking=True),
            labels=self.labels.to(device, non_blocking=True),
            candidate_ids=self.candidate_ids,
        )


class PointNeXtCollator:
    """Stack fixed-size clouds into a dense ``(B, N, 3)`` xyz tensor."""

    def __call__(self, samples: list[ClusterSample]) -> DenseBatch:
        xyz = np.stack([s.xyz for s in samples], axis=0)
        labels = np.fromiter((s.label for s in samples), dtype=np.float32, count=len(samples))
        return DenseBatch(
            xyz=torch.from_numpy(xyz),
            labels=torch.from_numpy(labels),
            candidate_ids=[s.candidate_id for s in samples],
        )


def build_collator(model_name: str) -> PointNeXtCollator:
    if model_name == "pointnext":
        return PointNeXtCollator()
    raise ValueError(f"Unknown model name for collator selection: {model_name!r}")
