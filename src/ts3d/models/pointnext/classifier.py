from __future__ import annotations

import torch
import torch.nn as nn

from ts3d.config.schemas import ModelConfig
from ts3d.data.collate import DenseBatch
from ts3d.models.base import BaseClassifier
from ts3d.models.pointnext.backbone import PointNeXtBackbone
from ts3d.models.registry import register_model


class PointNeXtClassifier(BaseClassifier):
    """PointNeXt-B backbone + MLP head producing a single logit per cluster."""

    def __init__(self, *, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = PointNeXtBackbone()
        c = self.backbone.out_channels
        self.head = nn.Sequential(
            nn.Linear(c, c // 2),
            nn.BatchNorm1d(c // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c // 2, 1),
        )

    def forward(self, batch: DenseBatch) -> torch.Tensor:
        emb = self.backbone(batch.xyz)
        return self.head(emb).squeeze(-1)


@register_model("pointnext")
def _build_pointnext(cfg: ModelConfig) -> PointNeXtClassifier:
    return PointNeXtClassifier(dropout=cfg.dropout)
