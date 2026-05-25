from __future__ import annotations

import torch
import torch.nn as nn

from ts3d.config.schemas import TrainConfig


def build_loss(cfg: TrainConfig) -> nn.Module:
    if cfg.loss == "bce":
        pw = (
            torch.tensor(cfg.pos_weight, dtype=torch.float32)
            if cfg.pos_weight is not None
            else None
        )
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    raise ValueError(f"Unknown loss: {cfg.loss!r}")
