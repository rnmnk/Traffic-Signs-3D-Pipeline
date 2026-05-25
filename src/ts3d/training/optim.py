from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR

from ts3d.config.schemas import OptimizerConfig, SchedulerConfig


def build_optimizer(cfg: OptimizerConfig, params: Iterable[torch.nn.Parameter]) -> Optimizer:
    if cfg.name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer: {cfg.name!r}")


def build_scheduler(
    cfg: SchedulerConfig, optimizer: Optimizer, *, total_epochs: int
) -> LRScheduler | None:
    if cfg.name == "none":
        return None
    if cfg.name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs), eta_min=cfg.min_lr)
    if cfg.name == "step":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    raise ValueError(f"Unknown scheduler: {cfg.name!r}")
