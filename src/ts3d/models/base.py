from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ts3d.data.collate import DenseBatch


class BaseClassifier(nn.Module, ABC):
    """All cluster classifiers emit a single logit per candidate (B,)."""

    @abstractmethod
    def forward(self, batch: DenseBatch) -> torch.Tensor:  # (B,)
        ...

    @torch.no_grad()
    def predict_proba(self, batch: DenseBatch) -> torch.Tensor:
        """Return sigmoid-activated scores in ``[0, 1]``."""
        logits = self.forward(batch)
        return torch.sigmoid(logits)
