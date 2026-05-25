from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path

import torch


class EarlyStopping:
    """Stop when the monitored metric stops improving for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1.")
        self._patience = int(patience)
        self._min_delta = float(min_delta)
        self._best: float | None = None
        self._bad_epochs = 0

    @property
    def should_stop(self) -> bool:
        return self._bad_epochs >= self._patience

    def step(self, current: float) -> bool:
        if self._best is None or current > self._best + self._min_delta:
            self._best = current
            self._bad_epochs = 0
            return True
        self._bad_epochs += 1
        return False

    @property
    def best(self) -> float | None:
        return self._best


@dataclass(order=True)
class _CkptEntry:
    score: float
    path: Path = field(compare=False)


class CheckpointManager:
    """Keep the top-k checkpoints by monitored score (higher is better)."""

    def __init__(self, directory: str | Path, top_k: int = 1) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._top_k = int(top_k)
        self._heap: list[_CkptEntry] = []

    @property
    def best_path(self) -> Path | None:
        if not self._heap:
            return None
        return max(self._heap, key=lambda e: e.score).path

    def save(self, state: dict, *, epoch: int, score: float) -> Path:
        last_path = self._dir / "last.ckpt"
        torch.save({"state": state, "epoch": epoch, "score": score}, last_path)

        path = self._dir / f"epoch_{epoch:04d}-score_{score:.4f}.ckpt"
        torch.save({"state": state, "epoch": epoch, "score": score}, path)

        heapq.heappush(self._heap, _CkptEntry(score=score, path=path))
        while len(self._heap) > self._top_k:
            worst = heapq.heappop(self._heap)
            worst.path.unlink(missing_ok=True)

        best = self.best_path
        if best is not None:
            best_link = self._dir / "best.ckpt"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(best.name)
        return path
