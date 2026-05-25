from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ts3d.config.schemas import ExperimentConfig
from ts3d.data.collate import DenseBatch
from ts3d.evaluation.metrics import BinaryClassificationMetrics
from ts3d.models.base import BaseClassifier
from ts3d.training.callbacks import CheckpointManager, EarlyStopping
from ts3d.training.tracking import Tracker
from ts3d.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EpochOutputs:
    loss: float
    metrics: dict[str, float]


class Trainer:
    """Fit/validate loop driven by validation AUPRC (higher is better)."""

    def __init__(
        self,
        *,
        model: BaseClassifier,
        cfg: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        loss_fn: torch.nn.Module,
        tracker: Tracker,
        checkpoint_dir: str | Path,
        metrics_path: str | Path | None = None,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",  # noqa: B008
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device)
        self.tracker = tracker
        self.device = torch.device(device)

        self._ckpt_mgr = CheckpointManager(checkpoint_dir, top_k=cfg.train.checkpoint_top_k)
        self._early = (
            EarlyStopping(cfg.train.early_stopping_patience)
            if cfg.train.early_stopping_patience
            else None
        )
        self._scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.train.mixed_precision and self.device.type == "cuda"
        )
        self._metrics_path = Path(metrics_path) if metrics_path is not None else None
        if self._metrics_path is not None:
            self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._metrics_path.write_text("")

    def _move(self, batch: DenseBatch) -> DenseBatch:
        return batch.to(self.device)

    def fit(self) -> Path:
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_out = self._train_epoch(epoch)
            val_out = self._validate_epoch(epoch)

            metrics_logged = {
                "train/loss": train_out.loss,
                "val/loss": val_out.loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"val/{k}": v for k, v in val_out.metrics.items()},
            }
            self.tracker.log_metrics(metrics_logged, step=epoch)
            self._print_epoch_summary(epoch, metrics_logged)
            self._append_metrics_jsonl(epoch, metrics_logged)

            monitor_val = val_out.metrics.get("auprc", float("nan"))
            improved = True
            if self._early is not None:
                improved = self._early.step(monitor_val)
            if improved:
                self._ckpt_mgr.save(self.model.state_dict(), epoch=epoch, score=monitor_val)

            if self.scheduler is not None:
                self.scheduler.step()

            if self._early is not None and self._early.should_stop:
                log.info("early_stop", epoch=epoch, best=self._early.best)
                break

        best = self._ckpt_mgr.best_path
        if best is None:
            raise RuntimeError("No checkpoint was saved; did training run at all?")
        return best

    def _train_epoch(self, epoch: int) -> EpochOutputs:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        n_samples = 0
        pbar = tqdm(
            self.train_loader,
            desc=f"train  e{epoch:03d}/{self.cfg.train.epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = self._move(batch)
            labels = batch.labels.float()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self._scaler.is_enabled()):
                logits = self.model(batch)
                loss = self.loss_fn(logits, labels)

            self._scaler.scale(loss).backward()
            if self.cfg.train.grad_clip_norm:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip_norm
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()

            bs = labels.numel()
            total_loss += float(loss.detach()) * bs
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += int((preds == labels).sum().item())
            n_samples += bs
            pbar.set_postfix(
                loss=f"{total_loss / max(n_samples, 1):.4f}",
                acc=f"{total_correct / max(n_samples, 1):.3f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return EpochOutputs(
            loss=total_loss / max(n_samples, 1),
            metrics={"acc": total_correct / max(n_samples, 1)},
        )

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> EpochOutputs:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        n_samples = 0
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        pbar = tqdm(
            self.val_loader,
            desc=f"val    e{epoch:03d}/{self.cfg.train.epochs}",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in pbar:
            batch = self._move(batch)
            labels = batch.labels.float()
            logits = self.model(batch)
            loss = self.loss_fn(logits, labels)
            total_loss += float(loss) * labels.numel()
            n_samples += labels.numel()

            scores = torch.sigmoid(logits).detach()
            preds = (scores >= 0.5).float()
            total_correct += int((preds == labels).sum().item())
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            pbar.set_postfix(
                loss=f"{total_loss / max(n_samples, 1):.4f}",
                acc=f"{total_correct / max(n_samples, 1):.3f}",
            )

        if n_samples == 0:
            return EpochOutputs(loss=float("nan"), metrics={})

        scores = np.concatenate(all_scores)
        labels_np = np.concatenate(all_labels).astype(int)
        m = BinaryClassificationMetrics.from_scores(labels_np, scores)
        return EpochOutputs(
            loss=total_loss / n_samples,
            metrics={
                "acc": total_correct / n_samples,
                "f1": m.f1,
                "precision": m.precision,
                "recall": m.recall,
                "auroc": m.auroc,
                "auprc": m.auprc,
            },
        )

    def _print_epoch_summary(self, epoch: int, metrics: dict[str, float]) -> None:
        keys = (
            "train/loss",
            "val/loss",
            "val/acc",
            "val/f1",
            "val/precision",
            "val/recall",
            "val/auroc",
            "val/auprc",
            "lr",
        )
        parts = [f"{k}={metrics[k]:.4f}" for k in keys if k in metrics]
        msg = f"[epoch {epoch:03d}/{self.cfg.train.epochs}] " + " | ".join(parts)
        tqdm.write(msg)
        log.info("epoch", epoch=epoch, **{k: round(v, 5) for k, v in metrics.items()})

    def _append_metrics_jsonl(self, epoch: int, metrics: dict[str, float]) -> None:
        if self._metrics_path is None:
            return
        record = {"epoch": int(epoch), **{k: float(v) for k, v in metrics.items()}}
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
