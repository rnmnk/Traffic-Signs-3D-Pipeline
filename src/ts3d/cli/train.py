from __future__ import annotations

import shutil
from pathlib import Path

import typer
from torch.utils.data import DataLoader

from ts3d.config.loader import load_experiment_config
from ts3d.data.collate import build_collator
from ts3d.data.dataset import CandidateClusterDataset
from ts3d.data.transforms import build_transform_pipeline
from ts3d.models.registry import build_model
from ts3d.training.losses import build_loss
from ts3d.training.optim import build_optimizer, build_scheduler
from ts3d.training.tracking import build_tracker
from ts3d.training.trainer import Trainer
from ts3d.utils.logging import configure_logging, get_logger
from ts3d.utils.paths import resolve_experiment_paths
from ts3d.utils.seed import seed_everything


def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML."),
    log_level: str = typer.Option("INFO", "--log-level"),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id override."),
) -> None:
    """Train a model according to the experiment configuration."""
    cfg = load_experiment_config(config)
    paths = resolve_experiment_paths(cfg.output_dir, cfg.experiment_name, run_id=run_id)
    configure_logging(level=log_level, file=paths.logs / "train.log")
    log = get_logger(__name__)
    seed_everything(cfg.seed)

    shutil.copy2(config, paths.root / "config.yaml")

    log.info("experiment_start", name=cfg.experiment_name, run_dir=str(paths.root))

    train_tfm = build_transform_pipeline(
        num_points=cfg.data.num_points,
        augment=cfg.data.augment,
        rotate_z=cfg.data.rotate_z,
        jitter_sigma=cfg.data.jitter_sigma,
        seed=cfg.seed,
    )
    val_tfm = build_transform_pipeline(
        num_points=cfg.data.num_points,
        augment=False,
        rotate_z=False,
        jitter_sigma=0.0,
        seed=cfg.seed,
    )

    train_ds = CandidateClusterDataset(
        cfg.data.root,
        "train",
        metadata_filename=cfg.data.metadata_filename,
        clusters_subdir=cfg.data.clusters_subdir,
        transform=train_tfm,
    )
    val_ds = CandidateClusterDataset(
        cfg.data.root,
        "val",
        metadata_filename=cfg.data.metadata_filename,
        clusters_subdir=cfg.data.clusters_subdir,
        transform=val_tfm,
    )

    collator = build_collator(cfg.model.name)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.val_batch_size or cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collator,
    )

    model = build_model(cfg.model)
    optimizer = build_optimizer(cfg.train.optimizer, model.parameters())
    scheduler = build_scheduler(cfg.train.scheduler, optimizer, total_epochs=cfg.train.epochs)
    loss_fn = build_loss(cfg.train)
    tracker = build_tracker(cfg.tracking)

    with tracker:
        tracker.start_run(cfg.tracking.run_name or cfg.experiment_name)
        tracker.log_params(_flatten(cfg.model_dump()))
        tracker.set_tags({"backbone": cfg.model.name})

        metrics_path = paths.artifacts / "metrics.jsonl"
        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            tracker=tracker,
            checkpoint_dir=paths.checkpoints,
            metrics_path=metrics_path,
        )
        best_ckpt = trainer.fit()

        tracker.log_artifact(paths.root / "config.yaml")
        tracker.log_artifact(best_ckpt, artifact_path="checkpoints")
        if metrics_path.is_file():
            tracker.log_artifact(metrics_path, artifact_path="artifacts")
        train_log = paths.logs / "train.log"
        if train_log.is_file():
            tracker.log_artifact(train_log, artifact_path="logs")

    log.info("experiment_done", best_ckpt=str(best_ckpt))
    typer.echo(f"best checkpoint: {best_ckpt}")


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out
