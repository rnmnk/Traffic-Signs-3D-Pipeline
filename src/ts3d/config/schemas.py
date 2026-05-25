from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class DataConfig(ConfigBase):
    root: Path
    metadata_filename: str = "metadata.json"
    clusters_subdir: str = "clusters"

    num_points: int = Field(1024, gt=0)

    augment: bool = True
    rotate_z: bool = True
    jitter_sigma: float = Field(0.01, ge=0.0)

    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True


class ModelConfig(ConfigBase):
    name: Literal["pointnext"] = "pointnext"
    dropout: float = Field(0.5, ge=0.0, lt=1.0)


class OptimizerConfig(ConfigBase):
    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = Field(1e-3, gt=0.0)
    weight_decay: float = Field(1e-4, ge=0.0)
    momentum: float = Field(0.9, ge=0.0, lt=1.0)


class SchedulerConfig(ConfigBase):
    name: Literal["none", "cosine", "step"] = "cosine"
    step_size: int = Field(30, gt=0)
    gamma: float = Field(0.1, gt=0.0)
    min_lr: float = Field(1e-6, ge=0.0)


class TrainConfig(ConfigBase):
    epochs: int = Field(100, gt=0)
    batch_size: int = Field(32, gt=0)
    val_batch_size: int | None = None

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    loss: Literal["bce"] = "bce"
    pos_weight: float | None = None

    grad_clip_norm: float | None = None
    mixed_precision: bool = True

    early_stopping_patience: int | None = 20

    checkpoint_top_k: int = Field(1, ge=1)


class EvalConfig(ConfigBase):
    threshold: float | Literal["fixed_recall"] = "fixed_recall"
    target_recall: float = Field(0.95, gt=0.0, le=1.0)


class TrackingConfig(ConfigBase):
    tracking_uri: str | None = None
    experiment: str = "traffic-signs-3d-pipeline"
    run_name: str | None = None
    log_model: bool = True


class ExperimentConfig(ConfigBase):
    experiment_name: str
    seed: int = 42
    output_dir: Path = Path("runs")

    data: DataConfig
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
