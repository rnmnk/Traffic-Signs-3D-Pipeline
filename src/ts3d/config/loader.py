from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ts3d.config.schemas import ExperimentConfig

T = TypeVar("T", bound=BaseModel)


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}.")
    return data


def load_config(path: str | Path, schema: type[T]) -> T:
    """Load a YAML file and validate it against ``schema``."""
    return schema.model_validate(_load_yaml(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a training/eval experiment YAML into ``ExperimentConfig``."""
    return load_config(path, ExperimentConfig)
