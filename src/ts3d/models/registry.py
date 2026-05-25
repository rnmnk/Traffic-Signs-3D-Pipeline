from __future__ import annotations

from collections.abc import Callable

from ts3d.config.schemas import ModelConfig
from ts3d.models.base import BaseClassifier

ModelFactory = Callable[[ModelConfig], BaseClassifier]

MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator registering a factory under ``name``."""

    def wrap(factory: ModelFactory) -> ModelFactory:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} already registered.")
        MODEL_REGISTRY[name] = factory
        return factory

    return wrap


def build_model(cfg: ModelConfig) -> BaseClassifier:
    if cfg.name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {cfg.name!r}. Registered: {sorted(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[cfg.name](cfg)
