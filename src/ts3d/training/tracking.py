from __future__ import annotations

from pathlib import Path
from typing import Any

from ts3d.config.schemas import TrackingConfig


class MLflowTracker:
    """MLflow-backed experiment tracker."""

    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    def start_run(self, run_name: str | None = None) -> None:
        self._mlflow.start_run(run_name=run_name)

    def end_run(self, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)

    def log_params(self, params: dict[str, Any]) -> None:
        safe = {
            k: v if v is None or isinstance(v, (str, int, float, bool)) else str(v)
            for k, v in params.items()
        }
        self._mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._mlflow.set_tags(tags)

    def __enter__(self) -> MLflowTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_run("FAILED" if exc_type is not None else "FINISHED")


def build_tracker(cfg: TrackingConfig) -> MLflowTracker:
    return MLflowTracker(experiment=cfg.experiment, tracking_uri=cfg.tracking_uri)


Tracker = MLflowTracker
