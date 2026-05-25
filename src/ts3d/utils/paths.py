from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    """Per-run directory layout."""

    root: Path
    checkpoints: Path
    logs: Path
    artifacts: Path

    @classmethod
    def create(
        cls, base: str | Path, experiment_name: str, *, run_id: str | None = None
    ) -> ExperimentPaths:
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        root = Path(base) / experiment_name / run_id
        paths = cls(
            root=root,
            checkpoints=root / "checkpoints",
            logs=root / "logs",
            artifacts=root / "artifacts",
        )
        for p in (paths.root, paths.checkpoints, paths.logs, paths.artifacts):
            p.mkdir(parents=True, exist_ok=True)
        return paths


def resolve_experiment_paths(
    base: str | Path, experiment_name: str, run_id: str | None = None
) -> ExperimentPaths:
    return ExperimentPaths.create(base, experiment_name, run_id=run_id)
