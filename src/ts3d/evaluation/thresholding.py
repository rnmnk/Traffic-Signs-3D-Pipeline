from __future__ import annotations

import numpy as np

from ts3d.config.schemas import EvalConfig
from ts3d.evaluation.metrics import PRCurve, compute_pr_curve


def _at_fixed_recall(curve: PRCurve, target: float) -> float:
    mask = curve.recall >= target
    if not mask.any():
        return 0.0
    candidate_idx = np.where(mask)[0]
    best = candidate_idx[int(np.argmax(curve.precision[candidate_idx]))]
    return float(curve.thresholds[min(best, len(curve.thresholds) - 1)])


def pick_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    cfg: EvalConfig,
) -> float:
    """Resolve the configured threshold strategy to a concrete float."""
    t = cfg.threshold
    if isinstance(t, float):
        return float(t)
    if t == "fixed_recall":
        return _at_fixed_recall(compute_pr_curve(labels, scores), cfg.target_recall)
    raise ValueError(f"Unknown threshold strategy: {t!r}")
