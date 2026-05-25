from __future__ import annotations

import numpy as np
import pytest

from ts3d.config.schemas import EvalConfig
from ts3d.evaluation.fp_analysis import analyze_fp_reduction
from ts3d.evaluation.metrics import BinaryClassificationMetrics, compute_pr_curve
from ts3d.evaluation.thresholding import pick_threshold


def test_perfect_classifier_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = BinaryClassificationMetrics.from_scores(labels, scores, threshold=0.5)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.auprc == pytest.approx(1.0)
    assert m.auroc == pytest.approx(1.0)
    assert m.tp == 2 and m.tn == 2 and m.fp == 0 and m.fn == 0


def test_threshold_fixed_recall_picks_separating_point():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    cfg = EvalConfig(threshold="fixed_recall", target_recall=1.0)
    t = pick_threshold(labels, scores, cfg)
    assert 0.2 < t <= 0.8


def test_threshold_fixed():
    labels = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    cfg = EvalConfig(threshold=0.42)
    assert pick_threshold(labels, scores, cfg) == pytest.approx(0.42)


def test_pr_curve_shapes_match():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    curve = compute_pr_curve(labels, scores)
    assert curve.precision.shape == curve.recall.shape == curve.thresholds.shape


def test_fp_reduction_report():
    import pandas as pd

    metadata = pd.DataFrame(
        [
            {"candidate_id": "a", "label": 1},
            {"candidate_id": "b", "label": 0},
            {"candidate_id": "c", "label": 0},
            {"candidate_id": "d", "label": 1},
        ]
    )
    scores = {"a": 0.9, "b": 0.1, "c": 0.6, "d": 0.8}
    report = analyze_fp_reduction(metadata=metadata, scores=scores, threshold=0.5)
    assert report.upstream_tp == 2
    assert report.upstream_fp == 2
    assert report.kept_tp == 2  # both TPs above threshold
    assert report.kept_fp == 1  # 'c' is above threshold
    assert report.fp_reduction == pytest.approx(0.5)
    assert report.tp_preservation == pytest.approx(1.0)
