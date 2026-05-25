from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class PRCurve:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "thresholds": self.thresholds.tolist(),
        }


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    auroc: float
    auprc: float
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_scores(
        cls,
        labels: np.ndarray,
        scores: np.ndarray,
        *,
        threshold: float = 0.5,
    ) -> BinaryClassificationMetrics:
        labels = np.asarray(labels).astype(int)
        scores = np.asarray(scores).astype(float)
        preds = (scores >= threshold).astype(int)

        has_both = len(np.unique(labels)) == 2
        auroc = float(roc_auc_score(labels, scores)) if has_both else float("nan")
        auprc = float(average_precision_score(labels, scores)) if has_both else float("nan")

        p = float(precision_score(labels, preds, zero_division=0))
        r = float(recall_score(labels, preds, zero_division=0))
        f1 = float(f1_score(labels, preds, zero_division=0))

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        return cls(
            threshold=float(threshold),
            precision=p,
            recall=r,
            f1=f1,
            auroc=auroc,
            auprc=auprc,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )


def compute_pr_curve(labels: np.ndarray, scores: np.ndarray) -> PRCurve:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    thresholds = np.concatenate([thresholds, np.array([1.0 + 1e-9])])
    return PRCurve(precision=precision, recall=recall, thresholds=thresholds)
