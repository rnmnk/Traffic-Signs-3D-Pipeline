from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class FPReductionReport:
    threshold: float
    upstream_tp: int
    upstream_fp: int
    kept_tp: int
    kept_fp: int
    rejected_tp: int
    rejected_fp: int
    tp_preservation: float  # kept_tp / upstream_tp
    fp_reduction: float  # rejected_fp / upstream_fp

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


def analyze_fp_reduction(
    *,
    metadata: pd.DataFrame,
    scores: dict[str, float],
    threshold: float,
) -> FPReductionReport:
    """Compute FP-reduction metrics from labeled metadata and predicted scores."""
    df = metadata.copy()
    df["score"] = df["candidate_id"].map(scores)
    df = df.dropna(subset=["score", "label"]).copy()
    df["label"] = df["label"].astype(int)

    tp_mask = df["label"] == 1
    fp_mask = df["label"] == 0
    kept = df["score"] >= threshold

    upstream_tp = int(tp_mask.sum())
    upstream_fp = int(fp_mask.sum())
    kept_tp = int((tp_mask & kept).sum())
    kept_fp = int((fp_mask & kept).sum())

    return FPReductionReport(
        threshold=float(threshold),
        upstream_tp=upstream_tp,
        upstream_fp=upstream_fp,
        kept_tp=kept_tp,
        kept_fp=kept_fp,
        rejected_tp=upstream_tp - kept_tp,
        rejected_fp=upstream_fp - kept_fp,
        tp_preservation=float(kept_tp / upstream_tp) if upstream_tp else float("nan"),
        fp_reduction=float((upstream_fp - kept_fp) / upstream_fp) if upstream_fp else float("nan"),
    )
