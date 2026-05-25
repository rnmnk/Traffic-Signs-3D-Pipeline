from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ts3d.config.schemas import EvalConfig
from ts3d.evaluation.metrics import BinaryClassificationMetrics
from ts3d.evaluation.thresholding import pick_threshold
from ts3d.models.base import BaseClassifier


@dataclass
class MetricsReport:
    metrics: BinaryClassificationMetrics
    threshold: float
    num_samples: int
    num_positives: int
    num_negatives: int
    candidate_ids: list[str]
    scores: list[float]
    labels: list[int]

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "threshold": self.threshold,
            "num_samples": self.num_samples,
            "num_positives": self.num_positives,
            "num_negatives": self.num_negatives,
            "metrics": asdict(self.metrics),
        }
        path.write_text(json.dumps(payload, indent=2))


class Evaluator:
    """Runs the model on a DataLoader and computes metrics + PR artifacts."""

    def __init__(self, model: BaseClassifier, device: torch.device | str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def score_loader(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[str]]:
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        cids: list[str] = []
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = batch.to(self.device)
            s = torch.sigmoid(self.model(batch)).cpu().numpy()
            scores.append(s)
            labels.append(batch.labels.cpu().numpy())
            cids.extend(batch.candidate_ids)
        return (
            np.concatenate(scores) if scores else np.zeros(0),
            np.concatenate(labels) if labels else np.zeros(0),
            cids,
        )

    def evaluate(
        self,
        loader: DataLoader,
        cfg: EvalConfig,
        *,
        threshold_override: float | None = None,
    ) -> MetricsReport:
        """Score the loader and compute metrics."""
        scores, labels, cids = self.score_loader(loader)
        labels_int = labels.astype(int)

        if threshold_override is not None:
            threshold = float(threshold_override)
        else:
            threshold = pick_threshold(labels_int, scores, cfg)
        m = BinaryClassificationMetrics.from_scores(labels_int, scores, threshold=threshold)

        return MetricsReport(
            metrics=m,
            threshold=threshold,
            num_samples=len(scores),
            num_positives=int((labels_int == 1).sum()),
            num_negatives=int((labels_int == 0).sum()),
            candidate_ids=cids,
            scores=scores.tolist(),
            labels=labels_int.tolist(),
        )

    def save_plots(self, report: MetricsReport, out_dir: str | Path) -> Path:
        """Write pr_curve.png, roc_curve.png, and confusion_matrix.png.

        Styled after ``eval_out_honest/analyze.py``: AUPRC/AUROC labels,
        filled curves, and a single operating-point marker at ``report.threshold``.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = np.asarray(report.labels, dtype=int)
        scores = np.asarray(report.scores)
        tau = float(report.threshold)

        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve

        p, r, _ = precision_recall_curve(labels, scores)
        pred = (scores >= tau).astype(int)
        tp = int(((labels == 1) & (pred == 1)).sum())
        fp = int(((labels == 0) & (pred == 1)).sum())
        fn = int(((labels == 1) & (pred == 0)).sum())
        tn = int(((labels == 0) & (pred == 0)).sum())
        op_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        op_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("Precision-Recall Curve")
        ax.plot(r, p, color="#1f77b4", lw=1.8, label=f"AUPRC = {report.metrics.auprc:.4f}")
        ax.fill_between(r, p, alpha=0.08, color="#1f77b4")
        ax.scatter(
            [op_r],
            [op_p],
            c="#d62728",
            marker="*",
            s=240,
            zorder=5,
            edgecolor="black",
            linewidth=0.6,
            label=f"t* = {tau:.3f}  (P={op_p:.3f}, R={op_r:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "pr_curve.png", dpi=240)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(labels, scores)
        op_x = 1.0 - (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title("ROC Curve")
        ax.plot(fpr, tpr, color="#1f77b4", lw=1.8, label=f"AUROC = {report.metrics.auroc:.4f}")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#888", label="AUROC = 0.5")
        ax.scatter(
            [op_x],
            [op_r],
            c="#d62728",
            marker="*",
            s=240,
            zorder=5,
            edgecolor="black",
            linewidth=0.6,
            label=f"t* = {tau:.3f}  (1-S={op_x:.3f}, R={op_r:.3f})",
        )
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Recall")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower right", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=240)
        plt.close(fig)

        mat = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f"Confusion Matrix (t* = {tau:.3f})")
        im = ax.imshow(mat, cmap="Blues", aspect="equal")
        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{cell_labels[i][j]}\n{mat[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=color,
                )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted Neg", "Predicted Pos"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual Neg", "Actual Pos"])
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=240)
        plt.close(fig)

        return out_dir
