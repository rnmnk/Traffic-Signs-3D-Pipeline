from __future__ import annotations

from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader

from ts3d.config.loader import load_experiment_config
from ts3d.data.collate import build_collator
from ts3d.data.dataset import CandidateClusterDataset
from ts3d.data.transforms import build_transform_pipeline
from ts3d.evaluation.evaluator import Evaluator
from ts3d.evaluation.fp_analysis import analyze_fp_reduction
from ts3d.evaluation.thresholding import pick_threshold
from ts3d.models.registry import build_model
from ts3d.utils.logging import configure_logging, get_logger


def _build_loader(cfg, split: str):
    tfm = build_transform_pipeline(
        num_points=cfg.data.num_points,
        augment=False,
        rotate_z=False,
        jitter_sigma=0.0,
        seed=cfg.seed,
    )
    ds = CandidateClusterDataset(
        cfg.data.root,
        split=split,  # type: ignore[arg-type]
        metadata_filename=cfg.data.metadata_filename,
        clusters_subdir=cfg.data.clusters_subdir,
        transform=tfm,
    )
    collator = build_collator(cfg.model.name)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.val_batch_size or cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collator,
    )
    return ds, loader


def main(
    config: Path = typer.Option(..., "--config", "-c"),
    ckpt: Path = typer.Option(..., "--ckpt"),
    split: str = typer.Option("test", "--split"),
    out_dir: Path = typer.Option(Path("eval_out"), "--out-dir"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Score a split and emit a metrics report + plots."""
    configure_logging(level=log_level)
    log = get_logger(__name__)

    cfg = load_experiment_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, loader = _build_loader(cfg, split)

    model = build_model(cfg.model)
    state = torch.load(str(ckpt), map_location="cpu")
    model.load_state_dict(state.get("state", state))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(model, device=device)

    if split == "val":
        threshold_override: float | None = None
    else:
        _, thr_loader = _build_loader(cfg, "val")
        thr_scores, thr_labels, _ = evaluator.score_loader(thr_loader)
        threshold_override = float(pick_threshold(thr_labels.astype(int), thr_scores, cfg.eval))
        log.info("threshold_selected_on_val", threshold=threshold_override)

    report = evaluator.evaluate(loader, cfg.eval, threshold_override=threshold_override)

    report.to_json(out_dir / "metrics.json")
    evaluator.save_plots(report, out_dir)

    scores_by_id = dict(zip(report.candidate_ids, report.scores, strict=True))
    fp_report = analyze_fp_reduction(
        metadata=ds.metadata.df,
        scores=scores_by_id,
        threshold=report.threshold,
    )
    fp_report.to_json(out_dir / "fp_reduction.json")

    scored_df = ds.metadata.df.assign(score=ds.metadata.df["candidate_id"].map(scores_by_id))
    scored_df.to_json(out_dir / "scored_metadata.json", orient="records", indent=2)

    log.info(
        "eval_done",
        threshold=report.threshold,
        f1=report.metrics.f1,
        auprc=report.metrics.auprc,
        fp_reduction=fp_report.fp_reduction,
        tp_preservation=fp_report.tp_preservation,
    )
    typer.echo(
        f"F1={report.metrics.f1:.3f} AUPRC={report.metrics.auprc:.3f} "
        f"FP-reduction={fp_report.fp_reduction:.3f} TP-preservation={fp_report.tp_preservation:.3f}"
    )
