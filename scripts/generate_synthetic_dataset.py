"""Generate a synthetic dataset for pipeline smoke-testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np

SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}


def _split_counts(total: int) -> dict[str, tuple[int, int]]:
    """Split `total` into per-split `(n_tp, n_fp)."""
    if total < len(SPLIT_RATIOS) * 2:
        raise ValueError(
            f"total={total} too small; need at least {len(SPLIT_RATIOS) * 2} clusters."
        )
    raw = {s: total * r for s, r in SPLIT_RATIOS.items()}
    counts: dict[str, tuple[int, int]] = {}
    for split, n in raw.items():
        per_class = max(1, round(n / 2))
        counts[split] = (per_class, per_class)
    return counts


def _sample_tp(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense planar rectangle (~0.4-0.9 m), N ~ U(250, 1200), thickness ~3 cm."""
    width = float(rng.uniform(0.40, 0.90))
    height = float(rng.uniform(0.40, 0.90))
    thickness = float(rng.uniform(0.02, 0.06))
    n = int(rng.integers(250, 1200))

    u = rng.uniform(-width / 2, width / 2, size=n)
    v = rng.uniform(-height / 2, height / 2, size=n)
    w = rng.normal(0.0, thickness / 3.0, size=n)
    pts = np.stack([u, w, v], axis=1)

    yaw = float(rng.uniform(-np.pi, np.pi))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pts = pts @ rot.T

    center = np.array(
        [rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(2.0, 4.0)],
        dtype=np.float64,
    )
    pts = pts + center

    bbox_size = np.array([max(width, thickness), max(width, thickness), height], dtype=np.float64)
    intensity = rng.integers(20_000, 65_535, size=n, dtype=np.uint32).astype(np.uint16)
    return pts, intensity, center, bbox_size


def _sample_fp(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sparse irregular blob, N ~ U(60, 500), extent ~0.1-0.45 m, no flat plane."""
    extents = rng.uniform(0.10, 0.45, size=3)
    n = int(rng.integers(60, 500))
    pts = rng.normal(0.0, 1.0, size=(n, 3)) * (extents / 2.5)

    if rng.random() < 0.5:
        pts[:, 2] *= rng.uniform(0.5, 1.8)

    center = np.array(
        [rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(0.5, 4.0)],
        dtype=np.float64,
    )
    pts = pts + center
    bbox_size = np.maximum(extents, 1e-3).astype(np.float64)
    intensity = rng.integers(0, 65_535, size=n, dtype=np.uint32).astype(np.uint16)
    return pts, intensity, center, bbox_size


def _write_laz(path: Path, xyz: np.ndarray, intensity: np.ndarray) -> None:
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = xyz.min(axis=0)
    las = laspy.LasData(header=header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.intensity = intensity
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--total",
        type=int,
        default=200,
        help="Total number of candidates (split 70/15/15 train/val/test, balanced TP/FP).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_counts = _split_counts(args.total)
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    args.out.mkdir(parents=True, exist_ok=True)

    for split, (n_tp, n_fp) in split_counts.items():
        for kind, n, sampler in (("tp", n_tp, _sample_tp), ("fp", n_fp, _sample_fp)):
            for i in range(n):
                cid = f"syn_{split}_{kind}_{i:04d}"
                xyz, intensity, center, size = sampler(rng)
                _write_laz(args.out / "clusters" / split / f"{cid}.laz", xyz, intensity)
                records.append(
                    {
                        "candidate_id": cid,
                        "split": split,
                        "label": 1 if kind == "tp" else 0,
                        "bbox3d_center_xyz": [float(x) for x in center],
                        "bbox3d_size_xyz": [float(x) for x in size],
                    }
                )

    rng_perm = np.random.default_rng(args.seed + 1)
    rng_perm.shuffle(records)
    (args.out / "metadata.json").write_text(json.dumps(records, indent=2))

    n_tp = sum(r["label"] == 1 for r in records)
    n_fp = sum(r["label"] == 0 for r in records)
    print(
        f"Wrote {len(records)} candidates ({n_tp} TP / {n_fp} FP) to {args.out} "
        f"across splits {split_counts}."
    )


if __name__ == "__main__":
    main()
