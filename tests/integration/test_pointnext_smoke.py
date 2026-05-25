from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ts3d.config.schemas import ModelConfig
from ts3d.data.collate import build_collator
from ts3d.data.dataset import CandidateClusterDataset
from ts3d.data.transforms import build_transform_pipeline
from ts3d.models.registry import build_model


def test_pointnext_one_step(synthetic_dataset: Path):
    tfm = build_transform_pipeline(
        num_points=128, augment=False, rotate_z=False, jitter_sigma=0.0, seed=0
    )
    ds = CandidateClusterDataset(
        synthetic_dataset,
        split="train",
        transform=tfm,
    )
    collator = build_collator("pointnext")
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collator)

    model = build_model(ModelConfig(name="pointnext", dropout=0.0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    batch = next(iter(loader))
    logits = model(batch)
    assert logits.shape == (2,)
    loss = loss_fn(logits, batch.labels.float())
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).item()
