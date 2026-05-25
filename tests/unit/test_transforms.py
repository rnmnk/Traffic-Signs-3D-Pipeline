from __future__ import annotations

import numpy as np

from ts3d.data.dataset import ClusterSample
from ts3d.data.transforms import (
    Jitter,
    NormalizeByBBox,
    RandomRotateZ,
    Recenter,
    Subsample,
    build_transform_pipeline,
)


def _sample(n: int = 256) -> ClusterSample:
    rng = np.random.default_rng(0)
    return ClusterSample(
        candidate_id="c",
        xyz=(rng.random((n, 3)).astype(np.float32) - 0.5) * 2
        + np.array([10.0, 5.0, 2.0], dtype=np.float32),
        bbox_center=np.array([10.0, 5.0, 2.0], dtype=np.float32),
        bbox_size=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        label=1,
    )


def test_recenter_normalizes_bbox_center_to_origin():
    s = Recenter()(_sample())
    assert np.abs(s.xyz.mean(axis=0)).max() < 0.5  # roughly around origin


def test_normalize_scales_within_unit_sphere():
    s = NormalizeByBBox()(Recenter()(_sample()))
    assert float(np.linalg.norm(s.xyz, axis=1).max()) <= 1.5


def test_rotate_z_preserves_z_and_norms():
    s = _sample()
    rot = RandomRotateZ(max_deg=90, rng=np.random.default_rng(42))(s)
    assert rot.xyz.shape == s.xyz.shape
    np.testing.assert_allclose(rot.xyz[:, 2], s.xyz[:, 2], atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(rot.xyz[:, :2], axis=1),
        np.linalg.norm(s.xyz[:, :2], axis=1),
        atol=1e-4,
    )


def test_jitter_is_bounded():
    s = _sample()
    j = Jitter(sigma=0.01, clip=0.05, rng=np.random.default_rng(0))(s)
    assert np.abs(j.xyz - s.xyz).max() <= 0.05 + 1e-6


def test_subsample_outputs_exact_count():
    s = _sample(256)
    out = Subsample(num_points=128)(s)
    assert out.xyz.shape == (128, 3)


def test_pipeline_composes():
    tfm = build_transform_pipeline(
        num_points=128, augment=True, rotate_z=True, jitter_sigma=0.01, seed=1
    )
    out = tfm(_sample(256))
    assert out.xyz.shape == (128, 3)


def test_subsample_deterministic_is_id_keyed():
    s = _sample(256)
    sub = Subsample(num_points=64, deterministic=True, salt=42)
    a = sub(s).xyz
    b = sub(s).xyz
    np.testing.assert_array_equal(a, b)


def test_subsample_deterministic_differs_per_id():
    s1 = _sample(256)
    s2 = _sample(256)
    object.__setattr__(s2, "candidate_id", "different_id")
    sub = Subsample(num_points=64, deterministic=True, salt=42)
    assert not np.array_equal(sub(s1).xyz, sub(s2).xyz)


def test_subsample_random_changes_each_call():
    s = _sample(256)
    sub = Subsample(num_points=64, rng=np.random.default_rng(0))
    a = sub(s).xyz
    b = sub(s).xyz
    assert not np.array_equal(a, b)


def test_eval_pipeline_is_deterministic_across_builds():
    s = _sample(256)
    tfm_a = build_transform_pipeline(
        num_points=128, augment=False, rotate_z=False, jitter_sigma=0.0, seed=42
    )
    tfm_b = build_transform_pipeline(
        num_points=128, augment=False, rotate_z=False, jitter_sigma=0.0, seed=42
    )
    np.testing.assert_array_equal(tfm_a(s).xyz, tfm_b(s).xyz)
