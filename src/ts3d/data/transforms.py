from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from ts3d.data.dataset import ClusterSample


class Transform(ABC):
    @abstractmethod
    def __call__(self, sample: ClusterSample) -> ClusterSample: ...


class Compose(Transform):
    def __init__(self, transforms: Iterable[Transform]) -> None:
        self._transforms = list(transforms)

    def __call__(self, sample: ClusterSample) -> ClusterSample:
        for t in self._transforms:
            sample = t(sample)
        return sample


class Recenter(Transform):
    """Translate ``xyz`` to the bbox centre (or to the cloud centroid)."""

    def __init__(self, use_bbox_center: bool = True) -> None:
        self._use_bbox = use_bbox_center

    def __call__(self, sample: ClusterSample) -> ClusterSample:
        center = sample.bbox_center if self._use_bbox else sample.xyz.mean(axis=0)
        xyz = (sample.xyz - center).astype(np.float32, copy=False)
        return replace(sample, xyz=xyz)


class NormalizeByBBox(Transform):
    """Scale ``xyz`` by the bbox diagonal."""

    def __init__(self, eps: float = 1e-6) -> None:
        self._eps = eps

    def __call__(self, sample: ClusterSample) -> ClusterSample:
        diag = float(np.linalg.norm(sample.bbox_size) + self._eps)
        xyz = (sample.xyz / diag).astype(np.float32, copy=False)
        return replace(sample, xyz=xyz)


class RandomRotateZ(Transform):
    """Random yaw rotation; pole-mounted signs are invariant under yaw."""

    def __init__(self, max_deg: float = 180.0, rng: np.random.Generator | None = None) -> None:
        self._max_rad = float(np.deg2rad(max_deg))
        self._rng = rng or np.random.default_rng()

    def __call__(self, sample: ClusterSample) -> ClusterSample:
        theta = float(self._rng.uniform(-self._max_rad, self._max_rad))
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = sample.xyz @ r.T
        return replace(sample, xyz=xyz.astype(np.float32, copy=False))


class Jitter(Transform):
    """Per-point additive Gaussian jitter."""

    def __init__(
        self, sigma: float = 0.01, clip: float = 0.05, rng: np.random.Generator | None = None
    ) -> None:
        self._sigma = float(sigma)
        self._clip = float(clip)
        self._rng = rng or np.random.default_rng()

    def __call__(self, sample: ClusterSample) -> ClusterSample:
        if self._sigma <= 0.0:
            return sample
        noise = self._rng.normal(0.0, self._sigma, size=sample.xyz.shape).astype(np.float32)
        np.clip(noise, -self._clip, self._clip, out=noise)
        return replace(sample, xyz=(sample.xyz + noise).astype(np.float32, copy=False))


class Subsample(Transform):
    """Resample each cloud to exactly ``num_points``."""

    def __init__(
        self,
        num_points: int,
        rng: np.random.Generator | None = None,
        *,
        deterministic: bool = False,
        salt: int = 0,
    ) -> None:
        if num_points <= 0:
            raise ValueError("num_points must be positive.")
        self._num_points = int(num_points)
        self._rng = rng or np.random.default_rng()
        self._deterministic = bool(deterministic)
        self._salt = int(salt)

    @staticmethod
    def _seed_from_id(candidate_id: str, salt: int) -> int:
        digest = hashlib.blake2b(
            candidate_id.encode("utf-8"), digest_size=8, salt=salt.to_bytes(8, "little")
        ).digest()
        return int.from_bytes(digest, "little", signed=False)

    def _rng_for(self, sample: ClusterSample) -> np.random.Generator:
        if not self._deterministic:
            return self._rng
        return np.random.default_rng(self._seed_from_id(sample.candidate_id, self._salt))

    def __call__(self, sample: ClusterSample) -> ClusterSample:
        n = sample.xyz.shape[0]
        if n == 0:
            raise ValueError(f"Empty point cloud for candidate {sample.candidate_id!r}.")
        rng = self._rng_for(sample)
        replace_ = n < self._num_points
        idx = rng.choice(n, self._num_points, replace=replace_)
        return replace(sample, xyz=sample.xyz[idx].astype(np.float32, copy=False))


def build_transform_pipeline(
    *,
    num_points: int,
    augment: bool,
    rotate_z: bool,
    jitter_sigma: float,
    seed: int | None = None,
) -> Compose:
    """Recenter -> normalize -> (optional augment) -> subsample."""
    rng = np.random.default_rng(seed)
    steps: list[Transform] = [Recenter(use_bbox_center=True), NormalizeByBBox()]
    if augment:
        if rotate_z:
            steps.append(RandomRotateZ(max_deg=180.0, rng=rng))
        if jitter_sigma > 0.0:
            steps.append(Jitter(sigma=jitter_sigma, rng=rng))
        steps.append(Subsample(num_points=num_points, rng=rng))
    else:
        steps.append(Subsample(num_points=num_points, deterministic=True, salt=int(seed or 0)))
    return Compose(steps)
