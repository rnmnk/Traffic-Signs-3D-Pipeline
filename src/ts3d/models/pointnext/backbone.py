from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sampling and grouping primitives
# ---------------------------------------------------------------------------


def farthest_point_sample(xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Iterative FPS. ``xyz``: (B, N, 3) -> indices (B, S)."""
    b, n, _ = xyz.shape
    device = xyz.device
    if num_samples >= n:
        return torch.arange(n, device=device).unsqueeze(0).expand(b, n).clone()

    idx = torch.zeros(b, num_samples, dtype=torch.long, device=device)
    distance = torch.full((b, n), 1e10, device=device)
    farthest = torch.randint(0, n, (b,), device=device, dtype=torch.long)
    batch_idx = torch.arange(b, device=device)
    for i in range(num_samples):
        idx[:, i] = farthest
        centroid = xyz[batch_idx, farthest].unsqueeze(1)
        d = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, d)
        farthest = torch.argmax(distance, dim=-1)
    return idx


def gather_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the point axis. ``x``: (B, N, C), ``idx``: (B, S) -> (B, S, C)."""
    _, _, c = x.shape
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, c))


def knn(xyz: torch.Tensor, queries: torch.Tensor, k: int) -> torch.Tensor:
    """k-nearest-neighbour indices: (B, S, k)."""
    k = min(k, xyz.shape[1])
    diff = queries.unsqueeze(2) - xyz.unsqueeze(1)
    dist2 = (diff * diff).sum(dim=-1)
    _, idx = dist2.topk(k, dim=-1, largest=False)
    return idx


def group_features(
    xyz: torch.Tensor,
    features: torch.Tensor | None,
    centers: torch.Tensor,
    knn_idx: torch.Tensor,
) -> torch.Tensor:
    """Build per-group input: relative xyz (+ features) -> (B, S, k, C')."""
    b, s, k = knn_idx.shape
    idx_flat = knn_idx.reshape(b, s * k)
    grouped_xyz = gather_points(xyz, idx_flat).reshape(b, s, k, 3)
    rel = grouped_xyz - centers.unsqueeze(2)
    if features is None:
        return rel
    grouped_f = gather_points(features, idx_flat).reshape(b, s, k, -1)
    return torch.cat([rel, grouped_f], dim=-1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SharedMLP(nn.Sequential):
    """Shared 1x1 Conv2d MLP over grouped point features (B, C, S, k)."""

    def __init__(self, channels: list[int], *, last_act: bool = True) -> None:
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            is_last = i == len(channels) - 2
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            if not is_last or last_act:
                layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SetAbstraction(nn.Module):
    """FPS-sample + kNN-group + 1-layer shared MLP + max-pool.

    Matches the official PointNeXt-B ``SetAbstraction`` (single-layer MLP,
    no residual connection).
    """

    def __init__(self, *, stride: int, k: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stride = stride
        self.k = k
        self.mlp = SharedMLP([in_channels + 3, out_channels], last_act=True)
        self.out_channels = out_channels

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = max(xyz.shape[1] // self.stride, 1)
        idx_centers = farthest_point_sample(xyz, num_samples)
        centers = gather_points(xyz, idx_centers)

        knn_idx = knn(xyz, centers, self.k)
        grouped = group_features(xyz, features, centers, knn_idx)
        grouped = grouped.permute(0, 3, 1, 2).contiguous()  # (B, C+3, S, k)

        out = self.mlp(grouped).max(dim=-1).values  # (B, C_out, S)
        return centers, out.permute(0, 2, 1).contiguous()


class InvResMLPBlock(nn.Module):
    """Inverted-residual MLP block (PointNeXt).

    Pipeline:
      1. kNN-group + 1-layer Conv2d + max-pool  (neighborhood aggregation)
      2. Pointwise Conv1d expansion (x4) + projection
      3. Residual addition + ReLU
    """

    def __init__(self, channels: int, k: int, expansion: int = 4) -> None:
        super().__init__()
        self.k = k
        self.local_mlp = SharedMLP([channels + 3, channels], last_act=True)
        hidden = channels * expansion
        self.pw = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        identity = features

        knn_idx = knn(xyz, xyz, self.k)
        grouped = group_features(xyz, features, xyz, knn_idx)  # (B, N, k, C+3)
        grouped = grouped.permute(0, 3, 1, 2).contiguous()  # (B, C+3, N, k)
        f = self.local_mlp(grouped).max(dim=-1).values  # (B, C, N)

        f = self.pw(f)  # (B, C, N)
        f = f.permute(0, 2, 1).contiguous()  # (B, N, C)
        return self.act(f + identity)


# ---------------------------------------------------------------------------
# PointNeXt-B encoder
# ---------------------------------------------------------------------------


# Per stage: 1 SetAbstraction + (BLOCKS[i] - 1) InvResMLP blocks.
# B variant: B=(1, 2, 1, 1) InvResMLP per stage, plus the SA layer.
_BLOCKS = (2, 3, 2, 2)
_WIDTH = 32
_K = 32
_STRIDE = 4


class PointNeXtBackbone(nn.Module):
    """PointNeXt-B encoder: stem -> 4 x (SA + InvResMLP*) -> global max-pool."""

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Conv1d(3, _WIDTH, 1, bias=True)

        self.sa_blocks = nn.ModuleList()
        self.inv_res_blocks = nn.ModuleList()

        prev_c = _WIDTH
        stage_channels: list[int] = []
        c = _WIDTH
        for _ in range(4):
            c *= 2
            stage_channels.append(c)

        for n_blocks, out_c in zip(_BLOCKS, stage_channels, strict=True):
            self.sa_blocks.append(
                SetAbstraction(stride=_STRIDE, k=_K, in_channels=prev_c, out_channels=out_c)
            )
            self.inv_res_blocks.append(
                nn.ModuleList(InvResMLPBlock(out_c, k=_K) for _ in range(n_blocks - 1))
            )
            prev_c = out_c

        self.out_channels = stage_channels[-1]

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """``xyz``: (B, N, 3) -> global embedding (B, C_out)."""
        features = self.stem(xyz.permute(0, 2, 1).contiguous())  # (B, C, N)
        features = features.permute(0, 2, 1).contiguous()  # (B, N, C)

        for sa, inv_blocks in zip(self.sa_blocks, self.inv_res_blocks, strict=True):
            xyz, features = sa(xyz, features)
            for blk in inv_blocks:
                features = blk(xyz, features)

        return features.max(dim=1).values
