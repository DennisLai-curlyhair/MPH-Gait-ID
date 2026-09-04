"""Standalone inference adapter for the LidarGait++ architecture.

The architecture and parameter layout follow the official OpenGait source at
commit f754f6f3831e9f83bb28f4e2f63dd43d8bcf9dc4. OpenGait and LidarGait++ are
not authored or owned by this project; their upstream terms remain applicable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lidargaitpp_official_utils as official_utils


def deterministic_farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
) -> torch.Tensor:
    """Evaluation-time FPS used by the released Fixed-Special5 evaluator."""

    device = xyz.device
    batch_size, point_count, channels = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, point_count), 1e10, device=device)
    coordinates = xyz[:, :, :3]
    farthest = coordinates.square().sum(dim=-1).max(dim=-1)[1]
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for index in range(npoint):
        centroids[:, index] = farthest
        centroid = coordinates[batch_indices, farthest].view(batch_size, 1, channels)
        squared_distance = (coordinates - centroid).square().sum(dim=-1)
        mask = squared_distance < distance
        distance[mask] = squared_distance[mask]
        farthest = distance.max(dim=-1)[1]
    return centroids


class SeparateFCs(nn.Module):
    """Byte-compatible parameter layout with pinned OpenGait SeparateFCs."""

    def __init__(
        self,
        parts_num: int,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
    ) -> None:
        super().__init__()
        self.p = int(parts_num)
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)
            )
        )
        self.norm = bool(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.permute(2, 0, 1).contiguous()
        weights = F.normalize(self.fc_bin, dim=1) if self.norm else self.fc_bin
        return values.matmul(weights).permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """Pinned OpenGait BNNeck layout retained for strict checkpoint loading."""

    def __init__(
        self,
        parts_num: int,
        in_channels: int,
        class_num: int,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.p = int(parts_num)
        self.class_num = int(class_num)
        self.norm = bool(norm)
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)
            )
        )
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, parts = x.shape
        values = self.bn1d(x.view(batch, -1)).view(batch, channels, parts)
        feature = values.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)
            weights = F.normalize(self.fc_bin, dim=1)
        else:
            weights = self.fc_bin
        logits = feature.matmul(weights)
        return (
            feature.permute(1, 2, 0).contiguous(),
            logits.permute(1, 2, 0).contiguous(),
        )


class LidarGaitPlusPlusInference(nn.Module):
    """Official LidarGait++ network without OpenGait's distributed trainer shell.

    Layer names and tensor flow match the pinned OpenGait LidarGaitPlusPlus
    implementation. The public embedding is the pre-BN part embedding flattened
    exactly as in the Fixed-Special5 unified evaluator.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        channel = int(cfg.get("channel", 16))
        fc_cfg = dict(cfg.get("SeparateFCs", {}))
        bn_cfg = dict(cfg.get("SeparateBNNecks", {}))
        out_channels = int(fc_cfg.get("in_channels", 256))
        scale_aware = bool(cfg.get("scale_aware", True))
        normalize_dp = bool(cfg.get("normalize_dp", True))
        sampling = str(cfg.get("sampling", "knn"))
        npoints = list(cfg.get("npoints", [512, 256, 128]))
        nsample = int(cfg.get("nsample", 32))
        in_channel = 4 if scale_aware else 3

        official_utils.farthest_point_sample = deterministic_farthest_point_sample
        abstraction = official_utils.PointNetSetAbstraction
        self.sa1 = abstraction(
            npoint=npoints[0], radius=0.1, nsample=nsample,
            in_channel=in_channel, mlp=[2 * channel, 2 * channel, 4 * channel],
            group_all=False, sampling=sampling, scale_aware=scale_aware,
            normalize_dp=normalize_dp,
        )
        self.sa2 = abstraction(
            npoint=npoints[1], radius=0.2, nsample=nsample,
            in_channel=4 * channel + in_channel,
            mlp=[4 * channel, 4 * channel, 8 * channel],
            group_all=False, sampling=sampling, scale_aware=scale_aware,
            normalize_dp=normalize_dp,
        )
        self.sa3 = abstraction(
            npoint=npoints[2], radius=0.4, nsample=nsample,
            in_channel=8 * channel + in_channel,
            mlp=[8 * channel, 8 * channel, 16 * channel],
            group_all=False, sampling=sampling, scale_aware=scale_aware,
            normalize_dp=normalize_dp,
        )
        self.sa4 = abstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=16 * channel + in_channel,
            mlp=[16 * channel, 16 * channel, out_channels],
            group_all=True, sampling=sampling, scale_aware=scale_aware,
            normalize_dp=normalize_dp,
        )
        pool_name = str(cfg.get("pool", "PPP_HAP"))
        if pool_name != "PPP_HAP":
            raise ValueError("Standalone release currently supports official PPP_HAP only")
        self.pool = official_utils.PPPooling(
            scale_aware=True,
            bin_num=list(cfg.get("scale", [1, 2, 4, 8, 16])),
        )
        self.BNNecks = SeparateBNNecks(
            parts_num=int(bn_cfg.get("parts_num", 31)),
            in_channels=int(bn_cfg.get("in_channels", 256)),
            class_num=int(bn_cfg.get("class_num", 14)),
        )
        self.FCs = SeparateFCs(
            parts_num=int(fc_cfg.get("parts_num", 31)),
            in_channels=int(fc_cfg.get("in_channels", 256)),
            out_channels=int(fc_cfg.get("out_channels", 256)),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        if points.ndim != 4 or points.shape[-1] != 4:
            raise ValueError(
                f"Expected scale-aware points [B,T,N,4], got {tuple(points.shape)}"
            )
        batch, frames, point_count, channels = points.shape
        xyz = points.reshape(batch * frames, point_count, channels)
        xyz = xyz.permute(0, 2, 1).contiguous()

        l1_xyz, l1_points = self.sa1(xyz, None)
        l1_points = torch.max(l1_points, dim=-2)[0]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = torch.max(l2_points, dim=-2)[0]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, dim=-2)[0]
        _, l4_points = self.sa4(l3_xyz, l3_points)

        pooled = self.pool(l4_points, l3_xyz)
        pooled = pooled.reshape(batch, frames, pooled.shape[1], pooled.shape[2])
        feature = pooled.max(dim=1)[0]
        part_embedding = self.FCs(feature)
        bn_embedding, logits = self.BNNecks(part_embedding)
        return {
            "embedding": part_embedding.flatten(1),
            "part_embedding": part_embedding,
            "bn_embedding": bn_embedding,
            "logits": logits,
        }


def build_model(cfg: dict[str, Any]) -> nn.Module:
    return LidarGaitPlusPlusInference(cfg)
