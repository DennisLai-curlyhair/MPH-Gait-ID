from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FramePointNet(nn.Module):
    """Shared PointNet frame encoder used by both comparison methods."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(in_channels)
        for width_value in channels:
            width = int(width_value)
            layers.extend(
                [
                    nn.Conv1d(previous, width, kernel_size=1, bias=False),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            previous = width
        self.net = nn.Sequential(*layers)
        self.out_channels = previous

    def encode_points(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points.transpose(1, 2).contiguous())

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.encode_points(points).max(dim=2)[0]


class TemporalPointNetV1(nn.Module):
    """The tuned TemporalPointNet control copied into the new protocol."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        point_channels: list[int] | tuple[int, ...] = (64, 128, 256, 512),
        embedding_dim: int = 256,
        temporal_pool: str = "max",
        dropout: float = 0.0,
        point_normalization: str = "center",
    ) -> None:
        super().__init__()
        if temporal_pool not in {"max", "mean"}:
            raise ValueError("temporal_pool must be 'max' or 'mean'")
        if point_normalization not in {"none", "center", "center_scale"}:
            raise ValueError("point_normalization must be none, center, or center_scale")

        self.temporal_pool = temporal_pool
        self.point_normalization = point_normalization
        self.frame_encoder = FramePointNet(
            in_channels=in_channels,
            channels=point_channels,
        )
        frame_dim = self.frame_encoder.out_channels
        self.embedding = nn.Sequential(
            nn.Linear(frame_dim, int(embedding_dim), bias=False),
            nn.BatchNorm1d(int(embedding_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )
        self.bnneck = nn.BatchNorm1d(int(embedding_dim))
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes), bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.point_normalization == "none":
            return points

        center = points.mean(dim=2, keepdim=True)
        points = points - center
        if self.point_normalization == "center_scale":
            scale = (
                points.pow(2)
                .sum(dim=-1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .sqrt()
            )
            points = points / scale.clamp_min(1e-6)
        return points

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        if points.dim() != 4:
            raise ValueError(
                f"Expected points with shape [B, T, N, C], got {tuple(points.shape)}"
            )

        points = self._normalize_points(points.float())
        batch_size, clip_len, num_points, channels = points.shape
        frame_points = points.reshape(batch_size * clip_len, num_points, channels)
        frame_features = self.frame_encoder(frame_points).reshape(
            batch_size,
            clip_len,
            -1,
        )

        if self.temporal_pool == "mean":
            sequence_feature = frame_features.mean(dim=1)
        else:
            sequence_feature = frame_features.max(dim=1)[0]

        raw_embedding = self.embedding(sequence_feature)
        embedding = self.bnneck(raw_embedding)
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "raw_embedding": raw_embedding,
            "normalized_embedding": F.normalize(embedding, dim=1),
        }




def build_model(cfg: dict[str, Any]) -> nn.Module:
    name = str(cfg.get("name", "pc_v1")).lower()
    if name not in {"pc_v1", "temporal_pointnet_v1"}:
        raise ValueError(f"Unknown PointNet-TMax model: {cfg.get('name')}")
    return TemporalPointNetV1(
        num_classes=int(cfg.get("num_classes", 29)),
        in_channels=int(cfg.get("in_channels", 3)),
        point_channels=cfg.get("point_channels", [64, 128, 256, 512]),
        embedding_dim=int(cfg.get("embedding_dim", 256)),
        temporal_pool=str(cfg.get("temporal_pool", "max")),
        dropout=float(cfg.get("dropout", 0.0)),
        point_normalization=str(cfg.get("point_normalization", "center")),
    )
