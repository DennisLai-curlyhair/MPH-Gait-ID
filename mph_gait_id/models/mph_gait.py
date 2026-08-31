from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mph_tokenizer import (
    FramePointStem,
    LocalWindowTokenizer,
    WindowPartition,
    _as_three_bools,
    _as_three_ints,
)


class MultiScalePointHierarchy(nn.Module):
    """Encode regular fine/coarse point windows within each frame."""

    def __init__(
        self,
        point_feature_dim: int,
        token_dim: int = 64,
        fine_grid: Sequence[int] = (2, 2, 4),
        coarse_grid: Sequence[int] = (1, 2, 2),
        shift_axes: Sequence[bool] = (False, True, True),
        use_coarse_windows: bool = True,
        use_shifted_windows: bool = True,
        use_token_transformer: bool = False,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
        transformer_mlp_ratio: float = 2.0,
        attention_dropout: float = 0.0,
        min_points_per_window: int = 2,
        token_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        fine = _as_three_ints(fine_grid, "fine_grid")
        coarse = _as_three_ints(coarse_grid, "coarse_grid")
        shifted_axes = _as_three_bools(shift_axes, "shift_axes")
        if token_dim % transformer_heads != 0:
            raise ValueError("token_dim must be divisible by transformer_heads")
        if use_token_transformer and transformer_depth <= 0:
            raise ValueError("transformer_depth must be positive")
        if not 0.0 <= token_dropout < 1.0:
            raise ValueError("token_dropout must be in [0, 1)")

        partitions = [WindowPartition(fine, False, shifted_axes, 0.0)]
        partition_names = ["fine_regular"]
        if use_shifted_windows:
            partitions.append(WindowPartition(fine, True, shifted_axes, 0.0))
            partition_names.append("fine_shifted")
        if use_coarse_windows:
            partitions.append(WindowPartition(coarse, False, shifted_axes, 1.0))
            partition_names.append("coarse_regular")
            if use_shifted_windows:
                partitions.append(WindowPartition(coarse, True, shifted_axes, 1.0))
                partition_names.append("coarse_shifted")

        self.use_coarse_windows = bool(use_coarse_windows)
        self.use_shifted_windows = bool(use_shifted_windows)
        self.use_token_transformer = bool(use_token_transformer)
        self.partition_names = tuple(partition_names)
        self.tokenizer = LocalWindowTokenizer(
            point_feature_dim=int(point_feature_dim),
            token_dim=int(token_dim),
            partitions=partitions,
            min_points_per_window=int(min_points_per_window),
        )
        if self.use_token_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(token_dim),
                nhead=int(transformer_heads),
                dim_feedforward=max(
                    int(token_dim),
                    int(round(token_dim * float(transformer_mlp_ratio))),
                ),
                dropout=float(attention_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_mixer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(transformer_depth),
                enable_nested_tensor=False,
            )
        else:
            self.token_mixer = nn.Identity()
        self.token_dropout = float(token_dropout)
        self.frame_projection = nn.Sequential(
            nn.Linear(int(token_dim) * 2, int(token_dim), bias=False),
            nn.LayerNorm(int(token_dim)),
            nn.ReLU(inplace=True),
        )
        self.output_dim = int(token_dim)
        self.num_tokens = sum(item.num_windows for item in partitions)

    def _apply_token_dropout(self, valid: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_dropout <= 0.0:
            return valid
        keep = torch.rand(valid.shape, device=valid.device) >= self.token_dropout
        dropped = valid & keep
        has_token = dropped.any(dim=1)
        if bool((~has_token).any().item()):
            dropped[~has_token] = valid[~has_token]
        return dropped

    def forward(
        self,
        coordinates: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, valid = self.tokenizer(coordinates, point_features)
        valid = self._apply_token_dropout(valid)
        if self.use_token_transformer:
            mixed = self.token_mixer(tokens, src_key_padding_mask=~valid)
        else:
            mixed = self.token_mixer(tokens)
        valid_float = valid.unsqueeze(-1).to(mixed.dtype)
        mean_token = (mixed * valid_float).sum(dim=1)
        mean_token = mean_token / valid_float.sum(dim=1).clamp_min(1.0)
        max_token = mixed.masked_fill(~valid.unsqueeze(-1), -1e4).max(dim=1)[0]
        frame_feature = self.frame_projection(
            torch.cat((mean_token, max_token), dim=1)
        )
        valid_ratio = valid_float.mean(dim=1).squeeze(-1)
        return frame_feature, valid_ratio


class MPHGait(nn.Module):
    """PointNet-TMax with a bounded multi-scale point-hierarchy residual."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        point_channels: Sequence[int] = (64, 128, 256, 512),
        embedding_dim: int = 256,
        temporal_pool: str = "max",
        dropout: float = 0.0,
        point_normalization: str = "center",
        token_dim: int = 64,
        fine_grid: Sequence[int] = (2, 2, 4),
        coarse_grid: Sequence[int] = (1, 2, 2),
        shift_axes: Sequence[bool] = (False, True, True),
        use_coarse_windows: bool = True,
        use_shifted_windows: bool = True,
        use_token_transformer: bool = False,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
        transformer_mlp_ratio: float = 2.0,
        attention_dropout: float = 0.0,
        min_points_per_window: int = 2,
        token_dropout: float = 0.0,
        residual_max_weight: float = 0.20,
        residual_weight_init: float = 0.05,
    ) -> None:
        super().__init__()
        if temporal_pool not in {"max", "mean"}:
            raise ValueError("temporal_pool must be max or mean")
        if point_normalization not in {"none", "center", "center_scale"}:
            raise ValueError("point_normalization must be none, center, or center_scale")
        if residual_max_weight <= 0.0:
            raise ValueError("residual_max_weight must be positive")
        if abs(residual_weight_init) >= residual_max_weight:
            raise ValueError("abs(residual_weight_init) must be below the cap")

        self.temporal_pool = str(temporal_pool)
        self.point_normalization = str(point_normalization)
        self.residual_max_weight = float(residual_max_weight)
        self.frame_encoder = FramePointStem(
            in_channels=int(in_channels), channels=point_channels
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
        self._initialize(self.frame_encoder)
        self._initialize(self.embedding)
        self._initialize(self.bnneck)
        self._initialize(self.classifier)

        self.hierarchical_encoder = MultiScalePointHierarchy(
            point_feature_dim=frame_dim,
            token_dim=int(token_dim),
            fine_grid=fine_grid,
            coarse_grid=coarse_grid,
            shift_axes=shift_axes,
            use_coarse_windows=bool(use_coarse_windows),
            use_shifted_windows=bool(use_shifted_windows),
            use_token_transformer=bool(use_token_transformer),
            transformer_depth=int(transformer_depth),
            transformer_heads=int(transformer_heads),
            transformer_mlp_ratio=float(transformer_mlp_ratio),
            attention_dropout=float(attention_dropout),
            min_points_per_window=int(min_points_per_window),
            token_dropout=float(token_dropout),
        )
        self.local_embedding = nn.Sequential(
            nn.Linear(
                self.hierarchical_encoder.output_dim,
                int(embedding_dim),
                bias=False,
            ),
            nn.LayerNorm(int(embedding_dim)),
        )
        initial_ratio = float(residual_weight_init) / self.residual_max_weight
        self.residual_logit = nn.Parameter(
            torch.tensor(math.atanh(initial_ratio), dtype=torch.float32)
        )
        self._initialize(self.hierarchical_encoder)
        self._initialize(self.local_embedding)

    @staticmethod
    def _initialize(root: nn.Module) -> None:
        for module in root.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def current_residual_weight(self) -> torch.Tensor:
        return self.residual_max_weight * torch.tanh(self.residual_logit)

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        if self.point_normalization == "none":
            return points
        centered = points - points.mean(dim=2, keepdim=True)
        if self.point_normalization == "center_scale":
            scale = (
                centered.pow(2)
                .sum(dim=-1, keepdim=True)
                .mean(dim=2, keepdim=True)
                .sqrt()
            )
            centered = centered / scale.clamp_min(1e-6)
        return centered

    def _temporal_pool(self, features: torch.Tensor) -> torch.Tensor:
        if self.temporal_pool == "mean":
            return features.mean(dim=1)
        return features.max(dim=1)[0]

    def _aggregate_local_frames(
        self,
        local_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Aggregate local frame features; temporal variants override this hook."""
        return self._temporal_pool(local_frames), {}

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        if points.dim() != 4:
            raise ValueError(f"Expected [B, T, N, C], got {tuple(points.shape)}")
        points = self._normalize_points(points.float())
        batch_size, clip_len, num_points, channels = points.shape
        frame_points = points.reshape(batch_size * clip_len, num_points, channels)
        point_features = self.frame_encoder(frame_points)
        global_frames = point_features.max(dim=2)[0]
        local_frames, valid_ratio = self.hierarchical_encoder(
            frame_points[..., :3], point_features
        )
        global_sequence = self._temporal_pool(
            global_frames.reshape(batch_size, clip_len, -1)
        )
        local_sequence, temporal_diagnostics = self._aggregate_local_frames(
            local_frames.reshape(batch_size, clip_len, -1)
        )
        base_raw_embedding = self.embedding(global_sequence)
        local_embedding = self.local_embedding(local_sequence)
        local_direction = F.normalize(local_embedding, dim=1)
        base_norm = base_raw_embedding.detach().norm(dim=1, keepdim=True)
        residual_delta = local_direction * base_norm.clamp_min(1e-6)
        residual_weight = self.current_residual_weight()
        raw_embedding = base_raw_embedding + residual_weight * residual_delta
        embedding = self.bnneck(raw_embedding)
        logits = self.classifier(embedding)
        outputs = {
            "logits": logits,
            "embedding": embedding,
            "raw_embedding": raw_embedding,
            "normalized_embedding": F.normalize(embedding, dim=1),
            "base_raw_embedding": base_raw_embedding,
            "local_embedding": local_embedding,
            "residual_delta": residual_delta,
            "residual_weight": residual_weight.expand(batch_size),
            "hierarchical_valid_token_ratio": valid_ratio.reshape(
                batch_size, clip_len
            ).mean(dim=1),
        }
        outputs.update(temporal_diagnostics)
        return outputs




def build_model(cfg: dict[str, Any]) -> nn.Module:
    name = str(cfg.get("name", "mph_gait")).lower()
    if name not in {
        "mph_gait",
        "pc_v1_hierarchical_multiscale_regular_transformer_v1",
    }:
        raise ValueError(f"Unknown MPH-Gait ID model: {cfg.get('name')}")
    return MPHGait(
        num_classes=int(cfg.get("num_classes", 29)),
        in_channels=int(cfg.get("in_channels", 3)),
        point_channels=cfg.get("point_channels", [64, 128, 256, 512]),
        embedding_dim=int(cfg.get("embedding_dim", 256)),
        temporal_pool=str(cfg.get("temporal_pool", "max")),
        dropout=float(cfg.get("dropout", 0.0)),
        point_normalization=str(cfg.get("point_normalization", "center")),
        token_dim=int(cfg.get("token_dim", 64)),
        fine_grid=cfg.get("fine_grid", [2, 2, 4]),
        coarse_grid=cfg.get("coarse_grid", [1, 2, 2]),
        shift_axes=cfg.get("shift_axes", [False, True, True]),
        use_coarse_windows=True,
        use_shifted_windows=False,
        use_token_transformer=True,
        transformer_depth=int(cfg.get("transformer_depth", 1)),
        transformer_heads=int(cfg.get("transformer_heads", 4)),
        transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 2.0)),
        attention_dropout=float(cfg.get("attention_dropout", 0.0)),
        min_points_per_window=int(cfg.get("min_points_per_window", 2)),
        token_dropout=float(cfg.get("token_dropout", 0.0)),
        residual_max_weight=float(cfg.get("residual_max_weight", 0.20)),
        residual_weight_init=float(cfg.get("residual_weight_init", 0.05)),
    )
