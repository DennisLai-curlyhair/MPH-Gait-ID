from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_three_ints(values: Sequence[int], name: str) -> tuple[int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain three positive integers")
    return result


def _as_three_bools(values: Sequence[bool], name: str) -> tuple[bool, bool, bool]:
    result = tuple(bool(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain three booleans")
    return result


class FramePointStem(nn.Module):
    """Shared per-point feature stem used by the global and local branches."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 512),
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

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points.transpose(1, 2).contiguous())


@dataclass(frozen=True)
class WindowPartition:
    grid: tuple[int, int, int]
    shifted: bool
    shift_axes: tuple[bool, bool, bool]
    level: float

    @property
    def offsets(self) -> tuple[float, float, float]:
        if not self.shifted:
            return (0.0, 0.0, 0.0)
        return tuple(
            0.5 if enabled and size > 1 else 0.0
            for enabled, size in zip(self.shift_axes, self.grid)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(
            size + (1 if offset > 0.0 else 0)
            for size, offset in zip(self.grid, self.offsets)
        )

    @property
    def num_windows(self) -> int:
        return math.prod(self.shape)


def _window_centers(partition: WindowPartition) -> torch.Tensor:
    axes = []
    for grid_size, partition_size, offset in zip(
        partition.grid,
        partition.shape,
        partition.offsets,
    ):
        indices = torch.arange(partition_size, dtype=torch.float32)
        centers = (indices + 0.5 - offset) / float(grid_size)
        axes.append(centers.clamp(0.0, 1.0))
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, 3)


def _window_metadata(partition: WindowPartition) -> torch.Tensor:
    centers = _window_centers(partition)
    cell_sizes = torch.tensor(
        [1.0 / float(value) for value in partition.grid],
        dtype=torch.float32,
    ).expand(centers.shape[0], -1)
    shifted = torch.full(
        (centers.shape[0], 1),
        float(partition.shifted),
        dtype=torch.float32,
    )
    level = torch.full(
        (centers.shape[0], 1),
        float(partition.level),
        dtype=torch.float32,
    )
    return torch.cat((centers, cell_sizes, shifted, level), dim=1)


class LocalWindowTokenizer(nn.Module):
    """Pool irregular points into overlapping axis-aligned local tokens."""

    def __init__(
        self,
        point_feature_dim: int,
        token_dim: int,
        partitions: Sequence[WindowPartition],
        min_points_per_window: int = 2,
    ) -> None:
        super().__init__()
        if min_points_per_window <= 0:
            raise ValueError("min_points_per_window must be positive")
        self.partitions = tuple(partitions)
        self.min_points_per_window = int(min_points_per_window)
        self.point_projection = nn.Sequential(
            nn.Conv1d(
                int(point_feature_dim),
                int(token_dim),
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(int(token_dim)),
            nn.ReLU(inplace=True),
        )
        self.content_score = nn.Conv1d(
            int(token_dim),
            1,
            kernel_size=1,
            bias=False,
        )
        self.position_strength = nn.Parameter(torch.tensor(0.0))
        self.metadata_projection = nn.Sequential(
            nn.Linear(8, int(token_dim), bias=False),
            nn.LayerNorm(int(token_dim)),
        )

        for index, partition in enumerate(self.partitions):
            self.register_buffer(
                f"window_centers_{index}",
                _window_centers(partition),
                persistent=False,
            )
            self.register_buffer(
                f"window_metadata_{index}",
                _window_metadata(partition),
                persistent=False,
            )

    @staticmethod
    def _routing_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        lower = coordinates.amin(dim=1, keepdim=True)
        upper = coordinates.amax(dim=1, keepdim=True)
        span = (upper - lower).clamp_min(1e-5)
        return ((coordinates - lower) / span).clamp(0.0, 1.0)

    @staticmethod
    def _flat_window_indices(
        routing: torch.Tensor,
        partition: WindowPartition,
    ) -> torch.Tensor:
        grid = routing.new_tensor(partition.grid)
        offsets = routing.new_tensor(partition.offsets)
        shape = partition.shape
        indices = torch.floor(routing * grid + offsets).long()
        maxima = indices.new_tensor(shape).view(1, 1, 3) - 1
        indices = torch.minimum(indices, maxima).clamp_min(0)
        return (
            indices[..., 0] * (shape[1] * shape[2])
            + indices[..., 1] * shape[2]
            + indices[..., 2]
        )

    def _tokenize_partition(
        self,
        routing: torch.Tensor,
        point_tokens: torch.Tensor,
        content_scores: torch.Tensor,
        partition: WindowPartition,
        partition_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_indices = self._flat_window_indices(routing, partition)
        membership = F.one_hot(
            flat_indices,
            num_classes=partition.num_windows,
        ).permute(0, 2, 1).bool()
        counts = membership.sum(dim=2)
        valid = counts >= self.min_points_per_window

        centers = getattr(self, f"window_centers_{partition_index}")
        centers = centers.to(device=routing.device, dtype=routing.dtype)
        cell_sizes = routing.new_tensor(
            [1.0 / float(value) for value in partition.grid]
        )
        relative = (
            routing[:, None, :, :] - centers[None, :, None, :]
        ) / cell_sizes[None, None, None, :]
        distance_penalty = relative.square().sum(dim=-1)
        strength = F.softplus(self.position_strength)
        scores = content_scores[:, None, :] - strength * distance_penalty
        scores = scores.masked_fill(~membership, -1e4)
        weights = torch.softmax(scores, dim=2)
        weights = weights * membership.to(weights.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

        tokens = torch.bmm(weights, point_tokens)
        metadata = getattr(self, f"window_metadata_{partition_index}")
        metadata = metadata.to(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + self.metadata_projection(metadata).unsqueeze(0)
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)
        return tokens, valid

    def forward(
        self,
        coordinates: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing = self._routing_coordinates(coordinates)
        projected = self.point_projection(point_features)
        point_tokens = projected.transpose(1, 2).contiguous()
        content_scores = self.content_score(projected).squeeze(1)

        token_rows = []
        valid_rows = []
        for index, partition in enumerate(self.partitions):
            tokens, valid = self._tokenize_partition(
                routing,
                point_tokens,
                content_scores,
                partition,
                index,
            )
            token_rows.append(tokens)
            valid_rows.append(valid)
        return torch.cat(token_rows, dim=1), torch.cat(valid_rows, dim=1)

