from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .types import Detection, SensorFrame


@dataclass(frozen=True)
class PreprocessResult:
    sampled_points_mm: np.ndarray
    foreground_points_mm: np.ndarray
    color_mask: np.ndarray
    cloud_mask: np.ndarray
    diagnostics: dict[str, Any]


def _bbox_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)] = True
    return mask


def _deterministic_sample(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if len(points) >= num_points:
        indices = np.linspace(0, len(points) - 1, num_points).round().astype(np.int64)
    else:
        repeats = int(math.ceil(num_points / len(points)))
        indices = np.tile(np.arange(len(points), dtype=np.int64), repeats)[:num_points]
    return np.asarray(points[indices, :3], dtype=np.float32)


def _cloud_mask_to_rgb(
    frame: SensorFrame,
    cloud: np.ndarray,
    cloud_mask: np.ndarray,
    color_shape: tuple[int, int],
) -> np.ndarray:
    """Map the final depth-filtered cloud mask back to RGB for display only."""

    color_height, color_width = color_shape
    cloud_height, cloud_width = cloud_mask.shape
    if frame.point_cloud_color_aligned and (cloud_height, cloud_width) == color_shape:
        return np.asarray(cloud_mask, dtype=bool).copy()

    projection = frame.metadata.get("pinhole_projection")
    if isinstance(projection, dict):
        reference_width, reference_height = projection.get(
            "reference_size", [color_width, color_height]
        )
        scale_x = color_width / max(float(reference_width), 1.0)
        scale_y = color_height / max(float(reference_height), 1.0)
        fx = float(projection["fx"]) * scale_x
        fy = float(projection["fy"]) * scale_y
        cx = float(projection["cx"]) * scale_x
        cy = float(projection["cy"]) * scale_y
        x_offset = float(projection.get("x_offset", 0.0)) * scale_x
        y_offset = float(projection.get("y_offset", 0.0)) * scale_y
        rows, columns = np.where(cloud_mask)
        selected = cloud[rows, columns]
        z_values = selected[:, 2]
        projectable = np.isfinite(selected).all(axis=1) & (z_values > 1e-6)
        selected = selected[projectable]
        projected_x = np.rint(
            fx * selected[:, 0] / selected[:, 2] + cx + x_offset
        ).astype(np.int32)
        projected_y = np.rint(
            fy * selected[:, 1] / selected[:, 2] + cy + y_offset
        ).astype(np.int32)
        inside = (
            (projected_x >= 0)
            & (projected_x < color_width)
            & (projected_y >= 0)
            & (projected_y < color_height)
        )
        result = np.zeros(color_shape, dtype=bool)
        result[projected_y[inside], projected_x[inside]] = True
        return result

    return cv2.resize(
        cloud_mask.astype(np.uint8),
        (color_width, color_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def filter_person_pointcloud(
    frame: SensorFrame,
    detection: Detection,
    num_points: int,
    near_mm: float = 400.0,
    far_mm: float = 6000.0,
    depth_margin_mm: float = 550.0,
    min_person_points: int = 96,
) -> PreprocessResult:
    color_height, color_width = frame.color_bgr.shape[:2]
    color_mask = _bbox_mask(
        (color_height, color_width),
        detection.bbox_xyxy,
    )
    if detection.mask is not None:
        instance = detection.mask
        if instance.shape != color_mask.shape:
            instance = cv2.resize(
                instance.astype(np.uint8),
                (color_width, color_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        color_mask &= instance

    cloud = np.asarray(frame.point_cloud_mm[..., :3], dtype=np.float32)
    cloud_height, cloud_width = cloud.shape[:2]
    if frame.point_cloud_color_aligned and (cloud_height, cloud_width) == (
        color_height,
        color_width,
    ):
        cloud_mask = color_mask
        alignment = "sdk_color_aligned"
    elif isinstance(frame.metadata.get("pinhole_projection"), dict):
        projection = frame.metadata["pinhole_projection"]
        reference_width, reference_height = projection.get(
            "reference_size", [color_width, color_height]
        )
        scale_x = color_width / max(float(reference_width), 1.0)
        scale_y = color_height / max(float(reference_height), 1.0)
        fx = float(projection["fx"]) * scale_x
        fy = float(projection["fy"]) * scale_y
        cx = float(projection["cx"]) * scale_x
        cy = float(projection["cy"]) * scale_y
        x_offset = float(projection.get("x_offset", 0.0)) * scale_x
        y_offset = float(projection.get("y_offset", 0.0)) * scale_y
        x_values = cloud[..., 0]
        y_values = cloud[..., 1]
        z_values = cloud[..., 2]
        projectable = np.isfinite(cloud).all(axis=2) & (z_values > 1e-6)
        rows, columns = np.where(projectable)
        projected_x = np.rint(
            fx * x_values[projectable] / z_values[projectable] + cx + x_offset
        ).astype(np.int32)
        projected_y = np.rint(
            fy * y_values[projectable] / z_values[projectable] + cy + y_offset
        ).astype(np.int32)
        inside = (
            (projected_x >= 0)
            & (projected_x < color_width)
            & (projected_y >= 0)
            & (projected_y < color_height)
        )
        cloud_mask = np.zeros((cloud_height, cloud_width), dtype=bool)
        selected_rows = rows[inside]
        selected_columns = columns[inside]
        cloud_mask[selected_rows, selected_columns] = color_mask[
            projected_y[inside], projected_x[inside]
        ]
        alignment = "replay_pinhole_reference_v1"
    else:
        cloud_mask = cv2.resize(
            color_mask.astype(np.uint8),
            (cloud_width, cloud_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        alignment = "legacy_grid_scaled"

    finite = np.isfinite(cloud).all(axis=2)
    depth = cloud[..., 2]
    valid = cloud_mask & finite & (depth >= near_mm) & (depth <= far_mm)
    before_depth_gate = int(valid.sum())
    if before_depth_gate:
        values = depth[valid]
        center_depth = float(np.median(values))
        robust_deviation = float(np.median(np.abs(values - center_depth)))
        adaptive_margin = max(float(depth_margin_mm), 4.0 * robust_deviation)
        valid &= np.abs(depth - center_depth) <= adaptive_margin
    else:
        center_depth = float("nan")
        adaptive_margin = float(depth_margin_mm)

    points = cloud[valid]
    if len(points) < min_person_points and detection.mask is not None:
        # A segmentation mask may be too strict on sparse depth boundaries.
        valid = (
            _bbox_mask((cloud_height, cloud_width), (
                int(detection.bbox_xyxy[0] * cloud_width / color_width),
                int(detection.bbox_xyxy[1] * cloud_height / color_height),
                int(detection.bbox_xyxy[2] * cloud_width / color_width),
                int(detection.bbox_xyxy[3] * cloud_height / color_height),
            ))
            & finite
            & (depth >= near_mm)
            & (depth <= far_mm)
        )
        if np.isfinite(center_depth):
            valid &= np.abs(depth - center_depth) <= adaptive_margin
        points = cloud[valid]

    if len(points) < min_person_points:
        raise ValueError(
            f"Person foreground has only {len(points)} valid points; "
            f"need at least {min_person_points}"
        )

    # Keep height range robust against isolated floor/background points.
    y_low, y_high = np.percentile(points[:, 1], [0.5, 99.5])
    x_low, x_high = np.percentile(points[:, 0], [0.5, 99.5])
    robust = points[
        (points[:, 1] >= y_low)
        & (points[:, 1] <= y_high)
        & (points[:, 0] >= x_low)
        & (points[:, 0] <= x_high)
    ]
    if len(robust) >= min_person_points:
        points = robust
    sampled = _deterministic_sample(points, int(num_points))
    rgb_foreground_mask = _cloud_mask_to_rgb(
        frame,
        cloud,
        valid,
        (color_height, color_width),
    )
    return PreprocessResult(
        sampled_points_mm=sampled,
        foreground_points_mm=np.asarray(points, dtype=np.float32),
        color_mask=rgb_foreground_mask,
        cloud_mask=valid,
        diagnostics={
            "alignment": alignment,
            "bbox_or_mask_points": before_depth_gate,
            "foreground_points": int(len(points)),
            "center_depth_mm": center_depth,
            "depth_margin_mm": adaptive_margin,
        },
    )
