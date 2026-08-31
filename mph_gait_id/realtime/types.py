from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeviceStatus:
    backend: str
    connected: bool
    device_count: int = 0
    serial: str | None = None
    message: str = ""
    dependency_ready: bool = False


@dataclass
class SensorFrame:
    index: int
    timestamp: float
    color_bgr: np.ndarray
    point_cloud_mm: np.ndarray
    point_cloud_color_aligned: bool
    depth_image: np.ndarray | None = None
    source_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_name: str = "person"
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    detector: str = ""
    person_count: int = 1


@dataclass
class PipelineSnapshot:
    state: str
    message: str
    timestamp: float
    color_bgr: np.ndarray | None = None
    person_points_mm: np.ndarray | None = None
    person_mask_rgb: np.ndarray | None = None
    detection: Detection | None = None
    result: dict[str, Any] | None = None
    fps: float = 0.0
    capture_fps: float = 0.0
    effective_sampling_fps: float = 0.0
    window_duration_s: float = 0.0
    mean_frame_gap_ms: float = 0.0
    max_frame_gap_ms: float = 0.0
    frame_read_ms: float = 0.0
    detection_ms: float = 0.0
    preprocessing_ms: float = 0.0
    model_encode_ms: float = 0.0
    gallery_match_ms: float = 0.0
    inference_ms: float = 0.0
    buffer_size: int = 0
    clip_len: int = 0
    frame_index: int = -1
    raw_point_count: int = 0
    person_point_count: int = 0
    detected_person_count: int = 0
    dropped_frames: int = 0
    operation: str = "recognize"
    enrollment_elapsed_s: float = 0.0
    enrollment_duration_s: float = 0.0
    enrollment_embeddings: int = 0
    enrollment_min_embeddings: int = 0
    device_status: DeviceStatus | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
