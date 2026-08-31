from __future__ import annotations

import platform
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from .types import PipelineSnapshot


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    """Return compact descriptive statistics without retaining sensor data."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": int(samples.size),
        "mean": float(samples.mean()),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "min": float(samples.min()),
        "max": float(samples.max()),
    }


@dataclass(frozen=True)
class BenchmarkProgress:
    phase: str
    elapsed_s: float
    remaining_s: float
    progress: float
    frame_count: int
    valid_frame_count: int


class LiveBenchmarkRecorder:
    """Collect timing-only live telemetry from the pipeline thread.

    The recorder deliberately keeps only scalar measurements, counters and
    model/result labels. RGB, depth maps and point-cloud arrays are never
    retained. Timing starts after the first valid person point cloud so model
    loading and time spent waiting for a subject do not consume the warm-up.
    """

    _TIMING_FIELDS = (
        "frame_read_ms",
        "detection_ms",
        "preprocessing_ms",
        "model_encode_ms",
        "gallery_match_ms",
        "inference_ms",
        "effective_sampling_fps",
        "capture_fps",
        "mean_frame_gap_ms",
        "max_frame_gap_ms",
        "person_point_count",
    )

    def __init__(
        self,
        warmup_seconds: float,
        duration_seconds: float,
        metadata: dict[str, Any],
        expected_person_id: str = "",
        posture: str = "stationary",
    ) -> None:
        if warmup_seconds < 0:
            raise ValueError("warmup_seconds must not be negative")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self.warmup_seconds = float(warmup_seconds)
        self.duration_seconds = float(duration_seconds)
        self.metadata = dict(metadata)
        self.expected_person_id = str(expected_person_id).strip()
        self.posture = str(posture or "stationary")
        self._lock = threading.RLock()
        self._first_valid_at: float | None = None
        self._measurement_started_at: float | None = None
        self._measurement_finished_at: float | None = None
        self._completed = False
        self._cancelled = False
        self._samples: dict[str, list[float]] = {
            field: [] for field in self._TIMING_FIELDS
        }
        self._states: Counter[str] = Counter()
        self._frame_count = 0
        self._valid_frame_count = 0
        self._inference_count = 0
        self._accepted_count = 0
        self._unknown_count = 0
        self._correct_count = 0
        self._incorrect_count = 0
        self._stable_count = 0
        self._multiple_people_frames = 0
        self._first_dropped: int | None = None
        self._last_dropped = 0
        self._device: dict[str, Any] = {}
        self._model_key = ""
        self._gpu_memory_tracking = False

    @property
    def completed(self) -> bool:
        with self._lock:
            return self._completed

    def accept(self, snapshot: PipelineSnapshot) -> None:
        now = time.perf_counter()
        with self._lock:
            if snapshot.device_status is not None:
                self._device = {
                    "backend": snapshot.device_status.backend,
                    "serial": snapshot.device_status.serial,
                    "message": snapshot.device_status.message,
                }
            if self._completed or snapshot.frame_index < 0:
                return

            valid = bool(
                snapshot.person_point_count > 0
                and snapshot.preprocessing_ms > 0.0
            )
            if self._first_valid_at is None:
                if not valid:
                    return
                self._first_valid_at = now

            measurement_start = self._first_valid_at + self.warmup_seconds
            measurement_end = measurement_start + self.duration_seconds
            if now < measurement_start:
                return
            if self._measurement_started_at is None:
                self._measurement_started_at = measurement_start
                if (
                    torch.cuda.is_available()
                    and str(self.metadata.get("device_requested", "auto")) != "cpu"
                ):
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        self._gpu_memory_tracking = True
                    except (RuntimeError, ValueError):
                        self._gpu_memory_tracking = False
            if now > measurement_end:
                self._measurement_finished_at = measurement_end
                self._completed = True
                return

            self._frame_count += 1
            self._states[str(snapshot.state)] += 1
            if valid:
                self._valid_frame_count += 1
            if snapshot.detected_person_count > 1:
                self._multiple_people_frames += 1
            if self._first_dropped is None:
                self._first_dropped = int(snapshot.dropped_frames)
            self._last_dropped = int(snapshot.dropped_frames)

            for field in (
                "frame_read_ms",
                "detection_ms",
                "capture_fps",
            ):
                value = float(getattr(snapshot, field, 0.0))
                if np.isfinite(value) and value >= 0.0:
                    self._samples[field].append(value)
            if valid:
                for field in (
                    "preprocessing_ms",
                    "effective_sampling_fps",
                    "mean_frame_gap_ms",
                    "max_frame_gap_ms",
                    "person_point_count",
                ):
                    value = float(getattr(snapshot, field, 0.0))
                    if np.isfinite(value) and value >= 0.0:
                        self._samples[field].append(value)

            if bool(snapshot.diagnostics.get("inference_ran")):
                self._inference_count += 1
                for field in (
                    "model_encode_ms",
                    "gallery_match_ms",
                    "inference_ms",
                ):
                    value = float(getattr(snapshot, field, 0.0))
                    if np.isfinite(value) and value >= 0.0:
                        self._samples[field].append(value)
                result = snapshot.result or {}
                self._model_key = str(result.get("model_key") or self._model_key)
                accepted = bool(result.get("accepted"))
                if accepted:
                    self._accepted_count += 1
                    if self.expected_person_id:
                        if str(result.get("person_id") or "") == self.expected_person_id:
                            self._correct_count += 1
                        else:
                            self._incorrect_count += 1
                else:
                    self._unknown_count += 1
                if bool(result.get("stable")):
                    self._stable_count += 1

    def progress(self) -> BenchmarkProgress:
        now = time.perf_counter()
        with self._lock:
            if self._first_valid_at is None:
                return BenchmarkProgress("waiting_person", 0.0, self.warmup_seconds, 0.0, 0, 0)
            warmup_elapsed = now - self._first_valid_at
            if warmup_elapsed < self.warmup_seconds:
                return BenchmarkProgress(
                    "warmup",
                    max(0.0, warmup_elapsed),
                    max(0.0, self.warmup_seconds - warmup_elapsed),
                    min(1.0, warmup_elapsed / max(self.warmup_seconds, 1e-9)),
                    0,
                    0,
                )
            started = self._measurement_started_at or (
                self._first_valid_at + self.warmup_seconds
            )
            elapsed = min(self.duration_seconds, max(0.0, now - started))
            phase = "complete" if self._completed else "measuring"
            return BenchmarkProgress(
                phase,
                elapsed,
                max(0.0, self.duration_seconds - elapsed),
                min(1.0, elapsed / self.duration_seconds),
                self._frame_count,
                self._valid_frame_count,
            )

    def cancel(self) -> None:
        with self._lock:
            if self._completed:
                return
            self._cancelled = True
            self._completed = True
            if self._measurement_started_at is not None:
                self._measurement_finished_at = time.perf_counter()

    def report(self) -> dict[str, Any] | None:
        with self._lock:
            if self._measurement_started_at is None or self._frame_count == 0:
                return None
            finished = self._measurement_finished_at or time.perf_counter()
            elapsed = max(
                1e-9,
                min(self.duration_seconds, finished - self._measurement_started_at),
            )
            invalid_count = max(0, self._frame_count - self._valid_frame_count)
            dropped_delta = max(
                0,
                self._last_dropped - int(self._first_dropped or 0),
            )
            distributions = {
                name: _distribution(list(values))
                for name, values in self._samples.items()
            }
            gpu_memory = {
                "tracked": self._gpu_memory_tracking,
                "allocated_mib": None,
                "reserved_mib": None,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            }
            if self._gpu_memory_tracking:
                try:
                    mib = float(1024 * 1024)
                    gpu_memory.update(
                        {
                            "allocated_mib": float(torch.cuda.memory_allocated() / mib),
                            "reserved_mib": float(torch.cuda.memory_reserved() / mib),
                            "peak_allocated_mib": float(
                                torch.cuda.max_memory_allocated() / mib
                            ),
                            "peak_reserved_mib": float(
                                torch.cuda.max_memory_reserved() / mib
                            ),
                        }
                    )
                except (RuntimeError, ValueError):
                    gpu_memory["tracked"] = False
            report = {
                "schema_version": 1,
                "test_id": datetime.now(timezone.utc).strftime(
                    "benchmark_%Y%m%dT%H%M%S%fZ"
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "scope": (
                    "stationary_efficiency_only"
                    if self.posture == "stationary"
                    else "live_motion_efficiency_and_result_behavior"
                ),
                "privacy": {
                    "rgb_saved": False,
                    "depth_saved": False,
                    "pointcloud_saved": False,
                    "embeddings_saved": False,
                    "numeric_telemetry_only": True,
                },
                "configuration": {
                    **self.metadata,
                    "posture": self.posture,
                    "expected_person_id": self.expected_person_id or None,
                    "warmup_seconds": self.warmup_seconds,
                    "requested_duration_seconds": self.duration_seconds,
                },
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "device": dict(self._device),
                },
                "summary": {
                    "completed": not self._cancelled,
                    "cancelled": self._cancelled,
                    "measured_seconds": float(elapsed),
                    "frames": self._frame_count,
                    "valid_frames": self._valid_frame_count,
                    "invalid_frames": invalid_count,
                    "end_to_end_fps": float(self._frame_count / elapsed),
                    "valid_pointcloud_fps": float(self._valid_frame_count / elapsed),
                    "valid_frame_percent": float(
                        100.0 * self._valid_frame_count / max(1, self._frame_count)
                    ),
                    "inference_count": self._inference_count,
                    "recognition_updates_per_second": float(
                        self._inference_count / elapsed
                    ),
                    "accepted_results": self._accepted_count,
                    "unknown_results": self._unknown_count,
                    "correct_results": self._correct_count,
                    "incorrect_results": self._incorrect_count,
                    "stable_results": self._stable_count,
                    "multiple_people_frames": self._multiple_people_frames,
                    "dropped_frame_delta": dropped_delta,
                    "model_key": self._model_key or None,
                },
                "latency_ms_and_rates": distributions,
                "gpu_memory": gpu_memory,
                "state_counts": dict(self._states),
                "interpretation": {
                    "accuracy_claim_allowed": bool(
                        self.posture != "stationary" and self.expected_person_id
                    ),
                    "note": (
                        "Stationary capture measures efficiency only; recognition "
                        "accuracy and gait stability must not be inferred."
                        if self.posture == "stationary"
                        else "Motion results are descriptive unless the same protocol "
                        "is repeated across subjects and sessions."
                    ),
                },
            }
            return report
