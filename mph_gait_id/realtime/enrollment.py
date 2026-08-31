from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..database import GalleryRepository
from ..runtime import EmbeddingBatch
from ..services import RegistrationService
from ..source_fingerprint import fingerprint_files


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"live_enroll_{timestamp}"


def replay_content_fingerprint(root: str | Path) -> str:
    """Fingerprint every RGB and raw-cloud frame used by synchronized replay."""

    replay_root = Path(root).expanduser().resolve()
    files = sorted(replay_root.glob("Color_image_*.png"))
    files.extend(sorted(replay_root.glob("Raw_data_*.npy")))
    if not files:
        raise FileNotFoundError(
            f"Replay fingerprint found no Color_image/Raw_data frames: {replay_root}"
        )
    return fingerprint_files(files, source_kind="synchronized-rgbd-replay")


def live_source_fingerprint(
    session_id: str,
    source_name: str,
    device_serial: str | None,
) -> str:
    value = f"{session_id}|{source_name}|{device_serial or 'unknown'}"
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class EnrollmentRequest:
    person_id: str
    display_name: str
    note: str = ""
    duration_s: float = 20.0
    warmup_s: float = 3.0
    min_embeddings: int = 5
    max_embeddings: int = 10

    def validate(self) -> None:
        if not self.person_id.strip():
            raise ValueError("Realtime enrollment requires a person ID")
        if not self.display_name.strip():
            raise ValueError("Realtime enrollment requires a display name")
        if self.duration_s <= 0:
            raise ValueError("Enrollment duration must be positive")
        if self.warmup_s < 0:
            raise ValueError("Enrollment warm-up cannot be negative")
        if self.min_embeddings <= 0:
            raise ValueError("Enrollment min embeddings must be positive")
        if self.max_embeddings < self.min_embeddings:
            raise ValueError(
                "Enrollment max embeddings must be at least min embeddings"
            )


@dataclass
class EnrollmentPass:
    pass_id: str
    requested_direction: str
    started_at: str
    start_frame: int
    embedding_start_index: int
    ended_at: str | None = None
    end_frame: int = -1
    frame_count: int = 0
    mean_person_points: float = 0.0
    observed_direction: str = "unknown"
    direction_displacement_mm: float = 0.0
    direction_monotonic_ratio: float = 0.0
    direction_match: bool = False
    quality_accepted: bool = False
    quality_message: str = "Pass has not ended"
    discarded: bool = False

    def public_record(self, embedding_count: int) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "requested_direction": self.requested_direction,
            "observed_direction": self.observed_direction,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "frame_count": int(self.frame_count),
            "embedding_count": int(embedding_count),
            "selectable": bool(not self.discarded and embedding_count > 0),
            "mean_person_points": float(self.mean_person_points),
            "direction_displacement_mm": float(self.direction_displacement_mm),
            "direction_monotonic_ratio": float(self.direction_monotonic_ratio),
            "direction_match": bool(self.direction_match),
            "quality_accepted": bool(self.quality_accepted),
            "quality_message": self.quality_message,
            "discarded": bool(self.discarded),
        }


class LiveEnrollmentCollector:
    """Collects a live walk and commits one model-scoped Gallery source."""

    def __init__(
        self,
        request: EnrollmentRequest,
        model_record: dict[str, Any],
        source_path: str | Path,
        source_fingerprint: str,
        source_kind: str,
        session_id: str,
        session_dir: str | Path,
        source_metadata: dict[str, Any],
    ) -> None:
        request.validate()
        self.request = request
        self.model_record = dict(model_record)
        self.source_path = Path(source_path).expanduser().resolve()
        self.source_fingerprint = str(source_fingerprint)
        self.source_kind = str(source_kind)
        self.session_id = str(session_id)
        self.session_dir = Path(session_dir).expanduser().resolve()
        self.source_metadata = dict(source_metadata)
        self.created_at = utc_now()
        self.embeddings: list[np.ndarray] = []
        self.windows: list[dict[str, Any]] = []
        self.passes: list[EnrollmentPass] = []
        self._active_pass: EnrollmentPass | None = None

    @property
    def count(self) -> int:
        return len(self.embeddings)

    @property
    def active_pass_id(self) -> str | None:
        return self._active_pass.pass_id if self._active_pass is not None else None

    def start_pass(
        self,
        requested_direction: str,
        start_frame: int,
    ) -> dict[str, Any]:
        if self._active_pass is not None:
            raise RuntimeError("An enrollment pass is already active")
        item = EnrollmentPass(
            pass_id=f"pass_{len(self.passes) + 1:03d}",
            requested_direction=str(requested_direction or "unknown"),
            started_at=utc_now(),
            start_frame=int(start_frame),
            embedding_start_index=self.count,
        )
        self.passes.append(item)
        self._active_pass = item
        return item.public_record(embedding_count=0)

    def end_pass(
        self,
        end_frame: int,
        frame_count: int,
        mean_person_points: float,
        observed_direction: str,
        direction_displacement_mm: float = 0.0,
        direction_monotonic_ratio: float = 0.0,
        discard: bool = False,
    ) -> dict[str, Any]:
        item = self._active_pass
        if item is None:
            raise RuntimeError("No enrollment pass is active")
        item.ended_at = utc_now()
        item.end_frame = int(end_frame)
        item.frame_count = int(frame_count)
        item.mean_person_points = float(mean_person_points)
        item.observed_direction = str(observed_direction or "unknown")
        item.direction_displacement_mm = float(direction_displacement_mm)
        item.direction_monotonic_ratio = float(direction_monotonic_ratio)
        item.discarded = bool(discard)
        embedding_count = self.count - item.embedding_start_index
        # The current checkpoints were trained with front-facing captures.
        # Centroid translation can check that the person approached the camera,
        # but front/back body orientation still relies on operator control.
        expected_direction = (
            "toward_camera"
            if item.requested_direction == "front_facing"
            else item.requested_direction
        )
        requires_direction = expected_direction not in {"automatic", "unknown", ""}
        item.direction_match = bool(
            not requires_direction
            or expected_direction == item.observed_direction
        )
        item.quality_accepted = bool(
            not item.discarded and embedding_count > 0 and item.direction_match
        )
        if item.discarded:
            item.quality_message = "Pass was discarded"
        elif embedding_count <= 0:
            item.quality_message = "Pass contains no gait embeddings"
        elif not item.direction_match:
            item.quality_message = (
                f"Movement warning: expected {expected_direction}, "
                f"observed {item.observed_direction}"
            )
        elif item.requested_direction == "front_facing":
            item.quality_message = (
                "Movement toward the camera was detected; front-facing orientation "
                "still relies on operator-controlled capture"
            )
        else:
            item.quality_message = "Direction and embedding checks passed"
        for window in self.windows[item.embedding_start_index :]:
            window["observed_direction"] = item.observed_direction
            window["direction_displacement_mm"] = item.direction_displacement_mm
            window["direction_monotonic_ratio"] = item.direction_monotonic_ratio
            window["direction_match"] = item.direction_match
            window["quality_accepted"] = item.quality_accepted
        if discard:
            del self.embeddings[item.embedding_start_index :]
            del self.windows[item.embedding_start_index :]
            embedding_count = 0
        self._active_pass = None
        return item.public_record(embedding_count=embedding_count)

    def pass_summaries(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for window in self.windows:
            pass_id = str(window.get("pass_id") or "")
            counts[pass_id] = counts.get(pass_id, 0) + 1
        return [
            item.public_record(embedding_count=counts.get(item.pass_id, 0))
            for item in self.passes
        ]

    def count_for_pass(self, pass_id: str) -> int:
        return sum(
            1
            for window in self.windows
            if str(window.get("pass_id") or "") == str(pass_id)
        )

    def add_embedding(
        self,
        embedding: np.ndarray,
        start_frame: int,
        end_frame: int,
        timestamp: float,
        effective_sampling_fps: float = 0.0,
        window_duration_s: float = 0.0,
        mean_frame_gap_ms: float = 0.0,
        max_frame_gap_ms: float = 0.0,
    ) -> None:
        if self._active_pass is None:
            raise RuntimeError("Enrollment embedding was produced outside an active pass")
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        expected = int(self.model_record["embedding_dim"])
        if vector.size != expected:
            raise ValueError(
                f"Enrollment embedding dimension mismatch: {vector.size} != {expected}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("Enrollment embedding contains NaN or Inf")
        self.embeddings.append(vector.copy())
        self.windows.append(
            {
                "window_index": self.count - 1,
                "start_frame_number": int(start_frame),
                "end_frame_number": int(end_frame),
                "captured_at": float(timestamp),
                "effective_sampling_fps": float(effective_sampling_fps),
                "window_duration_s": float(window_duration_s),
                "mean_frame_gap_ms": float(mean_frame_gap_ms),
                "max_frame_gap_ms": float(max_frame_gap_ms),
                "pass_id": self._active_pass.pass_id,
                "requested_direction": self._active_pass.requested_direction,
                "observed_direction": self._active_pass.observed_direction,
            }
        )

    def write_review_manifest(self, elapsed_s: float) -> None:
        self._write_json(
            "capture_manifest.json",
            {
                "status": "awaiting_review",
                "session_id": self.session_id,
                "created_at": self.created_at,
                "source_path": str(self.source_path),
                "source_fingerprint": self.source_fingerprint,
                "source_kind": self.source_kind,
                "model": self.model_record,
                "request": asdict(self.request),
                "elapsed_s": float(elapsed_s),
                "passes": self.pass_summaries(),
                "windows": self.windows,
                "raw_sensor_frames_saved": False,
            },
        )

    def write_abandoned_manifest(self, elapsed_s: float) -> None:
        self._write_json(
            "capture_manifest.json",
            {
                "status": "abandoned_without_gallery_commit",
                "session_id": self.session_id,
                "created_at": self.created_at,
                "source_path": str(self.source_path),
                "source_fingerprint": self.source_fingerprint,
                "source_kind": self.source_kind,
                "model": self.model_record,
                "request": asdict(self.request),
                "elapsed_s": float(elapsed_s),
                "passes": self.pass_summaries(),
                "windows": self.windows,
                "raw_sensor_frames_saved": False,
            },
        )

    def finalize(
        self,
        repository: GalleryRepository,
        elapsed_s: float,
        selected_pass_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._active_pass is not None:
            raise RuntimeError("End the active pass before committing enrollment")
        selectable_passes = {
            item.pass_id
            for item in self.passes
            if not item.discarded
            and (self.count_for_pass(item.pass_id) > 0)
        }
        recommended_passes = {
            item.pass_id
            for item in self.passes
            if not item.discarded and item.quality_accepted
        }
        selected = set(
            sorted(recommended_passes)
            if selected_pass_ids is None
            else selected_pass_ids
        )
        if not selected:
            raise ValueError("Select at least one pass before Gallery commit")
        unknown = selected - selectable_passes
        if unknown:
            raise ValueError(
                "Unknown, discarded, or empty pass IDs: "
                f"{sorted(unknown)}"
            )
        manual_overrides = selected - recommended_passes
        indices = [
            index
            for index, window in enumerate(self.windows)
            if str(window.get("pass_id")) in selected
        ]
        if len(indices) < self.request.min_embeddings:
            raise ValueError(
                "Selected passes do not contain enough valid gait clips: "
                f"{len(indices)} < {self.request.min_embeddings}"
            )
        matrix = np.stack([self.embeddings[index] for index in indices]).astype(np.float32)
        selected_windows = [dict(self.windows[index]) for index in indices]
        for window in selected_windows:
            window["manual_quality_override"] = bool(
                str(window.get("pass_id")) in manual_overrides
            )
        metadata = {
            **self.source_metadata,
            "source_fingerprint": self.source_fingerprint,
            "source_adapter": {
                "source_kind": self.source_kind,
                "source_fingerprint": self.source_fingerprint,
            },
            "realtime_enrollment": {
                "session_id": self.session_id,
                "created_at": self.created_at,
                "elapsed_s": float(elapsed_s),
                "available_embeddings": len(indices),
                "selected_pass_ids": sorted(selected),
                "manual_quality_override_pass_ids": sorted(manual_overrides),
                "passes": self.pass_summaries(),
                "request": asdict(self.request),
            },
        }
        batch = EmbeddingBatch(
            embeddings=matrix,
            windows=selected_windows,
            source=self.source_path,
            source_metadata=metadata,
            model=self.model_record,
        )
        self._write_json(
            "capture_manifest.json",
            {
                "status": "captured",
                "session_id": self.session_id,
                "created_at": self.created_at,
                "source_path": str(self.source_path),
                "source_fingerprint": self.source_fingerprint,
                "source_kind": self.source_kind,
                "model": self.model_record,
                "request": asdict(self.request),
                "elapsed_s": float(elapsed_s),
                "selected_pass_ids": sorted(selected),
                "manual_quality_override_pass_ids": sorted(manual_overrides),
                "passes": self.pass_summaries(),
                "windows": selected_windows,
                "raw_sensor_frames_saved": False,
            },
        )
        try:
            result = RegistrationService(repository).enroll(
                batch=batch,
                person_id=self.request.person_id.strip(),
                display_name=self.request.display_name.strip(),
                note=self.request.note.strip(),
                min_embeddings=self.request.min_embeddings,
                max_embeddings=self.request.max_embeddings,
                allow_duplicate_source=False,
            )
        except Exception as exc:
            self._write_json(
                "registration_error.json",
                {
                    "status": "failed",
                    "session_id": self.session_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        output = {
            **result,
            "state": "enrolled",
            "accepted": True,
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "elapsed_s": float(elapsed_s),
            "captured_embeddings": self.count,
            "selected_embeddings": len(indices),
            "selected_pass_ids": sorted(selected),
            "manual_quality_override_pass_ids": sorted(manual_overrides),
            "passes": self.pass_summaries(),
        }
        self._write_json("registration_result.json", output)
        return output

    def record_failure(self, error: Exception, elapsed_s: float, reason: str) -> None:
        self._write_json(
            "capture_manifest.json",
            {
                "status": "captured_not_registered",
                "session_id": self.session_id,
                "created_at": self.created_at,
                "source_path": str(self.source_path),
                "source_fingerprint": self.source_fingerprint,
                "source_kind": self.source_kind,
                "model": self.model_record,
                "request": asdict(self.request),
                "elapsed_s": float(elapsed_s),
                "passes": self.pass_summaries(),
                "windows": self.windows,
                "raw_sensor_frames_saved": False,
            },
        )
        self._write_json(
            "registration_error.json",
            {
                "status": "failed",
                "session_id": self.session_id,
                "reason": str(reason),
                "captured_embeddings": self.count,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        destination = self.session_dir / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
