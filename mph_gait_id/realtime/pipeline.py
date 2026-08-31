from __future__ import annotations

import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from ..controller import GaitApplicationController
from ..config import nested, resolve_system_path
from ..runtime import EmbeddingBatch, SystemModelRuntime
from ..services import RecognitionService
from .detection import (
    ReplayDepthPersonDetector,
    YoloPersonDetector,
    YoloSamPersonDetector,
)
from .devices import AzureKinectSource, FrameSource, ReplaySource
from .enrollment import (
    EnrollmentRequest,
    LiveEnrollmentCollector,
    live_source_fingerprint,
    new_session_id,
    replay_content_fingerprint,
)
from .preprocessing import filter_person_pointcloud
from .types import DeviceStatus, PipelineSnapshot


@dataclass(frozen=True)
class RealtimeConfig:
    bundle_id: str
    operation: str = "recognize"
    source_mode: str = "replay"
    replay_path: str = ""
    replay_fps: float = 10.0
    replay_loop: bool = True
    detector: str = "replay_depth"
    yolo_weights: str = ""
    sam_checkpoint: str = ""
    sam_model_type: str = "vit_b"
    sam_refresh_interval: int = 10
    detector_confidence: float = 0.35
    yolo_image_size: int = 640
    device: str = "auto"
    clip_len: int = 15
    inference_stride: int = 5
    num_points: int = 1024
    threshold: float | None = 0.55
    min_margin: float = 0.03
    top_k_per_identity: int = 3
    stability_windows: int = 3
    max_missing_frames: int = 4
    near_mm: float = 400.0
    far_mm: float = 8000.0
    depth_margin_mm: float = 550.0
    min_person_points: int = 96
    enrollment_person_id: str = ""
    enrollment_display_name: str = ""
    enrollment_note: str = ""
    enrollment_duration_s: float = 20.0
    enrollment_warmup_s: float = 3.0
    enrollment_min_embeddings: int = 5
    enrollment_max_embeddings: int = 10
    enrollment_stride: int = 0
    enrollment_session_root: str = ""
    guided_passes: bool = True
    preprocessing_profile_id: str = "person_foreground_pointcloud_v1"
    reject_multiple_people: bool = False
    multi_person_policy: str = "primary_salience"
    max_valid_frame_gap_s: float = 1.0


class RealtimePipeline:
    """Threaded detect-filter-buffer-infer-match pipeline."""

    def __init__(
        self,
        controller: GaitApplicationController,
        config: RealtimeConfig,
        metric_sink: Callable[[PipelineSnapshot], None] | None = None,
    ) -> None:
        self.controller = controller
        self.config = config
        self._snapshots: queue.Queue[PipelineSnapshot] = queue.Queue(maxsize=3)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: FrameSource | None = None
        self._runtime: SystemModelRuntime | None = None
        self._device_status: DeviceStatus | None = None
        self._sampled_frames: deque[np.ndarray] = deque(maxlen=config.clip_len)
        self._sampled_timestamps: deque[float] = deque(maxlen=config.clip_len)
        self._sampled_frame_indices: deque[int] = deque(maxlen=config.clip_len)
        self._recent_results: deque[dict[str, Any]] = deque(
            maxlen=max(1, config.stability_windows)
        )
        self._last_result: dict[str, Any] | None = None
        self._processed = 0
        self._dropped = 0
        self._missing = 0
        self._last_inference_ms = 0.0
        self._last_model_encode_ms = 0.0
        self._last_gallery_match_ms = 0.0
        self._metric_sink = metric_sink
        self._metric_sink_error: str | None = None
        self._session_id = new_session_id()
        self._source_path: Path | None = None
        self._source_fingerprint = ""
        self._collector: LiveEnrollmentCollector | None = None
        self._enrollment_person_seen_at: float | None = None
        self._enrollment_active_at: float | None = None
        self._enrollment_elapsed_offset_s = 0.0
        self._control_lock = threading.RLock()
        self._pass_start_requested = False
        self._pass_end_requested: bool | None = None
        self._finish_review_requested = False
        self._requested_direction = "front_facing"
        self._pass_frame_count = 0
        self._pass_point_total = 0
        self._pass_first_center_x: float | None = None
        self._pass_last_center_x: float | None = None
        self._pass_first_center_z: float | None = None
        self._pass_last_center_z: float | None = None
        self._pass_center_history: list[tuple[float, float]] = []
        self._pass_last_frame_index = -1
        self._review_ready = False
        self._review_elapsed_s = 0.0
        self._committed = False

    def _clear_sequence_state(self, clear_results: bool = False) -> None:
        self._sampled_frames.clear()
        self._sampled_timestamps.clear()
        self._sampled_frame_indices.clear()
        if clear_results:
            self._recent_results.clear()
            self._last_result = None

    @staticmethod
    def _selection_message(detection: Any, message: str) -> str:
        count = int(getattr(detection, "person_count", 1))
        if count <= 1:
            return message
        return f"Primary person selected from {count} detections | {message}"

    def _selection_diagnostics(self, detection: Any) -> dict[str, Any]:
        count = int(getattr(detection, "person_count", 1))
        return {
            "detected_person_count": count,
            "multi_person_policy": (
                "reject"
                if self.config.reject_multiple_people
                else self.config.multi_person_policy
            ),
            "primary_person_selected": bool(
                count > 1 and not self.config.reject_multiple_people
            ),
        }

    def _sampling_metrics(self) -> tuple[float, float, float, float]:
        if len(self._sampled_timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0
        timestamps = np.asarray(self._sampled_timestamps, dtype=np.float64)
        gaps = np.diff(timestamps)
        if not np.isfinite(gaps).all() or np.any(gaps <= 0.0):
            return 0.0, 0.0, 0.0, 0.0
        duration = float(timestamps[-1] - timestamps[0])
        effective_fps = float((len(timestamps) - 1) / duration)
        return (
            effective_fps,
            duration,
            float(gaps.mean() * 1000.0),
            float(gaps.max() * 1000.0),
        )

    def _timing_snapshot_fields(self) -> dict[str, float]:
        fps, duration, mean_gap_ms, max_gap_ms = self._sampling_metrics()
        return {
            "effective_sampling_fps": fps,
            "window_duration_s": duration,
            "mean_frame_gap_ms": mean_gap_ms,
            "max_frame_gap_ms": max_gap_ms,
        }

    @property
    def operation(self) -> str:
        return str(self.config.operation).strip().lower()

    def start_enrollment_pass(self, direction: str = "front_facing") -> None:
        if self.operation != "enroll" or not self.config.guided_passes:
            raise RuntimeError("Pass control is only available in guided enrollment")
        if self._review_ready:
            raise RuntimeError("This enrollment session is already awaiting review")
        with self._control_lock:
            if self._collector is not None and self._collector.active_pass_id is not None:
                raise RuntimeError("An enrollment pass is already recording")
            self._requested_direction = str(direction or "unknown")
            self._pass_start_requested = True
            self._pass_end_requested = None
            self._enrollment_person_seen_at = None

    def end_enrollment_pass(self, discard: bool = False) -> None:
        if self.operation != "enroll" or not self.config.guided_passes:
            raise RuntimeError("Pass control is only available in guided enrollment")
        with self._control_lock:
            self._pass_end_requested = bool(discard)

    def finish_enrollment_for_review(self) -> None:
        if self.operation != "enroll" or not self.config.guided_passes:
            raise RuntimeError("Review is only available in guided enrollment")
        with self._control_lock:
            self._finish_review_requested = True

    def enrollment_passes(self) -> list[dict[str, Any]]:
        return self._collector.pass_summaries() if self._collector is not None else []

    def _finish_review_worker_shutdown(self) -> None:
        if self.running:
            # The review snapshot may arrive before the source worker has
            # completed its finally block.
            self.stop(timeout=2.0)
        if self.running:
            raise RuntimeError("The capture worker is still stopping; try again")

    def commit_enrollment(self, selected_pass_ids: list[str]) -> dict[str, Any]:
        if not self._review_ready:
            raise RuntimeError("Finish capture and enter pass review before committing")
        self._finish_review_worker_shutdown()
        if self._committed:
            raise RuntimeError("This enrollment session has already been committed")
        if self._collector is None:
            raise RuntimeError("Enrollment collector is unavailable")
        result = self._collector.finalize(
            repository=self.controller.repository,
            elapsed_s=self._review_elapsed_s,
            selected_pass_ids=list(selected_pass_ids),
        )
        self._committed = True
        return result

    def abandon_enrollment(self) -> None:
        if not self._review_ready:
            raise RuntimeError("Finish capture and enter pass review before abandoning")
        self._finish_review_worker_shutdown()
        if self._committed:
            raise RuntimeError("This enrollment session was already committed")
        if self._collector is not None:
            self._collector.write_abandoned_manifest(self._review_elapsed_s)
        self._review_ready = False

    def resume_enrollment_capture(self) -> None:
        """Leave review and continue collecting passes in the same session."""

        if self.operation != "enroll" or not self.config.guided_passes:
            raise RuntimeError("Resume is only available in guided enrollment")
        if not self._review_ready:
            raise RuntimeError("Enter pass review before resuming capture")
        self._finish_review_worker_shutdown()
        if self._committed:
            raise RuntimeError("This enrollment session has already been committed")
        if self._collector is None:
            raise RuntimeError("Enrollment collector is unavailable")
        self._review_ready = False
        self._last_result = None
        self._clear_sequence_state(clear_results=True)
        with self._control_lock:
            self._pass_start_requested = False
            self._pass_end_requested = None
            self._finish_review_requested = False
        self.start()

    def _enrollment_request(self) -> EnrollmentRequest:
        return EnrollmentRequest(
            person_id=self.config.enrollment_person_id,
            display_name=self.config.enrollment_display_name,
            note=self.config.enrollment_note,
            duration_s=float(self.config.enrollment_duration_s),
            warmup_s=float(self.config.enrollment_warmup_s),
            min_embeddings=int(self.config.enrollment_min_embeddings),
            max_embeddings=int(self.config.enrollment_max_embeddings),
        )

    def _validate_config(self) -> None:
        if self.operation not in {"recognize", "enroll"}:
            raise ValueError(f"Unknown realtime operation: {self.config.operation}")
        if self.config.clip_len <= 0:
            raise ValueError("clip_len must be positive")
        if self.config.max_valid_frame_gap_s <= 0:
            raise ValueError("max_valid_frame_gap_s must be positive")
        if self.operation == "enroll":
            self._enrollment_request().validate()
        if self.config.multi_person_policy != "primary_salience":
            raise ValueError(
                "multi_person_policy must be 'primary_salience' in the "
                "single-target point-cloud pipeline"
            )

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("Realtime pipeline is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="gait-realtime-pipeline",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout)))
        if thread is not None and thread.is_alive() and self._source is not None:
            # Azure capture normally returns within one second. Closing is only
            # the fallback for a backend that does not honor that timeout.
            try:
                self._source.close()
            except Exception:
                pass
            thread.join(timeout=1.0)
        if thread is None or not thread.is_alive():
            self._thread = None

    def poll_latest(self) -> PipelineSnapshot | None:
        latest = None
        while True:
            try:
                latest = self._snapshots.get_nowait()
            except queue.Empty:
                return latest

    def _emit(self, snapshot: PipelineSnapshot) -> None:
        if self._metric_sink is not None:
            try:
                self._metric_sink(snapshot)
            except Exception as exc:
                # Telemetry must never stop camera capture or recognition.
                self._metric_sink_error = f"{type(exc).__name__}: {exc}"
        while True:
            try:
                self._snapshots.put_nowait(snapshot)
                return
            except queue.Full:
                try:
                    self._snapshots.get_nowait()
                except queue.Empty:
                    return

    def _build_source(self) -> FrameSource:
        if self.config.source_mode == "azure_kinect":
            return AzureKinectSource(fps=30)
        if self.config.source_mode == "replay":
            return ReplaySource(
                root=self.config.replay_path,
                fps=self.config.replay_fps,
                loop=self.config.replay_loop,
            )
        raise ValueError(f"Unknown realtime source mode: {self.config.source_mode}")

    def _build_detector(self):
        if self.config.detector == "yolo":
            return YoloPersonDetector(
                weights=self.config.yolo_weights,
                confidence=self.config.detector_confidence,
                image_size=self.config.yolo_image_size,
                device=self.config.device,
            )
        if self.config.detector == "yolo_seg":
            return YoloPersonDetector(
                weights=self.config.yolo_weights,
                confidence=self.config.detector_confidence,
                image_size=self.config.yolo_image_size,
                device=self.config.device,
                require_mask=True,
            )
        if self.config.detector == "yolo_sam":
            return YoloSamPersonDetector(
                yolo_weights=self.config.yolo_weights,
                sam_checkpoint=self.config.sam_checkpoint,
                sam_model_type=self.config.sam_model_type,
                confidence=self.config.detector_confidence,
                image_size=self.config.yolo_image_size,
                device=self.config.device,
                refresh_interval=self.config.sam_refresh_interval,
            )
        if self.config.detector == "replay_depth":
            if self.config.source_mode != "replay":
                raise ValueError(
                    "Replay depth fallback is diagnostic-only; Azure Kinect requires YOLO."
                )
            return ReplayDepthPersonDetector(
                near_mm=self.config.near_mm,
                far_mm=self.config.far_mm,
            )
        raise ValueError(f"Unknown person detector: {self.config.detector}")

    def _run(self) -> None:
        try:
            self._validate_config()
            self._emit(
                PipelineSnapshot(
                    state="starting",
                    message=(
                        "Loading model and opening source for realtime enrollment"
                        if self.operation == "enroll"
                        else "Loading model and opening source for realtime recognition"
                    ),
                    timestamp=time.time(),
                    clip_len=self.config.clip_len,
                    operation=self.operation,
                )
            )
            # Open/probe the sensor before allocating the gait model or detector.
            # Native Azure SDK calls stay in this worker so the Tk event loop
            # remains responsive while a busy camera backend is retried.
            self._source = self._build_source()
            self._device_status = self._source.open()
            self._emit(
                PipelineSnapshot(
                    state="source_ready",
                    message=f"{self._device_status.message}; loading gait model",
                    timestamp=time.time(),
                    clip_len=self.config.clip_len,
                    operation=self.operation,
                    device_status=self._device_status,
                )
            )
            self._runtime = SystemModelRuntime(
                bundle_id=self.config.bundle_id,
                device=self.config.device,
                model_store=self.controller.model_store,
                runtime_clip_len=self.config.clip_len,
                preprocessing_profile_id=self.config.preprocessing_profile_id,
            )
            self.controller.repository.initialize()
            detector = self._build_detector()
            self._prepare_session_provenance()
            self._loop(detector)
        except Exception as exc:
            self._emit(
                PipelineSnapshot(
                    state="error",
                    message=f"{type(exc).__name__}: {exc}",
                    timestamp=time.time(),
                    clip_len=self.config.clip_len,
                    operation=self.operation,
                    device_status=self._device_status,
                    diagnostics={"exception_type": type(exc).__name__},
                )
            )
        finally:
            if self._source is not None:
                try:
                    self._source.close()
                except Exception:
                    pass
            self._source = None

    def _prepare_session_provenance(self) -> None:
        assert self._runtime is not None
        session_root = (
            Path(self.config.enrollment_session_root).expanduser().resolve()
            if self.config.enrollment_session_root
            else resolve_system_path(
                nested(
                    self.controller.config,
                    "storage",
                    "live_enrollment_sessions",
                    "../data/live_enrollment_sessions",
                )
            )
        )
        session_dir = session_root / self._session_id
        if self.config.source_mode == "replay":
            self._source_path = Path(self.config.replay_path).expanduser().resolve()
            self._source_fingerprint = replay_content_fingerprint(self._source_path)
            source_kind = "realtime_replay_rgbd"
        else:
            self._source_path = session_dir
            self._source_fingerprint = live_source_fingerprint(
                self._session_id,
                "azure_kinect_dk",
                self._device_status.serial if self._device_status else None,
            )
            source_kind = "azure_kinect_live"
        if self.operation == "enroll" and self._collector is not None:
            # Reopening the capture worker preserves the
            # collector, session ID, passes, embeddings, and source fingerprint.
            self._enrollment_active_at = time.monotonic()
            return
        if self.operation == "enroll":
            self._collector = LiveEnrollmentCollector(
                request=self._enrollment_request(),
                model_record=self._runtime.model_record,
                source_path=self._source_path,
                source_fingerprint=self._source_fingerprint,
                source_kind=source_kind,
                session_id=self._session_id,
                session_dir=session_dir,
                source_metadata={
                    "source_type": self.config.source_mode,
                    "foreground_detector": self.config.detector,
                    "yolo_weights": self.config.yolo_weights,
                    "sam_checkpoint": (
                        self.config.sam_checkpoint
                        if self.config.detector == "yolo_sam"
                        else None
                    ),
                    "sam_model_type": (
                        self.config.sam_model_type
                        if self.config.detector == "yolo_sam"
                        else None
                    ),
                    "sam_refresh_interval": (
                        int(self.config.sam_refresh_interval)
                        if self.config.detector == "yolo_sam"
                        else None
                    ),
                    "device_serial": (
                        self._device_status.serial if self._device_status else None
                    ),
                    "clip_len": int(self.config.clip_len),
                    "enrollment_stride": int(
                        self.config.enrollment_stride or self.config.clip_len
                    ),
                    "raw_sensor_frames_saved": False,
                    "guided_passes": bool(self.config.guided_passes),
                    "reject_multiple_people": bool(
                        self.config.reject_multiple_people
                    ),
                    "multi_person_policy": self.config.multi_person_policy,
                },
            )
            if self.config.guided_passes:
                self._enrollment_active_at = time.monotonic()

    def _loop(self, detector: Any) -> None:
        assert self._source is not None
        assert self._runtime is not None
        times: deque[float] = deque(maxlen=30)
        while not self._stop.is_set():
            if self.operation == "enroll" and self.config.guided_passes:
                if self._process_guided_controls():
                    return
            if self._enrollment_expired():
                if self.config.guided_passes:
                    self._emit_enrollment_review("maximum_session_duration")
                else:
                    self._emit_enrollment_terminal("duration_complete")
                return
            loop_started = time.perf_counter()
            frame = self._source.read()
            frame_read_ms = (time.perf_counter() - loop_started) * 1000.0
            if frame is None:
                if self.config.source_mode == "replay" and not self.config.replay_loop:
                    if self.operation == "enroll":
                        if self.config.guided_passes:
                            self._emit_enrollment_review("replay_complete")
                        else:
                            self._emit_enrollment_terminal("replay_complete")
                        return
                    self._emit(
                        PipelineSnapshot(
                            state="complete",
                            message="Replay completed",
                            timestamp=time.time(),
                            clip_len=self.config.clip_len,
                            operation=self.operation,
                            device_status=self._device_status,
                        )
                    )
                    return
                continue
            times.append(loop_started)
            capture_fps = 0.0
            if len(times) >= 2:
                capture_fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)

            detection_started = time.perf_counter()
            detection = detector.detect(frame)
            detection_ms = (time.perf_counter() - detection_started) * 1000.0
            if detection is None:
                self._missing += 1
                if self.operation == "enroll" and (
                    (
                        not self.config.guided_passes
                        and self._enrollment_active_at is None
                    )
                    or (
                        self.config.guided_passes
                        and self._collector is not None
                        and self._collector.active_pass_id is None
                    )
                ):
                    self._enrollment_person_seen_at = None
                if self._missing > self.config.max_missing_frames:
                    self._clear_sequence_state(clear_results=True)
                self._emit(
                    PipelineSnapshot(
                        state="waiting_person",
                        message="Waiting for a person",
                        timestamp=time.time(),
                        color_bgr=frame.color_bgr,
                        result=self._enrollment_progress_result(),
                        fps=1.0 / max(time.perf_counter() - loop_started, 1e-6),
                        capture_fps=capture_fps,
                        frame_read_ms=frame_read_ms,
                        detection_ms=detection_ms,
                        buffer_size=len(self._sampled_frames),
                        clip_len=self.config.clip_len,
                        frame_index=frame.index,
                        raw_point_count=int(frame.point_cloud_mm.shape[0] * frame.point_cloud_mm.shape[1]),
                        dropped_frames=self._dropped,
                        **self._timing_snapshot_fields(),
                        **self._operation_snapshot_fields(),
                        device_status=self._device_status,
                    )
                )
                continue

            if self.config.reject_multiple_people and detection.person_count > 1:
                self._missing += 1
                self._clear_sequence_state(clear_results=True)
                self._emit(
                    PipelineSnapshot(
                        state="multiple_people",
                        message=(
                            f"Detected {detection.person_count} people; strict mode supports "
                            "one person in the walking area"
                        ),
                        timestamp=time.time(),
                        color_bgr=frame.color_bgr,
                        detection=detection,
                        result=self._enrollment_progress_result(),
                        fps=1.0 / max(time.perf_counter() - loop_started, 1e-6),
                        capture_fps=capture_fps,
                        frame_read_ms=frame_read_ms,
                        detection_ms=detection_ms,
                        buffer_size=0,
                        clip_len=self.config.clip_len,
                        frame_index=frame.index,
                        raw_point_count=int(
                            frame.point_cloud_mm.shape[0] * frame.point_cloud_mm.shape[1]
                        ),
                        detected_person_count=int(detection.person_count),
                        dropped_frames=self._dropped,
                        **self._timing_snapshot_fields(),
                        **self._operation_snapshot_fields(),
                        device_status=self._device_status,
                    )
                )
                continue

            preprocess_started = time.perf_counter()
            try:
                prepared = filter_person_pointcloud(
                    frame,
                    detection,
                    num_points=self.config.num_points,
                    near_mm=self.config.near_mm,
                    far_mm=self.config.far_mm,
                    depth_margin_mm=self.config.depth_margin_mm,
                    min_person_points=self.config.min_person_points,
                )
            except ValueError as exc:
                self._dropped += 1
                self._missing += 1
                if self._missing > self.config.max_missing_frames:
                    self._clear_sequence_state(clear_results=True)
                if self.operation == "enroll" and (
                    (
                        not self.config.guided_passes
                        and self._enrollment_active_at is None
                    )
                    or (
                        self.config.guided_passes
                        and self._collector is not None
                        and self._collector.active_pass_id is None
                    )
                ):
                    self._enrollment_person_seen_at = None
                self._emit(
                    PipelineSnapshot(
                        state="insufficient_points",
                        message=self._selection_message(detection, str(exc)),
                        timestamp=time.time(),
                        color_bgr=frame.color_bgr,
                        detection=detection,
                        result=self._enrollment_progress_result(),
                        capture_fps=capture_fps,
                        frame_read_ms=frame_read_ms,
                        detection_ms=detection_ms,
                        buffer_size=len(self._sampled_frames),
                        clip_len=self.config.clip_len,
                        frame_index=frame.index,
                        raw_point_count=int(frame.point_cloud_mm.shape[0] * frame.point_cloud_mm.shape[1]),
                        dropped_frames=self._dropped,
                        detected_person_count=int(detection.person_count),
                        **self._timing_snapshot_fields(),
                        **self._operation_snapshot_fields(),
                        device_status=self._device_status,
                        diagnostics=self._selection_diagnostics(detection),
                    )
                )
                continue
            preprocessing_ms = (time.perf_counter() - preprocess_started) * 1000.0
            self._missing = 0

            if self.operation == "enroll" and not self._enrollment_ready(frame.index):
                self._clear_sequence_state()
                result = self._enrollment_progress_result()
                remaining = float(result.get("warmup_remaining_s", 0.0))
                paused = str(result.get("state")) == "enrollment_paused"
                countdown_message = (
                    "Ready: return to the start position, then press Start pass"
                    if paused
                    else f"Pass starts in {remaining:.1f} s"
                )
                self._emit(
                    PipelineSnapshot(
                        state=("enrollment_paused" if paused else "enrollment_countdown"),
                        message=self._selection_message(
                            detection,
                            countdown_message,
                        ),
                        timestamp=time.time(),
                        color_bgr=frame.color_bgr,
                        person_points_mm=prepared.foreground_points_mm,
                        person_mask_rgb=prepared.color_mask,
                        detection=detection,
                        result=result,
                        fps=1.0 / max(time.perf_counter() - loop_started, 1e-6),
                        capture_fps=capture_fps,
                        frame_read_ms=frame_read_ms,
                        detection_ms=detection_ms,
                        preprocessing_ms=preprocessing_ms,
                        buffer_size=0,
                        clip_len=self.config.clip_len,
                        frame_index=frame.index,
                        raw_point_count=int(
                            frame.point_cloud_mm.shape[0] * frame.point_cloud_mm.shape[1]
                        ),
                        person_point_count=int(len(prepared.foreground_points_mm)),
                        dropped_frames=self._dropped,
                        detected_person_count=int(detection.person_count),
                        **self._timing_snapshot_fields(),
                        **self._operation_snapshot_fields(),
                        device_status=self._device_status,
                        diagnostics={
                            **prepared.diagnostics,
                            **self._selection_diagnostics(detection),
                        },
                    )
                )
                continue

            if self._sampled_timestamps:
                valid_gap = float(frame.timestamp) - self._sampled_timestamps[-1]
                if valid_gap <= 0.0 or valid_gap > float(
                    self.config.max_valid_frame_gap_s
                ):
                    self._clear_sequence_state(clear_results=True)
            self._sampled_frames.append(prepared.sampled_points_mm)
            self._sampled_timestamps.append(float(frame.timestamp))
            self._sampled_frame_indices.append(int(frame.index))
            self._processed += 1
            if (
                self.operation == "enroll"
                and self._collector is not None
                and self._collector.active_pass_id is not None
            ):
                self._pass_frame_count += 1
                self._pass_point_total += int(len(prepared.foreground_points_mm))
                center_x = float(np.median(prepared.foreground_points_mm[:, 0]))
                center_z = float(np.median(prepared.foreground_points_mm[:, 2]))
                if self._pass_first_center_x is None:
                    self._pass_first_center_x = center_x
                    self._pass_first_center_z = center_z
                self._pass_last_center_x = center_x
                self._pass_last_center_z = center_z
                self._pass_center_history.append((center_x, center_z))
                self._pass_last_frame_index = int(frame.index)

            result = self._last_result
            inference_ms = 0.0
            stride = (
                int(self.config.enrollment_stride or self.config.clip_len)
                if self.operation == "enroll"
                else int(self.config.inference_stride)
            )
            should_infer = (
                len(self._sampled_frames) == self.config.clip_len
                and self._processed % max(1, stride) == 0
            )
            if should_infer:
                inference_started = time.perf_counter()
                model_started = time.perf_counter()
                embedding = self._infer()
                model_encode_ms = (time.perf_counter() - model_started) * 1000.0
                gallery_match_ms = 0.0
                if self.operation == "enroll":
                    assert self._collector is not None
                    self._collector.model_record = dict(self._runtime.model_record)
                    timing = self._timing_snapshot_fields()
                    self._collector.add_embedding(
                        embedding=embedding,
                        start_frame=self._sampled_frame_indices[0],
                        end_frame=frame.index,
                        timestamp=frame.timestamp,
                        effective_sampling_fps=timing["effective_sampling_fps"],
                        window_duration_s=timing["window_duration_s"],
                        mean_frame_gap_ms=timing["mean_frame_gap_ms"],
                        max_frame_gap_ms=timing["max_frame_gap_ms"],
                    )
                    result = self._enrollment_progress_result()
                else:
                    match_started = time.perf_counter()
                    result = self._match(embedding, frame)
                    result = self._stabilize(result)
                    gallery_match_ms = (
                        time.perf_counter() - match_started
                    ) * 1000.0
                self._last_result = result
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                self._last_inference_ms = inference_ms
                self._last_model_encode_ms = model_encode_ms
                self._last_gallery_match_ms = gallery_match_ms

            elapsed = time.perf_counter() - loop_started
            if self.operation == "enroll":
                result = self._enrollment_progress_result()
                state = "enrolling"
                message = (
                    f"Recording {result.get('active_pass_id')}: "
                    f"{result['captured_embeddings']} valid embeddings"
                    if self.config.guided_passes
                    else (
                        f"Registering {result['display_name']}: "
                        f"{result['elapsed_s']:.1f}/{result['duration_s']:.1f} s, "
                        f"{result['captured_embeddings']} valid embeddings"
                    )
                )
            else:
                state = (
                    str(result.get("state", "recognized"))
                    if result is not None
                    else "collecting"
                )
                message = (
                    f"{result.get('display_name', 'Unknown')} "
                    f"({float(result.get('similarity', 0.0)):.3f})"
                    if result is not None
                    else f"Collecting gait frames {len(self._sampled_frames)}/{self.config.clip_len}"
                )
            message = self._selection_message(detection, message)
            self._emit(
                PipelineSnapshot(
                    state=state,
                    message=message,
                    timestamp=time.time(),
                    color_bgr=frame.color_bgr,
                    person_points_mm=prepared.foreground_points_mm,
                    person_mask_rgb=prepared.color_mask,
                    detection=detection,
                    result=result,
                    fps=1.0 / max(elapsed, 1e-6),
                    capture_fps=capture_fps,
                    frame_read_ms=frame_read_ms,
                    detection_ms=detection_ms,
                    preprocessing_ms=preprocessing_ms,
                    model_encode_ms=self._last_model_encode_ms,
                    gallery_match_ms=self._last_gallery_match_ms,
                    inference_ms=self._last_inference_ms,
                    buffer_size=len(self._sampled_frames),
                    clip_len=self.config.clip_len,
                    frame_index=frame.index,
                    raw_point_count=int(frame.point_cloud_mm.shape[0] * frame.point_cloud_mm.shape[1]),
                    person_point_count=int(len(prepared.foreground_points_mm)),
                    detected_person_count=int(detection.person_count),
                    dropped_frames=self._dropped,
                    **self._timing_snapshot_fields(),
                    **self._operation_snapshot_fields(),
                    device_status=self._device_status,
                    diagnostics={
                        **prepared.diagnostics,
                        "inference_ran": should_infer,
                        "detector": detection.detector,
                        **self._selection_diagnostics(detection),
                    },
                )
            )

    def _enrollment_ready(self, frame_index: int = -1) -> bool:
        if self.operation != "enroll":
            return True
        if self.config.guided_passes:
            assert self._collector is not None
            if self._collector.active_pass_id is not None:
                return True
            with self._control_lock:
                requested = self._pass_start_requested
                direction = self._requested_direction
            if not requested:
                self._enrollment_person_seen_at = None
                return False
            now = time.monotonic()
            if self._enrollment_person_seen_at is None:
                self._enrollment_person_seen_at = now
            if now - self._enrollment_person_seen_at < float(
                self.config.enrollment_warmup_s
            ):
                return False
            self._collector.start_pass(direction, start_frame=frame_index)
            with self._control_lock:
                self._pass_start_requested = False
            self._processed = 0
            self._clear_sequence_state()
            self._pass_frame_count = 0
            self._pass_point_total = 0
            self._pass_first_center_x = None
            self._pass_last_center_x = None
            self._pass_first_center_z = None
            self._pass_last_center_z = None
            self._pass_center_history = []
            self._pass_last_frame_index = int(frame_index)
            return True
        now = time.monotonic()
        if self._enrollment_person_seen_at is None:
            self._enrollment_person_seen_at = now
        warmup = float(self.config.enrollment_warmup_s)
        if now - self._enrollment_person_seen_at < warmup:
            return False
        if self._enrollment_active_at is None:
            self._enrollment_active_at = now
            assert self._collector is not None
            self._collector.start_pass("automatic", start_frame=frame_index)
            self._processed = 0
            self._clear_sequence_state()
            self._pass_frame_count = 0
            self._pass_point_total = 0
            self._pass_first_center_x = None
            self._pass_last_center_x = None
            self._pass_first_center_z = None
            self._pass_last_center_z = None
            self._pass_center_history = []
            self._pass_last_frame_index = int(frame_index)
        return True

    def _pass_motion_summary(self) -> tuple[str, float, float]:
        if len(self._pass_center_history) < 5:
            return "unknown", 0.0, 0.0
        centers = np.asarray(self._pass_center_history, dtype=np.float64)
        edge = max(2, min(10, len(centers) // 5))
        start = np.median(centers[:edge], axis=0)
        end = np.median(centers[-edge:], axis=0)
        delta_x = float(end[0] - start[0])
        delta_z = float(end[1] - start[1])
        lateral_strength = abs(delta_x) / 150.0
        depth_strength = abs(delta_z) / 250.0
        axis = 0 if lateral_strength >= depth_strength else 1
        delta = delta_x if axis == 0 else delta_z
        steps = np.diff(centers[:, axis])
        direction_sign = 1.0 if delta >= 0 else -1.0
        monotonic_ratio = float(np.mean(steps * direction_sign >= -10.0))
        displacement = float(abs(delta))
        if monotonic_ratio < 0.65:
            return "stationary_or_turning", displacement, monotonic_ratio
        if lateral_strength >= 1.0 and lateral_strength >= depth_strength:
            return (
                "left_to_right" if delta_x > 0 else "right_to_left",
                displacement,
                monotonic_ratio,
            )
        if depth_strength >= 1.0:
            return (
                "away_from_camera" if delta_z > 0 else "toward_camera",
                displacement,
                monotonic_ratio,
            )
        if delta_x >= 150.0:
            return "left_to_right", displacement, monotonic_ratio
        if delta_x <= -150.0:
            return "right_to_left", displacement, monotonic_ratio
        return "stationary_or_turning", displacement, monotonic_ratio

    def _close_guided_pass(self, discard: bool) -> dict[str, Any] | None:
        assert self._collector is not None
        if self._collector.active_pass_id is None:
            with self._control_lock:
                self._pass_start_requested = False
            self._enrollment_person_seen_at = None
            return None
        mean_points = (
            self._pass_point_total / max(1, self._pass_frame_count)
        )
        observed, displacement, monotonic_ratio = self._pass_motion_summary()
        result = self._collector.end_pass(
            end_frame=self._pass_last_frame_index,
            frame_count=self._pass_frame_count,
            mean_person_points=mean_points,
            observed_direction=observed,
            direction_displacement_mm=displacement,
            direction_monotonic_ratio=monotonic_ratio,
            discard=discard,
        )
        self._clear_sequence_state()
        self._processed = 0
        self._enrollment_person_seen_at = None
        self._pass_center_history = []
        return result

    def _process_guided_controls(self) -> bool:
        with self._control_lock:
            end_request = self._pass_end_requested
            finish_request = self._finish_review_requested
            self._pass_end_requested = None
            self._finish_review_requested = False
        if end_request is not None:
            self._close_guided_pass(discard=bool(end_request))
        if finish_request:
            self._close_guided_pass(discard=False)
            self._emit_enrollment_review("user_finished")
            return True
        return False

    def _enrollment_elapsed(self) -> float:
        if self._enrollment_active_at is None:
            return float(self._enrollment_elapsed_offset_s)
        return float(self._enrollment_elapsed_offset_s) + max(
            0.0,
            time.monotonic() - self._enrollment_active_at,
        )

    def _enrollment_expired(self) -> bool:
        return (
            self.operation == "enroll"
            and self._enrollment_active_at is not None
            and self._enrollment_elapsed() >= float(self.config.enrollment_duration_s)
        )

    def _enrollment_progress_result(self) -> dict[str, Any] | None:
        if self.operation != "enroll":
            return None
        request = self._enrollment_request()
        captured = self._collector.count if self._collector is not None else 0
        warmup_remaining = 0.0
        active_pass_id = (
            self._collector.active_pass_id if self._collector is not None else None
        )
        with self._control_lock:
            pass_requested = self._pass_start_requested
            requested_direction = self._requested_direction
        if self.config.guided_passes:
            if active_pass_id is not None:
                progress_state = "enrolling"
            elif pass_requested:
                progress_state = "enrollment_countdown"
                started = self._enrollment_person_seen_at or time.monotonic()
                warmup_remaining = max(
                    0.0,
                    float(self.config.enrollment_warmup_s)
                    - (time.monotonic() - started),
                )
            else:
                progress_state = "enrollment_paused"
        elif self._enrollment_active_at is None:
            progress_state = "enrollment_countdown"
            started = self._enrollment_person_seen_at or time.monotonic()
            warmup_remaining = max(
                0.0,
                float(self.config.enrollment_warmup_s)
                - (time.monotonic() - started),
            )
        else:
            progress_state = "enrolling"
        return {
            "operation": "enroll",
            "state": progress_state,
            "accepted": False,
            "person_id": request.person_id.strip(),
            "display_name": request.display_name.strip(),
            "elapsed_s": self._enrollment_elapsed(),
            "duration_s": float(request.duration_s),
            "warmup_remaining_s": warmup_remaining,
            "captured_embeddings": captured,
            "min_embeddings": int(request.min_embeddings),
            "max_embeddings": int(request.max_embeddings),
            "guided_passes": bool(self.config.guided_passes),
            "active_pass_id": active_pass_id,
            "requested_direction": requested_direction,
            "passes": (
                self._collector.pass_summaries()
                if self._collector is not None
                else []
            ),
            "top_candidates": [],
        }

    def _operation_snapshot_fields(self) -> dict[str, Any]:
        request = self._enrollment_request() if self.operation == "enroll" else None
        return {
            "operation": self.operation,
            "enrollment_elapsed_s": self._enrollment_elapsed(),
            "enrollment_duration_s": float(request.duration_s) if request else 0.0,
            "enrollment_embeddings": (
                self._collector.count if self._collector is not None else 0
            ),
            "enrollment_min_embeddings": (
                int(request.min_embeddings) if request else 0
            ),
        }

    def _emit_enrollment_review(self, reason: str) -> None:
        assert self._collector is not None
        elapsed = self._enrollment_elapsed()
        self._enrollment_elapsed_offset_s = elapsed
        self._enrollment_active_at = None
        self._review_elapsed_s = elapsed
        self._review_ready = True
        self._collector.write_review_manifest(elapsed_s=elapsed)
        passes = self._collector.pass_summaries()
        usable = [
            item
            for item in passes
            if not item.get("discarded") and int(item.get("embedding_count", 0)) > 0
        ]
        result = {
            **(self._enrollment_progress_result() or {}),
            "state": "enrollment_review",
            "accepted": False,
            "passes": passes,
            "usable_passes": len(usable),
            "reason": reason,
        }
        self._last_result = result
        self._emit(
            PipelineSnapshot(
                state="enrollment_review",
                message=(
                    f"Capture complete: review {len(passes)} passes before Gallery commit"
                ),
                timestamp=time.time(),
                result=result,
                buffer_size=0,
                clip_len=self.config.clip_len,
                dropped_frames=self._dropped,
                **self._operation_snapshot_fields(),
                device_status=self._device_status,
                diagnostics={
                    "completion_reason": reason,
                    "session_id": self._session_id,
                },
            )
        )

    def _emit_enrollment_terminal(self, reason: str) -> None:
        assert self._collector is not None
        if self._stop.is_set():
            return
        if self._collector.active_pass_id is not None:
            self._close_guided_pass(discard=False)
        elapsed = self._enrollment_elapsed()
        try:
            result = self._collector.finalize(
                repository=self.controller.repository,
                elapsed_s=elapsed,
            )
            state = "enrolled"
            message = (
                f"Enrollment complete: {result['display_name']} | "
                f"stored {result['stored_embeddings']} embeddings"
            )
        except Exception as exc:
            self._collector.record_failure(
                error=exc,
                elapsed_s=elapsed,
                reason=reason,
            )
            result = {
                **(self._enrollment_progress_result() or {}),
                "state": "enrollment_failed",
                "accepted": False,
                "error": str(exc),
                "reason": reason,
            }
            state = "enrollment_failed"
            message = f"Enrollment not saved: {exc}"
        self._last_result = result
        self._emit(
            PipelineSnapshot(
                state=state,
                message=message,
                timestamp=time.time(),
                result=result,
                buffer_size=len(self._sampled_frames),
                clip_len=self.config.clip_len,
                dropped_frames=self._dropped,
                **self._operation_snapshot_fields(),
                device_status=self._device_status,
                diagnostics={
                    "completion_reason": reason,
                    "session_id": self._session_id,
                },
            )
        )

    def _infer(self) -> np.ndarray:
        assert self._runtime is not None
        inputs = torch.from_numpy(np.stack(self._sampled_frames).astype(np.float32))
        embedding = self._runtime.adapter.encode_tensor(inputs)[0].numpy().astype(np.float32)
        if embedding.size != int(self._runtime.model_record["embedding_dim"]):
            self._runtime.model_record = self._runtime._build_model_record(
                embedding_dim=int(embedding.size)
            )
        return embedding

    def _match(self, embedding: np.ndarray, frame: Any) -> dict[str, Any]:
        assert self._runtime is not None
        assert self._source_path is not None
        batch = EmbeddingBatch(
            embeddings=embedding.reshape(1, -1),
            windows=[
                {
                    "window_index": self._processed,
                    "start_frame_number": self._sampled_frame_indices[0],
                    "end_frame_number": frame.index,
                }
            ],
            source=self._source_path,
            source_metadata={
                "source_fingerprint": self._source_fingerprint,
                "source_type": self.config.source_mode,
                "frame_index": frame.index,
                "clip_len": self.config.clip_len,
                "preprocessing_profile_id": self.config.preprocessing_profile_id,
                "effective_sampling_fps": self._sampling_metrics()[0],
                "window_duration_s": self._sampling_metrics()[1],
                "mean_frame_gap_ms": self._sampling_metrics()[2],
                "max_frame_gap_ms": self._sampling_metrics()[3],
            },
            model=self._runtime.model_record,
        )
        try:
            return RecognitionService(self.controller.repository).recognize(
                batch,
                top_k_per_identity=self.config.top_k_per_identity,
                threshold=self.config.threshold,
                min_margin=self.config.min_margin,
                allow_source_overlap=False,
            )
        except ValueError as exc:
            if "No compatible gallery embeddings" not in str(exc):
                raise
            return {
                "operation": "recognize",
                "state": "no_gallery",
                "accepted": False,
                "person_id": None,
                "display_name": "Gallery empty",
                "similarity": 0.0,
                "top_candidates": [],
                "warning": str(exc),
                "model_key": self._runtime.model_record["model_key"],
            }

    def _stabilize(self, result: dict[str, Any]) -> dict[str, Any]:
        self._recent_results.append(result)
        identities = [
            str(item.get("person_id"))
            for item in self._recent_results
            if item.get("accepted") and item.get("person_id") is not None
        ]
        stable_count = 0
        stable_identity = None
        if identities:
            stable_identity, stable_count = Counter(identities).most_common(1)[0]
        required = min(
            max(1, self.config.stability_windows),
            self._recent_results.maxlen or 1,
        )
        output = dict(result)
        output["stability_count"] = stable_count
        output["stability_required"] = required
        output["stable_identity"] = stable_identity if stable_count >= required else None
        output["stable"] = bool(
            result.get("accepted")
            and stable_count >= required
            and str(result.get("person_id")) == stable_identity
        )
        if result.get("accepted") and stable_count < required:
            output["state"] = "accumulating"
        elif output["stable"]:
            output["state"] = "stable"
        return output
