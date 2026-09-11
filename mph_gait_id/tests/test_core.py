from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.database import GalleryRepository
from mph_gait_id.i18n import I18n
from mph_gait_id.realtime.benchmark import LiveBenchmarkRecorder
from mph_gait_id.realtime.enrollment import (
    EnrollmentRequest,
    LiveEnrollmentCollector,
)
from mph_gait_id.realtime.pipeline import (
    RealtimeConfig,
    RealtimePipeline,
)
from mph_gait_id.realtime.preprocessing import filter_person_pointcloud
from mph_gait_id.realtime.types import (
    Detection,
    PipelineSnapshot,
    SensorFrame,
)
from mph_gait_id.realtime.ui_page import RealtimePage
from mph_gait_id.runtime import EmbeddingBatch
from mph_gait_id.services import RecognitionService, RegistrationService
import mph_gait_id.app as cli_app


class FakeRepository:
    def __init__(self, gallery: list[dict[str, object]]) -> None:
        self.gallery = gallery

    def initialize(self) -> None:
        return None

    def load_gallery(self, _model_key: str) -> list[dict[str, object]]:
        return self.gallery


def gallery_item(
    person_id: str,
    display_name: str,
    embedding: list[float],
) -> dict[str, object]:
    return {
        "person_id": person_id,
        "display_name": display_name,
        "embedding": np.asarray(embedding, dtype=np.float32),
        "source_path": f"/gallery/{person_id}",
        "source_fingerprint": f"fingerprint-{person_id}",
    }


def model_record(model_key: str = "test-model", processing: str = "person_foreground_pointcloud_v1") -> dict[str, object]:
    return {
        "model_key": model_key,
        "bundle_id": "pointnet_tmax_fixed_special5_seed0_split0",
        "method_key": "pc_v1",
        "display_name": "PointNet-TMax",
        "architecture": "pc_v1",
        "input_type": "pointcloud",
        "input_mode": "raw_pointcloud",
        "fold": 0,
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "a" * 64,
        "embedding_dim": 2,
        "compatibility": {
            "clip_len": 15,
            "preprocessing_profile_id": processing,
        },
    }


def embedding_batch(source: Path, fingerprint: str, model: dict[str, object]) -> EmbeddingBatch:
    return EmbeddingBatch(
        embeddings=np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
        windows=[{"window_index": 0}, {"window_index": 1}],
        source=source,
        source_metadata={"source_fingerprint": fingerprint},
        model=model,
    )


class RecognitionPolicyTest(unittest.TestCase):
    def test_low_margin_is_candidate_but_not_accepted(self) -> None:
        repository = FakeRepository(
            [
                gallery_item("P001", "Alice", [1.0, 0.0]),
                gallery_item("P002", "Bob", [0.999, 0.0447]),
            ]
        )
        batch = EmbeddingBatch(
            embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            windows=[{"window_index": 0}],
            source=Path("/probe/new-sequence"),
            source_metadata={"source_fingerprint": "probe-fingerprint"},
            model={
                "model_key": "test-model",
                "input_type": "pointcloud",
                "input_mode": "raw_pointcloud",
            },
        )
        result = RecognitionService(repository).recognize(
            batch,
            threshold=0.5,
            min_margin=0.03,
        )
        self.assertEqual(result["state"], "low_confidence")
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["person_id"])
        self.assertEqual(result["candidate_person_id"], "P001")

    def test_rejected_candidates_do_not_become_stable(self) -> None:
        pipeline = RealtimePipeline(
            controller=object(),
            config=RealtimeConfig(bundle_id="unused", stability_windows=3),
        )
        for _ in range(3):
            result = pipeline._stabilize(
                {
                    "state": "low_confidence",
                    "accepted": False,
                    "person_id": None,
                    "candidate_person_id": "P001",
                }
            )
        self.assertFalse(result["stable"])
        self.assertIsNone(result["stable_identity"])


class RealtimeTimingTest(unittest.TestCase):
    def test_effective_sampling_metrics_use_valid_frame_timestamps(self) -> None:
        pipeline = RealtimePipeline(
            controller=object(),
            config=RealtimeConfig(bundle_id="unused", clip_len=3),
        )
        pipeline._sampled_timestamps.extend([10.0, 10.1, 10.2])
        fps, duration, mean_gap, max_gap = pipeline._sampling_metrics()
        self.assertAlmostEqual(fps, 10.0)
        self.assertAlmostEqual(duration, 0.2)
        self.assertAlmostEqual(mean_gap, 100.0)
        self.assertAlmostEqual(max_gap, 100.0)

    def test_non_monotonic_timestamps_are_rejected(self) -> None:
        pipeline = RealtimePipeline(
            controller=object(),
            config=RealtimeConfig(bundle_id="unused", clip_len=3),
        )
        pipeline._sampled_timestamps.extend([10.0, 10.1, 9.9])
        self.assertEqual(pipeline._sampling_metrics(), (0.0, 0.0, 0.0, 0.0))


class LiveBenchmarkTest(unittest.TestCase):
    def test_recorder_keeps_numeric_telemetry_without_sensor_arrays(self) -> None:
        recorder = LiveBenchmarkRecorder(
            warmup_seconds=0,
            duration_seconds=60,
            metadata={"bundle_id": "mph", "model_display_name": "MPH-Gait ID"},
            posture="stationary",
        )
        recorder.accept(
            PipelineSnapshot(
                state="recognized",
                message="test",
                timestamp=1.0,
                frame_index=1,
                person_point_count=1024,
                frame_read_ms=12.0,
                detection_ms=20.0,
                preprocessing_ms=3.0,
                model_encode_ms=8.0,
                gallery_match_ms=2.0,
                inference_ms=10.0,
                capture_fps=25.0,
                effective_sampling_fps=24.0,
                mean_frame_gap_ms=41.0,
                max_frame_gap_ms=45.0,
                result={"accepted": False, "model_key": "mph-key"},
                diagnostics={"inference_ran": True},
            )
        )
        recorder.accept(
            PipelineSnapshot(
                state="waiting_person",
                message="test",
                timestamp=2.0,
                frame_index=2,
                frame_read_ms=13.0,
                detection_ms=21.0,
                capture_fps=24.0,
            )
        )
        recorder.cancel()
        report = recorder.report()

        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["summary"]["frames"], 2)
        self.assertEqual(report["summary"]["valid_frames"], 1)
        self.assertEqual(report["summary"]["inference_count"], 1)
        self.assertEqual(report["latency_ms_and_rates"]["model_encode_ms"]["p50"], 8.0)
        self.assertEqual(report["scope"], "stationary_efficiency_only")
        self.assertTrue(report["privacy"]["numeric_telemetry_only"])
        self.assertFalse(report["privacy"]["rgb_saved"])
        self.assertNotIn("color_bgr", str(report))
        self.assertNotIn("person_points_mm", str(report))

    def test_metric_sink_receives_every_snapshot_even_when_ui_queue_drops(self) -> None:
        received: list[int] = []
        pipeline = RealtimePipeline(
            controller=object(),
            config=RealtimeConfig(bundle_id="unused"),
            metric_sink=lambda snapshot: received.append(snapshot.frame_index),
        )
        for index in range(6):
            pipeline._emit(
                PipelineSnapshot(
                    state="collecting",
                    message="test",
                    timestamp=float(index),
                    frame_index=index,
                )
            )

        self.assertEqual(received, list(range(6)))
        latest = pipeline.poll_latest()
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.frame_index, 5)


class RealtimeEnrollmentPolicyTest(unittest.TestCase):
    def test_primary_person_is_selected_by_default_without_rejecting_frame(self) -> None:
        pipeline = RealtimePipeline(
            controller=object(),
            config=RealtimeConfig(bundle_id="unused"),
        )
        detection = Detection(
            bbox_xyxy=(10, 20, 100, 220),
            confidence=0.9,
            person_count=2,
        )
        self.assertFalse(pipeline.config.reject_multiple_people)
        self.assertIn(
            "Primary person selected from 2 detections",
            pipeline._selection_message(detection, "Collecting"),
        )
        diagnostics = pipeline._selection_diagnostics(detection)
        self.assertTrue(diagnostics["primary_person_selected"])
        self.assertEqual(diagnostics["multi_person_policy"], "primary_salience")

    def test_multi_person_enrollment_switch_maps_to_rejection_policy(self) -> None:
        self.assertFalse(
            RealtimePage._should_reject_multiple_people("enroll", True, False)
        )
        self.assertTrue(
            RealtimePage._should_reject_multiple_people("enroll", False, False)
        )
        self.assertTrue(
            RealtimePage._should_reject_multiple_people("recognize", True, True)
        )

    def test_review_can_resume_same_collector_and_session(self) -> None:
        pipeline = RealtimePipeline(
            controller=object(),
            config=RealtimeConfig(
                bundle_id="unused",
                operation="enroll",
                enrollment_person_id="P001",
                enrollment_display_name="Test person",
            ),
        )
        collector = object()
        pipeline._collector = collector  # type: ignore[assignment]
        pipeline._review_ready = True
        pipeline._last_result = {"state": "enrollment_review"}
        pipeline._sampled_frames.append(
            np.zeros((1024, 3), dtype=np.float32)
        )
        started: list[bool] = []
        pipeline.start = lambda: started.append(True)  # type: ignore[method-assign]

        pipeline.resume_enrollment_capture()

        self.assertFalse(pipeline._review_ready)
        self.assertIs(pipeline._collector, collector)
        self.assertEqual(started, [True])
        self.assertEqual(len(pipeline._sampled_frames), 0)


class FakeVar:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class RealtimeUiStartupTest(unittest.TestCase):
    def test_azure_detector_values_follow_current_locale(self) -> None:
        for locale_name in ("zh_TW", "en"):
            page = object.__new__(RealtimePage)
            page.i18n = I18n(locale_name)
            page._build_option_labels()

            self.assertEqual(list(page.pass_direction_labels.values()), ["front_facing"])

            labels = page._detector_values_for_source("azure_kinect")

            self.assertEqual(
                [page.detector_labels[label] for label in labels],
                ["yolo_seg", "yolo", "yolo_sam"],
            )

    def test_azure_start_does_not_probe_device_on_tk_callback(self) -> None:
        page = object.__new__(RealtimePage)
        page._refresh_readiness = lambda: None
        page._readiness_blocker = ""
        page._pending_review_result = None
        page.pipeline = None
        page._operation = lambda: "enroll"  # type: ignore[method-assign]
        page.operation_labels = {"Enrollment": "enroll"}
        page.source_labels = {"Azure": "azure_kinect"}
        page.source_var = FakeVar("Azure")
        page.status_var = FakeVar()
        page.result_var = FakeVar()
        page.result_detail_var = FakeVar()
        page.device_status = object()
        page.device_var_text = FakeVar()
        page.update_idletasks = lambda: None  # type: ignore[method-assign]
        page.detector_labels = {"Replay detector": "replay_depth"}
        page.detector_var = FakeVar("Replay detector")
        page.yolo_weights_var = FakeVar("")
        page.sam_checkpoint_var = FakeVar("")
        page.replay_path_var = FakeVar(".")
        page._bundle_id = lambda required=True: "bundle"  # type: ignore[method-assign]
        page.controller = SimpleNamespace(
            model_store=SimpleNamespace(
                get=lambda _bundle_id: SimpleNamespace(data={"num_points": 1024})
            )
        )
        page.enrollment_person_id_var = FakeVar("")
        page.enrollment_name_var = FakeVar("")
        page.enrollment_min_var = FakeVar(2)
        page.enrollment_max_var = FakeVar(5)
        rejection: list[tuple[str, str]] = []
        page._reject_start = (  # type: ignore[method-assign]
            lambda title, detail: rejection.append((title, detail)) or False
        )

        def fail_if_probed(*_args: object, **_kwargs: object) -> None:
            self.fail("Azure Kinect probe ran synchronously on the Tk callback")

        page._check_device = fail_if_probed  # type: ignore[method-assign]

        self.assertFalse(page._start_impl())
        self.assertIsNone(page.device_status)
        self.assertIn("background", str(page.device_var_text.get()))
        self.assertEqual(rejection[0][0], "Enrollment identity required")


class PointCloudOverlayTest(unittest.TestCase):
    def test_rgb_overlay_mask_uses_depth_filtered_pointcloud(self) -> None:
        cloud = np.zeros((3, 3, 3), dtype=np.float32)
        cloud[..., 2] = 2000.0
        cloud[0, 0, 2] = 9000.0
        frame = SensorFrame(
            index=1,
            timestamp=1.0,
            color_bgr=np.zeros((3, 3, 3), dtype=np.uint8),
            point_cloud_mm=cloud,
            point_cloud_color_aligned=True,
        )
        result = filter_person_pointcloud(
            frame,
            Detection(bbox_xyxy=(0, 0, 3, 3), confidence=0.9),
            num_points=4,
            near_mm=400.0,
            far_mm=8000.0,
            min_person_points=1,
        )

        self.assertEqual(result.color_mask.shape, (3, 3))
        self.assertFalse(bool(result.color_mask[0, 0]))
        self.assertTrue(bool(result.color_mask[1, 1]))


class ThresholdConfigTest(unittest.TestCase):
    def test_threshold_is_selected_by_model_and_clip_length(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            controller = GaitApplicationController(
                database_path=root / "gallery.sqlite3",
                output_root=root / "outputs",
            )
            controller.config["recognition"]["provisional_thresholds"]["pc_v1"] = {
                "15": 0.51,
                "30": 0.61,
            }
            value_t15 = controller.provisional_threshold(
                "pointnet_tmax_fixed_special5_seed0_split0",
                clip_len=15,
            )
            value_t30 = controller.provisional_threshold(
                "pointnet_tmax_fixed_special5_seed0_split0",
                clip_len=30,
            )
            self.assertEqual(value_t15, 0.51)
            self.assertEqual(value_t30, 0.61)
            self.assertEqual(
                controller.processing_version_id(),
                "person_foreground_pointcloud_v1",
            )


class CompatibilityAndGalleryTest(unittest.TestCase):
    def test_cli_window_size_and_drop_first_are_bound_to_runtime_key(self) -> None:
        captured: dict[str, object] = {}

        class FakeRuntime:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        args = SimpleNamespace(
            bundle="pointnet_tmax_fixed_special5_seed0_split0",
            device="cpu",
            batch_size=1,
            num_workers=0,
            window_size=30,
            drop_first_frames=7,
        )
        config = {
            "runtime": {},
            "realtime": {"processing_version_id": "person_foreground_pointcloud_v1"},
        }
        with patch.object(cli_app, "SystemModelRuntime", FakeRuntime):
            cli_app._runtime(args, config, object())
        self.assertEqual(captured["runtime_clip_len"], 30)
        self.assertEqual(captured["runtime_drop_first_frames"], 7)

    def test_gallery_model_keys_are_filtered_by_processing_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repository = GalleryRepository(Path(raw) / "gallery.sqlite3")
            repository.initialize()
            repository.upsert_model(model_record("model-v1", "processing-v1"))
            repository.upsert_model(model_record("model-v2", "processing-v2"))
            keys = repository.model_keys_for_bundle(
                "pointnet_tmax_fixed_special5_seed0_split0",
                clip_len=15,
                preprocessing_profile_id="processing-v2",
            )
            self.assertEqual(keys, ["model-v2"])

    def test_public_model_rename_preserves_model_key_and_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repository = GalleryRepository(Path(raw) / "gallery.sqlite3")
            repository.initialize()
            previous = model_record("mph-model")
            previous["bundle_id"] = "mph_gait_fixed_special5_seed0_split0"
            previous["method_key"] = "mph_gait"
            previous["display_name"] = "MPH-Gait Fixed-Special5 seed0 split0"
            repository.upsert_model(previous)

            renamed = dict(previous)
            renamed["display_name"] = "MPH-Gait ID"
            repository.upsert_model(renamed)

            with repository.connect() as connection:
                row = connection.execute(
                    "SELECT model_key, display_name, config_json FROM model_versions"
                ).fetchone()
            self.assertEqual(row["model_key"], "mph-model")
            self.assertEqual(row["display_name"], "MPH-Gait ID")
            self.assertEqual(
                json.loads(row["config_json"])["display_name"],
                "MPH-Gait ID",
            )


class EnrollmentIntegrityTest(unittest.TestCase):
    def test_multi_source_failure_rolls_back_every_gallery_write(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            repository = GalleryRepository(root / "gallery.sqlite3")
            repository.initialize()
            original_add = repository.add_embeddings
            calls = 0

            def fail_second(**kwargs: object) -> list[int]:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated write failure")
                return original_add(**kwargs)

            repository.add_embeddings = fail_second  # type: ignore[method-assign]
            batches = [
                embedding_batch(root / "source-1", "fingerprint-1", model_record()),
                embedding_batch(root / "source-2", "fingerprint-2", model_record()),
            ]
            with self.assertRaises(RuntimeError):
                RegistrationService(repository).enroll_many(
                    batches,
                    person_id="P001",
                    display_name="Test person",
                    min_embeddings=1,
                    max_embeddings_per_source=2,
                )
            with repository.connect() as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM persons").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM gallery_embeddings").fetchone()[0], 0)

    def test_direction_warning_can_be_manually_committed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            collector = LiveEnrollmentCollector(
                request=EnrollmentRequest(
                    person_id="P001",
                    display_name="Test person",
                    min_embeddings=1,
                    max_embeddings=2,
                ),
                model_record=model_record(),
                source_path=root / "live-source",
                source_fingerprint="fingerprint-live",
                source_kind="azure_kinect_live",
                session_id="session-test",
                session_dir=root / "session",
                source_metadata={},
            )
            collector.start_pass("toward_camera", start_frame=1)
            collector.add_embedding(np.asarray([1.0, 0.0]), 1, 15, 10.0)
            result = collector.end_pass(
                end_frame=20,
                frame_count=20,
                mean_person_points=1000.0,
                observed_direction="stationary_or_turning",
            )
            self.assertFalse(result["quality_accepted"])
            self.assertTrue(result["selectable"])
            repository = GalleryRepository(root / "gallery.sqlite3")
            repository.initialize()
            committed = collector.finalize(
                repository,
                elapsed_s=5.0,
                selected_pass_ids=["pass_001"],
            )
            self.assertEqual(
                committed["manual_quality_override_pass_ids"],
                ["pass_001"],
            )

    def test_front_facing_movement_warning_remains_manually_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            collector = LiveEnrollmentCollector(
                request=EnrollmentRequest(
                    person_id="P001",
                    display_name="Test person",
                    min_embeddings=1,
                    max_embeddings=2,
                ),
                model_record=model_record(),
                source_path=root / "live-source",
                source_fingerprint="fingerprint-front-facing",
                source_kind="azure_kinect_live",
                session_id="session-front-facing",
                session_dir=root / "session",
                source_metadata={},
            )
            collector.start_pass("front_facing", start_frame=1)
            collector.add_embedding(np.asarray([1.0, 0.0]), 1, 15, 10.0)
            result = collector.end_pass(
                end_frame=20,
                frame_count=20,
                mean_person_points=1000.0,
                observed_direction="stationary_or_turning",
            )
            self.assertFalse(result["quality_accepted"])
            self.assertTrue(result["selectable"])
            repository = GalleryRepository(root / "gallery.sqlite3")
            repository.initialize()
            collector.finalize(
                repository,
                elapsed_s=5.0,
                selected_pass_ids=["pass_001"],
            )
            with repository.connect() as connection:
                stored_direction = connection.execute(
                    "SELECT direction FROM gallery_embeddings LIMIT 1"
                ).fetchone()[0]
            self.assertEqual(stored_direction, "front_facing")


class I18nTest(unittest.TestCase):
    def test_language_choice_is_persisted_outside_project_config(self) -> None:
        previous = os.environ.get("MPH_GAIT_ID_SETTINGS")
        try:
            with tempfile.TemporaryDirectory() as raw:
                settings = Path(raw) / "settings.json"
                os.environ["MPH_GAIT_ID_SETTINGS"] = str(settings)
                first = I18n("zh_TW")
                self.assertEqual(first.tr("app.title"), "MPH-Gait ID 點雲步態身分辨識系統")
                first.set_locale("en")
                second = I18n("auto")
                self.assertEqual(second.locale, "en")
                self.assertEqual(
                    second.tr("app.title"),
                    "MPH-Gait ID",
                )
        finally:
            if previous is None:
                os.environ.pop("MPH_GAIT_ID_SETTINGS", None)
            else:
                os.environ["MPH_GAIT_ID_SETTINGS"] = previous


if __name__ == "__main__":
    unittest.main()
