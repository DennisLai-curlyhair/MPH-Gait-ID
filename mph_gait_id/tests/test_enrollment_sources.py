from __future__ import annotations

import json
from pathlib import Path
import queue
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mph_gait_id.database import GalleryRepository
from mph_gait_id.enrollment_sources import EnrollmentSourceLibrary, ForegroundCapture
from mph_gait_id.gallery_lifecycle import GalleryLifecycle
from mph_gait_id.realtime.enrollment import EnrollmentRequest, LiveEnrollmentCollector
from mph_gait_id.realtime.pipeline import RealtimeConfig, RealtimePipeline
from mph_gait_id.realtime.types import Detection, SensorFrame
from mph_gait_id.tests.test_core import model_record


class EnrollmentSourcesTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repository = GalleryRepository(self.root / "gallery.sqlite3")
        self.library = EnrollmentSourceLibrary(self.repository)
        self.library.initialize()

    def capture(self, **limits):
        capture = ForegroundCapture(self.library, min_free_bytes=0, **limits)
        def cleanup():
            try:
                capture.abandon()
            except RuntimeError:
                pass
        self.addCleanup(cleanup)
        return capture

    def collector(self, capture=None, **kwargs):
        return LiveEnrollmentCollector(
            request=EnrollmentRequest("P1", "Alice", min_embeddings=1, max_embeddings=1,
                                      save_foreground=capture is not None),
            model_record=model_record(), source_path=self.root / "live-source",
            source_fingerprint="fingerprint-live", source_kind="azure_kinect_live",
            session_id="session-test", session_dir=self.root / "session",
            source_metadata={}, foreground_capture=capture, **kwargs)

    def add_pass(self, collector, discard=False):
        pass_id = collector.start_pass("automatic", start_frame=10)["pass_id"]
        clouds = []
        for i in range(2):
            points = np.arange((1300 + i) * 3, dtype=np.float32).reshape(-1, 3) + 1000
            clouds.append(points.copy())
            collector.add_foreground(points, frame_index=10 + i, timestamp=100 + i / 15,
                                     segment_start=i == 0, metadata={"quality": {"count": len(points)}})
            points[:] = 0  # Native camera buffers may be reused immediately.
        collector.add_embedding(np.array([1., 0.], dtype=np.float32), 10, 11, 100.1)
        collector.end_pass(11, 2, 1300.5, "unknown", discard=discard)
        return pass_id, clouds

    def commit(self):
        capture = self.capture()
        collector = self.collector(capture)
        pass_id, clouds = self.add_pass(collector)
        result = collector.finalize(self.repository, 10., [pass_id])
        return result["foreground_source_id"], clouds

    def test_database_names_with_same_stem_have_isolated_source_roots(self):
        other = EnrollmentSourceLibrary(GalleryRepository(self.root / "gallery.db"))
        self.assertNotEqual(other.root, self.library.root)

    def test_default_off_does_not_write_foreground(self):
        collector = self.collector()
        pass_id, _ = self.add_pass(collector)
        result = collector.finalize(self.repository, 10., [pass_id])
        self.assertNotIn("foreground_source_id", result)
        self.assertFalse(self.library.root.exists())
        self.assertEqual(self.library.list_sources(), [])

    def test_committed_clouds_preserve_counts_metric_coordinates_and_hashes(self):
        source_id, clouds = self.commit()
        manifest = self.library.manifest(source_id)
        self.assertEqual(manifest["units"], "mm")
        self.assertFalse(manifest["centered"])
        self.assertFalse(manifest["scale_normalized"])
        self.assertFalse(manifest["raw_sensor_frames_saved"])
        for frame, original in zip(manifest["frames"], clouds):
            np.testing.assert_array_equal(self.library.load_frame(source_id, frame), original)
        self.assertEqual([f["points"] for f in manifest["frames"]], [1300, 1301])
        self.assertEqual(len(list(self.library.root.rglob("*.npy"))), 2)
        row = self.library.list_sources()[0]
        self.assertEqual(row["frame_count"], 2)
        self.assertEqual(row["pass_count"], 1)
        self.assertGreater(row["size_bytes"], sum(x.nbytes for x in clouds))
        with self.repository.connect() as con:
            metadata = json.loads(con.execute("SELECT metadata_json FROM gallery_embeddings").fetchone()[0])
        self.assertEqual(metadata["source_metadata"]["foreground_source_id"], source_id)

    def test_resume_keeps_passes_and_only_selected_passes_are_published(self):
        capture = self.capture()
        collector = self.collector(capture)
        self.add_pass(collector)
        collector.write_review_manifest(10.)
        self.assertIsNone(capture.thread)
        selected, clouds = self.add_pass(collector)
        self.add_pass(collector, discard=True)
        result = collector.finalize(self.repository, 20., [selected])
        manifest = self.library.manifest(result["foreground_source_id"])
        self.assertEqual({f["pass_id"] for f in manifest["frames"]}, {selected})
        self.assertEqual(len(list(self.library.root.rglob("*.npy"))), len(clouds))
        self.assertFalse(capture.stage.exists())

    def test_abandon_cleans_frames_and_releases_writer(self):
        capture = self.capture()
        collector = self.collector(capture)
        self.add_pass(collector)
        collector.write_review_manifest(10.)
        collector.write_abandoned_manifest(10.)
        self.assertFalse(capture.stage.exists())
        self.assertEqual(self.library.list_sources(), [])
        self.capture().abandon()

    def test_source_failure_rolls_back_gallery_and_retry_works(self):
        capture = self.capture()
        collector = self.collector(capture)
        pass_id, _ = self.add_pass(collector)
        with self.repository.connect() as con:
            con.execute("CREATE TRIGGER fail_source BEFORE INSERT ON enrollment_sources "
                        "BEGIN SELECT RAISE(ABORT, 'source blocked'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "source blocked"):
            collector.finalize(self.repository, 10., [pass_id])
        with self.repository.connect() as con:
            for table in ("persons", "gallery_embeddings", "enrollment_sources", "gallery_transfer_aliases"):
                self.assertEqual(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
            con.execute("DROP TRIGGER fail_source")
        self.assertFalse(self.library.record_dir(capture.source_id).exists())
        self.assertTrue(capture.stage.exists())
        collector.finalize(self.repository, 10., [pass_id])
        self.assertEqual(len(self.library.list_sources()), 1)

    def test_gallery_failure_rolls_back_prepared_source(self):
        capture = self.capture()
        collector = self.collector(capture)
        pass_id, _ = self.add_pass(collector)
        with self.repository.connect() as con:
            con.execute("CREATE TRIGGER fail_embedding BEFORE INSERT ON gallery_embeddings "
                        "BEGIN SELECT RAISE(ABORT, 'embedding blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            collector.finalize(self.repository, 10., [pass_id])
        self.assertEqual(self.library.list_sources(), [])
        self.assertFalse(self.library.record_dir(capture.source_id).exists())

    def test_limits_and_write_failure_prevent_registration(self):
        for limits, message in [({"session_limit_bytes": 1}, "per-session"),
                                ({"library_limit_bytes": 1}, "library storage")]:
            with self.subTest(limits=limits):
                capture = self.capture(**limits)
                capture.append("pass_001", np.ones((10, 3)), frame_index=1, timestamp=1.,
                               segment_start=True, metadata={})
                with self.assertRaisesRegex(RuntimeError, message):
                    capture.pause()
                capture.abandon()
                self.assertFalse(capture.stage.exists())

    def test_disk_full_is_reported(self):
        capture = self.capture()
        with patch("mph_gait_id.enrollment_sources.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
            capture.append("pass_001", np.ones((10, 3)), frame_index=1, timestamp=1.,
                           segment_start=True, metadata={})
            with self.assertRaisesRegex(RuntimeError, "free disk"):
                capture.pause()
        self.assertEqual(self.library.list_sources(), [])

    def test_queue_full_does_not_silently_drop_frames(self):
        capture = self.capture()
        with patch.object(capture.queue, "put_nowait", side_effect=queue.Full):
            with self.assertRaisesRegex(RuntimeError, "queue is full"):
                capture.append("pass_001", np.ones((10, 3)), frame_index=1, timestamp=1.,
                               segment_start=True, metadata={})

    def test_single_writer_blocks_cleanup_and_concurrent_sessions(self):
        self.capture()
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.capture()
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.library.cleanup_uncommitted()

    def test_tamper_and_traversal_are_rejected(self):
        source_id, _ = self.commit()
        manifest = self.library.manifest(source_id)
        frame = manifest["frames"][0]
        with self.assertRaises(ValueError):
            self.library.load_frame(source_id, {**frame, "file": "../outside.npy"})
        (self.library.record_dir(source_id) / frame["file"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.library.load_frame(source_id, frame)
        (self.library.record_dir(source_id) / "manifest.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.library.manifest(source_id)

    def test_source_delete_preserves_embeddings(self):
        source_id, _ = self.commit()
        before = self.repository.load_gallery("test-model")
        self.library.delete(source_id)
        after = self.repository.load_gallery("test-model")
        self.assertEqual(len(before), len(after))
        np.testing.assert_array_equal(before[0]["embedding"], after[0]["embedding"])
        self.assertFalse(self.library.record_dir(source_id).exists())
        self.assertEqual(self.library.list_sources(), [])

    def test_person_rename_delete_and_id_reuse_do_not_rebind_sources(self):
        source_id, _ = self.commit()
        edits = GalleryLifecycle(self.repository)
        edits.apply(edits.preview("P1"), "rename", "Alice updated")
        self.assertEqual(self.library.list_sources()[0]["current_name"], "Alice updated")
        edits.apply(edits.preview("P1"), "delete_person")
        self.repository.upsert_person("P1", "Bob")
        row = self.library.list_sources()[0]
        self.assertIsNone(row["current_person_id"])
        self.assertEqual(row["captured_name"], "Alice")
        self.assertEqual(self.library.manifest(source_id)["captured_name"], "Alice")

    def test_cleanup_recovers_interrupted_delete_and_removes_only_orphans(self):
        source_id, _ = self.commit()
        directory = self.library.record_dir(source_id)
        trash = self.library.root / "trash" / source_id
        trash.parent.mkdir()
        directory.rename(trash)
        orphan = self.library.root / "staging" / "interrupted"
        orphan.mkdir(parents=True)
        (orphan / "private.npy").write_bytes(b"partial")
        self.library.cleanup_uncommitted()
        self.assertTrue(directory.exists())
        self.assertFalse(orphan.exists())
        self.library.manifest(source_id)

    def test_delete_db_failure_restores_source_files(self):
        source_id, _ = self.commit()
        with self.repository.connect() as con:
            con.execute("CREATE TRIGGER prevent_source_delete BEFORE DELETE ON enrollment_sources "
                        "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.library.delete(source_id)
        self.library.manifest(source_id)

    def test_precommit_report_failure_leaves_review_retryable(self):
        capture = self.capture()
        collector = self.collector(capture)
        pass_id, _ = self.add_pass(collector)
        with patch.object(collector, "_write_json", side_effect=OSError("report disk")):
            with self.assertRaises(OSError):
                collector.finalize(self.repository, 10., [pass_id])
        self.assertFalse(self.library.record_dir(capture.source_id).exists())
        collector.finalize(self.repository, 10., [pass_id])
        self.assertEqual(len(self.library.list_sources()), 1)

    def test_postcommit_report_failure_is_warning_not_duplicate_retry(self):
        capture = self.capture()
        collector = self.collector(capture)
        pass_id, _ = self.add_pass(collector)
        original = collector._write_json
        def fail_result(name, value):
            if name == "registration_result.json":
                raise OSError("report disk")
            original(name, value)
        with patch.object(collector, "_write_json", side_effect=fail_result):
            result = collector.finalize(self.repository, 10., [pass_id])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["report_warning"], "report disk")
        self.assertEqual(len(self.library.list_sources()), 1)

    def test_clock_and_frame_counter_resets_are_marked_as_segment_boundaries(self):
        capture = self.capture()
        for index, timestamp in [(10, 100.), (11, 100.1), (0, 100.2), (1, 99.)]:
            capture.append("pass_001", np.ones((10, 3)), frame_index=index,
                           timestamp=timestamp, segment_start=False, metadata={})
        capture.pause()
        self.assertEqual([f["segment_start"] for f in capture.frames], [True, False, True, True])

    def test_capture_error_cleans_temp_sources_without_gallery_writes(self):
        capture = self.capture()
        collector = self.collector(capture)
        self.add_pass(collector)
        controller = SimpleNamespace(repository=self.repository)
        pipeline = RealtimePipeline(controller, RealtimeConfig(bundle_id="test", operation="enroll",
                                    enrollment_person_id="P1", enrollment_display_name="Alice"))
        pipeline._collector = collector
        source = SimpleNamespace(open=lambda: SimpleNamespace(message="test"),
                                 close=lambda: None)
        with patch.object(pipeline, "_build_source", return_value=source), \
             patch("mph_gait_id.realtime.pipeline.SystemModelRuntime"), \
             patch.object(pipeline, "_build_detector"), \
             patch.object(pipeline, "_prepare_session_provenance"), \
             patch.object(pipeline, "_loop", side_effect=RuntimeError("capture failed")):
            controller.model_store = object()
            pipeline._run()
        self.assertFalse(capture.stage.exists())
        self.assertEqual(self.library.list_sources(), [])
        self.assertEqual(pipeline.poll_latest().state, "error")

    def test_pipeline_records_foreground_not_fixed_point_input(self):
        capture = self.capture()
        collector = self.collector(capture)
        controller = SimpleNamespace(repository=self.repository)
        config = RealtimeConfig(bundle_id="test", operation="enroll", source_mode="replay",
                                replay_loop=False, guided_passes=False, num_points=1024,
                                clip_len=2, enrollment_stride=1, enrollment_warmup_s=0,
                                enrollment_duration_s=100, enrollment_min_embeddings=1,
                                save_enrollment_foreground=True)
        pipeline = RealtimePipeline(controller, config)
        pipeline._collector = collector
        pipeline._runtime = SimpleNamespace(model_record=model_record())
        xyz = np.zeros((60, 60, 3), dtype=np.float32)
        xyz[:, :, 0] = np.arange(60)[None, :] * 10
        xyz[:, :, 1] = np.arange(60)[:, None] * 10
        xyz[:, :, 2] = 2000
        frames = [SensorFrame(i, 100 + i / 15, np.zeros((60, 60, 3), dtype=np.uint8),
                              xyz.copy(), True) for i in range(3)]
        iterator = iter(frames)
        pipeline._source = SimpleNamespace(read=lambda: next(iterator, None))
        detector = SimpleNamespace(detect=lambda _frame: Detection((0, 0, 60, 60), .9))
        with patch.object(pipeline, "_infer", return_value=np.array([1., 0.], dtype=np.float32)):
            pipeline._loop(detector)
        self.assertTrue(pipeline._committed)
        source = self.library.list_sources()[0]
        manifest = self.library.manifest(source["source_id"])
        self.assertEqual(len(manifest["frames"]), 3)
        self.assertTrue(all(f["points"] > 1024 for f in manifest["frames"]))
        self.assertEqual([f["segment_start"] for f in manifest["frames"]], [True, False, False])


if __name__ == "__main__":
    unittest.main()
