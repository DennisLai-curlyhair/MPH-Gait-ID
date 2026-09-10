from __future__ import annotations

from dataclasses import replace
import gc
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mph_gait_id.database import GalleryRepository
from mph_gait_id.enrollment_sources import EnrollmentSourceLibrary, ForegroundCapture
from mph_gait_id.gallery_lifecycle import GalleryLifecycle
from mph_gait_id.gallery_transfer import GalleryTransfer
from mph_gait_id.model_record import build_model_record
from mph_gait_id.model_store import descriptor_dimension
from mph_gait_id.realtime.preprocessing import _deterministic_sample
from mph_gait_id.source_registration import RegistrationTarget, SourceRegistrationService, plan_windows
from mph_gait_id.tests.test_gallery_transfer import TestStore


class Store(TestStore):
    def get(self, key):
        return self.items[key]

    def verify(self, key):
        return {**super().verify(key), "errors": ["checkpoint verification failed"]}


class SourceRegistrationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repository = GalleryRepository(self.root / "gallery.sqlite3")
        self.library = EnrollmentSourceLibrary(self.repository)
        self.library.initialize()
        self.store = Store(self.root / "bundles", "first")
        self.store.items.update(Store(self.store.root, "second", b"other-weights").items)
        for i, (key, bundle) in enumerate(self.store.items.items()):
            self.store.items[key] = replace(bundle, data={**bundle.data, "num_points": 4 + i,
                "clip_len": 2 + i, "drop_first_frames": 30}, model={**bundle.model, "embedding_dim": 2 + i})
        self.repository.upsert_person("P1", "Alice")
        capture = ForegroundCapture(self.library, min_free_bytes=0, queue_size=32)
        try:
            for pass_id, count in [("pass_001", 6), ("pass_002", 4)]:
                for i in range(count):
                    points = np.arange((8 + i) * 3, dtype=np.float32).reshape(-1, 3) + 1000
                    capture.append(pass_id, points, frame_index=i, timestamp=i / 15.,
                                   segment_start=i == 0, metadata={})
            prepared = capture.prepare({"pass_001", "pass_002"}, dict(session_id="session-one",
                capture={"preprocessing_profile_id": "foreground-v1",
                         "filter_parameters": {"max_valid_frame_gap_s": 1.0}}, passes=[]))
            with self.repository.connect() as con:
                prepared.attach(con, "P1", "Alice", [])
            self.source_id = capture.source_id
            prepared.finish()
        finally:
            if not capture.closed:
                capture.abandon()
        self.inputs = []
        self.loaded = []
        self.active_runtimes = 0
        self.hook = lambda _key, _tensor: None

        test = self
        class Runtime:
            def __init__(self, *, bundle_id, model_store, runtime_clip_len, preprocessing_profile_id, **_kwargs):
                test.assertEqual(test.active_runtimes, 0)
                test.active_runtimes += 1
                self.bundle_id = bundle_id
                test.loaded.append(bundle_id)
                bundle = model_store.get(bundle_id)
                self.model_record = build_model_record(bundle, model_store, runtime_clip_len, 30, preprocessing_profile_id)
                self.adapter = self

            def encode_tensor(self, tensor):
                test.inputs.append((self.bundle_id, tensor.clone()))
                test.hook(self.bundle_id, tensor)
                count = self.model_record["embedding_dim"]
                return torch.arange(1, count + 1, dtype=torch.float32).reshape(1, -1)

            def __del__(self):
                test.active_runtimes -= 1

        self.service = SourceRegistrationService(self.library, self.store, "foreground-v1", runtime_factory=Runtime)
        self.targets = [RegistrationTarget("first", 2), RegistrationTarget("second", 3)]

    def run_job(self, **kwargs):
        return self.service.run(self.source_id, kwargs.pop("targets", self.targets),
            kwargs.pop("pass_ids", ["pass_001", "pass_002"]), device="cpu", **kwargs)

    def count(self):
        with self.repository.connect() as con:
            return con.execute("SELECT COUNT(*) FROM gallery_embeddings").fetchone()[0]

    def last_report(self):
        with self.repository.connect() as con:
            return json.loads(con.execute("SELECT report_json FROM enrollment_source_jobs ORDER BY rowid DESC LIMIT 1").fetchone()[0])

    def test_two_models_point_counts_keys_and_no_additional_frame_drop(self):
        report = self.run_job()
        self.assertEqual(report["embeddings_added"], 8)
        self.assertEqual(self.loaded, ["first", "second"])
        self.assertEqual(self.active_runtimes, 0)
        self.assertEqual([i["embeddings_added"] for i in report["targets"]], [5, 3])
        self.assertEqual(self.inputs[0][1].shape, (2, 4, 3))
        self.assertEqual(self.inputs[-1][1].shape, (3, 5, 3))
        manifest = self.library.manifest(self.source_id)
        expected = _deterministic_sample(self.library.load_frame(self.source_id, manifest["frames"][0]), 4)
        np.testing.assert_array_equal(self.inputs[0][1][0], expected)
        with self.repository.connect() as con:
            rows = list(con.execute("SELECT * FROM gallery_embeddings"))
            self.assertEqual({r["embedding_dim"] for r in rows}, {2, 3})
            self.assertEqual(len({r["model_key"] for r in rows}), 2)
            for row in rows:
                self.assertAlmostEqual(float(np.linalg.norm(np.frombuffer(row["embedding"], dtype=np.float32))), 1., places=6)
                self.assertEqual(row["session_id"], "session-one")
        self.assertEqual(self.last_report()["status"], "completed")

    def test_repeat_and_inactive_passes_are_skipped_without_loading_models(self):
        self.run_job()
        with self.repository.connect() as con:
            con.execute("UPDATE gallery_embeddings SET active=0")
        self.loaded.clear()
        report = self.run_job()
        self.assertEqual(report["embeddings_added"], 0)
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.count(), 8)

    def test_partial_pass_selection_can_add_remaining_passes_later(self):
        first = self.run_job(pass_ids=["pass_001"])
        second = self.run_job()
        self.assertEqual(first["embeddings_added"], 5)
        self.assertEqual(second["embeddings_added"], 3)
        self.assertEqual(self.count(), 8)

    def test_cap_and_short_pass_reporting(self):
        report = self.run_job(max_windows_per_pass=1, targets=[RegistrationTarget("first", 5)])
        self.assertEqual(report["embeddings_added"], 1)
        self.assertEqual(report["targets"][0]["passes"][1]["status"], "too_short")

    def test_cancellation_between_models_saves_no_features(self):
        cancel = threading.Event()
        def hook(key, _tensor):
            if key == "second":
                cancel.set()
        self.hook = hook
        report = self.run_job(cancel=cancel)
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.active_runtimes, 0)
        self.assertEqual(self.last_report()["status"], "cancelled")
        self.assertTrue(self.library.manifest(self.source_id))

    def test_second_model_failure_preserves_existing_gallery_and_sources(self):
        self.run_job(targets=[self.targets[0]], pass_ids=["pass_001"])
        def fail(key, _tensor):
            if key == "second":
                raise RuntimeError("second model failed")
        self.hook = fail
        with self.assertRaisesRegex(RuntimeError, "second model failed"):
            self.run_job()
        self.assertEqual(self.count(), 3)
        self.assertEqual(self.last_report()["embeddings_added"], 0)
        gc.collect()
        self.assertEqual(self.active_runtimes, 0)

    def test_sqlite_second_model_failure_rolls_back_first_model(self):
        with self.repository.connect() as con:
            con.execute("CREATE TRIGGER fail_second BEFORE INSERT ON gallery_embeddings "
                        "WHEN NEW.embedding_dim=3 BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.run_job()
        self.assertEqual(self.count(), 0)
        with self.repository.connect() as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0], 0)
        self.assertEqual(self.last_report()["status"], "failed")

    def test_bad_hash_and_coordinate_contract_fail_before_model_loading(self):
        self.store.get("second").checkpoint.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "verification"):
            self.run_job()
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.count(), 0)

    def test_official_nested_part_head_descriptor_dimension(self):
        bundle = replace(self.store.get("first"), architecture="lidargaitpp_official",
                         model={"SeparateFCs": {"out_channels": 128, "parts_num": 7}})
        self.assertEqual(descriptor_dimension(bundle), 896)
        self.assertEqual(descriptor_dimension(replace(bundle, model={})), 7936)

    def test_profile_mismatch_is_not_silently_relabelled(self):
        self.service.profile = "different-foreground"
        with self.assertRaisesRegex(ValueError, "profile"):
            self.run_job()
        self.assertEqual(self.loaded, [])

    def test_source_frame_tamper_rolls_back(self):
        manifest = self.library.manifest(self.source_id)
        (self.library.record_dir(self.source_id) / manifest["frames"][0]["file"]).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_job()
        self.assertEqual(self.count(), 0)

    def test_owner_delete_id_reuse_is_never_silent_reassignment(self):
        lifecycle = GalleryLifecycle(self.repository)
        lifecycle.apply(lifecycle.preview("P1"), "delete_person")
        self.repository.upsert_person("P1", "Bob")
        with self.assertRaisesRegex(ValueError, "owner"):
            self.run_job()
        self.assertEqual(self.count(), 0)

    def test_owner_change_during_inference_aborts(self):
        def hook(_key, _tensor):
            with self.repository.connect() as con:
                con.execute("UPDATE persons SET display_name='Changed' WHERE person_id='P1'")
        self.hook = hook
        with self.assertRaisesRegex(ValueError, "owner changed"):
            self.run_job()
        self.assertEqual(self.count(), 0)

    def test_checkpoint_and_axis_changes_during_job_abort(self):
        def hook(_key, _tensor):
            self.store.get("second").data["coordinate_adapter"] = "none"
        self.hook = hook
        with self.assertRaisesRegex(ValueError, "bundle changed"):
            self.run_job()
        self.assertEqual(self.count(), 0)

    def test_writer_lease_blocks_source_delete_and_cleanup_during_encoding(self):
        def hook(_key, _tensor):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.library.delete(self.source_id)
            with self.assertRaisesRegex(RuntimeError, "busy"):
                self.library.cleanup_uncommitted()
        self.hook = hook
        self.run_job()
        # Lease released after completion.
        self.library.cleanup_uncommitted()

    def test_transfer_roundtrip_still_skips_registered_passes(self):
        self.run_job()
        transfer = GalleryTransfer(self.repository, self.store, "foreground-v1")
        archive = self.root / "test.mphgallery"
        transfer.export(archive)
        # Simulate restore on the same device: local source media is retained.
        with self.repository.connect() as con:
            con.execute("DELETE FROM gallery_embeddings")
            con.execute("DELETE FROM gallery_transfer_aliases WHERE kind='embedding'")
        preview = transfer.preview(archive)
        transfer.import_archive(archive, preview, {p["uid"]: p["target_id"] for p in preview["persons"]})
        self.loaded.clear()
        self.assertEqual(self.run_job()["embeddings_added"], 0)
        self.assertEqual(self.loaded, [])

    def test_original_stage_b_session_is_also_duplicate(self):
        record = build_model_record(self.store.get("first"), self.store, 2, 30, "foreground-v1")
        self.repository.upsert_model(record)
        self.repository.add_embeddings("P1", record["model_key"], np.array([[1, 0]], dtype=np.float32),
            self.root / "original-capture", "original-live-session", "azure_kinect_live", 1.,
            [{"window": {"pass_id": "pass_001"}, "source_metadata": {
                "realtime_enrollment": {"session_id": "session-one"}, "foreground_source_id": self.source_id}}])
        result = self.run_job(targets=[self.targets[0]])
        self.assertEqual(result["embeddings_added"], 2)


class WindowPlannerTest(unittest.TestCase):
    def test_no_cross_pass_segment_reverse_time_or_gap_and_no_padding(self):
        frames = [dict(pass_id="p", timestamp=i / 15, sensor_frame_index=i, points=10, segment_start=False)
                  for i in range(10)]
        frames[3]["segment_start"] = True
        frames[6]["timestamp"] = 5.
        frames[9]["pass_id"] = "other"
        windows = plan_windows(frames, "p", 3, 10, 1.)
        self.assertEqual([[f["sensor_frame_index"] for f in w] for w in windows], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(plan_windows(frames, "p", 30, 10, 1.), [])

    def test_even_window_cap(self):
        frames = [dict(pass_id="p", timestamp=i / 15, sensor_frame_index=i, points=10, segment_start=False)
                  for i in range(20)]
        windows = plan_windows(frames, "p", 2, 2, 1.)
        self.assertEqual([w[0]["sensor_frame_index"] for w in windows], [0, 18])


if __name__ == "__main__":
    unittest.main()
