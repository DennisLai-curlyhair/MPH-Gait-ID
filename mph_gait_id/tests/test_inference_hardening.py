from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from mph_gait_id.checkpoint_io import load_checkpoint
from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.dataio import PointCloudWindowDataset, build_windows
from mph_gait_id.database import GalleryRepository
from mph_gait_id.embedding_contract import (
    CONTRACT_FIELDS, LEGACY_V030_ENCODER, MIGRATION_TARGET_ENCODER, encoder_hash,
)
from mph_gait_id.gallery_migration import migrate_gallery
from mph_gait_id.gallery_transfer import GalleryTransfer, read_archive
from mph_gait_id.model_adapter import ModelAdapter
from mph_gait_id.model_record import build_model_record
from mph_gait_id.model_store import ModelStore
from mph_gait_id.realtime.detection import restore_person_mask, YoloPersonDetector, verified_yolo_checkpoint
from mph_gait_id.realtime.pipeline import RealtimeConfig, RealtimePipeline
from mph_gait_id.realtime.types import SensorFrame
from mph_gait_id.report_io import write_result
from mph_gait_id.scripts.download_assets import download_url
from mph_gait_id.services import RegistrationService
from mph_gait_id.tests.test_core import embedding_batch, model_record
from mph_gait_id.tests.test_gallery_transfer import TestStore


class UnsafeObject:
    def __reduce__(self):
        return eval, ("1 + 1",)


class CheckpointTest(unittest.TestCase):
    def test_restricted_loader_rejects_executable_objects(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "unsafe.pt"
            torch.save({"model": {"weight": torch.ones(1)}, "metadata": UnsafeObject()}, path)
            with self.assertRaisesRegex(ValueError, "Unrestricted pickle"):
                load_checkpoint(path)

    def test_old_torch_is_blocked_before_loading(self):
        with patch("torch.__version__", "2.5.1"), patch("torch.load") as loader:
            with self.assertRaisesRegex(RuntimeError, "upgrade"):
                load_checkpoint("unused.pt")
            loader.assert_not_called()

    def test_state_dict_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "state.pt"
            for state in ({}, {"x": torch.tensor([float("nan")])}, {"x": "not a tensor"}):
                torch.save({"model": state}, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)
            torch.save({"model": {"x": torch.ones(2)}, "config": {"data": {"clip_len": 15}}}, path)
            self.assertTrue(torch.equal(load_checkpoint(path)["model"]["x"], torch.ones(2)))

    def test_unknown_yolo_bytes_never_reach_loader(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "yolov8n.pt"
            path.write_bytes(b"untrusted checkpoint")
            with self.assertRaisesRegex(ValueError, "Unverified YOLO"):
                verified_yolo_checkpoint(str(path))

    def test_download_checks_hash_before_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "weights.pt"
            target.write_bytes(b"existing")
            def download(_url, path):
                Path(path).write_bytes(b"bad download")
            with patch("urllib.request.urlretrieve", side_effect=download):
                with self.assertRaisesRegex(RuntimeError, "SHA256"):
                    download_url("https://example.invalid/file", target, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(len(list(Path(raw).iterdir())), 1)


class MaskTest(unittest.TestCase):
    def test_letterboxed_landscape_and_portrait(self):
        original = np.zeros((720, 1280), dtype=np.uint8)
        original[100:620, 400:800] = 1
        letterbox = np.zeros((384, 640), dtype=np.float32)
        letterbox[62:322, 200:400] = 1
        np.testing.assert_array_equal(restore_person_mask(letterbox, 720, 1280), original)
        np.testing.assert_array_equal(restore_person_mask(letterbox.T, 1280, 720), original.T)
        np.testing.assert_array_equal(restore_person_mask(original, 720, 1280), original)

    def test_detection_requests_original_resolution_masks(self):
        detector = object.__new__(YoloPersonDetector)
        detector.confidence, detector.image_size = .35, 640
        detector.device, detector.bbox_padding, detector.require_mask = "cpu", 0, True
        class Boxes:
            conf = torch.tensor([.9])
            xyxy = torch.tensor([[400., 100., 800., 620.]])
            def __len__(self):
                return 1
        mask = torch.zeros(1, 384, 640)
        mask[0, 62:322, 200:400] = 1
        detector.model = SimpleNamespace(predict=Mock(return_value=[
            SimpleNamespace(boxes=Boxes(), masks=SimpleNamespace(data=mask))]))
        frame = SensorFrame(0, 0., np.zeros((720, 1280, 3), dtype=np.uint8), np.empty((0, 3)), True)
        result = detector.detect(frame)
        self.assertTrue(detector.model.predict.call_args.kwargs["retina_masks"])
        self.assertEqual(np.where(result.mask)[0].min(), 100)
        detector.model.predict.return_value[0].masks = None
        with self.assertRaisesRegex(RuntimeError, "instance mask"):
            detector.detect(frame)


class InputValidationTest(unittest.TestCase):
    def test_short_windows_are_not_padded(self):
        paths = [Path(f"clear_data_{i}.npy") for i in range(20)]
        with self.assertRaisesRegex(ValueError, "Insufficient frames"):
            build_windows(paths, 15, 5, 10)
        self.assertEqual(len(build_windows(paths, 15, 5, 0)), 2)

    def test_bad_frames_do_not_turn_into_valid_zero_inputs(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "clear_data_0.npy"
            for array in (np.empty((0, 3)), np.ones((5, 2)), np.ones(4),
                          np.full((6, 3), np.nan), np.zeros((4, 3)),
                          np.array([object()], dtype=object)):
                np.save(path, array)
                dataset = PointCloudWindowDataset(path.parent, 8, 1, 1, 0)
                with self.assertRaises(ValueError):
                    dataset[0]
            np.save(path, np.array([[1, 2, 3], [4, 5, 6], [np.nan, 0, 0]], dtype=np.float32))
            result = dataset[0]["points"]
            self.assertEqual(tuple(result.shape), (1, 8, 3))
            self.assertTrue(bool(torch.isfinite(result).all()))

    def test_live_input_and_zero_descriptor_validation(self):
        adapter = object.__new__(ModelAdapter)
        adapter.bundle = SimpleNamespace(data={"coordinate_adapter": "none"}, architecture="pc_v1")
        for value in (torch.zeros(1, 15, 32, 3), torch.full((1, 15, 32, 3), float("nan")),
                      torch.empty(1, 0, 32, 3)):
            with self.assertRaises(ValueError):
                adapter._prepare_inputs(value)
        adapter.device, adapter.amp_enabled = torch.device("cpu"), False
        adapter._forward_embedding = lambda *_a, **_kw: torch.zeros(1, 256)
        with self.assertRaisesRegex(RuntimeError, "zero-length"):
            adapter.encode_tensor(torch.randn(15, 32, 3))


class StabilityTest(unittest.TestCase):
    def test_acceptance_requires_full_stability_and_resets_on_switch(self):
        pipeline = RealtimePipeline(object(), RealtimeConfig(bundle_id="unused", stability_windows=3))
        def feed(person):
            return pipeline._stabilize(dict(accepted=True, person_id=person, display_name=person, state="stable"))
        for _ in range(2):
            result = feed("A")
            self.assertTrue(result["window_accepted"])
            self.assertFalse(result["accepted"])
            self.assertEqual(result["candidate_person_id"], "A")
            self.assertIsNone(result["person_id"])
            self.assertEqual(result["state"], "accumulating")
        self.assertTrue(feed("A")["accepted"])
        self.assertFalse(feed("B")["accepted"])
        self.assertFalse(feed("B")["accepted"])
        self.assertTrue(feed("B")["accepted"])
        pipeline._stabilize(dict(accepted=False, person_id=None, state="unknown"))
        self.assertFalse(feed("B")["accepted"])


class RegistrationSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = GalleryRepository(self.root / "gallery.sqlite3")
        self.batch = embedding_batch(self.root / "source", "same-content", model_record())
        self.service = RegistrationService(self.repo)

    def enroll(self, person="A", **kwargs):
        return self.service.enroll(self.batch, person, person, min_embeddings=1, **kwargs)

    def test_inactive_source_cannot_be_rebound_even_with_duplicate_override(self):
        self.enroll()
        with self.repo.connect() as con:
            con.execute("UPDATE gallery_embeddings SET active=0")
        with self.assertRaises(ValueError):
            self.enroll()
        with self.assertRaisesRegex(ValueError, "another person"):
            self.enroll("B", allow_duplicate_source=True)
        self.assertIsNone(self.repo.get_person("B"))

    def test_zero_descriptors_cannot_be_registered(self):
        self.batch.embeddings[:] = 0
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.enroll()
        self.assertIsNone(self.repo.get_person("A"))

    def test_transaction_rechecks_duplicate_if_preflight_misses_it(self):
        self.enroll()
        with patch.object(self.repo, "find_enrolled_source", return_value=None):
            with self.assertRaisesRegex(ValueError, "already enrolled"):
                self.enroll()
        self.assertEqual(len(self.repo.load_gallery("test-model")), 2)

    def test_report_failure_keeps_committed_registration(self):
        result = self.enroll()
        controller = object.__new__(GaitApplicationController)
        prepared = SimpleNamespace(metadata=lambda: {})
        with patch.object(controller, "_save_result", side_effect=OSError("disk unavailable")):
            outcome = controller._finish("enroll", result, prepared, 0.)
        self.assertEqual(outcome.result["stored_embeddings"], 2)
        self.assertIn("report_warning", outcome.result)
        self.assertNotIn("result_path", outcome.result)
        self.assertEqual(len(self.repo.load_gallery("test-model")), 2)

    def test_atomic_report_failure_does_not_advertise_missing_file(self):
        result = {"operation": "enroll"}
        with patch("mph_gait_id.report_io.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                write_result(self.root / "reports", result)
        self.assertNotIn("result_path", result)
        self.assertFalse((self.root / "reports" / "result.json").exists())
        self.assertEqual(list((self.root / "reports").iterdir()), [])
        write_result(self.root / "good", result)
        self.assertTrue(Path(result["result_path"]).is_file())


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = TestStore(self.root / "bundles")
        self.bundle = next(iter(self.store.items.values()))
        self.record = build_model_record(self.bundle, self.store, 15, 0, "foreground-v1")
        self.repo = GalleryRepository(self.root / "gallery.sqlite3")
        self.repo.initialize()
        self.transfer = GalleryTransfer(self.repo, self.store, "foreground-v1")

    def make_legacy(self, saved_spec=True):
        legacy = {**self.record, "model_key": "legacy-model"}
        legacy.pop("inference_spec")
        legacy["compatibility"] = {k: v for k, v in self.record["compatibility"].items()
                                   if k not in CONTRACT_FIELDS}
        self.repo.upsert_person("A", "Alice")
        self.repo.upsert_model(legacy)
        self.ids = self.repo.add_embeddings("A", "legacy-model", np.array([[1., 0.]], np.float32),
            self.root / "source", "content", "folder", 1., [{}])
        spec = {**self.record["inference_spec"], "encoder_sha256": LEGACY_V030_ENCODER}
        with self.repo.connect() as con:
            con.execute("UPDATE gallery_embeddings SET active=0")
            if saved_spec:
                con.execute("INSERT INTO gallery_transfer_model_specs VALUES (?,?)",
                            ("legacy-model", json.dumps(spec)))

    def test_axis_model_and_input_changes_create_distinct_keys(self):
        variants = [
            replace(self.bundle, data={**self.bundle.data, "coordinate_adapter": "none"}),
            replace(self.bundle, data={**self.bundle.data, "num_points": 512}),
            replace(self.bundle, model={**self.bundle.model, "point_normalization": "scale"}),
        ]
        for bundle in variants:
            record = build_model_record(bundle, self.store, 15, 0, "foreground-v1")
            self.assertNotEqual(record["model_key"], self.record["model_key"])
        lidar = next(b for b in ModelStore().bundles().values() if b.architecture == "lidargaitpp_official")
        self.assertEqual(build_model_record(lidar, ModelStore(), 15, 0, "foreground-v1")["embedding_dim"], 7936)

    def test_runtime_cache_changes_when_bundle_input_contract_changes(self):
        controller = object.__new__(GaitApplicationController)
        self.store.get = lambda key: self.store.items[key]
        controller.model_store, controller.config = self.store, {}
        controller._runtime_cache, controller._runtime_lock = {}, threading.RLock()
        with patch("mph_gait_id.controller.SystemModelRuntime") as runtime:
            runtime.side_effect = [object(), object()]
            first = controller._runtime("mph", "cpu", 1, 0)
            self.assertIs(first, controller._runtime("mph", "cpu", 1, 0))
            self.bundle.data["coordinate_adapter"] = "none"
            self.assertIsNot(first, controller._runtime("mph", "cpu", 1, 0))
            self.assertEqual(runtime.call_count, 2)

    def test_migration_target_is_the_tested_encoder(self):
        self.assertEqual(encoder_hash(), MIGRATION_TARGET_ENCODER)

    def test_legacy_archive_blocks_an_unreviewed_encoder_change(self):
        self.make_legacy()
        archive = self.root / "old.mphgallery"
        self.transfer.export(archive)
        with patch("mph_gait_id.gallery_transfer._encoder_hash", return_value="unreviewed"):
            self.assertEqual(self.transfer.preview(archive)["compatible_embeddings"], 0)

    def test_legacy_migration_preserves_ids_vectors_and_inactive_status(self):
        self.make_legacy()
        preview = migrate_gallery(self.repo.path, self.store)
        self.assertEqual(preview["models"][0]["status"], "ready")
        with self.repo.connect() as con:
            before = dict(con.execute("SELECT * FROM gallery_embeddings").fetchone())
        result = migrate_gallery(self.repo.path, self.store, apply=True)
        self.assertTrue(Path(result["backup"]).is_file())
        with self.repo.connect() as con:
            after = dict(con.execute("SELECT * FROM gallery_embeddings").fetchone())
        for field in ("embedding_id", "embedding", "person_id", "active", "source_fingerprint"):
            self.assertEqual(before[field], after[field])
        self.assertEqual(after["model_key"], self.record["model_key"])
        self.assertEqual(migrate_gallery(self.repo.path, self.store, apply=True)["models"], [])
        archive = self.root / "new.mphgallery"
        self.transfer.export(archive)
        self.assertEqual(self.transfer.preview(archive)["compatible_embeddings"], 1)

    def test_missing_legacy_provenance_is_not_guessed(self):
        self.make_legacy(saved_spec=False)
        before = self.repo.path.read_bytes()
        result = migrate_gallery(self.repo.path, self.store)
        self.assertEqual(result["models"][0]["status"], "blocked")
        self.assertEqual(self.repo.path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "blocked models"):
            migrate_gallery(self.repo.path, self.store, apply=True)
        archive = self.root / "unknown.mphgallery"
        self.transfer.export(archive)
        self.assertIsNone(read_archive(archive)["records"]["models"][0]["bundle_spec"])
        self.assertEqual(self.transfer.preview(archive)["compatible_embeddings"], 0)

    def test_migration_rolls_back_on_database_write_failure(self):
        self.make_legacy()
        with patch.object(GalleryRepository, "upsert_model", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                migrate_gallery(self.repo.path, self.store, apply=True)
        with self.repo.connect() as con:
            row = con.execute("SELECT model_key,active FROM gallery_embeddings").fetchone()
            self.assertEqual(tuple(row), ("legacy-model", 0))
        self.assertEqual(len(list(self.root.glob("*.before-contract-v2-*.sqlite3"))), 1)

    def test_v030_archive_imports_and_reexports_under_new_contract(self):
        self.make_legacy()
        archive = self.root / "old.mphgallery"
        self.transfer.export(archive)
        other = GalleryRepository(self.root / "other.sqlite3")
        other.initialize()
        transfer = GalleryTransfer(other, self.store, "foreground-v1")
        preview = transfer.preview(archive)
        result = transfer.import_archive(archive, preview, {p["uid"]: p["person_id"] for p in preview["persons"]})
        self.assertEqual(result["inserted"], 1)
        transfer.export(self.root / "reexport.mphgallery")
        self.assertEqual(transfer.preview(self.root / "reexport.mphgallery")["compatible_embeddings"], 1)

    def test_legacy_changed_axis_blocks_migration(self):
        self.make_legacy()
        self.bundle.data["coordinate_adapter"] = "none"
        self.assertEqual(migrate_gallery(self.repo.path, self.store)["models"][0]["status"], "blocked")
