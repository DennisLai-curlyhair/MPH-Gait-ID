from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from mph_gait_id.database import GalleryRepository
from mph_gait_id.gallery_transfer import GalleryTransfer, read_archive, _encoder_hash
from mph_gait_id.model_record import build_model_record
from mph_gait_id.model_store import ModelBundle


class TestStore:
    __test__ = False

    def __init__(self, root: Path, bundle_id: str = "mph", weights: bytes = b"test-not-a-model"):
        self.root = root
        folder = root / bundle_id
        folder.mkdir(parents=True)
        checkpoint = folder / "test.pt"
        checkpoint.write_bytes(weights)
        bundle = ModelBundle(
            bundle_id=bundle_id, display_name="Test MPH", method_key="mph_gait", description="Test",
            architecture="mph_gait", input_type="pointcloud", mode="raw_pointcloud", channels=3,
            data={"num_points": 1024, "clip_len": 15, "drop_first_frames": 0,
                  "coordinate_adapter": "kinect_xyz_mm_to_forward_lateral_height_m_v1"},
            model={"embedding_dim": 2, "point_normalization": "center"}, training={"fold": -1},
            checkpoint=checkpoint, checkpoint_sha256=hashlib.sha256(weights).hexdigest(),
            manifest_path=folder / "bundle.yaml",
        )
        self.items = {bundle_id: bundle}

    def bundles(self):
        return self.items.copy()

    def refresh(self):
        return self.bundles()

    def verify(self, key):
        item = self.items[key]
        return {"valid": item.checkpoint.is_file() and
                hashlib.sha256(item.checkpoint.read_bytes()).hexdigest() == item.checkpoint_sha256}


class GalleryTransferTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a = GalleryRepository(self.root / "a" / "gallery.sqlite3")
        self.b = GalleryRepository(self.root / "b" / "gallery.sqlite3")
        self.a.initialize()
        self.b.initialize()
        self.sa = TestStore(self.root / "a" / "bundles")
        self.sb = TestStore(self.root / "b" / "bundles")
        self.ta = GalleryTransfer(self.a, self.sa, "foreground-v1")
        self.tb = GalleryTransfer(self.b, self.sb, "foreground-v1")
        bundle = next(iter(self.sa.items.values()))
        self.model = build_model_record(bundle, self.sa, 15, 0, "foreground-v1")
        self.a.upsert_person("P1", "Alice", note="C2 enrollment")
        self.a.upsert_model(self.model)
        self.ids = self.a.add_embeddings(
            "P1", self.model["model_key"], np.array([[1, 0], [0.6, 0.8]], dtype=np.float32),
            self.root / "private-sensor-folder", "sha256:source", "azure-live", 0.9,
            [{"window": {"window_index": i, "pass_id": "pass1", "start_frame_number": i * 15},
              "source_metadata": {"private_path": "PRIVATE_PATH_SENTINEL",
                                  "device_serial": "PRIVATE_SERIAL_SENTINEL",
                                  "realtime_enrollment": {"session_id": "session1"}}} for i in range(2)],
        )
        with self.a.connect() as con:
            con.execute("UPDATE gallery_embeddings SET active=0 WHERE embedding_id=?", (self.ids[1],))
        self.archive = self.root / "gallery.mphgallery"
        self.ta.export(self.archive)

    def import_b(self, path=None, mapping=None):
        path = path or self.archive
        preview = self.tb.preview(path)
        if mapping is None:
            mapping = {p["uid"]: p["target_id"] for p in preview["persons"]}
        return self.tb.import_archive(path, preview, mapping)

    def mutate_archive(self, change):
        with zipfile.ZipFile(self.archive) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        records = json.loads(contents["records.json"])
        change(records, contents)
        contents["records.json"] = json.dumps(records).encode()
        manifest = json.loads(contents["manifest.json"])
        manifest["sha256"] = {name: hashlib.sha256(contents[name]).hexdigest()
                              for name in ("records.json", "embeddings.bin")}
        contents["manifest.json"] = json.dumps(manifest).encode()
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, content in contents.items():
                archive.writestr(name, content)

    def test_round_trip_matches_scores_and_inactive_states(self):
        before = self.a.load_gallery(self.model["model_key"])
        result = self.import_b()
        self.assertEqual(result["inserted"], 2)
        after = self.b.load_gallery(self.model["model_key"])
        np.testing.assert_array_equal(before[0]["embedding"], after[0]["embedding"])
        self.assertEqual(len(after), 1)
        self.assertTrue(after[0]["source_path"].startswith("gallery-source:"))
        query = np.array([0.8, 0.6], dtype=np.float32)
        self.assertEqual(float(before[0]["embedding"] @ query), float(after[0]["embedding"] @ query))
        with sqlite3.connect(result["backup"]) as con:
            self.assertEqual(con.execute("SELECT COUNT(*) FROM persons").fetchone()[0], 0)
        self.assertTrue(Path(result["retained_archive"]).is_file())

    def test_export_does_not_contain_paths_device_serials_or_weights(self):
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(set(archive.namelist()), {"manifest.json", "records.json", "embeddings.bin"})
            data = archive.read("records.json")
        for text in ("PRIVATE_PATH_SENTINEL", "PRIVATE_SERIAL_SENTINEL", str(self.root), "private-sensor-folder"):
            self.assertNotIn(text.encode(), data)

    def test_duplicate_import_never_reactivates_records(self):
        self.import_b()
        with self.b.connect() as con:
            con.execute("UPDATE gallery_embeddings SET active=0")
        result = self.import_b()
        self.assertEqual(result["duplicates"], 2)
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(self.b.load_gallery(self.model["model_key"]), [])

    def test_reexport_to_third_computer_preserves_identity(self):
        self.import_b()
        export_b = self.root / "b.mphgallery"
        self.tb.export(export_b)
        result = self.import_b(export_b)
        self.assertEqual(result["duplicates"], 2)
        a_uids = [p["uid"] for p in read_archive(self.archive)["records"]["persons"]]
        b_uids = [p["uid"] for p in read_archive(export_b)["records"]["persons"]]
        self.assertEqual(a_uids, b_uids)

    def test_renamed_bundle_and_different_device_paths_are_compatible(self):
        self.sb = TestStore(self.root / "renamed" / "bundles", bundle_id="custom-name")
        self.tb = GalleryTransfer(self.b, self.sb, "foreground-v1")
        result = self.import_b()
        self.assertEqual(result["inserted"], 2)
        keys = self.b.model_keys_for_bundle("custom-name", clip_len=15)
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(self.b.load_gallery(keys[0])), 1)
        # Local metadata must be accepted by the ordinary enrollment service too.
        record = build_model_record(next(iter(self.sb.items.values())), self.sb, 15, 0, "foreground-v1")
        self.b.upsert_model(record)

    def test_missing_weights_are_retained_then_can_be_imported_later(self):
        saved = self.sb.items
        self.sb.items = {}
        result = self.import_b()
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(result["skipped"], 2)
        self.sb.items = saved
        result = self.import_b(result["retained_archive"])
        self.assertEqual(result["inserted"], 2)

    def test_axis_profile_checkpoint_or_encoder_change_is_rejected(self):
        for variant in ("axis", "profile", "checkpoint", "encoder"):
            with self.subTest(variant=variant):
                self.sb = TestStore(self.root / variant / "bundles")
                self.tb = GalleryTransfer(self.b, self.sb, "other-profile" if variant == "profile" else "foreground-v1")
                bundle = next(iter(self.sb.items.values()))
                if variant == "axis":
                    bundle.data["coordinate_adapter"] = "identity"
                elif variant == "checkpoint":
                    bundle.checkpoint.write_bytes(b"modified")
                with patch("mph_gait_id.gallery_transfer._encoder_hash", return_value="different") if variant == "encoder" else nullcontext():
                    self.assertEqual(self.tb.preview(self.archive)["compatible_embeddings"], 0)

    def test_person_conflict_requires_explicit_mapping(self):
        self.b.upsert_person("P1", "Different person")
        preview = self.tb.preview(self.archive)
        self.assertEqual(preview["persons"][0]["status"], "conflict")
        with self.assertRaisesRegex(ValueError, "every imported person"):
            self.tb.import_archive(self.archive, preview, {})
        self.import_b(mapping={preview["persons"][0]["uid"]: "IMPORTED-P1"})
        self.assertEqual(self.b.get_person("P1")["display_name"], "Different person")
        self.assertEqual(self.b.get_person("IMPORTED-P1")["display_name"], "Alice")

    def test_explicit_merge_preserves_local_name_note_and_status(self):
        self.b.upsert_person("LOCAL", "Alice-local", "Keep this note")
        with self.b.connect() as con:
            con.execute("UPDATE persons SET status='inactive'")
        preview = self.tb.preview(self.archive)
        self.import_b(mapping={preview["persons"][0]["uid"]: "LOCAL"})
        self.assertEqual(self.b.get_person("LOCAL")["status"], "inactive")
        self.assertEqual(self.b.get_person("LOCAL")["note"], "Keep this note")
        self.assertEqual(self.b.get_person("LOCAL")["display_name"], "Alice-local")

    def test_preview_does_not_change_gallery_rows(self):
        with self.b.connect() as con:
            before = self.tb._state(con)
        self.tb.preview(self.archive)
        with self.b.connect() as con:
            self.assertEqual(before, self.tb._state(con))

    def test_stale_preview_is_rejected(self):
        preview = self.tb.preview(self.archive)
        self.b.upsert_person("P2", "New local")
        with self.assertRaisesRegex(ValueError, "Gallery changed"):
            self.tb.import_archive(self.archive, preview, {preview["persons"][0]["uid"]: "P1"})
        self.assertIsNone(self.b.get_person("P1"))

    def test_changed_archive_after_preview_is_rejected(self):
        preview = self.tb.preview(self.archive)
        self.mutate_archive(lambda records, _: records["persons"][0].update(note="changed"))
        with self.assertRaisesRegex(ValueError, "Archive changed"):
            self.tb.import_archive(self.archive, preview, {preview["persons"][0]["uid"]: "P1"})

    def test_failure_rolls_back_people_models_and_embeddings(self):
        preview = self.tb.preview(self.archive)
        with self.b.connect() as con:
            con.execute("""CREATE TRIGGER fail_second BEFORE INSERT ON gallery_embeddings
                        WHEN (SELECT COUNT(*) FROM gallery_embeddings) > 0
                        BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "simulated failure"):
            self.tb.import_archive(self.archive, preview, {preview["persons"][0]["uid"]: "P1"})
        with self.b.connect() as con:
            for table in ("persons", "model_versions", "gallery_embeddings", "gallery_transfer_aliases", "gallery_transfer_model_specs"):
                self.assertEqual(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_bad_checksum_is_rejected(self):
        with zipfile.ZipFile(self.archive) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        contents["embeddings.bin"] = b"bad"
        with zipfile.ZipFile(self.archive, "w") as archive:
            for key, value in contents.items():
                archive.writestr(key, value)
        with self.assertRaisesRegex(ValueError, "Checksum"):
            self.tb.preview(self.archive)

    def test_traversal_member_is_rejected_without_extraction(self):
        with zipfile.ZipFile(self.archive, "a") as archive:
            archive.writestr("../escaped", b"payload")
        with self.assertRaisesRegex(ValueError, "archive members"):
            self.tb.preview(self.archive)
        self.assertFalse((self.root.parent / "escaped").exists())

    def test_nan_zero_wrong_dimensions_and_foreign_keys_are_rejected(self):
        original = self.archive.read_bytes()
        changes = [
            lambda _r, c: c.update({"embeddings.bin": np.array([np.nan, 0, 0.6, 0.8], dtype="<f4").tobytes()}),
            lambda _r, c: c.update({"embeddings.bin": b"\0" * 16}),
            lambda r, _c: r["models"][0]["compatibility"].update(embedding_dim=4),
            lambda r, _c: r["embeddings"][0].update(person_uid="unknown"),
            lambda r, _c: r["embeddings"][0].update(active=2),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.archive.write_bytes(original)
                self.mutate_archive(change)
                with self.assertRaises(ValueError):
                    self.tb.preview(self.archive)

    def test_size_limit_is_checked_before_reading_payload(self):
        with patch("mph_gait_id.gallery_transfer.MAX_BYTES", 50):
            with self.assertRaisesRegex(ValueError, "512 MiB"):
                read_archive(self.archive)

    def test_line_endings_do_not_change_encoder_fingerprint(self):
        expected = _encoder_hash()
        original = Path.read_bytes
        with patch.object(Path, "read_bytes", lambda path: original(path).replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")):
            self.assertEqual(expected, _encoder_hash())

    def test_reexport_preserves_spec_even_if_local_weights_are_removed(self):
        self.import_b()
        self.sb.items = {}
        other = self.root / "no-weights.mphgallery"
        self.tb.export(other)
        self.assertEqual(read_archive(self.archive)["records"]["models"][0]["bundle_spec"],
                         read_archive(other)["records"]["models"][0]["bundle_spec"])

    def test_encoding_change_does_not_relabel_imported_features_on_export(self):
        self.import_b()
        other = self.root / "new-encoder.mphgallery"
        with patch("mph_gait_id.gallery_transfer._encoder_hash", return_value="changed"):
            self.tb.export(other)
            self.assertEqual(self.tb.preview(other)["compatible_embeddings"], 0)

    def test_deleted_imported_embedding_is_not_silently_restored(self):
        self.import_b()
        with self.b.connect() as con:
            con.execute("DELETE FROM gallery_embeddings")
        with self.assertRaisesRegex(ValueError, "removed"):
            self.import_b()

    def test_transfer_never_loads_checkpoint_pickle(self):
        with patch("torch.load", side_effect=AssertionError("Checkpoint load attempted")):
            self.ta.export(self.archive)
            self.import_b()

    def test_source_assigned_to_other_person_rolls_back(self):
        self.import_b()
        self.mutate_archive(lambda r, _c: (
            r["persons"][0].update(uid="other-person", person_id="P2"),
            [row.update(uid="other-" + row["uid"], person_uid="other-person") for row in r["embeddings"]],
        ))
        with self.assertRaisesRegex(ValueError, "another person"):
            self.import_b()
        self.assertIsNone(self.b.get_person("P2"))

    def test_empty_gallery_can_be_exported_and_imported(self):
        empty = self.root / "empty.mphgallery"
        self.tb.export(empty)
        self.assertEqual(self.import_b(empty)["inserted"], 0)


if __name__ == "__main__":
    unittest.main()
