from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sqlite3
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch
import uuid
import zipfile

import numpy as np

from mph_gait_id.database import GalleryRepository
from mph_gait_id.enrollment_sources import EnrollmentSourceLibrary, ForegroundCapture, _Lease, _json
from mph_gait_id.gallery_lifecycle import GalleryLifecycle
from mph_gait_id.source_transfer import SourceTransfer, SourceTransferCancelled


class SourceTransferTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.a = EnrollmentSourceLibrary(GalleryRepository(self.root / "a" / "gallery.sqlite3"))
        self.b = EnrollmentSourceLibrary(GalleryRepository(self.root / "b" / "gallery.sqlite3"))
        self.ta = SourceTransfer(self.a, min_free_bytes=0)
        self.tb = SourceTransfer(self.b, min_free_bytes=0)
        self.sid = self.capture()
        self.archive = self.root / "sources.mphsources"
        self.ta.export(self.archive, [self.sid])

    def capture(self, offset=0, person="P1", fortran=False):
        self.a.repository.upsert_person(person, "Alice" if person == "P1" else "Bob")
        capture = ForegroundCapture(self.a, min_free_bytes=0, queue_size=32)
        try:
            for pid in ("pass_001", "pass_002"):
                for i in range(4):
                    points = np.arange((8+i)*3, dtype=np.float32).reshape(-1, 3) + 1000 + offset
                    if fortran:
                        points = np.asfortranarray(points)
                    capture.append(pid, points, frame_index=i, timestamp=i / 15.,
                                   segment_start=i == 0, metadata={"private_path": "PRIVATE_SENTINEL"})
            prepared = capture.prepare({"pass_001", "pass_002"}, {
                "session_id": str(uuid.uuid4()),
                "capture": {"preprocessing_profile_id": "foreground-v1",
                            "preprocessing_source_sha256": "a" * 64,
                            "filter_parameters": {"max_valid_frame_gap_s": 1.},
                            "yolo_weights": "/PRIVATE_SENTINEL/yolo.pt", "device_serial": "PRIVATE_SENTINEL"},
                "model": {"checkpoint": "/PRIVATE_SENTINEL/model.pt"},
                "passes": [{"pass_id": "pass_001", "requested_direction": "left_to_right"},
                           {"pass_id": "pass_002", "requested_direction": "right_to_left"}],
                "windows": [{"private_path": "PRIVATE_SENTINEL"}],
                "consent": "operator_opt_in_at_enrollment_start",
                "selected_pass_ids": ["pass_001", "pass_002"],
            })
            with self.a.repository.connect() as con:
                prepared.attach(con, person, "Alice" if person == "P1" else "Bob", [])
            sid = capture.source_id
            prepared.finish()
            return sid
        finally:
            if not capture.closed:
                capture.abandon()

    def import_b(self, path=None, preview=None, mapping=None, **kwargs):
        path = path or self.archive
        preview = preview or self.tb.preview(path)
        mapping = mapping or {p["uid"]: p["target_id"] for p in preview["persons"]}
        return self.tb.import_archive(path, preview, mapping, **kwargs)

    def rewrite(self, change):
        with zipfile.ZipFile(self.archive) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
        manifest = json.loads(contents["manifest.json"])
        change(manifest, contents)
        contents["manifest.json"] = _json(manifest)
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)

    def assert_empty_b(self):
        self.assertEqual(self.b.list_sources(), [])
        self.assertEqual(self.b.repository.list_persons(), [])
        self.assertEqual(list((self.b.root / "records").glob("*")), [])
        self.assertEqual(list((self.b.root / "staging").glob("*")), [])

    def test_round_trip_points_timing_and_identity_are_exact(self):
        report = self.import_b()
        self.assertEqual(report["inserted"], 1)
        original, imported = self.a.manifest(self.sid), self.b.manifest(self.sid)
        for key in ("source_id", "created_at", "person_uid", "units", "coordinate_convention", "storage_stage"):
            self.assertEqual(imported[key], original[key])
        self.assertEqual(imported["metadata"]["session_id"], original["metadata"]["session_id"])
        self.assertEqual(imported["metadata"]["passes"], original["metadata"]["passes"])
        for old, new in zip(original["frames"], imported["frames"]):
            self.assertEqual({k: v for k, v in old.items() if k != "metadata"},
                             {k: v for k, v in new.items() if k != "metadata"})
            np.testing.assert_array_equal(self.a.load_frame(self.sid, old), self.b.load_frame(self.sid, new))
        self.assertEqual(self.b.list_sources()[0]["current_name"], "Alice")
        with sqlite3.connect(report["backup"]) as con:
            self.assertEqual(con.execute("SELECT count(*) FROM enrollment_sources").fetchone()[0], 0)
        with self.b.repository.connect() as con:
            self.assertEqual(con.execute("SELECT count(*) FROM gallery_embeddings").fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT count(*) FROM enrollment_source_transfers").fetchone()[0], 1)

    def test_export_excludes_local_paths_models_rgb_and_embeddings(self):
        with zipfile.ZipFile(self.archive) as archive:
            payload = archive.read("manifest.json")
            self.assertNotIn(b"PRIVATE_SENTINEL", payload)
            self.assertNotIn(b"original_embedding_ids", payload)
            self.assertNotIn(str(self.root).encode(), payload)
            self.assertEqual(len(archive.namelist()), 9)
            self.assertTrue(all(n == "manifest.json" or n.endswith(".npy") for n in archive.namelist()))

    def test_preview_does_not_import_data(self):
        preview = self.tb.preview(self.archive)
        self.assertEqual(preview["frames"], 8)
        self.assertEqual(preview["sources"][0]["status"], "new")
        self.assert_empty_b()

    def test_duplicates_reexport_and_different_archive_path(self):
        self.import_b()
        other = self.root / "renamed.mphsources"
        self.tb.export(other, [self.sid])
        report = self.import_b(other)
        self.assertEqual((report["inserted"], report["duplicates"]), (0, 1))
        self.assertEqual(len(self.b.list_sources()), 1)

    def test_delete_source_then_import_restores_it(self):
        self.import_b()
        self.b.delete(self.sid)
        self.assertEqual(self.import_b()["inserted"], 1)
        self.assertTrue(self.b.load_frame(self.sid, self.b.manifest(self.sid)["frames"][0]).size)

    def test_person_id_collision_can_map_to_new_id(self):
        self.b.repository.upsert_person("P1", "Different person")
        preview = self.tb.preview(self.archive)
        self.assertEqual(preview["persons"][0]["status"], "conflict")
        uid = preview["persons"][0]["uid"]
        self.import_b(preview=preview, mapping={uid: "P2"})
        self.assertEqual(self.b.repository.get_person("P1")["display_name"], "Different person")
        self.assertEqual(self.b.list_sources()[0]["current_person_id"], "P2")
        self.assertEqual(self.b.manifest(self.sid)["captured_person_id"], "P1")

    def test_explicit_merge_preserves_current_name_and_inactive_state(self):
        self.b.repository.upsert_person("P2", "Current name")
        with self.b.repository.connect() as con:
            con.execute("UPDATE persons SET status='inactive'")
        p = self.tb.preview(self.archive)
        self.import_b(preview=p, mapping={p["persons"][0]["uid"]: "P2"})
        self.assertEqual(self.b.repository.get_person("P2")["display_name"], "Current name")
        self.assertEqual(self.b.repository.get_person("P2")["status"], "inactive")

    def test_deleted_owner_needs_explicit_restore_and_unused_id(self):
        self.import_b()
        lifecycle = GalleryLifecycle(self.b.repository)
        lifecycle.apply(lifecycle.preview("P1"), "delete_person")
        self.b.repository.upsert_person("P1", "Another person")
        p = self.tb.preview(self.archive)
        uid = p["persons"][0]["uid"]
        self.assertEqual(p["persons"][0]["status"], "deleted")
        with self.assertRaisesRegex(ValueError, "explicit restore"):
            self.import_b(preview=p, mapping={uid: "P2"})
        with self.assertRaisesRegex(ValueError, "explicit restore"):
            self.import_b(preview=p, mapping={uid: "P1"}, restore_deleted=True)
        report = self.import_b(preview=p, mapping={uid: "P2"}, restore_deleted=True)
        self.assertEqual(report["restored_persons"], 1)
        self.assertEqual(self.b.list_sources()[0]["current_person_id"], "P2")
        self.assertEqual(self.b.repository.get_person("P1")["display_name"], "Another person")

    def test_linked_owner_cannot_be_remapped(self):
        self.import_b()
        p = self.tb.preview(self.archive)
        with self.assertRaisesRegex(ValueError, "cannot be reassigned"):
            self.import_b(preview=p, mapping={p["persons"][0]["uid"]: "P2"})
        self.assertIsNone(self.b.repository.get_person("P2"))

    def test_archive_and_database_changes_require_new_preview(self):
        p = self.tb.preview(self.archive)
        self.b.repository.upsert_person("P9", "Other")
        with self.assertRaisesRegex(ValueError, "preview again"):
            self.import_b(preview=p)
        p = self.tb.preview(self.archive)
        self.rewrite(lambda m, c: m.update(created_at="changed"))
        with self.assertRaisesRegex(ValueError, "Archive changed"):
            self.import_b(preview=p)

    def test_bad_frame_checksum_is_rejected(self):
        def change(m, c):
            name = next(n for n in c if n.endswith(".npy"))
            c[name] = c[name][:-1] + bytes([c[name][-1] ^ 1])
        self.rewrite(change)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.tb.preview(self.archive)
        self.assert_empty_b()

    def test_unexpected_traversal_and_symlink_members_are_rejected(self):
        original = self.archive.read_bytes()
        for name in ("../escaped.npy", "/tmp/escaped.npy", "C:\\escaped.npy", "models/model.pt"):
            with self.subTest(name=name):
                self.archive.write_bytes(original)
                with zipfile.ZipFile(self.archive, "a") as archive:
                    archive.writestr(name, b"x")
                with self.assertRaises(ValueError):
                    self.tb.preview(self.archive)
        self.archive.write_bytes(original)
        with zipfile.ZipFile(self.archive, "a") as archive:
            member = zipfile.ZipInfo("link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(member, "../outside")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.tb.preview(self.archive)
        self.assert_empty_b()

    def test_duplicate_zip_and_json_fields_are_rejected(self):
        original = self.archive.read_bytes()
        import warnings
        with warnings.catch_warnings(), zipfile.ZipFile(self.archive, "a") as archive:
            warnings.simplefilter("ignore")
            archive.writestr("manifest.json", b"{}")
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            self.tb.preview(self.archive)
        self.archive.write_bytes(original)
        with zipfile.ZipFile(self.archive) as archive:
            items = {n: archive.read(n) for n in archive.namelist()}
        items["manifest.json"] = items["manifest.json"].replace(b'{', b'{"schema":"bad",', 1)
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, data in items.items():
                archive.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.tb.preview(self.archive)

    def test_invalid_npy_shapes_objects_nonfinite_and_truncation_rejected(self):
        original = self.archive.read_bytes()
        def npy(value):
            stream = io.BytesIO()
            np.save(stream, value)
            return stream.getvalue()
        huge = io.BytesIO()
        np.lib.format.write_array_header_1_0(huge, dict(descr="<f4", fortran_order=False, shape=(10**12, 3)))
        for payload in (npy(np.zeros((8, 4), np.float32)), npy(np.zeros((8, 3), np.float64)),
                        npy(np.full((8, 3), np.nan, np.float32)), npy(np.zeros((8, 3), object)),
                        npy(np.zeros((8, 3), np.float32))[:-1], huge.getvalue()):
            with self.subTest(size=len(payload)):
                self.archive.write_bytes(original)
                def change(m, c):
                    source = m["sources"][0]
                    frame = source["frames"][0]
                    frame["size_bytes"] = len(payload)
                    frame["sha256"] = hashlib.sha256(payload).hexdigest()
                    c[f"sources/{source['source_id']}/{frame['file']}"] = payload
                self.rewrite(change)
                with self.assertRaises(ValueError):
                    self.tb.preview(self.archive)
        self.assert_empty_b()

    def test_fortran_layout_capture_is_portable(self):
        sid = self.capture(offset=1, fortran=True)
        self.ta.export(self.archive, [sid])
        self.import_b()
        for frame in self.a.manifest(sid)["frames"]:
            np.testing.assert_array_equal(self.a.load_frame(sid, frame), self.b.load_frame(sid, frame))

    def test_missing_temporal_boundary_and_wrong_coordinate_convention_rejected(self):
        original = self.archive.read_bytes()
        for update in (lambda s: s["frames"][4].update(segment_start=False),
                       lambda s: s.update(units="m"), lambda s: s.update(centered=True),
                       lambda s: s["metadata"]["capture"].pop("preprocessing_profile_id")):
            self.archive.write_bytes(original)
            self.rewrite(lambda m, c: update(m["sources"][0]))
            with self.assertRaises(ValueError):
                self.tb.preview(self.archive)

    def test_same_uuid_changed_content_and_new_uuid_same_content_conflict(self):
        self.import_b()
        original = self.archive.read_bytes()
        self.rewrite(lambda m, c: m["sources"][0]["metadata"].update(session_id="wrong-session"))
        p = self.tb.preview(self.archive)
        self.assertEqual(p["sources"][0]["status"], "conflict")
        with self.assertRaisesRegex(ValueError, "collision"):
            self.import_b(preview=p)
        self.archive.write_bytes(original)
        def change(m, c):
            source = m["sources"][0]
            old, new = source["source_id"], str(uuid.uuid4())
            source["source_id"] = new
            for name in list(c):
                if name.startswith(f"sources/{old}/"):
                    c[name.replace(old, new)] = c.pop(name)
        self.rewrite(change)
        with self.assertRaisesRegex(ValueError, "collision"):
            self.import_b()

    def test_skip_person_with_conflicting_source_still_imports_other_person(self):
        self.import_b()
        second = self.capture(offset=8, person="P2")
        self.ta.export(self.archive, [self.sid, second])
        self.rewrite(lambda m, c: m["sources"][0]["metadata"].update(session_id="wrong"))
        p = self.tb.preview(self.archive)
        mapping = {r["uid"]: None if r["person_id"] == "P1" else "P2" for r in p["persons"]}
        result = self.import_b(preview=p, mapping=mapping)
        self.assertEqual((result["inserted"], result["skipped"]), (1, 1))

    def test_cancel_during_export_preserves_existing_destination(self):
        before = self.archive.read_bytes()
        cancel = threading.Event()
        with self.assertRaises(SourceTransferCancelled):
            self.ta.export(self.archive, [self.sid], cancel=cancel, progress=lambda _: cancel.set())
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_cancel_during_import_rolls_back_and_removes_staging(self):
        cancel = threading.Event()
        with self.assertRaises(SourceTransferCancelled):
            self.import_b(cancel=cancel, progress=lambda _: cancel.set())
        self.assert_empty_b()

    def test_second_sqlite_insert_failure_rolls_back_files_and_people(self):
        second = self.capture(offset=8, person="P2")
        self.ta.export(self.archive, [self.sid, second])
        with self.b.repository.connect() as con:
            con.execute(f"CREATE TRIGGER fail_source BEFORE INSERT ON enrollment_sources WHEN NEW.source_id='{second}' "
                        "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.import_b()
        self.assert_empty_b()
        with self.b.repository.connect() as con:
            self.assertEqual(con.execute("SELECT count(*) FROM gallery_transfer_aliases").fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT count(*) FROM enrollment_source_transfers").fetchone()[0], 0)

    def test_storage_limits_and_writer_lease_prevent_import(self):
        p = self.tb.preview(self.archive)
        self.tb.library_limit = 1
        with self.assertRaisesRegex(OSError, "storage limit"):
            self.import_b(preview=p)
        self.assert_empty_b()
        with _Lease(self.b.root):
            with self.assertRaises(RuntimeError):
                self.tb.preview(self.archive)
        with patch("mph_gait_id.source_transfer.MAX_MANIFEST", 1):
            with self.assertRaisesRegex(ValueError, "manifest exceeds"):
                self.tb.preview(self.archive)

    def test_corrupted_local_duplicate_requires_repair_not_silent_skip(self):
        self.import_b()
        frame = self.b.manifest(self.sid)["frames"][0]
        (self.b.record_dir(self.sid) / frame["file"]).unlink()
        with self.assertRaises((ValueError, OSError)):
            self.tb.preview(self.archive)
        self.b.delete(self.sid)
        self.assertEqual(self.import_b()["inserted"], 1)

    def test_export_rejects_uncommitted_and_orphan_sources(self):
        with self.assertRaises(ValueError):
            self.ta.export(self.archive, [str(uuid.uuid4())])
        lifecycle = GalleryLifecycle(self.a.repository)
        lifecycle.apply(lifecycle.preview("P1"), "delete_person")
        with self.assertRaisesRegex(ValueError, "owner was deleted"):
            self.ta.export(self.archive, [self.sid])


class SourceTransferRegistrationTest(unittest.TestCase):
    def test_transferred_sources_reencode_identically_and_gallery_transfer_deduplicates(self):
        from mph_gait_id.tests.test_source_registration import SourceRegistrationTest, Store
        from mph_gait_id.gallery_transfer import GalleryTransfer
        from mph_gait_id.source_registration import SourceRegistrationService
        fixture = SourceRegistrationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.run_job()
        tensors = [t.clone() for _, t in fixture.inputs]
        weights = Store(fixture.root / "portable-bundles", "first")
        weights.items = fixture.store.items.copy()
        source_path, gallery_path = fixture.root / "source.mphsources", fixture.root / "gallery.mphgallery"
        SourceTransfer(fixture.library, min_free_bytes=0).export(source_path, [fixture.source_id])
        GalleryTransfer(fixture.repository, fixture.store, "foreground-v1").export(gallery_path)
        for order in ("source-first", "gallery-first", "reencode"):
            with self.subTest(order=order):
                library = EnrollmentSourceLibrary(GalleryRepository(fixture.root / order / "gallery.sqlite3"))
                sources = SourceTransfer(library, min_free_bytes=0)
                gallery = GalleryTransfer(library.repository, weights, "foreground-v1")
                def import_gallery():
                    p = gallery.preview(gallery_path)
                    return gallery.import_archive(gallery_path, p, {r["uid"]: r["target_id"] for r in p["persons"]})
                if order == "gallery-first":
                    import_gallery()
                p = sources.preview(source_path)
                sources.import_archive(source_path, p, {r["uid"]: r["target_id"] for r in p["persons"]})
                if order == "source-first":
                    import_gallery()
                service = SourceRegistrationService(library, fixture.store, "foreground-v1",
                                                     runtime_factory=fixture.service.runtime_factory)
                fixture.inputs.clear()
                report = service.run(fixture.source_id, fixture.targets, ["pass_001", "pass_002"], device="cpu")
                if order == "reencode":
                    self.assertEqual(report["embeddings_added"], original["embeddings_added"])
                    for before, (_, after) in zip(tensors, fixture.inputs):
                        np.testing.assert_array_equal(before, after)
                else:
                    self.assertEqual(report["embeddings_added"], 0)


if __name__ == "__main__":
    unittest.main()
