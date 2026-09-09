from __future__ import annotations

import sqlite3
import unittest

import numpy as np

from mph_gait_id.gallery_lifecycle import GalleryLifecycle
from mph_gait_id.gallery_transfer import read_archive
from mph_gait_id.tests import test_gallery_transfer as transfer_fixture


class GalleryManagementTransferTest(unittest.TestCase):
    setUp = transfer_fixture.GalleryTransferTest.setUp
    import_b = transfer_fixture.GalleryTransferTest.import_b
    mutate_archive = transfer_fixture.GalleryTransferTest.mutate_archive

    def lifecycle_b(self):
        return GalleryLifecycle(self.b)

    def test_rename_imported_person_preserves_uid_and_old_import_keeps_local_name(self):
        self.import_b()
        edits = self.lifecycle_b()
        edits.apply(edits.preview("P1"), "rename", "Alice-local")
        self.assertEqual(self.import_b()["duplicates"], 2)
        self.assertEqual(self.b.get_person("P1")["display_name"], "Alice-local")
        path = self.root / "renamed.mphgallery"
        self.tb.export(path)
        before = read_archive(self.archive)["records"]
        after = read_archive(path)["records"]
        self.assertEqual(before["persons"][0]["uid"], after["persons"][0]["uid"])
        self.assertEqual([e["vector"] for e in before["embeddings"]],
                         [e["vector"] for e in after["embeddings"]])

    def test_deleted_imported_fragment_is_skipped_without_restoration(self):
        self.import_b()
        edits = self.lifecycle_b()
        row = self.b.list_gallery_passes("P1", [self.model["model_key"]])[0]
        fragment = {key: row[key] for key in (
            "model_key", "source_path", "session_id", "pass_id", "direction")}
        result = edits.apply(edits.preview("P1", fragment), "delete_fragment")
        self.assertEqual(result["deleted_embeddings"], 2)
        self.assertEqual(self.tb.preview(self.archive)["deleted_embeddings"], 2)
        result = self.import_b()
        self.assertEqual((result["inserted"], result["deleted_skipped"]), (0, 2))
        self.assertIsNotNone(self.b.get_person("P1"))
        self.assertEqual(self.b.load_gallery(self.model["model_key"]), [])

    def test_deleted_person_cannot_be_remapped_after_readable_id_reuse(self):
        self.import_b()
        edits = self.lifecycle_b()
        edits.apply(edits.preview("P1"), "delete_person")
        self.b.upsert_person("P1", "Different person")
        preview = self.tb.preview(self.archive)
        person = preview["persons"][0]
        self.assertEqual(person["status"], "deleted")
        for target in ("P1", "RESTORED-P1"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "deleted person"):
                self.tb.import_archive(self.archive, preview, {person["uid"]: target})
        result = self.tb.import_archive(self.archive, preview, {person["uid"]: None})
        self.assertEqual(result["deleted_skipped"], 2)
        self.assertEqual(self.b.get_person("P1")["display_name"], "Different person")
        self.assertIsNone(self.b.get_person("RESTORED-P1"))
        self.assertEqual(self.b.load_gallery(self.model["model_key"]), [])

    def test_native_legacy_export_then_id_reuse_gets_new_stable_uid(self):
        old_uid = read_archive(self.archive)["records"]["persons"][0]["uid"]
        self.import_b()
        # Stage-A exports originally derived UIDs without persisting alias rows.
        with self.a.connect() as con:
            con.execute("DELETE FROM gallery_transfer_aliases")
        edits = GalleryLifecycle(self.a)
        edits.apply(edits.preview("P1"), "delete_person")
        self.a.upsert_person("P1", "New P1")
        first, second = self.root / "new1.mphgallery", self.root / "new2.mphgallery"
        self.ta.export(first)
        self.ta.export(second)
        uid = read_archive(first)["records"]["persons"][0]["uid"]
        self.assertNotEqual(old_uid, uid)
        self.assertEqual(uid, read_archive(second)["records"]["persons"][0]["uid"])
        self.assertEqual(self.ta.preview(self.archive)["persons"][0]["status"], "deleted")
        # A second PC that still has the old P1 must request an explicit decision.
        self.assertEqual(self.tb.preview(first)["persons"][0]["status"], "conflict")
        self.assertEqual(self.b.get_person("P1")["display_name"], "Alice")

    def test_new_enrollment_after_fragment_delete_can_be_exported(self):
        self.import_b()
        edits = self.lifecycle_b()
        row = self.b.list_gallery_passes("P1", [self.model["model_key"]])[0]
        fragment = {key: row[key] for key in (
            "model_key", "source_path", "session_id", "pass_id", "direction")}
        edits.apply(edits.preview("P1", fragment), "delete_fragment")
        self.b.add_embeddings("P1", self.model["model_key"], np.array([[0, 1]], dtype=np.float32),
                              self.root / "new-pass", "new-pass", "folder", 1.,
                              [{"window": {"window_index": 0}}])
        path = self.root / "fresh.mphgallery"
        self.tb.export(path)
        result = self.import_b(path)
        self.assertEqual((result["inserted"], result["duplicates"], result["deleted_skipped"]), (0, 1, 0))
        self.assertEqual(self.import_b()["deleted_skipped"], 2)
        self.assertEqual(len(self.b.load_gallery(self.model["model_key"])), 1)

    def test_deleting_merged_person_retires_every_known_alias(self):
        self.b.upsert_person("LOCAL", "Local name")
        first_uid = read_archive(self.archive)["records"]["persons"][0]["uid"]
        self.import_b(mapping={first_uid: "LOCAL"})
        self.mutate_archive(lambda records, _: (
            records["persons"][0].update(uid="second-origin-person", person_id="P2"),
            [e.update(uid="second-" + e["uid"], person_uid="second-origin-person",
                      source_fingerprint="second-source") for e in records["embeddings"]],
        ))
        self.import_b(mapping={"second-origin-person": "LOCAL"})
        edits = self.lifecycle_b()
        edits.apply(edits.preview("LOCAL"), "delete_person")
        with self.b.connect() as con:
            uids = {r[0] for r in con.execute("SELECT uid FROM gallery_transfer_tombstones WHERE kind='person'")}
        self.assertTrue({first_uid, "second-origin-person"}.issubset(uids))
        self.assertEqual(self.tb.preview(self.archive)["persons"][0]["status"], "deleted")

    def test_rollback_restores_aliases_and_leaves_no_tombstones(self):
        self.import_b()
        edits = self.lifecycle_b()
        with self.b.connect() as con:
            con.execute("CREATE TRIGGER prevent_delete BEFORE DELETE ON persons "
                        "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
            before = self.tb._state(con)
        with self.assertRaises(sqlite3.IntegrityError):
            edits.apply(edits.preview("P1"), "delete_person")
        with self.b.connect() as con:
            self.assertEqual(before, self.tb._state(con))
            self.assertEqual(con.execute("SELECT COUNT(*) FROM gallery_transfer_tombstones").fetchone()[0], 0)
        self.assertEqual(self.import_b()["duplicates"], 2)

    def test_delete_after_preview_invalidates_saved_import_plan(self):
        self.import_b()
        preview = self.tb.preview(self.archive)
        edits = self.lifecycle_b()
        edits.apply(edits.preview("P1"), "delete_person")
        with self.assertRaisesRegex(ValueError, "Gallery changed"):
            self.tb.import_archive(self.archive, preview, {preview["persons"][0]["uid"]: "P1"})

    def test_management_backup_contains_predelete_transfer_state(self):
        self.import_b()
        edits = self.lifecycle_b()
        with self.b.connect() as con:
            before = self.tb._state(con)
        result = edits.apply(edits.preview("P1"), "delete_person")
        with sqlite3.connect(result["backup_path"]) as con:
            self.assertEqual(self.tb._state(con), before)


if __name__ == "__main__":
    unittest.main()
