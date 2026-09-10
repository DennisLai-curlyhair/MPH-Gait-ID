from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.gallery_lifecycle import GalleryLifecycle
from mph_gait_id.gallery_transfer import read_archive
from mph_gait_id import gallery_transfer_cli
from mph_gait_id.tests import test_gallery_transfer as fixture


class GalleryRestoreTest(unittest.TestCase):
    setUp = fixture.GalleryTransferTest.setUp
    import_b = fixture.GalleryTransferTest.import_b

    def delete_person(self, repository=None):
        edits = GalleryLifecycle(repository or self.b)
        edits.apply(edits.preview("P1"), "delete_person")

    def delete_fragment(self, repository=None):
        repository = repository or self.b
        edits = GalleryLifecycle(repository)
        row = repository.list_gallery_passes("P1", [self.model["model_key"]])[0]
        fragment = {key: row[key] for key in (
            "model_key", "source_path", "session_id", "pass_id", "direction")}
        edits.apply(edits.preview("P1", fragment), "delete_fragment")

    def restore(self, transfer=None, mapping=None, enabled=True):
        transfer = transfer or self.tb
        preview = transfer.preview(self.archive)
        if mapping is None:
            mapping = {p["uid"]: p["target_id"] for p in preview["persons"]}
        return transfer.import_archive(self.archive, preview, mapping, restore_deleted=enabled)

    def state(self, transfer=None):
        transfer = transfer or self.tb
        with transfer.repository.connect() as con:
            return transfer._state(con)

    def audit(self, repository=None):
        with (repository or self.b).connect() as con:
            return [dict(row) for row in con.execute("SELECT * FROM gallery_transfer_restorations")]

    def test_native_export_delete_fragment_restore_preserves_vectors_and_states(self):
        with self.a.connect() as con:
            before = [(bytes(r[0]), r[1]) for r in con.execute(
                "SELECT embedding,active FROM gallery_embeddings ORDER BY embedding_id")]
        self.delete_fragment(self.a)
        self.assertEqual(self.ta.preview(self.archive)["persons"][0]["restorable_embeddings"], 2)
        result = self.restore(self.ta)
        self.assertEqual((result["inserted"], result["restored_persons"], result["restored_embeddings"]), (2, 0, 2))
        with self.a.connect() as con:
            after = [(bytes(r[0]), r[1]) for r in con.execute(
                "SELECT embedding,active FROM gallery_embeddings ORDER BY embedding_id")]
        self.assertEqual(before, after)
        query = np.array([0.8, 0.6], dtype=np.float32)
        self.assertEqual(float(np.frombuffer(before[0][0], dtype=np.float32) @ query),
                         float(self.a.load_gallery(self.model["model_key"])[0]["embedding"] @ query))

    def test_native_person_restore_reexport_preserves_uid_and_backup(self):
        uid = read_archive(self.archive)["records"]["persons"][0]["uid"]
        self.delete_person(self.a)
        before = self.state(self.ta)
        result = self.restore(self.ta)
        self.assertEqual((result["restored_persons"], result["restored_embeddings"]), (1, 2))
        self.assertEqual(self.a.get_person("P1")["display_name"], "Alice")
        path = self.root / "restored.mphgallery"
        self.ta.export(path)
        self.assertEqual(read_archive(path)["records"]["persons"][0]["uid"], uid)
        self.assertEqual(self.restore(self.ta)["duplicates"], 2)
        self.assertEqual(len(self.audit(self.a)), 3)
        with sqlite3.connect(result["backup"]) as con:
            self.assertEqual(self.ta._state(con), before)
        self.assertTrue(all(row["archive_sha256"] == Path(result["retained_archive"]).stem
                            for row in self.audit(self.a)))

    def test_restore_does_not_reset_local_name_person_status_or_inactive_entries(self):
        self.import_b()
        self.delete_fragment()
        with self.b.connect() as con:
            con.execute("UPDATE persons SET display_name='Local Alice', status='inactive'")
        self.restore()
        with self.b.connect() as con:
            con.execute("UPDATE gallery_embeddings SET active=0")
        result = self.restore()
        self.assertEqual((result["duplicates"], result["restored_embeddings"]), (2, 0))
        self.assertEqual(self.b.get_person("P1")["display_name"], "Local Alice")
        self.assertEqual(self.b.get_person("P1")["status"], "inactive")
        with self.b.connect() as con:
            self.assertEqual(con.execute("SELECT SUM(active) FROM gallery_embeddings").fetchone()[0], 0)

    def test_default_still_skips_deleted_features_and_requires_person_opt_in(self):
        self.import_b()
        self.delete_fragment()
        self.assertEqual(self.restore(enabled=False)["deleted_skipped"], 2)
        self.assertEqual(self.audit(), [])
        self.delete_person()
        with self.assertRaisesRegex(ValueError, "explicit restore"):
            self.restore(enabled=False)

    def test_reused_id_cannot_be_restored_or_merged_but_new_id_works(self):
        self.import_b()
        self.delete_person()
        self.b.upsert_person("P1", "Different person")
        preview = self.tb.preview(self.archive)
        uid = preview["persons"][0]["uid"]
        self.assertTrue(preview["persons"][0]["target_exists"])
        before = self.state()
        with self.assertRaisesRegex(ValueError, "unused, distinct"):
            self.restore()
        self.assertEqual(self.state(), before)
        result = self.restore(mapping={uid: "RESTORED-P1"})
        self.assertEqual((result["restored_persons"], result["restored_embeddings"]), (1, 2))
        self.assertEqual(self.b.get_person("P1")["display_name"], "Different person")
        self.assertEqual(self.b.get_person("RESTORED-P1")["display_name"], "Alice")
        with self.b.connect() as con:
            self.assertEqual(con.execute("SELECT DISTINCT person_id FROM gallery_embeddings").fetchall()[0][0], "RESTORED-P1")
        # The archived old UID now follows the restored ID, not the reused visible ID.
        self.assertEqual(self.import_b()["duplicates"], 2)

    def test_multiple_deleted_people_may_not_restore_to_one_new_id(self):
        self.a.upsert_person("P2", "Bob")
        self.ta.export(self.archive)
        self.import_b()
        self.delete_person()
        edits = GalleryLifecycle(self.b)
        edits.apply(edits.preview("P2"), "delete_person")
        plan = self.tb.preview(self.archive)
        before = self.state()
        with self.assertRaisesRegex(ValueError, "unused, distinct"):
            self.restore(mapping={p["uid"]: "NEW" for p in plan["persons"]})
        self.assertEqual(self.state(), before)

    def test_empty_person_can_be_restored(self):
        self.a.upsert_person("P2", "Bob")
        self.ta.export(self.archive)
        self.import_b()
        edits = GalleryLifecycle(self.b)
        edits.apply(edits.preview("P2"), "delete_person")
        result = self.restore()
        self.assertEqual((result["restored_persons"], result["restored_embeddings"]), (1, 0))
        self.assertEqual(self.b.get_person("P2")["display_name"], "Bob")

    def test_restore_failure_rolls_back_person_markers_and_audit(self):
        self.import_b()
        self.delete_person()
        with self.b.connect() as con:
            con.execute("CREATE TRIGGER block_restore BEFORE INSERT ON gallery_embeddings "
                        "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
        before = self.state()
        with self.assertRaises(sqlite3.IntegrityError):
            self.restore()
        self.assertEqual(self.state(), before)
        self.assertIsNone(self.b.get_person("P1"))
        self.assertEqual(self.audit(), [])

    def test_missing_model_retains_deleted_vectors_until_explicit_later_restore(self):
        self.import_b()
        self.delete_person()
        saved = self.sb.items.copy()
        self.sb.items.clear()
        result = self.restore()
        self.assertEqual((result["inserted"], result["restored_persons"], result["restored_embeddings"]), (0, 1, 0))
        self.sb.items.update(saved)
        self.assertEqual(self.restore(enabled=False)["deleted_skipped"], 2)
        self.assertEqual(self.restore()["restored_embeddings"], 2)

    def test_skip_person_does_not_clear_deletion_markers(self):
        self.import_b()
        self.delete_person()
        before = self.state()
        uid = self.tb.preview(self.archive)["persons"][0]["uid"]
        result = self.restore(mapping={uid: None})
        self.assertEqual((result["inserted"], result["restored_persons"], result["restored_embeddings"]), (0, 0, 0))
        self.assertEqual(self.state(), before)

    def test_restore_flag_does_not_bypass_stale_preview(self):
        self.import_b()
        self.delete_person()
        plan = self.tb.preview(self.archive)
        self.b.upsert_person("P1", "New owner")
        before = self.state()
        with self.assertRaisesRegex(ValueError, "Gallery changed"):
            self.tb.import_archive(self.archive, plan, {plan["persons"][0]["uid"]: "P1"}, restore_deleted=True)
        self.assertEqual(self.state(), before)

    def test_restored_person_can_be_deleted_and_explicitly_restored_again(self):
        self.import_b()
        self.delete_person()
        self.restore()
        self.delete_person()
        self.assertEqual(self.tb.preview(self.archive)["persons"][0]["status"], "deleted")
        with self.assertRaisesRegex(ValueError, "explicit restore"):
            self.restore(enabled=False)
        result = self.restore()
        self.assertEqual((result["restored_persons"], result["restored_embeddings"]), (1, 2))
        self.assertEqual(len(self.audit()), 6)

    def test_restore_does_not_bypass_source_ownership_collision(self):
        self.import_b()
        self.delete_person()
        self.b.upsert_person("P2", "Bob")
        self.b.add_embeddings("P2", self.model["model_key"], np.array([[1, 0]], dtype=np.float32),
                              self.root / "same-source", "sha256:source", "folder", 1., [{}])
        before = self.state()
        with self.assertRaisesRegex(ValueError, "another person"):
            self.restore()
        self.assertEqual(self.state(), before)
        self.assertIsNone(self.b.get_person("P1"))

    def test_unarchived_deleted_features_stay_deleted(self):
        self.import_b()
        self.b.add_embeddings("P1", self.model["model_key"], np.array([[0, 1]], dtype=np.float32),
                              self.root / "later-pass", "later-pass", "folder", 1., [{}])
        # Persist the later feature's UID, but do not add it to the archive to restore.
        self.tb.export(self.root / "later.mphgallery")
        self.delete_person()
        self.assertEqual(self.restore()["restored_embeddings"], 2)
        with self.b.connect() as con:
            self.assertGreater(con.execute("SELECT COUNT(*) FROM gallery_transfer_tombstones WHERE kind='embedding'").fetchone()[0], 0)
            self.assertEqual(con.execute("SELECT COUNT(*) FROM gallery_embeddings").fetchone()[0], 2)

    def test_restore_uses_explicit_boolean_and_controller_forwards_it(self):
        self.import_b()
        self.delete_person()
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.restore(enabled="false")
        plan = self.tb.preview(self.archive)
        controller = SimpleNamespace(gallery_transfer=lambda: self.tb)
        result = GaitApplicationController.import_gallery(
            controller, self.archive, plan, {plan["persons"][0]["uid"]: "P1"}, restore_deleted=True)
        self.assertEqual(result["restored_persons"], 1)

    def test_cli_requires_apply_and_exposes_restore_flag(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            gallery_transfer_cli.main(["import", str(self.archive), "--preview-file", "unused", "--restore-deleted"])
        self.import_b()
        self.delete_person()
        plan = self.tb.preview(self.archive)
        path = self.root / "preview.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        controller = SimpleNamespace(import_gallery=lambda *args, **kwargs: self.tb.import_archive(*args, **kwargs))
        with patch.object(gallery_transfer_cli, "GaitApplicationController", return_value=controller), patch("builtins.print"):
            self.assertEqual(gallery_transfer_cli.main([
                "import", str(self.archive), "--preview-file", str(path), "--restore-deleted", "--apply"]), 0)
        self.assertEqual(self.b.get_person("P1")["display_name"], "Alice")


if __name__ == "__main__":
    unittest.main()
