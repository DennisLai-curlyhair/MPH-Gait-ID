from __future__ import annotations

import sqlite3
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.database import GalleryRepository
from mph_gait_id.gallery_lifecycle import GalleryLifecycle
from mph_gait_id.i18n import I18n
from mph_gait_id.ui.gallery_manager import GalleryManagerPage
from mph_gait_id.ui.main_window import GaitIdentityWindow
from mph_gait_id.tests.test_core import model_record


class GalleryLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = GalleryRepository(self.root / "gallery.sqlite3")
        self.repo.initialize()
        self.edits = GalleryLifecycle(self.repo)
        for key in ("model-a", "model-b"):
            self.repo.upsert_model(model_record(key))
        self.repo.upsert_person("p007", "Original")
        self.repo.upsert_person("p008", "Other")

    def add(self, person="p007", model="model-a", source="source-a", active=True,
            session="session", pass_id="pass-1", direction="front_facing"):
        ids = self.repo.add_embeddings(
            person, model, np.asarray([[1, 0]], dtype=np.float32),
            self.root / source, source, "folder", 1.0,
            [{"window": {"pass_id": pass_id, "requested_direction": direction},
              "source_metadata": {"realtime_enrollment": {"session_id": session}}}],
        )
        if not active:
            with self.repo.connect() as conn:
                conn.execute("UPDATE gallery_embeddings SET active=0 WHERE embedding_id=?", ids)
        return ids[0]

    def rows(self):
        with self.repo.connect() as conn:
            return [dict(row) for row in conn.execute(
                "SELECT * FROM gallery_embeddings ORDER BY embedding_id")]

    def fragment(self, source="source-a"):
        item = next(row for row in self.repo.list_gallery_passes("p007", ["model-a"])
                    if row["source_path"] == str(self.root / source))
        return {key: item[key] for key in (
            "model_key", "source_path", "session_id", "pass_id", "direction")}

    def assert_backup(self, result, count, name="Original"):
        path = Path(result["backup_path"])
        self.assertEqual(path.parent, self.root / "backups")
        conn = sqlite3.connect(path)
        try:
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM gallery_embeddings").fetchone()[0], count)
            self.assertEqual(conn.execute("SELECT display_name FROM persons WHERE person_id='p007'").fetchone()[0], name)
        finally:
            conn.close()

    def test_rename_preserves_features_and_status_across_models(self):
        self.add()
        self.add(model="model-b")
        self.add(active=False)
        before = self.rows()
        result = self.edits.apply(self.edits.preview("p007"), "rename", " 新名稱 ")
        self.assertEqual(self.rows(), before)
        self.assertEqual(self.repo.get_person("p007")["display_name"], "新名稱")
        for key in ("model-a", "model-b"):
            self.assertEqual(self.repo.load_gallery(key)[0]["display_name"], "新名稱")
        self.assert_backup(result, 3)

    def test_rename_does_not_reactivate_person_or_embeddings(self):
        self.add(active=False)
        with self.repo.connect() as conn:
            conn.execute("UPDATE persons SET status='inactive' WHERE person_id='p007'")
        self.edits.apply(self.edits.preview("p007"), "rename", "New")
        self.assertEqual(self.repo.get_person("p007")["status"], "inactive")
        self.assertEqual(self.rows()[0]["active"], 0)

    def test_fragment_deletion_is_scoped_to_person_model_source_and_direction(self):
        removed = {self.add(), self.add(active=False)}
        self.add(source="source-b")
        self.add(model="model-b")
        self.add(person="p008")
        self.add(direction="back_facing")
        fragment = self.fragment()
        fragment["direction"] = "front_facing"
        preview = self.edits.preview("p007", fragment)
        self.assertEqual((preview["active"], preview["inactive"]), (1, 1))
        result = self.edits.apply(preview, "delete_fragment")
        self.assertEqual(result["deleted_embeddings"], 2)
        self.assertEqual(len(self.rows()), 4)
        self.assertFalse(removed.intersection(row["embedding_id"] for row in self.rows()))
        self.assertIsNotNone(self.repo.get_person("p007"))
        self.assert_backup(result, 6)

    def test_legacy_null_and_empty_metadata_share_exact_displayed_scope(self):
        self.add(session=None, pass_id=None, direction=None, active=False)
        empty_id = self.add(session=None, pass_id=None, direction=None)
        self.add(source="source-b", session=None, pass_id=None, direction=None)
        with self.repo.connect() as conn:
            conn.execute("UPDATE gallery_embeddings SET session_id='', pass_id='', "
                         "direction='legacy_or_unknown' WHERE embedding_id=?", (empty_id,))
        rows = self.repo.list_gallery_passes("p007", ["model-a"])
        self.assertEqual(len(rows), 2)
        preview = self.edits.preview("p007", self.fragment())
        self.assertEqual(preview["total"], 2)
        self.edits.apply(preview, "delete_fragment")
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]["source_fingerprint"], "source-b")

    def test_full_delete_frees_id_across_all_models_but_retains_other_people(self):
        self.add(active=False)
        self.add(model="model-b")
        self.add(person="p008")
        result = self.edits.apply(self.edits.preview("p007"), "delete_person")
        self.assertIsNone(self.repo.get_person("p007"))
        self.assertEqual([row["person_id"] for row in self.rows()], ["p008"])
        self.assert_backup(result, 3)
        self.repo.upsert_person("p007", "Different person")
        self.add()
        self.assertEqual(self.repo.get_person("p007")["display_name"], "Different person")

    def test_zero_embedding_person_remains_manageable(self):
        self.add(active=False)
        self.edits.apply(self.edits.preview("p007", self.fragment()), "delete_fragment")
        self.assertEqual(self.repo.list_persons(["model-a"], include_inactive=True), [])
        self.assertIn("p007", [p["person_id"] for p in self.repo.list_persons(include_inactive=True)])
        self.edits.apply(self.edits.preview("p007"), "rename", "No features")
        self.edits.apply(self.edits.preview("p007"), "delete_person")
        self.assertIsNone(self.repo.get_person("p007"))

    def test_stale_confirmation_rejects_new_embeddings_without_backup_or_delete(self):
        self.add()
        preview = self.edits.preview("p007")
        self.add(model="model-b")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.edits.apply(preview, "delete_person")
        self.assertEqual(len(self.rows()), 2)
        self.assertFalse((self.root / "backups").exists())

    def test_stale_confirmation_rejects_activation_changes(self):
        self.add(active=False)
        preview = self.edits.preview("p007")
        with self.repo.connect() as conn:
            conn.execute("UPDATE gallery_embeddings SET active=1")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.edits.apply(preview, "delete_person")
        self.assertEqual(len(self.rows()), 1)

    def test_backup_failure_leaves_database_unchanged(self):
        self.add()
        before = self.rows()
        with patch.object(self.edits, "_backup", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.edits.apply(self.edits.preview("p007"), "delete_person")
        self.assertEqual(self.rows(), before)
        self.assertIsNotNone(self.repo.get_person("p007"))

    def test_delete_failure_rolls_back_embeddings_and_keeps_backup(self):
        self.add()
        with self.repo.connect() as conn:
            conn.execute("CREATE TRIGGER prevent_person_delete BEFORE DELETE ON persons "
                         "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        before = self.rows()
        with self.assertRaises(sqlite3.IntegrityError):
            self.edits.apply(self.edits.preview("p007"), "delete_person")
        self.assertEqual(self.rows(), before)
        self.assertIsNotNone(self.repo.get_person("p007"))
        self.assertEqual(len(list((self.root / "backups").glob("*.sqlite3"))), 1)

    def test_backup_includes_committed_wal(self):
        keeper = self.repo.connect()
        try:
            keeper.execute("PRAGMA journal_mode=WAL")
            self.add(active=False)
            result = self.edits.apply(self.edits.preview("p007"), "delete_person")
            self.assert_backup(result, 1)
        finally:
            keeper.close()

    def test_empty_name_and_wrong_action_scope_rejected(self):
        self.add()
        with self.assertRaises(ValueError):
            self.edits.apply(self.edits.preview("p007"), "rename", "  ")
        with self.assertRaises(ValueError):
            self.edits.apply(self.edits.preview("p007", self.fragment()), "delete_person")
        self.assertEqual(len(self.rows()), 1)

    def test_controller_rejects_wrong_person_or_model(self):
        self.add()
        controller = object.__new__(GaitApplicationController)
        controller.repository = self.repo
        controller._compatible_model_keys = Mock(return_value=["model-a"])
        row = self.repo.list_gallery_passes("p007", ["model-a"])[0]
        self.assertEqual(controller.preview_fragment_delete("bundle", "p007", row, 15)["total"], 1)
        with self.assertRaises(ValueError):
            controller.preview_fragment_delete("bundle", "p008", row, 15)
        with self.assertRaises(ValueError):
            controller.preview_fragment_delete("bundle", "p007", {**row, "model_key": "model-b"}, 15)

    def test_hidden_gallery_ui_rename_delete_refresh_and_localization(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        root.withdraw()
        self.addCleanup(root.destroy)
        self.add(active=False)
        controller = object.__new__(GaitApplicationController)
        controller.repository = self.repo
        controller.config = {}
        controller.model_store = Mock()
        controller._compatible_model_keys = Mock(return_value=["model-a"])
        controller.available_bundles = Mock(return_value={
            "bundle": SimpleNamespace(display_name="Test", checkpoint_sha256="a" * 64)})
        controller.processing_version_name = Mock(return_value="Test processing")
        i18n = I18n("en")
        i18n.locale = "en"
        root._gait_i18n = i18n
        on_changed = Mock()
        page = GalleryManagerPage(root, controller, on_changed, lambda: (True, ""))
        i18n.apply(page)
        root.update_idletasks()
        self.assertTrue(page.person_tree.exists("p007"))
        page.person_tree.selection_set("p007")
        with patch("mph_gait_id.ui.gallery_manager.simpledialog.askstring", return_value="Renamed"), \
             patch("mph_gait_id.ui.gallery_manager.messagebox.showinfo"), \
             patch("mph_gait_id.ui.gallery_manager.messagebox.showerror") as errors:
            page._edit_person("rename")
            self.assertEqual(self.repo.get_person("p007")["display_name"], "Renamed")
            self.assertEqual(page.person_tree.item("p007", "values")[1], "Renamed")
            page.pass_tree.selection_set(page.pass_tree.get_children()[0])
            with patch("mph_gait_id.ui.gallery_manager.messagebox.askyesno", return_value=True):
                page._delete_fragment()
            self.assertEqual(self.rows(), [])
            self.assertTrue(page.show_all_people.get())
            self.assertTrue(page.person_tree.exists("p007"))
            with patch("mph_gait_id.ui.gallery_manager.simpledialog.askstring", return_value="p007"):
                page._edit_person("delete_person")
            self.assertFalse(page.person_tree.exists("p007"))
            self.assertIsNone(self.repo.get_person("p007"))
            errors.assert_not_called()
        self.assertEqual(on_changed.call_count, 3)
        i18n.locale = "zh_TW"
        i18n.apply(page)
        buttons = [child for group in page.winfo_children() for section in group.winfo_children()
                   for child in section.winfo_children() if child.winfo_class() == "TButton"]
        self.assertIn("修改人物名稱（所有模型）", [button.cget("text") for button in buttons])


class GalleryEditGuardTest(unittest.TestCase):
    def test_main_window_guards_live_benchmark_worker_and_review(self):
        window = SimpleNamespace(
            realtime_page=SimpleNamespace(pipeline=SimpleNamespace(running=False), _pending_review_result=None),
            performance_page=SimpleNamespace(pipeline=SimpleNamespace(running=False)),
            worker=SimpleNamespace(busy=False), i18n=SimpleNamespace(locale="en", tr=lambda key: key),
        )
        self.assertTrue(GaitIdentityWindow._gallery_transfer_available(window)[0])
        for target, key, value in (
            (window.realtime_page.pipeline, "running", True),
            (window.performance_page.pipeline, "running", True),
            (window.worker, "busy", True),
            (window.realtime_page, "_pending_review_result", {}),
        ):
            old = getattr(target, key)
            setattr(target, key, value)
            self.assertFalse(GaitIdentityWindow._gallery_transfer_available(window)[0])
            setattr(target, key, old)

    def test_wrong_confirmation_id_does_not_delete(self):
        page = SimpleNamespace(
            _selected_person_id=lambda: "p007", _edit_allowed=lambda: True,
            controller=Mock(), _scope_text=lambda _: "scope", _text=lambda zh, en: en,
            _apply_edit=Mock(),
        )
        page.controller.preview_person_edit.return_value = {"person_id": "p007"}
        with patch("mph_gait_id.ui.gallery_manager.simpledialog.askstring", return_value="p008"), \
             patch("mph_gait_id.ui.gallery_manager.messagebox.showinfo"):
            GalleryManagerPage._edit_person(page, "delete_person")
        page._apply_edit.assert_not_called()

    def test_permission_is_rechecked_before_applying(self):
        page = SimpleNamespace(_edit_allowed=lambda: False, controller=Mock())
        GalleryManagerPage._apply_edit(page, {}, "delete_person")
        page.controller.apply_person_edit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
