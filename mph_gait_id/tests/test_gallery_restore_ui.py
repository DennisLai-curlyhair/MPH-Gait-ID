from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.ui.gallery_transfer import GalleryTransferDialog


class GalleryRestoreUiTest(unittest.TestCase):
    def make_dialog(self):
        flag = Mock()
        flag.get.return_value = False
        flag.set.side_effect = lambda value: setattr(flag.get, "return_value", value)
        dialog = SimpleNamespace(
            worker=SimpleNamespace(busy=False), operation="preview", restore_deleted=flag,
            restore_check=Mock(), people={}, person_map={}, person_tree=Mock(), model_tree=Mock(),
            status=Mock(), notes=Mock(), apply_button=Mock(), edit_buttons=[Mock(), Mock(), Mock()],
            controller=Mock(), path="test.mphgallery", preview=None, title=lambda: "Transfer",
            tr=lambda en, zh: en, _start=Mock(),
        )
        for name in ("_restore_changed", "_show_preview", "_set_target", "_actions", "_import", "_new_id", "_merge", "_skip"):
            setattr(dialog, name, MethodType(getattr(GalleryTransferDialog, name), dialog))
        dialog._selected = Mock(return_value=None)
        plan = {
            "persons": [
                {"uid": "deleted-free", "person_id": "P1", "display_name": "Alice", "status": "deleted",
                 "target_id": "P1", "target_exists": False, "restorable_embeddings": 2},
                {"uid": "deleted-used", "person_id": "P2", "display_name": "Bob", "status": "deleted",
                 "target_id": "P2", "target_exists": True, "restorable_embeddings": 3},
                {"uid": "linked", "person_id": "P3", "display_name": "Charlie", "status": "linked",
                 "target_id": "P3", "target_exists": True, "restorable_embeddings": 1},
            ],
            "models": {}, "compatible_embeddings": 6, "unavailable_embeddings": 0,
            "known_embedding_ids": 0, "deleted_embeddings": 6,
        }
        dialog._show_preview(plan)
        return dialog

    def test_restore_defaults_off_and_toggle_only_selects_unused_deleted_ids(self):
        dialog = self.make_dialog()
        self.assertFalse(dialog.restore_deleted.get())
        self.assertEqual(dialog.person_map, {"deleted-free": None, "deleted-used": None, "linked": "P3"})
        dialog.restore_deleted.set(True)
        dialog._restore_changed()
        self.assertEqual(dialog.person_map, {"deleted-free": "P1", "deleted-used": None, "linked": "P3"})
        dialog.person_tree.set.assert_any_call("deleted-used", "status", "Deleted: choose new ID")
        dialog.restore_deleted.set(False)
        dialog._restore_changed()
        self.assertIsNone(dialog.person_map["deleted-free"])
        self.assertEqual(dialog.person_map["linked"], "P3")

    def test_occupied_id_requires_new_id_not_merge(self):
        dialog = self.make_dialog()
        dialog.restore_deleted.set(True)
        dialog._selected.return_value = "deleted-used"
        dialog._actions()
        dialog.edit_buttons[0].configure.assert_called_with(state="normal")
        dialog.edit_buttons[1].configure.assert_called_with(state="disabled")
        dialog.controller.get_person.return_value = {"person_id": "P2"}
        with patch("mph_gait_id.ui.gallery_transfer.simpledialog.askstring", return_value="P2"), \
                patch("mph_gait_id.ui.gallery_transfer.messagebox.showerror") as error:
            dialog._new_id()
        error.assert_called_once()
        self.assertIsNone(dialog.person_map["deleted-used"])
        dialog.controller.get_person.return_value = None
        with patch("mph_gait_id.ui.gallery_transfer.simpledialog.askstring", return_value="RESTORED-P2"):
            dialog._new_id()
        self.assertEqual(dialog.person_map["deleted-used"], "RESTORED-P2")

    def test_restore_requires_additional_confirmation_and_captures_option(self):
        dialog = self.make_dialog()
        dialog.restore_deleted.set(True)
        dialog._restore_changed()
        with patch("mph_gait_id.ui.gallery_transfer.messagebox.askyesno", return_value=False) as confirm:
            dialog._import()
        self.assertIn("1 / 3", confirm.call_args.args[1])
        dialog._start.assert_not_called()
        with patch("mph_gait_id.ui.gallery_transfer.messagebox.askyesno", side_effect=[True, True]) as confirm:
            dialog._import()
        self.assertEqual(confirm.call_count, 2)
        task = dialog._start.call_args.args[1]
        dialog.restore_deleted.set(False)
        dialog.person_map["deleted-free"] = None
        task()
        self.assertTrue(dialog.controller.import_gallery.call_args.kwargs["restore_deleted"])
        self.assertEqual(dialog.controller.import_gallery.call_args.args[2]["deleted-free"], "P1")

    def test_existing_person_deleted_fragment_is_also_confirmed(self):
        dialog = self.make_dialog()
        dialog.restore_deleted.set(True)
        # Keep deleted people skipped; only P3's deleted fragment is selected.
        with patch("mph_gait_id.ui.gallery_transfer.messagebox.askyesno", return_value=False) as confirm:
            dialog._import()
        self.assertIn("0 / 1", confirm.call_args.args[1])
        dialog._start.assert_not_called()

    def test_busy_and_completed_dialog_cannot_start_another_import(self):
        dialog = self.make_dialog()
        for busy, operation in ((True, "preview"), (False, "import")):
            dialog.worker.busy, dialog.operation = busy, operation
            dialog._import()
        dialog._start.assert_not_called()

    def test_unchecked_restore_passes_false_and_keeps_deleted_people_skipped(self):
        dialog = self.make_dialog()
        with patch("mph_gait_id.ui.gallery_transfer.messagebox.askyesno", return_value=True) as confirm:
            dialog._import()
        self.assertEqual(confirm.call_count, 1)
        dialog._start.call_args.args[1]()
        self.assertFalse(dialog.controller.import_gallery.call_args.kwargs["restore_deleted"])
        self.assertIsNone(dialog.controller.import_gallery.call_args.args[2]["deleted-free"])


if __name__ == "__main__":
    unittest.main()
