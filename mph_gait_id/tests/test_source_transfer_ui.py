from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.database import GalleryRepository
from mph_gait_id.i18n import I18n
from mph_gait_id.ui.enrollment_sources import EnrollmentSourcesPage
from mph_gait_id.ui.source_transfer import SourceTransferDialog


class SourceTransferUiTest(unittest.TestCase):
    def test_transfer_dialog_blocks_other_source_actions_even_at_preview(self):
        page = SimpleNamespace(worker=Mock(busy=False), registration_dialog=None,
                               transfer_dialog=Mock())
        page.transfer_dialog.winfo_exists.return_value = True
        self.assertTrue(EnrollmentSourcesPage.busy.fget(page))
        page.transfer_dialog.winfo_exists.return_value = False
        self.assertFalse(EnrollmentSourcesPage.busy.fget(page))

    def test_close_during_import_cancels_without_destroying_window(self):
        dialog = SimpleNamespace(running=True, cancel_event=threading.Event(), close_requested=False,
                                 status=Mock(), destroy=Mock(), tr=lambda en, zh: en)
        SourceTransferDialog.close(dialog)
        self.assertTrue(dialog.cancel_event.is_set())
        self.assertTrue(dialog.close_requested)
        dialog.destroy.assert_not_called()

    def test_import_snapshots_tk_values_before_background_job(self):
        dialog = SimpleNamespace(preview={"test": "preview"}, running=False,
            path="test.mphsources", person_map={"uid": "P1"}, restore_var=Mock(),
            title=lambda: "Source transfer", tr=lambda en, zh: en, _start=Mock(),
            cancel_event=threading.Event(), _service=Mock())
        dialog.restore_var.get.return_value = True
        with patch("mph_gait_id.ui.source_transfer.messagebox.askyesno", return_value=True):
            SourceTransferDialog._import(dialog)
        task = dialog._start.call_args.args[1]
        dialog.person_map["uid"] = "CHANGED"
        dialog.restore_var.get.side_effect = AssertionError("Tk accessed from worker")
        progress = Mock()
        task(progress)
        dialog._service.return_value.import_archive.assert_called_once_with(
            "test.mphsources", dialog.preview, {"uid": "P1"}, restore_deleted=True,
            cancel=dialog.cancel_event, progress=progress)

    def test_ui_preserves_configured_storage_limits(self):
        dialog = SimpleNamespace(page=SimpleNamespace(library=object()),
                                 storage_options={"library_limit_bytes": 12345, "min_free_bytes": 42})
        with patch("mph_gait_id.ui.source_transfer.SourceTransfer") as service:
            SourceTransferDialog._service(dialog)
        service.assert_called_once_with(dialog.page.library, library_limit_bytes=12345, min_free_bytes=42)

    def test_tk_register_callback_remains_unmodified(self):
        self.assertIs(SourceTransferDialog._register, tk.Misc._register)

    def test_import_dialog_layout_and_conflict_actions_in_both_languages(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        try:
            with tempfile.TemporaryDirectory() as directory:
                controller = SimpleNamespace(repository=GalleryRepository(Path(directory) / "gallery.sqlite3"))
                root.geometry("1100x700")
                root._gait_i18n = I18n("en")
                page = EnrollmentSourcesPage(root, controller, can_edit=lambda: (True, ""))
                page.pack(fill="both", expand=True)
                for language in ("en", "zh_TW"):
                    root._gait_i18n.set_locale(language)
                    page.set_locale(language)
                    with patch.object(SourceTransferDialog, "_choose"):
                        dialog = SourceTransferDialog(page, "import")
                        page.transfer_dialog = dialog
                        root.update()
                        dialog.operation = "preview"
                        dialog._show_preview({"persons": [
                            dict(uid="a", person_id="P1", display_name="Alice", status="new", target_id="P1"),
                            dict(uid="b", person_id="P1", display_name="Bob", status="new", target_id="P1"),
                        ], "sources": [], "frames": 0})
                        root.update()
                        self.assertEqual(dialog.person_map, {"a": "P1", "b": None})
                        self.assertTrue(page.busy)
                        self.assertTrue(dialog.source_tree.cget("xscrollcommand"))
                        self.assertTrue(dialog.person_tree.cget("yscrollcommand"))
                        self.assertLessEqual(dialog.close_button.winfo_rootx() + dialog.close_button.winfo_width(),
                                             dialog.winfo_rootx() + dialog.winfo_width())
                        dialog.close()
                        self.assertFalse(page.busy)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
