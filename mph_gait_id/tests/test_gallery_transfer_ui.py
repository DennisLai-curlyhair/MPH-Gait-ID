from __future__ import annotations

from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.ui.gallery_transfer import GalleryTransferDialog
from mph_gait_id.ui.main_window import GaitIdentityWindow


class TransferUiGuardTest(unittest.TestCase):
    def test_live_or_offline_work_blocks_transfer(self):
        host = SimpleNamespace(worker=SimpleNamespace(busy=False),
                               i18n=SimpleNamespace(locale="en"))
        self.assertTrue(GaitIdentityWindow._gallery_transfer_available(host)[0])
        host.worker.busy = True
        self.assertFalse(GaitIdentityWindow._gallery_transfer_available(host)[0])
        host.worker.busy = False
        host.realtime_page = SimpleNamespace(pipeline=SimpleNamespace(running=True))
        self.assertFalse(GaitIdentityWindow._gallery_transfer_available(host)[0])

    def test_conflicts_default_to_skip_in_preview(self):
        fake = SimpleNamespace(
            people={}, person_map={}, person_tree=Mock(), model_tree=Mock(), status=Mock(),
            notes=Mock(), apply_button=Mock(), tr=lambda en, zh: en,
            restore_deleted=Mock(), restore_check=Mock(),
        )
        GalleryTransferDialog._show_preview(fake, {
            "persons": [{"uid": "a", "person_id": "P1", "display_name": "A", "status": "conflict", "target_id": "P1"},
                        {"uid": "b", "person_id": "P2", "display_name": "B", "status": "new", "target_id": "P2"},
                        {"uid": "c", "person_id": "P3", "display_name": "C", "status": "deleted", "target_id": "P3"}],
            "models": {}, "compatible_embeddings": 0, "unavailable_embeddings": 0, "known_embedding_ids": 0,
        })
        self.assertEqual(fake.person_map, {"a": None, "b": "P2", "c": None})


class TransferDialogLayoutTest(unittest.TestCase):
    def test_dialog_builds_with_scrollbars_in_both_languages(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        root.withdraw()
        try:
            for locale in ("en", "zh_TW"):
                with patch.object(GalleryTransferDialog, "_choose"):
                    dialog = GalleryTransferDialog(root, Mock(), "import", locale)
                    dialog.geometry("640x420")
                    root.update()
                    self.assertTrue(dialog.person_tree.cget("xscrollcommand"))
                    self.assertTrue(dialog.person_tree.cget("yscrollcommand"))
                    self.assertGreater(dialog.person_tree.winfo_width(), 100)
                    dialog.close()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
