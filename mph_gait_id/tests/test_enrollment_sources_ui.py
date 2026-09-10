from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock

from mph_gait_id.database import GalleryRepository
from mph_gait_id.i18n import I18n
from mph_gait_id.realtime.ui_page import RealtimePage
from mph_gait_id.ui.enrollment_sources import EnrollmentSourcesPage


class SourceUiGuardTest(unittest.TestCase):
    def test_review_actions_are_blocked_during_background_commit(self):
        page = SimpleNamespace(commit_pending=True, pipeline=Mock(), _review_window=Mock())
        for action in (RealtimePage._commit_enrollment_review, RealtimePage._resume_enrollment_review,
                       RealtimePage._abandon_enrollment_review, RealtimePage._close_enrollment_review):
            action(page)
        page.pipeline.commit_enrollment.assert_not_called()
        page.pipeline.resume_enrollment_capture.assert_not_called()
        page.pipeline.abandon_enrollment.assert_not_called()
        page._review_window.destroy.assert_not_called()

    def test_source_page_builds_in_both_languages(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        try:
            root.geometry("1100x650")
            with tempfile.TemporaryDirectory() as raw:
                controller = SimpleNamespace(repository=GalleryRepository(Path(raw) / "gallery.sqlite3"))
                root._gait_i18n = I18n("en")
                page = EnrollmentSourcesPage(root, controller, can_edit=lambda: (True, ""))
                page.pack(fill="both", expand=True)
                for language in ("en", "zh_TW"):
                    root._gait_i18n.set_locale(language)
                    page.set_locale(language)
                    root.update()
                    self.assertTrue(page.tree.cget("xscrollcommand"))
                    self.assertTrue(page.tree.cget("yscrollcommand"))
                    self.assertGreater(page.canvas.winfo_width(), 100)
                    self.assertGreater(page.canvas.winfo_height(), 100)
                page.pause()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
