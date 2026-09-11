from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

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


class SourcePlaybackTimerTest(unittest.TestCase):
    def setUp(self):
        # Use real Tcl callbacks without requiring a Tk window or display.
        self.interpreter = tk.Tcl()
        self.page = object.__new__(EnrollmentSourcesPage)
        self.page.tk = self.interpreter.tk
        self.page.master = None
        self.page._tclCommands = []
        self.page.timer = None
        self.page.index = 0
        self.page.frames = [
            {"timestamp": i * 0.01, "segment_start": i == 0} for i in range(3)
        ]
        self.page.seek = Mock()
        self.page.detail = Mock()
        self.page._t = lambda zh, en: en
        self.page._report_exception = Mock()
        self.addCleanup(self._cleanup_timers)

    def _cleanup_timers(self):
        for timer in self.interpreter.call("after", "info"):
            self.page.after_cancel(timer)
        self.page._report_exception.assert_not_called()

    def _advance_timer(self):
        self.interpreter.call("after", 20)
        self.interpreter.eval("update")

    def test_tk_callback_registration_is_not_overridden(self):
        self.assertIs(EnrollmentSourcesPage._register, tk.Misc._register)

    def test_play_advances_to_end_and_can_restart(self):
        self.page._play()
        self.assertIsNotNone(self.page.timer)
        self._advance_timer()
        self.assertEqual(self.page.index, 1)
        self.page.seek.set.assert_called_with(1)
        self._advance_timer()
        self.assertEqual(self.page.index, 2)
        self.assertIsNone(self.page.timer)
        self.page._play()
        self.assertEqual(self.page.index, 0)
        self.page.seek.set.assert_called_with(0)
        self.assertIsNotNone(self.page.timer)

    def test_pause_cancels_callback_and_repeated_play_keeps_one_timer(self):
        self.page._play()
        self.page._play()
        self.assertEqual(len(self.interpreter.call("after", "info")), 1)
        self.page.pause()
        self.assertIsNone(self.page.timer)
        self._advance_timer()
        self.assertEqual(self.page.index, 0)
        self.assertFalse(self.interpreter.call("after", "info"))

    def test_sampling_break_does_not_schedule_playback(self):
        self.page.frames[1]["segment_start"] = True
        self.page._play()
        self.assertIsNone(self.page.timer)
        self.page.detail.set.assert_called_with("Sampling break; select Next to continue.")

    def test_empty_source_does_not_schedule_playback(self):
        self.page.frames = []
        self.page._play()
        self.assertIsNone(self.page.timer)

    def test_source_mutation_can_schedule_worker_poll(self):
        self.page.can_edit = lambda: (True, "")
        self.page.worker = Mock(busy=False)
        self.page.summary = Mock()
        task = Mock()
        with patch("mph_gait_id.ui.enrollment_sources.messagebox.askyesno", return_value=True):
            self.page._mutate(task, "Confirm")
        self.page.worker.start.assert_called_once()
        self.assertEqual(len(self.interpreter.call("after", "info")), 1)

    def test_model_registration_action_still_opens_dialog(self):
        self.page.registration_dialog = None
        self.page.can_edit = lambda: (True, "")
        self.page.worker = Mock(busy=False)
        self.page.source_id = "source-test"
        with patch("mph_gait_id.ui.source_registration.SourceRegistrationDialog") as dialog:
            self.page._open_model_registration()
        dialog.assert_called_once_with(self.page)
        self.assertIs(self.page.registration_dialog, dialog.return_value)


if __name__ == "__main__":
    unittest.main()
