from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.i18n import I18n
from mph_gait_id.realtime.types import PipelineSnapshot, DeviceStatus
from mph_gait_id.realtime.ui_page import RealtimePage
from mph_gait_id.ui.workflow import snapshot_feedback


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        settings = patch.dict(os.environ, {"MPH_GAIT_ID_SETTINGS": str(Path(temporary.name) / "settings.json")})
        settings.start()
        self.addCleanup(settings.stop)

    def test_missing_gallery_is_not_an_unknown_candidate(self):
        snapshot = PipelineSnapshot(state="no_gallery", message="", timestamp=0,
            result={"state": "no_gallery", "display_name": "Unknown",
                    "candidate_display_name": "Gallery empty", "accepted": False})
        title, detail = snapshot_feedback(snapshot, I18n("en").tr)
        self.assertEqual(title, "No compatible Gallery")
        self.assertIn("enroll", detail)

    def test_sensor_state_overrides_old_identity_result(self):
        for state in ("waiting_person", "insufficient_points", "multiple_people", "error"):
            snapshot = PipelineSnapshot(state=state, message="sensor error", timestamp=0,
                result={"accepted": True, "display_name": "Alice"})
            title, _ = snapshot_feedback(snapshot, I18n("en").tr)
            self.assertNotEqual(title, "Alice")

    def test_names_are_not_translated_and_decisions_are_not_recomputed(self):
        snapshot = PipelineSnapshot(state="stable", message="", timestamp=0,
            result={"accepted": True, "display_name": "Gallery Manager", "similarity": 0.1})
        for locale in ("en", "zh_TW"):
            title, _ = snapshot_feedback(snapshot, I18n(locale).tr)
            self.assertEqual(title, "Gallery Manager")
            self.assertTrue(snapshot.result["accepted"])


class RealtimeWorkflowTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.directory = directory
        settings = patch.dict(os.environ, {"MPH_GAIT_ID_SETTINGS": str(directory / "settings.json")})
        settings.start()
        self.addCleanup(settings.stop)
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(args)
        self.controller = GaitApplicationController(database_path=directory / "gallery.sqlite3",
                                                     output_root=directory / "outputs")
        self.root._gait_i18n = I18n("en")
        self.page = RealtimePage(self.root, self.controller)
        self.page.pack(fill="both", expand=True)
        self.root.geometry("1200x800")
        self.page.source_var.set(self.page._option_label(self.page.source_labels, "azure_kinect"))
        self.page._source_changed()
        self.page.operation_var.set(self.page._option_label(self.page.operation_labels, "enroll"))
        self.page.enrollment_person_id_var.set("test-id")
        self.page.enrollment_name_var.set("Gallery Manager")
        self.page._operation_changed()
        self.root.update()
        self.addCleanup(self.close)

    def close(self):
        for timer in self.root.tk.call("after", "info"):
            self.root.tk.call("after", "cancel", timer)
        self.root.destroy()
        self.assertFalse(self.errors, self.errors)

    def pipeline(self):
        pipe = SimpleNamespace(running=True, operation="enroll", poll_latest=Mock(return_value=None),
            start_enrollment_pass=Mock(), end_enrollment_pass=Mock(),
            finish_enrollment_for_review=Mock(), enrollment_passes=Mock(return_value=[]))
        self.page.pipeline = pipe
        self.page._session_active = True
        self.page._set_running(True)
        return pipe

    def snapshot(self, state, result_state=None, **result):
        snapshot = PipelineSnapshot(state=state, message="", timestamp=time.time(),
            frame_index=1 if state not in {"starting", "source_ready"} else -1,
            clip_len=15, operation="enroll", result=dict(operation="enroll",
                state=result_state or state, display_name="Gallery Manager", **result))
        self.page._show_snapshot(snapshot)
        return snapshot

    def test_preflight_does_not_need_gallery_for_enrollment(self):
        self.assertFalse(self.page.start_button.instate(["disabled"]))
        self.assertIn("not required", self.page.readiness_vars["gallery"].get())
        self.assertIn("background", self.page.readiness_vars["source"].get())
        self.page.enrollment_name_var.set("")
        self.root.update()
        self.assertTrue(self.page.start_button.instate(["disabled"]))
        self.assertIn("required", self.page.readiness_vars["identity"].get())
        self.page.clip_len_var.set("")
        self.root.update()
        self.assertIn("Check", self.page.readiness_vars["settings"].get())

    def test_sam_readiness_is_path_only_but_detector_loader_verifies_hash(self):
        from mph_gait_id.realtime.detection import resolve_sam_checkpoint
        checkpoint = self.directory / "sam.pth"
        checkpoint.write_bytes(b"not a trusted checkpoint")
        self.page.detector_var.set(self.page._option_label(self.page.detector_labels, "yolo_sam"))
        self.page.sam_checkpoint_var.set(str(checkpoint))
        with patch("mph_gait_id.realtime.detection._checkpoint_sha256",
                   side_effect=AssertionError("UI hashed a large checkpoint")):
            self.page._refresh_readiness()
        self.assertFalse(self.page._readiness_blocker)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            resolve_sam_checkpoint(checkpoint)

    def test_small_window_keeps_both_preview_viewports(self):
        self.root.geometry("980x640")
        self.root.update()
        self.assertGreater(self.page.rgb_label.winfo_height(), 40)
        self.assertGreater(self.page.cloud_label.winfo_height(), 40)

    def test_pass_buttons_wait_for_ready_and_acknowledgement(self):
        pipe = self.pipeline()
        self.snapshot("source_ready")
        self.assertTrue(self.page.start_pass_button.instate(["disabled"]))
        self.page._start_enrollment_pass()
        pipe.start_enrollment_pass.assert_not_called()
        self.snapshot("waiting_person", "enrollment_paused")
        self.assertFalse(self.page.start_pass_button.instate(["disabled"]))
        self.page._start_enrollment_pass()
        self.page._start_enrollment_pass()
        pipe.start_enrollment_pass.assert_called_once()
        self.assertEqual(self.page._pass_action, "start")
        self.page._update_pass_controls()
        self.assertTrue(self.page.start_pass_button.instate(["disabled"]))
        self.page._end_enrollment_pass(False)
        self.page._end_enrollment_pass(False)
        pipe.end_enrollment_pass.assert_called_once_with(discard=False)
        self.assertEqual(self.page._pass_action, "end")
        self.snapshot("waiting_person", "enrollment_paused")
        self.assertFalse(self.page.start_pass_button.instate(["disabled"]))
        self.page._finish_enrollment_for_review()
        self.page._finish_enrollment_for_review()
        pipe.finish_enrollment_for_review.assert_called_once()
        self.assertTrue(self.page.finish_review_button.instate(["disabled"]))

    def test_configuration_restores_original_states_not_defaults(self):
        self.page.device_var.set("cpu")
        self.page.stride_var.set(3)
        yolo_state = str(self.page.yolo_entry.cget("state"))
        pipe = self.pipeline()
        self.assertTrue(self.page.check_device_button.instate(["disabled"]))
        self.assertTrue(self.page.bundle_combo.instate(["disabled"]))
        self.page.i18n.set_locale("zh_TW")
        self.page.set_locale("zh_TW")
        self.assertTrue(self.page.start_pass_button.instate(["disabled"]))
        pipe.running = False
        self.page._session_active = False
        self.page._set_running(False)
        self.root.update()
        self.assertEqual(self.page.device_var.get(), "cpu")
        self.assertEqual(self.page.stride_var.get(), 3)
        self.assertEqual(str(self.page.yolo_entry.cget("state")), yolo_state)

    def test_review_locks_configuration_and_remembers_selection(self):
        self.page._pending_review_result = dict(display_name="Gallery Manager",
            usable_passes=1, captured_embeddings=2, passes=[dict(
                pass_id="pass-1", quality_accepted=True, selectable=True, embedding_count=2)])
        self.page._set_running(False)
        self.assertTrue(self.page.bundle_combo.instate(["disabled"]))
        self.page._open_enrollment_review()
        self.page._review_selection_vars["pass-1"].set(False)
        self.assertTrue(self.page._review_commit_button.instate(["disabled"]))
        self.page._close_enrollment_review()
        self.page._open_enrollment_review()
        self.assertFalse(self.page._review_selection_vars["pass-1"].get())
        self.page._review_selection_vars["pass-1"].set(True)
        self.page.commit_pending = True
        self.page._update_review_controls()
        self.assertTrue(self.page._review_commit_button.instate(["disabled"]))
        self.assertIn("Completing", self.page._review_status_var.get())
        self.page.commit_pending = False
        self.page._update_review_controls()
        self.assertFalse(self.page._review_commit_button.instate(["disabled"]))

    def test_stop_does_not_unlock_until_worker_exits(self):
        pipe = self.pipeline()
        self.snapshot("enrolling", active_pass_id="pass-1")
        gate = threading.Event()
        self.addCleanup(gate.set)
        def stop():
            gate.wait(2)
            pipe.running = False
        pipe.stop = stop
        self.page.stop()
        self.assertTrue(self.page._stop_pending)
        self.assertTrue(self.page.start_button.instate(["disabled"]))
        self.assertTrue(self.page.stop_button.instate(["disabled"]))
        self.page._poll()
        self.assertTrue(self.page._session_active)
        self.page.i18n.set_locale("zh_TW")
        self.page.set_locale("zh_TW")
        self.assertEqual(self.page.result_var.get(), self.page.i18n.tr("workflow.stopping"))
        gate.set()
        self.page.stop_worker._thread.join(timeout=3)
        self.page._poll()
        self.root.update()
        self.assertFalse(self.page._session_active)
        self.assertFalse(self.page._stop_pending)
        self.assertFalse(self.page.start_button.instate(["disabled"]))
        self.assertNotIn("Recording", self.page.pass_status_var.get())

    def test_terminal_snapshot_waits_for_cleanup_before_unlocking(self):
        pipe = self.pipeline()
        self.page._open_enrollment_review = Mock()
        snapshot = PipelineSnapshot(state="enrollment_review", message="", timestamp=time.time(),
            operation="enroll", result={"state": "enrollment_review", "passes": []})
        pipe.poll_latest.return_value = snapshot
        self.page._poll()
        self.assertTrue(self.page._session_active)
        self.assertTrue(self.page.bundle_combo.instate(["disabled"]))
        pipe.running = False
        pipe.poll_latest.return_value = None
        self.page._poll()
        self.assertFalse(self.page._session_active)
        self.assertIsNotNone(self.page._pending_review_result)
        self.assertTrue(self.page.start_button.instate(["disabled"]))
        self.assertFalse(self.page.finish_review_button.instate(["disabled"]))

    def test_failed_commit_keeps_review_and_can_retry(self):
        self.page.pipeline = SimpleNamespace(running=False, commit_enrollment=Mock(side_effect=OSError("test failure")))
        self.page._pending_review_result = dict(display_name="Alice", passes=[
            dict(pass_id="one", selectable=True, quality_accepted=True, embedding_count=2)])
        self.page._open_enrollment_review()
        with patch("mph_gait_id.realtime.ui_page.messagebox.showerror"):
            self.page._commit_enrollment_review()
            self.assertTrue(self.page.commit_pending)
            self.assertTrue(self.page._review_commit_button.instate(["disabled"]))
            self.page.commit_worker._thread.join(timeout=3)
            self.page._poll_commit()
        self.assertFalse(self.page.commit_pending)
        self.assertIsNotNone(self.page._pending_review_result)
        self.assertFalse(self.page._review_commit_button.instate(["disabled"]))
        self.assertTrue(self.page._review_selection_vars["one"].get())

    def test_device_probe_is_background_and_blocks_start(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        def probe(**_kwargs):
            gate.wait(2)
            return DeviceStatus(backend="test", connected=False, message="Not connected")
        with patch("mph_gait_id.realtime.ui_page.probe_azure_kinect", side_effect=probe):
            self.page._check_device()
            self.assertTrue(self.page._device_check_pending)
            self.assertTrue(self.page.start_button.instate(["disabled"]))
            gate.set()
            self.page.device_worker._thread.join(timeout=3)
            self.page._poll()
        self.assertEqual(self.page.readiness_vars["source"].get(), "Not connected")


if __name__ == "__main__":
    unittest.main()
