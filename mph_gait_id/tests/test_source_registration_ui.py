from __future__ import annotations

import threading
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.source_registration import RegistrationTarget
from mph_gait_id.ui.main_window import GaitIdentityWindow
from mph_gait_id.ui.source_registration import SourceRegistrationDialog
from mph_gait_id.ui.workers import BackgroundWorker, WorkerMessage


class SourceRegistrationUiTest(unittest.TestCase):
    def test_open_transfer_preview_blocks_starting_source_job(self):
        window = SimpleNamespace(gallery_page=SimpleNamespace(
            transfer_dialog=Mock(winfo_exists=Mock(return_value=True), worker=SimpleNamespace(busy=False))),
            i18n=SimpleNamespace(locale="en"), _gallery_transfer_available=Mock(return_value=(True, "")))
        self.assertFalse(GaitIdentityWindow._source_edit_available(window)[0])

    def test_source_job_blocks_camera_and_gallery_operations(self):
        window = SimpleNamespace(sources_page=SimpleNamespace(busy=True), worker=SimpleNamespace(busy=False),
                                 i18n=SimpleNamespace(locale="en"))
        window._source_operation_running = lambda: GaitIdentityWindow._source_operation_running(window)
        for method in (GaitIdentityWindow._camera_available_for_realtime,
                       GaitIdentityWindow._camera_available_for_benchmark,
                       GaitIdentityWindow._gallery_transfer_available):
            self.assertFalse(method(window)[0])

    def test_start_snapshots_selection_and_does_not_read_tk_variables_from_worker(self):
        variables = [Mock(get=Mock(return_value=v)) for v in (True, "15", True, "30", True, "10", "cpu")]
        first, first_t, second, second_t, selected_pass, maximum, device = variables
        fake = SimpleNamespace(running=False, worker=Mock(busy=False),
            page=Mock(can_edit=Mock(return_value=(True, ""))), title=lambda: "Test",
            targets=[("pc", first, first_t), ("mph", second, second_t)],
            passes=[("pass_001", selected_pass)], maximum=maximum, device=device,
            cancel_event=threading.Event(), _lock_form=Mock(), start_button=Mock(), cancel_button=Mock(),
            bar=Mock(), status=Mock(), result_tree=Mock(get_children=Mock(return_value=())),
            _append=Mock(), _t=lambda zh, en: en, after=Mock(), poll=Mock(),
            controller=Mock(), source_id="source-id")
        with patch("mph_gait_id.ui.source_registration.messagebox.askyesno", return_value=True):
            SourceRegistrationDialog.start(fake)
        self.assertTrue(fake.running)
        fake.page.prepare_encoding.assert_called_once()
        task = fake.worker.start.call_args.args[0]
        for variable in variables:
            variable.get.side_effect = AssertionError("Tk read from worker")
        progress = Mock()
        task(progress)
        fake.controller.register_from_source.assert_called_once_with("source-id",
            [RegistrationTarget("pc", 15), RegistrationTarget("mph", 30)], ["pass_001"],
            device="cpu", max_windows_per_pass=10, cancel=fake.cancel_event, progress=progress)

    def test_close_requests_cancellation_without_destroying_busy_dialog(self):
        fake = SimpleNamespace(running=True, worker=Mock(busy=True), cancel=Mock(), destroy=Mock())
        SourceRegistrationDialog.close(fake)
        fake.cancel.assert_called_once()
        fake.destroy.assert_not_called()

    def test_cancelled_report_is_not_presented_as_success(self):
        report = dict(status="cancelled", embeddings_added=0, job_id="job-test",
                      targets=[dict(bundle_id="mph", clip_len=15, status="not_saved",
                                    embeddings_added=0, passes=[])])
        fake = SimpleNamespace(
            worker=Mock(drain=Mock(return_value=[WorkerMessage("result", report)])),
            running=True, bar=Mock(), _lock_form=Mock(), start_button=Mock(), owner_available=True,
            cancel_button=Mock(), status=Mock(), _append=Mock(),
            result_tree=Mock(get_children=Mock(return_value=())),
            _t=lambda zh, en: en, page=Mock(), after=Mock(), poll=Mock())
        SourceRegistrationDialog.poll(fake)
        self.assertFalse(fake.running)
        self.assertIn("Cancelled", fake.status.set.call_args.args[0])
        self.assertIn("Embeddings added: 0", fake.status.set.call_args.args[0])
        self.assertEqual(fake.result_tree.insert.call_args.kwargs["values"],
                         ("mph", 15, "Not saved", 0))

    def test_late_progress_does_not_overwrite_cancelling_status(self):
        cancel = threading.Event()
        cancel.set()
        fake = SimpleNamespace(worker=Mock(drain=Mock(return_value=[WorkerMessage("progress", "1/20")])),
            cancel_event=cancel, status=Mock(), _append=Mock(), after=Mock(), poll=Mock())
        SourceRegistrationDialog.poll(fake)
        fake.status.set.assert_not_called()
        fake._append.assert_called_once_with("1/20")

    def test_dialog_bilingual_scrollable_layout(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        try:
            page = ttk.Frame(root)
            page.controller = Mock()
            page.controller.available_bundles.return_value = {
                "pc": SimpleNamespace(input_type="pointcloud", data={"clip_len": 15, "num_points": 1024},
                    display_name="PointNet-TMax", bundle_id="pc", checkpoint_sha256="a" * 64)}
            page.library = Mock()
            page.library.manifest.return_value = {"frames": [{"pass_id": "pass_001"}] * 30}
            page.library.list_sources.return_value = [dict(source_id="test", current_person_id="P1",
                current_name="Alice", captured_person_id="P1", captured_name="Alice")]
            page.source_id = "test"
            page.worker = BackgroundWorker()
            for language in ("en", "zh"):
                page._t = lambda zh, en: zh if language == "zh" else en
                dialog = SourceRegistrationDialog(page)
                dialog.geometry("650x500")
                root.update()
                self.assertEqual(len(dialog.targets), 1)
                self.assertGreater(dialog.result_tree.winfo_width(), 100)
                dialog.notebook.select(1)
                root.update()
                self.assertTrue(dialog.log.cget("yscrollcommand"))
                self.assertGreater(dialog.log.winfo_width(), 100)
                self.assertLessEqual(dialog.start_button.winfo_rooty() + dialog.start_button.winfo_height(),
                                     dialog.winfo_rooty() + dialog.winfo_height())
                dialog.close()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
