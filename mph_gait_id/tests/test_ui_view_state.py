from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.i18n import I18n
from mph_gait_id.realtime.ui_page import RealtimePage
from mph_gait_id.ui.enrollment_sources import EnrollmentSourcesPage
from mph_gait_id.ui.gallery_manager import GalleryManagerPage
from mph_gait_id.ui.performance_benchmark import PerformanceBenchmarkPage


class Tree:
    """Small Treeview double with selection invalidation on row deletion."""
    def __init__(self, rows=(), selection=()):
        self.rows = dict.fromkeys(rows)
        self.selected = tuple(selection)
        self.focused = next(iter(selection), "")
        self.scroll = [0.25, 0.5]

    def get_children(self):
        return tuple(self.rows)

    def selection(self):
        return self.selected

    def selection_set(self, values):
        self.selected = (values,) if isinstance(values, str) else tuple(values)

    def selection_add(self, value):
        self.selected += (value,)

    def focus(self, value=None):
        if value is not None:
            self.focused = value
        return self.focused

    def exists(self, value):
        return value in self.rows

    def delete(self, *values):
        for value in values:
            self.rows.pop(value, None)
        self.selected = tuple(x for x in self.selected if x in self.rows)
        self.focused = ""
        self.scroll = [0, 0]

    def insert(self, _parent, _position, *, iid, values):
        self.rows[iid] = values

    def xview(self):
        return (self.scroll[0], 1)

    def yview(self):
        return (self.scroll[1], 1)

    def xview_moveto(self, value):
        self.scroll[0] = value

    def yview_moveto(self, value):
        self.scroll[1] = value


class ViewStateTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        settings = patch.dict(os.environ, {"MPH_GAIT_ID_SETTINGS": str(Path(temporary.name) / "settings.json")})
        settings.start()
        self.addCleanup(settings.stop)
        self.tcl = tk.Tcl()
        self.i18n = I18n("en")

    def var(self, value=""):
        return tk.StringVar(master=self.tcl, value=value)

    def test_realtime_locale_changes_labels_not_operation_or_identity(self):
        page = object.__new__(RealtimePage)
        page.i18n = self.i18n
        page._build_option_labels()
        for name, value in (("operation", "enroll"), ("source", "replay"),
                            ("detector", "yolo_seg"), ("pass_direction", "front_facing")):
            labels = getattr(page, name + "_labels")
            setattr(page, name + "_var", self.var(page._option_label(labels, value)))
        for name in ("source_combo", "detector_combo", "pass_direction_combo", "recognize_radio",
                     "enroll_radio", "allow_multi_enrollment_check", "save_foreground_check",
                     "rgb_pointcloud_check", "start_button", "_source_changed", "_operation_changed"):
            setattr(page, name, Mock())
        page.enrollment_name_var = self.var("Gallery Manager")
        page.yolo_weights_var = self.var("my-detector.pt")
        page._last_snapshot = object()
        page.result_var = self.var("Gallery Manager")
        page.pipeline = object()
        pipeline = page.pipeline
        for locale in ("zh_TW", "en", "zh_TW"):
            self.i18n.set_locale(locale)
            page.set_locale(locale)
            self.assertEqual(page._operation(), "enroll")
            self.assertEqual(page.detector_labels[page.detector_var.get()], "yolo_seg")
            self.assertEqual(page.source_labels[page.source_var.get()], "replay")
            self.assertEqual(page.enrollment_name_var.get(), "Gallery Manager")
            self.assertEqual(page.yolo_weights_var.get(), "my-detector.pt")
            self.assertEqual(page.result_var.get(), "Gallery Manager")
            self.assertIs(page.pipeline, pipeline)
        page._source_changed.assert_not_called()
        page._operation_changed.assert_not_called()

    def test_tab_model_refresh_does_not_reapply_defaults(self):
        for cls in (RealtimePage, PerformanceBenchmarkPage):
            with self.subTest(page=cls.__name__):
                page = object.__new__(cls)
                page.bundle_var = self.var("old label")
                page.bundle_labels = {"old label": "model-id"}
                page.bundle_combo = Mock()
                page.controller = Mock(config={})
                page.controller.available_bundles.return_value = {
                    "model-id": SimpleNamespace(display_name="New display", checkpoint_sha256="abc12345",
                                                input_type="pointcloud", data={}, model={})}
                page._bundle_changed = Mock()
                page._refresh_gallery_summary = Mock()
                page.refresh_bundles()
                self.assertEqual(page.bundle_labels[page.bundle_var.get()], "model-id")
                page._bundle_changed.assert_not_called()
                page._refresh_gallery_summary.assert_called_once()
                page.bundle_var.set("missing")
                page.refresh_bundles()
                page._bundle_changed.assert_called_once()
                page._bundle_changed.reset_mock()
                page.controller.available_bundles.return_value["model-id"].checkpoint_sha256 = "new-full-hash"
                page.refresh_bundles()
                page._bundle_changed.assert_called_once()

    def source_page(self):
        page = object.__new__(EnrollmentSourcesPage)
        page.tree = Tree(("source-a", "source-b"), ("source-a", "source-b"))
        page.source_id = "source-a"
        page.manifest = {"frames": [{"pass_id": "pass-2", "timestamp": i} for i in range(5)]}
        page.frames = page.manifest["frames"]
        page.index = 3
        page.pass_var = self.var("pass-2")
        page.zoom = self.var("1.5")
        page.pass_combo, page.canvas, page.seek = Mock(), Mock(), Mock()
        page.summary, page.detail = self.var(), self.var()
        page.pause, page._show = Mock(), Mock()
        page.i18n = self.i18n
        page._usage = None
        page.library = Mock()
        page.library.list_sources.return_value = [dict(
            source_id=key, current_person_id="P001", current_name="Gallery Manager",
            pass_count=2, frame_count=5, size_bytes=100, created_at="today")
            for key in ("source-b", "source-a")]
        page.library.usage.return_value = dict(source_count=2, committed_bytes=100, disk_bytes=100)
        return page

    def test_source_refresh_preserves_multiselection_pass_and_frame(self):
        page = self.source_page()
        manifest, frames = page.manifest, page.frames
        page.refresh()
        page._select()  # Tk also queues a selection event when rows are restored.
        self.assertEqual(page.tree.selection(), ("source-a", "source-b"))
        self.assertEqual(page.tree.focus(), "source-a")
        self.assertEqual(page.tree.scroll, [0.25, 0.5])
        self.assertEqual((page.pass_var.get(), page.index, page.zoom.get()), ("pass-2", 3, "1.5"))
        self.assertIs(page.manifest, manifest)
        self.assertIs(page.frames, frames)
        page.library.manifest.assert_not_called()
        page.seek.set.assert_not_called()

    def test_deleted_source_clears_obsolete_preview(self):
        page = self.source_page()
        page.library.list_sources.return_value = []
        page.refresh()
        self.assertIsNone(page.source_id)
        self.assertEqual(page.frames, [])
        self.assertEqual(page.tree.selection(), ())
        self.assertEqual(page.pass_var.get(), "")
        self.assertEqual(page.index, 0)

    def test_failed_source_query_keeps_previous_view(self):
        page = self.source_page()
        page.library.list_sources.side_effect = OSError("unavailable")
        page.refresh()
        self.assertEqual(page.index, 3)
        self.assertEqual(page.source_id, "source-a")
        self.assertEqual(page.summary.get(), "unavailable")

    def gallery_page(self):
        page = object.__new__(GalleryManagerPage)
        page.i18n = self.i18n
        page.person_tree = Tree(("P001", "P002"), ("P001",))
        page.pass_tree = Tree(("pass-row-0", "pass-row-1"), ("pass-row-1",))
        page.bundle_var = self.var("bundle")
        page.bundle_labels = {"bundle": "model-id"}
        page.clip_len_var = self.var("15")
        page.processing_version_var = self.var("v1")
        page.summary_var, page.person_detail_var = self.var(), self.var()
        page._summary, page._person = {}, {}
        page._pass_scope = ("P001", "model-id", 15)
        first = dict(model_key="hash-a", source_path="/a", session_id="session", pass_id="pass-1", direction="front")
        second = dict(model_key="hash-a", source_path="/b", session_id="session", pass_id="pass-2", direction="front")
        page._pass_rows = {"pass-row-0": first, "pass-row-1": second}
        page.controller = Mock()
        page.controller.get_person.return_value = {"display_name": "Gallery Manager"}
        page.controller.list_gallery_passes.return_value = [second, first]
        return page

    def test_gallery_preserves_logical_pass_after_reordering(self):
        page = self.gallery_page()
        page._load_passes()
        self.assertEqual(page.pass_tree.selection(), ("pass-row-0",))
        self.assertEqual(page._pass_rows["pass-row-0"]["pass_id"], "pass-2")
        self.assertEqual(page.pass_tree.scroll, [0.25, 0.5])
        page._load_passes()
        self.assertEqual(page.pass_tree.selection(), ("pass-row-0",))

    def test_different_gallery_scope_does_not_reselect_another_person_pass(self):
        page = self.gallery_page()
        page.person_tree.selection_set("P002")
        page._load_passes()
        self.assertEqual(page.pass_tree.selection(), ())
        self.assertEqual(page.pass_tree.scroll, [0, 0])

    def test_gallery_locale_does_not_refresh_or_modify_names(self):
        page = self.gallery_page()
        page._load_passes()
        page.refresh = Mock()
        page.export_button, page.import_button, page.processing_version_combo = Mock(), Mock(), Mock()
        page.controller.processing_version_name.return_value = "處理版本"
        self.i18n.set_locale("zh_TW")
        page.set_locale("zh_TW")
        self.assertIn("Gallery Manager", page.person_detail_var.get())
        self.assertEqual(page.pass_tree.selection(), ("pass-row-0",))
        page.refresh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
