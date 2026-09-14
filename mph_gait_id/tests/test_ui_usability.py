from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

import numpy as np

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.i18n import I18n
from mph_gait_id.realtime.ui_page import RealtimePage
from mph_gait_id.realtime.types import PipelineSnapshot
from mph_gait_id.ui.layout import SplitPane
from mph_gait_id.ui.main_window import GaitIdentityWindow
from mph_gait_id.ui.preferences import LayoutPreferences, read_layout, safe_geometry, write_layout
from mph_gait_id.ui.preview import PreviewViewOptions
from mph_gait_id.ui.usability import CollapsibleSection, TableBrowser, change_preview_view, icon_button


class PreferenceFileTest(unittest.TestCase):
    def test_geometry_clamps_offscreen_and_small_screen(self):
        self.assertEqual(safe_geometry("1500x1000-1600+900", (1280, 800)), "1260x720+0+20")
        self.assertEqual(safe_geometry("1x1+9999-50", (1600, 1000)), "980x640+610+0")
        for value in (None, {}, "invalid", "9" * 6000 + "x800"):
            self.assertEqual(safe_geometry(value, (1600, 1000)), "1380x860+10+10")

    def test_bounded_schema_read_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefs.json"
            self.assertEqual(read_layout(path), {})
            for content in ("{bad", "[]", '{"version": 9}', " " * 70000):
                path.write_text(content)
                self.assertEqual(read_layout(path), {})
            value = {"version": 1, "geometry": "1200x800+0+0"}
            write_layout(path, value)
            self.assertEqual(read_layout(path), value)
            with patch("mph_gait_id.ui.preferences.os.replace", side_effect=OSError("read-only")):
                with self.assertRaises(OSError):
                    write_layout(path, {"version": 1})
            self.assertEqual(read_layout(path), value)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_preview_is_nonmutating_and_resettable(self):
        original = PreviewViewOptions()
        view = original
        for action in ("rotate_right", "flip_h", "down", "zoom_in"):
            view = change_preview_view(view, action)
        self.assertEqual(original, PreviewViewOptions())
        self.assertEqual(view.rotation, 90)
        self.assertTrue(view.flip_horizontal)
        self.assertEqual(view.pan_y, 24)
        self.assertEqual(change_preview_view(view, "reset"), original)

    def test_realtime_render_does_not_change_input_or_fail_offscreen(self):
        points = np.array([[-100, -400, 1900], [100, 200, 2100], [20, 300, 2000]], dtype=np.float32)
        before = points.copy()
        plain = np.asarray(RealtimePage._render_points(points, (320, 240)))
        default = np.asarray(RealtimePage._render_points(points, (320, 240), view_options=PreviewViewOptions()))
        np.testing.assert_array_equal(plain, default)
        flipped = np.asarray(RealtimePage._render_points(points, (320, 240),
            view_options=PreviewViewOptions(flip_horizontal=True, rotation=90)))
        self.assertTrue(np.any(plain != flipped))
        away = np.asarray(RealtimePage._render_points(points, (320, 240),
            view_options=PreviewViewOptions(pan_x=10000)))
        self.assertTrue(np.all(away == (24, 27, 31)))
        np.testing.assert_array_equal(points, before)


class UsabilityGuiTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        settings = patch.dict(os.environ, {"MPH_GAIT_ID_SETTINGS": str(self.path / "settings.json")})
        settings.start()
        self.addCleanup(settings.stop)
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)
        self.i18n = I18n("en")
        self.root._gait_i18n = self.i18n
        self.addCleanup(self.close)

    def close(self):
        for timer in self.root.tk.call("after", "info"):
            self.root.tk.call("after", "cancel", timer)
        self.root.destroy()
        self.assertFalse(self.errors, self.errors)

    def table(self):
        tree = ttk.Treeview(self.root, columns=("id", "name", "count"), show="headings")
        browser = TableBrowser(self.root, tree, self.i18n, search_columns=("id", "name"), numeric_columns=("count",))
        browser.pack(fill="x")
        tree.pack(fill="both", expand=True)
        for iid, name, count in (("p10", "Gallery Manager", 100), ("p2", "Alice", 2), ("p1", "Amy", 10)):
            tree.insert("", "end", iid=iid, values=(iid, name, count))
        browser.after_refresh()
        self.root.update()
        return browser, tree

    def test_filter_sort_preserves_ids_clears_hidden_selection(self):
        browser, tree = self.table()
        tree.selection_set("p10")
        tree.focus("p10")
        browser.sort("id")
        self.assertEqual(tree.get_children(), ("p1", "p2", "p10"))
        self.assertEqual(tree.selection(), ("p10",))
        browser.sort("count")
        self.assertEqual(tree.get_children(), ("p2", "p1", "p10"))
        browser.sort("count")
        self.assertEqual(tree.get_children(), ("p10", "p1", "p2"))
        browser.query.set("alICE")
        browser.apply()
        self.assertEqual(tree.get_children(), ("p2",))
        self.assertEqual(tree.selection(), ())
        self.assertEqual(tree.focus(), "")
        browser.query.set("")
        browser.apply()
        self.assertEqual(tree.selection(), ())
        self.assertEqual(len(tree.get_children()), 3)

    def test_refresh_deletes_hidden_rows_and_keeps_filter_sort(self):
        browser, tree = self.table()
        browser.query.set("p2")
        browser.apply()
        browser.before_refresh()
        tree.delete(*tree.get_children())
        for iid in ("p2", "p10"):
            tree.insert("", "end", iid=iid, values=(iid, "Updated", 3))
        browser.after_refresh()
        self.assertEqual(tree.get_children(), ("p2",))
        self.assertFalse(tree.exists("p1"))
        self.assertIn("1/2", browser.count.get())

    def test_translation_preserves_sort_and_identity_data(self):
        browser, tree = self.table()
        browser.sort("id")
        for locale in ("zh_TW", "en"):
            self.i18n.set_locale(locale)
            self.i18n.apply(self.root)
            browser.update_count()
            self.assertEqual(tree.set("p10", "name"), "Gallery Manager")
            self.assertTrue(tree.heading("id", "command"))
            self.assertEqual(tree.get_children(), ("p1", "p2", "p10"))

    def test_preferences_restore_reset_allowlist_and_invalid_values(self):
        preferences = LayoutPreferences(self.root, self.i18n.settings_path, "1200x800")
        preferences.configure_window()
        pane = SplitPane(self.root, orient="horizontal", fraction=0.4)
        pane.view_id = "test.pane"
        pane.pack(fill="both", expand=True)
        tree = ttk.Treeview(pane, columns=("id",), show="headings")
        tree.column("id", minwidth=80, width=120, stretch=False)
        tree.view_id = "test.tree"
        section = CollapsibleSection(pane, "ui.advanced_inference", view_id="test.advanced")
        model_value = tk.StringVar(value="secret-person-data")
        ttk.Entry(section.content, textvariable=model_value).pack()
        pane.add(tree)
        pane.add(section)
        preferences.attach()
        self.root.update()
        tree.column("id", width=230)
        section.set_expanded(True)
        pane.sashpos(0, 600)
        preferences.save()
        raw = preferences.path.read_text()
        self.assertNotIn("secret-person-data", raw)
        self.assertEqual(set(json.loads(raw)), {"version", "geometry", "maximized", "columns", "panes", "sections"})
        tree.column("id", width=90)
        section.set_expanded(False)
        restored = LayoutPreferences(self.root, self.i18n.settings_path, "1200x800")
        restored.attach()
        self.root.update()
        self.assertEqual(tree.column("id", "width"), 230)
        self.assertTrue(section.expanded.get())
        self.assertAlmostEqual(pane.sashpos(0) / pane.winfo_width(), 0.5, delta=0.02)
        restored.apply({"panes": {"test.pane": float("nan")}, "columns": {"test.tree": {"id": -1}},
                        "sections": {"test.advanced": "false"}})
        self.assertEqual(tree.column("id", "width"), 230)
        preferences.reset()
        self.root.update()
        self.assertEqual(tree.column("id", "width"), 120)
        self.assertFalse(section.expanded.get())
        self.assertEqual(model_value.get(), "secret-person-data")

    def test_tooltip_is_bilingual_and_cleanup_cancels_timer(self):
        button = icon_button(self.root, "reset", lambda: None, self.i18n)
        button.pack()
        self.root.update()
        for locale in ("zh_TW", "en"):
            self.i18n.set_locale(locale)
            self.assertEqual(button._tooltip.text(), self.i18n.tr("tool.reset"))
        button._tooltip.schedule()
        button.destroy()
        self.root.update()

    def test_unvisited_pane_keeps_saved_ratio_until_mapped(self):
        prefs = LayoutPreferences(self.root, self.i18n.settings_path, "1200x800")
        prefs.configure_window()
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
        notebook.add(ttk.Frame(notebook), text="First")
        pane = SplitPane(notebook, orient="horizontal", fraction=0.4)
        pane.view_id = "hidden"
        pane.add(ttk.Frame(pane))
        pane.add(ttk.Frame(pane))
        notebook.add(pane, text="Hidden")
        prefs.saved = {"panes": {"hidden": 0.65}}
        prefs.attach()
        self.root.update()
        self.assertIn("hidden", prefs.pending)
        self.assertEqual(prefs.capture()["panes"]["hidden"], 0.65)
        notebook.select(pane)
        self.root.update()
        self.assertNotIn("hidden", prefs.pending)
        self.assertAlmostEqual(pane.sashpos(0) / pane.winfo_width(), 0.65, delta=0.02)

    def test_full_window_restore_does_not_restore_model_or_identity(self):
        controller = GaitApplicationController(database_path=self.path / "gallery.sqlite3",
                                               output_root=self.path / "outputs")
        window = GaitIdentityWindow(self.root, controller)
        self.root.update()
        page = window.realtime_page
        page.clip_len_var.set(30)
        page.threshold_var.set(0.83)
        page.enrollment_name_var.set("private")
        page._reveal_settings("settings")
        self.root.update()
        self.assertTrue(page.advanced.expanded.get())
        with patch("mph_gait_id.ui.main_window.messagebox.askyesno", return_value=True):
            window._reset_layout()
        self.root.update()
        self.assertEqual(page.clip_len_var.get(), 30)
        self.assertEqual(page.threshold_var.get(), 0.83)
        self.assertEqual(page.enrollment_name_var.get(), "private")
        self.assertNotIn("private", window.layout_preferences.path.read_text())
        self.assertFalse(page.advanced.expanded.get())
        window.layout_preferences.save()

        window.mode_notebook.select(window.realtime_page_host)
        self.root.geometry("980x640")
        self.root.update()
        snapshot = PipelineSnapshot(state="waiting_person", message="", timestamp=0,
            color_bgr=np.full((480, 640, 3), 80, dtype=np.uint8),
            person_points_mm=np.array([[-100, -400, 2000], [100, 400, 2100]], dtype=np.float32))
        for _ in range(3):
            page._show_images(snapshot)
            self.root.update()
        self.assertLessEqual(page._cloud_photo.height(), page.cloud_label.winfo_height())
        self.assertLessEqual(page._rgb_photo.height(), page.rgb_label.winfo_height())
        self.assertGreater(page._cloud_photo.height(), 40)
        self.root.geometry("1380x860")
        self.root.update()
        self.root.after(160, self.root.quit)
        self.root.mainloop()
        self.assertEqual(page._cloud_photo.width(), page.cloud_label.winfo_width() - 16)
        self.assertEqual(page._cloud_photo.height(), page.cloud_label.winfo_height() - 16)
        page._config_lock.set_locked(True)
        page._reveal_settings("settings")
        self.root.update()
        self.assertTrue(page.threshold_spin.instate(["disabled"]))
        self.assertFalse(page.advanced.toggle.instate(["disabled"]))
        self.assertFalse(page.settings_links["settings"].instate(["disabled"]))
        page._config_lock.set_locked(False)


if __name__ == "__main__":
    unittest.main()
