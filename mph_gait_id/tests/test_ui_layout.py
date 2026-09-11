from __future__ import annotations

import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.i18n import I18n
from mph_gait_id.ui.layout import preserve_layout
from mph_gait_id.ui.main_window import GaitIdentityWindow


class DesktopLayoutTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        self.addCleanup(self.close_window)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        settings = patch.dict(os.environ, {"MPH_GAIT_ID_SETTINGS": str(self.directory / "settings.json")})
        settings.start()
        self.addCleanup(settings.stop)
        self.errors = []
        self.root.report_callback_exception = lambda *error: self.errors.append(error)

    def close_window(self):
        for timer in self.root.tk.call("after", "info"):
            self.root.tk.call("after", "cancel", timer)
        self.root.destroy()
        self.assertFalse(self.errors, self.errors)

    def test_variable_labels_are_data_unless_explicitly_registered(self):
        i18n = I18n("en")
        name = tk.StringVar(master=self.root, value="Gallery Manager")
        status = tk.StringVar(master=self.root, value="Ready")
        ttk.Label(self.root, textvariable=name).pack()
        ttk.Label(self.root, textvariable=status).pack()
        button = ttk.Button(self.root, text="Refresh")
        button.pack()
        i18n.register_ui_variables(status)
        widths = []
        for locale in ("en", "zh_TW", "en"):
            i18n.set_locale(locale)
            i18n.apply(self.root)
            self.root.update_idletasks()
            widths.append(button.winfo_reqwidth())
            self.assertEqual(name.get(), "Gallery Manager")
            self.assertEqual(status.get(), i18n.tr("Ready"))
        self.assertEqual(len(set(widths)), 1)

    def test_layout_restore_preserves_splitter_and_scroll(self):
        self.root.geometry("900x500")
        pane = ttk.Panedwindow(self.root, orient="horizontal")
        pane.pack(fill="both", expand=True)
        tree = ttk.Treeview(pane, columns=("value",), show="headings", height=4)
        pane.add(tree)
        pane.add(ttk.Frame(pane))
        for index in range(100):
            tree.insert("", "end", iid=str(index), values=(str(index),))
        self.root.update()
        pane.sashpos(0, 350)
        tree.yview_moveto(0.4)
        restore = preserve_layout(self.root)
        position = tree.yview()[0]
        pane.sashpos(0, 450)
        tree.yview_moveto(0)
        restore()
        self.assertEqual(pane.sashpos(0), 350)
        self.assertAlmostEqual(tree.yview()[0], position)

    def test_navigation_width_is_stable_across_languages(self):
        i18n = I18n("en")
        notebook = ttk.Notebook(self.root)
        notebook.pack()
        for key in ("nav.offline", "nav.realtime", "nav.gallery", "nav.benchmark", "nav.sources"):
            notebook.add(ttk.Frame(notebook, width=10, height=10), text=key)
        widths = []
        for locale in ("en", "zh_TW", "en"):
            i18n.set_locale(locale)
            i18n.apply(notebook)
            self.root.update_idletasks()
            widths.append(notebook.winfo_reqwidth())
        self.assertLessEqual(max(widths) - min(widths), 2)

    def test_all_pages_resize_and_switch_without_resetting_inputs(self):
        controller = GaitApplicationController(database_path=self.directory / "gallery.sqlite3",
                                               output_root=self.directory / "outputs")
        window = GaitIdentityWindow(self.root, controller)
        self.root.update()
        realtime = window.realtime_page
        realtime.clip_len_var.set(30)
        realtime.threshold_var.set(0.83)
        realtime.enrollment_person_id_var.set("custom-id")
        realtime.enrollment_name_var.set("Gallery Manager")
        window.person_name_var.set("Gallery Manager")
        window.performance_page.duration_var.set(45)
        for scaling, geometry in ((1.33, "1380x860"), (1.33, "980x640"), (1.67, "1280x800")):
            self.root.tk.call("tk", "scaling", scaling)
            self.root.geometry(geometry)
            self.root.update()
            for tab in window.mode_notebook.tabs():
                window.mode_notebook.select(tab)
                self.root.update()
                for locale in ("en", "zh_TW", "en"):
                    window.i18n.set_locale(locale)
                    window.language_var.set(window.i18n.language_choice())
                    window._on_language_changed()
                    self.root.update()
                    self.assertEqual(realtime.clip_len_var.get(), 30)
                    self.assertAlmostEqual(realtime.threshold_var.get(), 0.83)
                    self.assertEqual(realtime.enrollment_name_var.get(), "Gallery Manager")
                    self.assertEqual(window.person_name_var.get(), "Gallery Manager")
                    self.assertEqual(window.performance_page.duration_var.get(), 45)
                    self.assertFalse(self.errors)
                # Content requests may differ, but the notebook must remain within the window.
                self.assertLessEqual(window.mode_notebook.winfo_width(), self.root.winfo_width())
                if tab == str(window.offline_page):
                    self.assertGreater(window.preview.image_label.winfo_height(), 70)
                if tab == str(window.performance_page_host):
                    self.assertGreater(window.performance_page.tree.winfo_height(), 40)
        self.assertTrue(realtime.candidates.cget("xscrollcommand"))
        self.assertTrue(window.performance_page.tree.cget("xscrollcommand"))
        self.assertGreater(window.sources_page.canvas.winfo_height(), 70)


if __name__ == "__main__":
    unittest.main()
