from __future__ import annotations

import ast
import os
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mph_gait_id.i18n import I18n, LANGUAGE_CHOICES
from mph_gait_id.ui.main_window import GaitIdentityWindow
from mph_gait_id.ui.performance_benchmark import PerformanceBenchmarkPage


class PageLocalizationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        settings = patch.dict(os.environ, {
            "MPH_GAIT_ID_SETTINGS": str(Path(temporary.name) / "settings.json"),
        })
        settings.start()
        self.addCleanup(settings.stop)
        self.i18n = I18n("zh_TW")

    def test_performance_bilingual_labels_are_registered_for_live_switching(self):
        path = Path(__file__).resolve().parents[1] / "ui" / "performance_benchmark.py"
        pairs = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_tr" and len(node.args) == 2
                    and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                            for arg in node.args)):
                pairs.append(tuple(arg.value for arg in node.args))
        self.assertGreater(len(pairs), 20)
        for language in ("en", "zh_TW", "en"):
            self.i18n.set_locale(language)
            for zh, en in pairs:
                expected = zh if language == "zh_TW" else en
                with self.subTest(language=language, label=zh):
                    self.assertEqual(self.i18n.tr(zh), expected)
                    self.assertEqual(self.i18n.tr(en), expected)

    def test_offline_static_chinese_labels_are_translatable(self):
        path = Path(__file__).resolve().parents[1] / "ui" / "main_window.py"
        labels = []
        for function in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(function, ast.FunctionDef) or not function.name.startswith("_build_"):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg not in {"text", "headings"}:
                        continue
                    for value in ast.walk(keyword.value):
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            if any("\u4e00" <= char <= "\u9fff" for char in value.value):
                                labels.append(value.value)
        self.assertGreater(len(labels), 30)
        self.i18n.set_locale("en")
        for label in labels:
            with self.subTest(label=label):
                translated = self.i18n.tr(label)
                self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in translated), translated)

    def test_performance_switch_does_not_rebuild_or_reset_state(self):
        page = SimpleNamespace(i18n=Mock(), pipeline=object(), recorder=object(),
                               _reports={"test": {"score": 0.75}})
        before = (page.pipeline, page.recorder, page._reports)
        PerformanceBenchmarkPage.set_locale(page, "en")
        page.i18n.apply.assert_called_once_with(page)
        self.assertEqual(before, (page.pipeline, page.recorder, page._reports))

    def test_actual_tabs_switch_both_directions_without_losing_results(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        self.addCleanup(root.destroy)
        root.withdraw()
        root._gait_i18n = self.i18n
        window = SimpleNamespace(
            root=root, i18n=self.i18n, language_var=tk.StringVar(master=root),
            status_var=tk.StringVar(master=root, value="系統就緒"),
            database_var=tk.StringVar(master=root, value="Gallery 尚未載入"),
            _build_sidebar=Mock(), _build_content=Mock(),
        )
        notebook = ttk.Notebook(root)
        notebook.pack()
        offline = ttk.Frame(notebook)
        notebook.add(offline, text="nav.offline")
        GaitIdentityWindow._build_offline_layout(window, offline)
        controller = SimpleNamespace(config={})
        with patch.object(PerformanceBenchmarkPage, "refresh_bundles"):
            performance = PerformanceBenchmarkPage(notebook, controller)
        notebook.add(performance, text="nav.benchmark")
        window.performance_page = performance
        performance.bundle_var.set("test bundle")
        performance.duration_var.set(45)
        performance._reports["sample"] = {"untouched": True}
        performance.tree.insert("", "end", iid="sample", values=("12:00", "MPH-Gait"))
        performance.tree.selection_set("sample")
        self.i18n.apply(root)

        def texts(widget):
            values = []
            try:
                values.append(str(widget.cget("text")))
            except tk.TclError:
                pass
            for child in widget.winfo_children():
                values.extend(texts(child))
            return values

        for language in ("en", "zh_TW", "en", "zh_TW"):
            window.language_var.set(next(label for label, value in LANGUAGE_CHOICES.items()
                                         if value == language))
            GaitIdentityWindow._on_language_changed(window)
            root.update_idletasks()
            for key in ("benchmark.settings", "benchmark.live", "benchmark.results",
                        "benchmark.start", "benchmark.stop", "benchmark.export"):
                self.assertIn(self.i18n.tr(key), texts(performance))
            self.assertIn(self.i18n.tr("offline.title"), texts(offline))
            self.assertEqual(root.title(), self.i18n.tr("app.title"))
            self.assertEqual(performance.tree.heading("time", "text"), self.i18n.tr("Time"))
            self.assertEqual(notebook.tab(offline, "text"), self.i18n.tr("nav.offline"))
            self.assertEqual(performance.bundle_var.get(), "test bundle")
            self.assertEqual(performance.duration_var.get(), 45)
            self.assertEqual(performance.tree.selection(), ("sample",))
            self.assertEqual(performance._reports, {"sample": {"untouched": True}})


if __name__ == "__main__":
    unittest.main()
