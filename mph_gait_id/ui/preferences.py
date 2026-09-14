"""Versioned local layout preferences, separate from inference configuration."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
import tkinter as tk
from tkinter import ttk

from .layout import SplitPane
from .usability import CollapsibleSection


def safe_geometry(value, screen, minimum=(980, 640)):
    value = str(value)
    match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)?([+-]\d+)?", value) if len(value) < 80 else None
    if not match:
        match = re.fullmatch(r"(\d+)x(\d+)([+-]\d+)?([+-]\d+)?", "1380x860")
    sw, sh = screen
    limit_w, limit_h = max(1, sw - 20), max(1, sh - 80)
    width = min(limit_w, max(minimum[0], int(match[1])))
    height = min(limit_h, max(minimum[1], int(match[2])))
    x = min(max(0, int(match[3] or 10)), max(0, sw - width - 10))
    y = min(max(0, int(match[4] or 10)), max(0, sh - height - 60))
    return f"{width}x{height}+{x}+{y}"


def read_layout(path):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            text = stream.read(65537)
        if len(text) > 65536:
            return {}
        data = json.loads(text)
        return data if isinstance(data, dict) and data.get("version") == 1 else {}
    except (OSError, ValueError, RecursionError):
        return {}


def write_layout(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".ui-layout-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _finite(value, low, high):
    return type(value) in (int, float) and low <= value <= high and math.isfinite(value)


class LayoutPreferences:
    def __init__(self, root, settings_path, default_geometry):
        self.root = root
        path = Path(settings_path)
        self.path = path.with_name(path.stem + "_layout.json")
        self.default_geometry = default_geometry
        self.saved = read_layout(self.path)
        self.widgets = {}
        self.defaults = {}
        self.pending = {}
        self._normal_geometry = default_geometry

    def _geometry(self, value):
        return safe_geometry(value, (self.root.winfo_screenwidth(), self.root.winfo_screenheight()))

    def configure_window(self):
        self.root.minsize(min(980, max(1, self.root.winfo_screenwidth() - 20)),
                          min(640, max(1, self.root.winfo_screenheight() - 80)))
        geometry = self._geometry(self.saved.get("geometry", self.default_geometry))
        self.root.geometry(geometry)
        self._normal_geometry = geometry

    def attach(self):
        def visit(widget):
            key = getattr(widget, "view_id", None)
            if key:
                self.widgets[key] = widget
                if isinstance(widget, ttk.Treeview):
                    self.defaults[key] = {c: widget.column(c, "width") for c in widget.cget("columns")}
                elif isinstance(widget, CollapsibleSection):
                    self.defaults[key] = widget.default_expanded
                elif isinstance(widget, ttk.Panedwindow):
                    self.defaults[key] = getattr(widget, "_fraction", 0.5)
                    widget.bind("<Map>", lambda _e, k=key: self.root.after_idle(lambda: self._apply_pane(k)), add="+")
            for child in widget.winfo_children():
                if not isinstance(child, tk.Toplevel):
                    visit(child)
        visit(self.root)
        self.apply(self.saved)
        self.root.bind("<Configure>", self._track_window, add="+")
        self.root.after_idle(self._restore_maximized)

    def _restore_maximized(self):
        if self.saved.get("maximized") is True:
            try:
                self.root.state("zoomed")
            except tk.TclError:
                pass

    def _track_window(self, event):
        if event.widget == self.root and self.root.state() == "normal":
            self._normal_geometry = self.root.geometry()

    def apply(self, data):
        panes = data.get("panes", {})
        columns = data.get("columns", {})
        sections = data.get("sections", {})
        for key, widget in self.widgets.items():
            if isinstance(widget, ttk.Panedwindow):
                fraction = panes.get(key) if isinstance(panes, dict) else None
                if _finite(fraction, 0.05, 0.95):
                    self.pending[key] = fraction
                    self.root.after_idle(lambda k=key: self._apply_pane(k))
            elif isinstance(widget, ttk.Treeview):
                widths = columns.get(key, {}) if isinstance(columns, dict) else {}
                if isinstance(widths, dict):
                    for column in widget.cget("columns"):
                        width = widths.get(column)
                        if type(width) is int and 30 <= width <= 1600:
                            widget.column(column, width=max(width, widget.column(column, "minwidth")))
            elif isinstance(widget, CollapsibleSection):
                expanded = sections.get(key) if isinstance(sections, dict) else None
                if type(expanded) is bool:
                    widget.set_expanded(expanded)

    def _apply_pane(self, key):
        widget = self.widgets[key]
        if key not in self.pending or not widget.winfo_ismapped() or len(widget.panes()) != 2:
            return
        extent = widget.winfo_height() if str(widget.cget("orient")) == "vertical" else widget.winfo_width()
        if extent < 50:
            return
        widget.sashpos(0, round(extent * self.pending.pop(key)))
        if isinstance(widget, SplitPane):
            widget._placed = True
            widget._fit()

    def capture(self):
        data = {"version": 1, "geometry": self._geometry(self._normal_geometry),
                "maximized": self.root.state() == "zoomed", "panes": {}, "columns": {}, "sections": {}}
        for key, widget in self.widgets.items():
            if isinstance(widget, ttk.Treeview):
                data["columns"][key] = {column: max(30, min(1600, widget.column(column, "width")))
                                        for column in widget.cget("columns")}
            elif isinstance(widget, CollapsibleSection):
                data["sections"][key] = widget.expanded.get()
            elif isinstance(widget, ttk.Panedwindow):
                if key in self.pending:
                    data["panes"][key] = self.pending[key]
                elif len(widget.panes()) == 2 and widget.winfo_width() > 1 and widget.winfo_height() > 1:
                    extent = widget.winfo_height() if str(widget.cget("orient")) == "vertical" else widget.winfo_width()
                    data["panes"][key] = max(0.05, min(0.95, widget.sashpos(0) / extent))
        return data

    def save(self):
        write_layout(self.path, self.capture())

    def reset(self):
        self.pending.clear()
        self.root.state("normal")
        self.root.geometry(self._geometry(self.default_geometry))
        self._normal_geometry = self._geometry(self.default_geometry)
        data = {"panes": {}, "columns": {}, "sections": {}}
        for key, widget in self.widgets.items():
            category = "columns" if isinstance(widget, ttk.Treeview) else (
                "sections" if isinstance(widget, CollapsibleSection) else "panes")
            data[category][key] = self.defaults[key]
        self.apply(data)
        # Let geometry settle before persisting the reset pane proportions.
        self.root.update_idletasks()
        self.save()
