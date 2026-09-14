"""Presentation controls; never change sources, model settings or database IDs."""
from __future__ import annotations

import re
from dataclasses import replace
import tkinter as tk
from tkinter import ttk

from .layout import ScrollPosition, WrappedLabel
from .preview import PreviewViewOptions


class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text = widget, text
        self.timer = self.window = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<FocusIn>", self.schedule, add="+")
        for event in ("<Leave>", "<FocusOut>", "<ButtonPress>", "<Unmap>", "<Destroy>"):
            widget.bind(event, self.hide, add="+")

    def schedule(self, _event=None):
        self.hide()
        self.timer = self.widget.after(500, self.show)

    def show(self):
        self.timer = None
        if not self.widget.winfo_ismapped():
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        ttk.Label(self.window, text=self.text(), padding=6).pack()
        self.window.update_idletasks()
        x = min(self.widget.winfo_rootx(), self.widget.winfo_screenwidth() - self.window.winfo_reqwidth())
        y = min(self.widget.winfo_rooty() + self.widget.winfo_height() + 4,
                self.widget.winfo_screenheight() - self.window.winfo_reqheight())
        self.window.geometry(f"+{max(0, x)}+{max(0, y)}")

    def hide(self, _event=None):
        if self.timer is not None:
            self.widget.after_cancel(self.timer)
            self.timer = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


SYMBOLS = {
    "play": "\u25b6", "pause": "\u2016", "previous": "\u25c0|", "next": "|\u25b6",
    "rotate_left": "\u21b6", "rotate_right": "\u21b7", "flip_h": "\u2194",
    "flip_v": "\u2195", "left": "\u2190", "right": "\u2192", "up": "\u2191",
    "down": "\u2193", "zoom_in": "+", "zoom_out": "\u2212", "reset": "\u21ba",
}


def icon_button(parent, action, command, i18n):
    button = ttk.Button(parent, text=SYMBOLS[action], command=command, width=3, style="Tool.TButton")
    button._tooltip = Tooltip(button, lambda: i18n.tr("tool." + action))
    return button


def change_preview_view(view, action):
    if action == "reset":
        return PreviewViewOptions()
    if action in ("rotate_left", "rotate_right"):
        return replace(view, rotation=(view.rotation + (-90 if action == "rotate_left" else 90)) % 360)
    if action in ("flip_h", "flip_v"):
        name = "flip_horizontal" if action == "flip_h" else "flip_vertical"
        return replace(view, **{name: not getattr(view, name)})
    if action in ("left", "right", "up", "down"):
        name = "pan_x" if action in ("left", "right") else "pan_y"
        return replace(view, **{name: getattr(view, name) + (-24 if action in ("left", "up") else 24)})
    if action in ("zoom_in", "zoom_out"):
        return replace(view, zoom=max(0.25, min(4.0, view.zoom * (1.2 if action == "zoom_in" else 1 / 1.2))))
    raise ValueError(f"Unknown preview action: {action}")


def preview_toolbar(parent, callback, i18n):
    toolbar = ttk.Frame(parent)
    for index, action in enumerate(("rotate_left", "rotate_right", "flip_h", "flip_v", "reset",
                                    "left", "right", "up", "down")):
        icon_button(toolbar, action, lambda a=action: callback(a), i18n).grid(
            row=index // 5, column=index % 5, padx=2, pady=2)
    return toolbar


class CollapsibleSection(ttk.Frame):
    def __init__(self, parent, title, *, view_id, expanded=False):
        super().__init__(parent, style="Panel.TFrame")
        self.view_id = view_id
        self.default_expanded = expanded
        self.expanded = tk.BooleanVar(self, value=expanded)
        self.toggle = ttk.Checkbutton(self, text=title, variable=self.expanded, command=self.update_visibility)
        self.toggle.grid(row=0, column=0, sticky="ew")
        self.content = ttk.Frame(self, style="Panel.TFrame", padding=(4, 8))
        self.content.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.update_visibility()

    def update_visibility(self):
        if self.expanded.get():
            self.content.grid()
        else:
            self.content.grid_remove()

    def set_expanded(self, value):
        self.expanded.set(bool(value))
        self.update_visibility()


def natural_key(value):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


class TableBrowser(ttk.Frame):
    """Filter by ID/name and sort existing rows without replacing their identifiers."""

    def __init__(self, parent, tree, i18n, *, search_columns, numeric_columns=()):
        super().__init__(parent)
        self.tree, self.i18n = tree, i18n
        self.search_columns, self.numeric_columns = search_columns, numeric_columns
        self.query = tk.StringVar(self)
        self.count = tk.StringVar(self)
        self.rows = []
        self.sort_column = None
        self.descending = False
        self._pending = None
        ttk.Label(self, text="ui.search").grid(row=0, column=0, padx=(0, 6))
        self.entry = ttk.Entry(self, textvariable=self.query, width=14)
        self.entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(self, text="ui.clear", command=lambda: self.query.set(""), width=6).grid(row=0, column=2, padx=4)
        WrappedLabel(self, textvariable=self.count).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        self.columnconfigure(1, weight=1)
        self.query.trace_add("write", self._schedule)
        self.entry.bind("<Escape>", lambda _event: self.query.set(""))
        tree.bind("<<TreeviewSelect>>", lambda _event: self.update_count(), add="+")
        self.bind("<Destroy>", self._destroy, add="+")
        for column in tree.cget("columns"):
            tree.heading(column, command=lambda c=column: self.sort(c))
        self.update_count()

    def _destroy(self, event):
        if event.widget == self and self._pending is not None:
            self.after_cancel(self._pending)
            self._pending = None

    def _schedule(self, *_args):
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(120, self.apply)

    def before_refresh(self):
        # Detached rows are absent from get_children() but still own their IIDs.
        visible = set(self.tree.get_children())
        for iid in self.rows:
            if self.tree.exists(iid) and iid not in visible:
                self.tree.delete(iid)
        self.rows = []

    def after_refresh(self):
        self.rows = list(self.tree.get_children())
        self.apply()

    def sort(self, column):
        self.descending = not self.descending if column == self.sort_column else False
        self.sort_column = column
        self.apply()

    def _key(self, iid):
        value = self.tree.set(iid, self.sort_column)
        if self.sort_column in self.numeric_columns:
            try:
                return (0, float(value.replace(",", "").split("/")[0].strip()))
            except ValueError:
                return (1, 0)
        return natural_key(value)

    def apply(self):
        if self._pending is not None:
            self.after_cancel(self._pending)
            self._pending = None
        position = ScrollPosition.capture(self.tree)
        query = self.query.get().strip().casefold()
        rows = [iid for iid in self.rows if self.tree.exists(iid)]
        visible = [iid for iid in rows if not query or any(
            query in self.tree.set(iid, column).casefold() for column in self.search_columns)]
        if self.sort_column:
            visible.sort(key=self._key, reverse=self.descending)
        selected = self.tree.selection()
        survivors = [iid for iid in selected if iid in visible]
        if tuple(survivors) != selected:
            self.tree.selection_set(survivors)
        if self.tree.focus() not in visible:
            self.tree.focus("")
        for iid in rows:
            self.tree.detach(iid)
        for iid in visible:
            self.tree.move(iid, "", "end")
        position.restore(self.tree)
        self.update_count()

    def update_count(self):
        visible = self.tree.get_children()
        text = self.i18n.tr("ui.rows", shown=len(visible), total=len(self.rows),
                            selected=sum(iid in visible for iid in self.tree.selection()))
        if self.sort_column:
            text += " | " + str(self.tree.heading(self.sort_column, "text")) + (" \u2193" if self.descending else " \u2191")
        self.count.set(text)
