"""Presentation-only sizing and view-state helpers for the desktop pages."""
from __future__ import annotations

from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import font as tkfont, ttk


def reserve_text_width(widget: tk.Misc, texts: tuple[str, ...]) -> None:
    """Reserve the larger translation without changing commands or variables."""
    if getattr(widget, "_responsive_wrap", False):
        return
    try:
        if str(widget.cget("textvariable")) or str(widget.cget("image")):
            return
    except tk.TclError:
        pass
    if not isinstance(widget, (ttk.Button, ttk.Label, ttk.Checkbutton, ttk.Radiobutton)):
        return
    try:
        if isinstance(widget, ttk.Label) and float(widget.cget("wraplength") or 0):
            return
        style = ttk.Style(widget)
        name = widget.cget("style") or widget.winfo_class()
        font_name = style.lookup(name, "font") or "TkDefaultFont"
        if isinstance(widget, ttk.Label) and widget.cget("font"):
            font_name = widget.cget("font")
        signature = (str(font_name), str(widget.tk.call("tk", "scaling")), texts)
        if getattr(widget, "_text_width_signature", None) == signature:
            return
        font = tkfont.Font(root=widget, font=font_name)
        width = math.ceil(max(font.measure(text) for text in texts) / max(1, font.measure("0")))
        width = max(width, abs(int(widget.cget("width") or 0)))
        widget.configure(width=width)
        widget._text_width_signature = signature
    except (tk.TclError, ValueError):
        pass


def reserve_heading_width(tree: ttk.Treeview, column: str, texts: tuple[str, ...]) -> None:
    style = ttk.Style(tree)
    font_name = style.lookup("Treeview.Heading", "font") or "TkDefaultFont"
    signature = (str(font_name), str(tree.tk.call("tk", "scaling")), texts)
    cache = getattr(tree, "_heading_widths", {})
    if cache.get(column) == signature:
        return
    font = tkfont.Font(root=tree, font=font_name)
    minimum = max(tree.column(column, "minwidth"), max(font.measure(text) for text in texts) + 18)
    tree.column(column, minwidth=minimum, width=max(tree.column(column, "width"), minimum))
    cache[column] = signature
    tree._heading_widths = cache


class WrappedLabel(ttk.Label):
    """Wrap long status/path text to the allocated grid cell, not its request."""

    _responsive_wrap = True

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        kwargs.setdefault("width", 1)
        kwargs.setdefault("wraplength", 320)
        kwargs.setdefault("anchor", "w")
        kwargs.setdefault("justify", "left")
        super().__init__(master, **kwargs)
        self.bind("<Configure>", self._resize, add="+")

    def _resize(self, event: tk.Event) -> None:
        width = max(1, event.width - 4)
        if abs(float(self.cget("wraplength")) - width) >= 2:
            self.configure(wraplength=width)


class SplitPane(ttk.Panedwindow):
    """Two resizable panes with a useful first layout and nonzero viewports."""

    def __init__(self, master, *, fraction=0.6, minimum=(150, 100), **kwargs):
        super().__init__(master, **kwargs)
        self._fraction = fraction
        self._minimum = minimum
        self._placed = False
        self.bind("<Configure>", self._fit, add="+")
        self.bind("<Map>", self._fit, add="+")
        self.bind("<ButtonRelease-1>", self._fit, add="+")

    def _fit(self, _event=None):
        if len(self.panes()) != 2 or not self.winfo_ismapped():
            return
        extent = self.winfo_height() if str(self.cget("orient")) == "vertical" else self.winfo_width()
        if extent < 50:
            return
        current = self.sashpos(0) if self._placed else round(extent * self._fraction)
        self._placed = True
        scale = min(1, (extent - 6) / sum(self._minimum))
        lower = round(self._minimum[0] * scale)
        upper = extent - 6 - round(self._minimum[1] * scale)
        position = max(lower, min(upper, current))
        if position != self.sashpos(0):
            self.sashpos(0, position)


@dataclass
class ScrollPosition:
    x: float = 0.0
    y: float = 0.0

    @classmethod
    def capture(cls, widget) -> "ScrollPosition":
        return cls(widget.xview()[0], widget.yview()[0])

    def restore(self, widget) -> None:
        widget.xview_moveto(self.x)
        widget.yview_moveto(self.y)


def preserve_layout(root: tk.Misc):
    """Return a one-shot restore callback; no application state is inspected."""
    panes, scrolls = [], []

    def visit(widget):
        if isinstance(widget, ttk.Panedwindow) and widget.winfo_ismapped():
            panes.append((widget, [widget.sashpos(i) for i in range(len(widget.panes()) - 1)]))
        if isinstance(widget, (tk.Canvas, tk.Text, tk.Listbox, ttk.Treeview)):
            scrolls.append((widget, ScrollPosition.capture(widget)))
        for child in widget.winfo_children():
            visit(child)

    visit(root)

    def restore():
        for widget, positions in panes:
            try:
                for i, position in enumerate(positions):
                    widget.sashpos(i, position)
            except tk.TclError:
                pass
        for widget, position in scrolls:
            try:
                position.restore(widget)
            except tk.TclError:
                pass

    return restore
