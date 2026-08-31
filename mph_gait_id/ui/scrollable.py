from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """A vertically scrollable ttk frame for controls that exceed window height."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        width: int = 390,
        canvas_background: str = "#ffffff",
        frame_style: str = "Panel.TFrame",
    ) -> None:
        super().__init__(master, style=frame_style)
        self.canvas = tk.Canvas(
            self,
            width=width,
            background=canvas_background,
            borderwidth=0,
            highlightthickness=0,
        )
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style=frame_style, padding=14)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.content, anchor="nw"
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._match_content_width)
        self.after_idle(self.bind_mousewheel_tree)

    def bind_mousewheel_tree(self) -> None:
        """Bind wheel scrolling after all static child controls have been created."""

        def bind(widget: tk.Misc) -> None:
            if not isinstance(widget, (tk.Listbox, tk.Text, ttk.Treeview)):
                widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
                widget.bind("<Button-4>", self._on_mousewheel, add="+")
                widget.bind("<Button-5>", self._on_mousewheel, add="+")
            for child in widget.winfo_children():
                bind(child)

        bind(self)

    def _update_scroll_region(self, _event: tk.Event[tk.Misc]) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _match_content_width(self, event: tk.Event[tk.Misc]) -> None:
        self.canvas.itemconfigure(self._window_id, width=max(1, int(event.width)))

    def _on_mousewheel(self, event: tk.Event[tk.Misc]) -> str:
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0))
            units = -max(-3, min(3, int(delta / 120))) if delta else 0
        if units:
            self.canvas.yview_scroll(units, "units")
        return "break"
