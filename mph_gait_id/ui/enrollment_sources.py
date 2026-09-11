"""Foreground-source browser. Playback never runs an identity encoder."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from PIL import ImageTk

from ..enrollment_sources import EnrollmentSourceLibrary
from ..i18n import I18n
from .workers import BackgroundWorker


class EnrollmentSourcesPage(ttk.Frame):
    def __init__(self, master, controller, can_edit: Callable[[], tuple[bool, str]],
                 on_gallery_changed: Callable[[], None] | None = None,
                 prepare_encoding: Callable[[], None] | None = None) -> None:
        super().__init__(master, padding=10)
        self.controller = controller
        self.on_gallery_changed = on_gallery_changed or (lambda: None)
        self.prepare_encoding = prepare_encoding or (lambda: None)
        self.registration_dialog = None
        self.transfer_dialog = None
        self.library = EnrollmentSourceLibrary(controller.repository)
        self.can_edit = can_edit
        self.i18n = getattr(self.winfo_toplevel(), "_gait_i18n", I18n("en"))
        self.worker = BackgroundWorker()
        self.source_id = None
        self.manifest = None
        self.frames = []
        self.index = 0
        self.timer = None
        self.photo = None
        self._labels = []
        self.summary = tk.StringVar()
        self.detail = tk.StringVar()
        self.pass_var = tk.StringVar()
        self.zoom = tk.DoubleVar(value=1.0)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for i, (zh, en, command) in enumerate([
            ("重新整理", "Refresh", self.refresh),
            ("刪除選取來源", "Delete source", self._delete),
            ("清理未提交暫存", "Clean uncommitted files", self._cleanup),
            ("多模型特徵註冊", "Register to models", self._open_model_registration),
        ]):
            button = ttk.Button(toolbar, command=command)
            button.grid(row=0, column=i, padx=4)
            self._labels.append((button, zh, en))
        for col, (zh, en, operation) in enumerate([
            ("匯出選取來源", "Export selected sources", "export"),
            ("匯入來源點雲", "Import sources", "import"),
        ]):
            button = ttk.Button(toolbar, command=lambda op=operation: self._open_transfer(op))
            button.grid(row=1, column=col * 2, columnspan=2, sticky="ew", padx=4, pady=(6, 0))
            self._labels.append((button, zh, en))
        ttk.Label(self, textvariable=self.summary).grid(row=2, column=0, sticky="w", pady=6)
        panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        panes.grid(row=1, column=0, sticky="nsew")
        listing, preview = ttk.Frame(panes), ttk.Frame(panes)
        panes.add(listing, weight=2)
        panes.add(preview, weight=3)
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(listing, columns=("person", "name", "passes", "frames", "size", "date"),
                                 show="headings", selectmode="extended")
        for key, width in [("person", 100), ("name", 140), ("passes", 60), ("frames", 75),
                           ("size", 85), ("date", 170)]:
            self.tree.column(key, width=width, minwidth=width, stretch=False)
        vs = ttk.Scrollbar(listing, command=self.tree.yview)
        hs = ttk.Scrollbar(listing, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._select())
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(1, weight=1)
        controls = ttk.Frame(preview)
        controls.grid(row=0, column=0, sticky="ew", padx=8)
        self.pass_combo = ttk.Combobox(controls, textvariable=self.pass_var, state="readonly", width=12)
        self.pass_combo.grid(row=0, column=0, padx=4)
        self.pass_combo.bind("<<ComboboxSelected>>", lambda _e: self._select_pass())
        for col, (zh, en, command) in enumerate([
            ("播放", "Play", self._play), ("暫停", "Pause", self.pause),
            ("上一幀", "Previous", lambda: self._step(-1)),
            ("下一幀", "Next", lambda: self._step(1)),
        ], start=1):
            button = ttk.Button(controls, command=command, width=9)
            button.grid(row=0, column=col, padx=2)
            self._labels.append((button, zh, en))
        ttk.Scale(controls, from_=0.25, to=3.0, variable=self.zoom,
                  command=lambda _v: self._show()).grid(row=1, column=0, columnspan=5, sticky="ew")
        self.canvas = tk.Canvas(preview, background="#181b1f", highlightthickness=0, width=500, height=420)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        self.canvas.bind("<Configure>", lambda _e: self._show())
        self.seek = ttk.Scale(preview, from_=0, to=0, command=self._seek)
        self.seek.grid(row=2, column=0, sticky="ew", padx=8)
        ttk.Label(preview, textvariable=self.detail, wraplength=550).grid(row=3, column=0, sticky="w", padx=8)
        self.set_locale(self.i18n.locale)
        self.refresh()

    def _t(self, zh, en):
        return zh if self.i18n.locale == "zh_TW" else en

    def set_locale(self, _locale):
        for widget, zh, en in self._labels:
            widget.configure(text=self._t(zh, en))
        for key, zh, en in [("person", "人物 ID", "Person ID"), ("name", "姓名", "Name"),
                            ("passes", "片段", "Passes"), ("frames", "幀數", "Frames"),
                            ("size", "大小 MiB", "Size MiB"), ("date", "建立 UTC", "Created UTC")]:
            self.tree.heading(key, text=self._t(zh, en))

    def refresh(self):
        self.pause()
        self.source_id, self.manifest, self.frames = None, None, []
        self.pass_combo.configure(values=[])
        self.pass_var.set("")
        self.canvas.delete("all")
        self.detail.set("")
        self.tree.delete(*self.tree.get_children())
        try:
            for row in self.library.list_sources():
                linked = row["current_person_id"] is not None
                self.tree.insert("", "end", iid=row["source_id"], values=(
                    row["current_person_id"] if linked else row["captured_person_id"] + " *",
                    row["current_name"] if linked else row["captured_name"],
                    row["pass_count"], row["frame_count"], f'{row["size_bytes"] / 2**20:.1f}', row["created_at"]))
            usage = self.library.usage()
            self.summary.set(self._t("來源 / 已提交 / 磁碟總量", "Sources / committed / disk total") +
                             f': {usage["source_count"]} / {usage["committed_bytes"] / 2**20:.1f} MiB / '
                             f'{usage["disk_bytes"] / 2**20:.1f} MiB')
        except Exception as exc:
            self.summary.set(str(exc))

    def _select(self):
        self.pause()
        selection = self.tree.selection()
        if not selection:
            return
        try:
            self.source_id = selection[0]
            self.manifest = self.library.manifest(self.source_id)
            passes = list(dict.fromkeys(frame["pass_id"] for frame in self.manifest["frames"]))
            self.pass_combo.configure(values=passes)
            self.pass_var.set(passes[0])
            self._select_pass()
        except Exception as exc:
            self.frames = []
            self.canvas.delete("all")
            self.detail.set(str(exc))

    def _select_pass(self):
        self.pause()
        self.frames = [f for f in (self.manifest or {}).get("frames", []) if f["pass_id"] == self.pass_var.get()]
        self.index = 0
        self.seek.configure(to=max(0, len(self.frames) - 1))
        self.seek.set(0)
        self._show()

    def _show(self):
        if not self.frames or not self.source_id:
            return
        try:
            from ..realtime.ui_page import RealtimePage
            frame = self.frames[self.index]
            points = self.library.load_frame(self.source_id, frame)
            size = (max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height()))
            image = RealtimePage._render_points(points, size, zoom=self.zoom.get())
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.delete("all")
            self.canvas.create_image(size[0] / 2, size[1] / 2, image=self.photo)
            self.detail.set(f'{self.index + 1}/{len(self.frames)} | {frame["points"]:,} points | '
                            f't={frame["timestamp"]:.3f} s | XYZ mm | {self.source_id}')
        except Exception as exc:
            self.pause()
            self.canvas.delete("all")
            self.detail.set(str(exc))

    def pause(self):
        if self.timer is not None:
            self.after_cancel(self.timer)
            self.timer = None

    def _step(self, delta):
        self.pause()
        if self.frames:
            self.index = min(max(self.index + delta, 0), len(self.frames) - 1)
            self.seek.set(self.index)

    def _seek(self, value):
        self.index = min(max(round(float(value)), 0), max(0, len(self.frames) - 1))
        self._show()

    def _play(self):
        self.pause()
        if not self.frames:
            return
        if self.index >= len(self.frames) - 1:
            self.index = 0
            self.seek.set(0)
        self._schedule()

    def _schedule(self):
        if self.index + 1 >= len(self.frames):
            return
        current, following = self.frames[self.index], self.frames[self.index + 1]
        if following["segment_start"]:
            self.detail.set(self._t("到達取樣中斷處；按下一幀繼續。", "Sampling break; select Next to continue."))
            return
        gap = following["timestamp"] - current["timestamp"]
        self.timer = self.after(max(10, min(2000, round(gap * 1000))), self._tick)

    def _tick(self):
        self.timer = None
        self.index += 1
        self.seek.set(self.index)
        self._schedule()

    @property
    def busy(self):
        return self.worker.busy or bool(
            self.transfer_dialog is not None and self.transfer_dialog.winfo_exists()) or bool(
            self.registration_dialog is not None and self.registration_dialog.running)

    def _open_transfer(self, operation):
        if self.transfer_dialog is not None and self.transfer_dialog.winfo_exists():
            self.transfer_dialog.lift()
            return
        allowed, reason = self.can_edit()
        if not allowed or self.busy:
            messagebox.showwarning(self._t("暫時無法操作", "Unavailable"), reason or "Operation in progress", parent=self)
            return
        selected = tuple(self.tree.selection())
        if operation == "export" and not selected:
            messagebox.showinfo(self._t("選取來源", "Select sources"),
                               self._t("請先選取要匯出的來源。", "Select the source recordings to export."), parent=self)
            return
        self.pause()
        try:
            from .source_transfer import SourceTransferDialog
            self.transfer_dialog = SourceTransferDialog(self, operation, selected)
        except Exception as exc:
            messagebox.showerror(self._t("操作失敗", "Operation failed"), str(exc), parent=self)

    def _open_model_registration(self):
        if self.registration_dialog is not None and self.registration_dialog.winfo_exists():
            self.registration_dialog.lift()
            return
        allowed, reason = self.can_edit()
        if not allowed or self.busy:
            messagebox.showwarning(self._t("暫時無法操作", "Unavailable"), reason or "Operation in progress", parent=self)
            return
        if self.source_id is None or len(self.tree.selection()) != 1:
            messagebox.showinfo(self._t("選取來源", "Select source"),
                               self._t("請選取單筆註冊來源。", "Select exactly one enrollment source."), parent=self)
            return
        try:
            from .source_registration import SourceRegistrationDialog
            self.pause()
            self.registration_dialog = SourceRegistrationDialog(self)
        except Exception as exc:
            messagebox.showerror(self._t("操作失敗", "Operation failed"), str(exc), parent=self)

    def _mutate(self, task, question):
        allowed, reason = self.can_edit()
        if not allowed or self.worker.busy:
            messagebox.showwarning(self._t("暫時無法操作", "Unavailable"), reason or "Operation in progress", parent=self)
            return
        if not messagebox.askyesno(self._t("確認", "Confirm"), question, parent=self):
            return
        self.pause()
        self.worker.start(lambda _progress: task())
        self.summary.set(self._t("處理中...", "Working..."))
        self.after(100, self._poll)

    def _delete(self):
        selection = self.tree.selection()
        if len(selection) > 1:
            messagebox.showinfo(self._t("選取來源", "Select source"),
                               self._t("刪除時請選取單筆來源。", "Select exactly one recording to delete."), parent=self)
            return
        if selection:
            self._mutate(lambda: self.library.delete(selection[0]), self._t(
                "永久刪除此來源點雲？Gallery 特徵不會刪除，無法從特徵還原點雲。",
                "Permanently delete this foreground recording? Gallery embeddings remain; they cannot reconstruct these points."))

    def _cleanup(self):
        self._mutate(self.library.cleanup_uncommitted, self._t(
            "清除中斷／未提交的來源暫存？已提交來源與 Gallery 特徵會保留。",
            "Remove interrupted/uncommitted source files? Committed sources and Gallery embeddings remain."))

    def _poll(self):
        for message in self.worker.drain():
            if message.kind == "error":
                messagebox.showerror(self._t("操作失敗", "Operation failed"), message.payload["message"], parent=self)
            if message.kind in {"result", "error"}:
                self.refresh()
                return
        self.after(100, self._poll)
