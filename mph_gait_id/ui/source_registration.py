"""Multi-model source enrollment dialog; all Tk operations stay on the UI thread."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from ..source_registration import RegistrationTarget
from .scrollable import ScrollableFrame
from .layout import WrappedLabel


class SourceRegistrationDialog(tk.Toplevel):
    def __init__(self, page):
        manifest = page.library.manifest(page.source_id)
        row = next(r for r in page.library.list_sources() if r["source_id"] == page.source_id)
        bundles = page.controller.available_bundles(refresh=True)
        super().__init__(page)
        self.page = page
        self.controller = page.controller
        self.worker = page.worker
        self.source_id = page.source_id
        self.manifest = manifest
        self.cancel_event = threading.Event()
        self.running = False
        self.after_id = None
        self.targets = []
        self.passes = []
        self.title(self._t("多模型特徵註冊", "Multi-model Gallery enrollment"))
        self.geometry("850x700")
        self.minsize(600, 430)
        self.transient(page.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=2)
        scroll = ScrollableFrame(self, width=780, frame_style="TFrame")
        scroll.grid(row=0, column=0, sticky="nsew")
        form = scroll.content
        form.columnconfigure(0, weight=1)
        self.owner_available = row["current_person_id"] is not None
        owner = f'{row["current_person_id"] or row["captured_person_id"]} / {row["current_name"] or row["captured_name"]}'
        ttk.Label(form, text=owner).grid(row=0, column=0, sticky="w")
        ttk.Label(form, text=self.source_id).grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Label(form, text=self._t("註冊片段", "Enrollment passes")).grid(row=2, column=0, sticky="w")
        passbox = ttk.Frame(form)
        passbox.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        pass_ids = list(dict.fromkeys(f["pass_id"] for f in self.manifest["frames"]))
        for i, pass_id in enumerate(pass_ids):
            count = sum(f["pass_id"] == pass_id for f in self.manifest["frames"])
            selected = tk.BooleanVar(value=True)
            ttk.Checkbutton(passbox, text=f"{pass_id} ({count} frames)", variable=selected).grid(
                row=i, column=0, sticky="w")
            self.passes.append((pass_id, selected))

        modelbox = ttk.Frame(form)
        modelbox.grid(row=4, column=0, sticky="ew", pady=6)
        modelbox.columnconfigure(0, weight=1)
        for col, heading in enumerate([self._t("模型 / 權重", "Model / checkpoint"), "T", "N"]):
            ttk.Label(modelbox, text=heading).grid(row=0, column=col, sticky="w", padx=4)
        for i, bundle in enumerate(bundles.values(), 1):
            if bundle.input_type != "pointcloud":
                continue
            selected = tk.BooleanVar(value=False)
            length = tk.StringVar(value=str(bundle.data.get("clip_len", 15)))
            label = f'{bundle.display_name}\n{bundle.bundle_id} | {bundle.checkpoint_sha256[:12]}'
            ttk.Checkbutton(modelbox, text=label, variable=selected).grid(row=i, column=0, sticky="w", pady=4)
            ttk.Spinbox(modelbox, from_=1, to=300, textvariable=length, width=6).grid(row=i, column=1, padx=4)
            ttk.Label(modelbox, text=str(bundle.data.get("num_points", 1024))).grid(row=i, column=2, padx=4)
            self.targets.append((bundle.bundle_id, selected, length))
        options = ttk.Frame(form)
        options.grid(row=5, column=0, sticky="ew", pady=8)
        ttk.Label(options, text=self._t("每片段最多特徵數", "Maximum embeddings per pass")).grid(row=0, column=0)
        self.maximum = tk.StringVar(value="10")
        ttk.Spinbox(options, from_=1, to=100, textvariable=self.maximum, width=6).grid(row=0, column=1, padx=8)
        ttk.Label(options, text=self._t("運算裝置", "Device")).grid(row=0, column=2)
        self.device = tk.StringVar(value="auto")
        ttk.Combobox(options, textvariable=self.device, values=("auto", "cpu", "cuda"),
                     state="readonly", width=8).grid(row=0, column=3, padx=8)

        output = ttk.Frame(self, padding=(10, 0))
        output.grid(row=1, column=0, sticky="nsew")
        output.columnconfigure(0, weight=1)
        output.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(output)
        self.notebook = notebook
        notebook.grid(row=0, column=0, sticky="nsew")
        results = ttk.Frame(notebook)
        logs = ttk.Frame(notebook)
        notebook.add(results, text=self._t("各模型結果", "Model results"))
        notebook.add(logs, text=self._t("執行紀錄", "Activity log"))
        for host in (results, logs):
            host.columnconfigure(0, weight=1)
            host.rowconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(results, columns=("model", "length", "state", "count"),
                                       show="headings", height=6)
        for key, title, width in (
            ("model", self._t("模型", "Model"), 300), ("length", "T", 50),
            ("state", self._t("狀態", "Status"), 170), ("count", self._t("新增特徵", "Added embeddings"), 130)):
            self.result_tree.heading(key, text=title)
            self.result_tree.column(key, width=width, minwidth=50)
        vertical = ttk.Scrollbar(results, command=self.result_tree.yview)
        horizontal = ttk.Scrollbar(results, orient="horizontal", command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.log = tk.Text(logs, height=8, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(logs, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.status = tk.StringVar(value=self._t("選取來源片段與目標模型", "Select source passes and target models"))
        WrappedLabel(output, textvariable=self.status).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.bar = ttk.Progressbar(self, mode="indeterminate")
        self.bar.grid(row=2, column=0, sticky="ew", padx=10, pady=6)
        commands = ttk.Frame(self, padding=10)
        commands.grid(row=3, column=0, sticky="ew")
        self.start_button = ttk.Button(commands, text=self._t("開始多模型註冊", "Register selected models"), command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(commands, text=self._t("取消工作", "Cancel job"), command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        ttk.Button(commands, text=self._t("關閉", "Close"), command=self.close).pack(side="right")
        self.form = form
        self.form_states = []
        if not self.owner_available:
            self.start_button.configure(state="disabled")
            self._append(self._t("此來源人物已刪除。請先還原原人物；不會自動綁定同 ID 的新人物。",
                "Source owner was deleted. Restore the original person first; a reused ID is not ownership."))

    def _t(self, zh, en):
        return self.page._t(zh, en)

    def _append(self, value):
        self.log.configure(state="normal")
        self.log.insert("end", str(value) + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _lock_form(self, locked):
        if locked:
            def visit(widget):
                for child in widget.winfo_children():
                    if isinstance(child, (ttk.Checkbutton, ttk.Spinbox, ttk.Combobox)):
                        self.form_states.append((child, child.cget("state")))
                        child.configure(state="disabled")
                    visit(child)
            visit(self.form)
        else:
            for widget, state in self.form_states:
                widget.configure(state=state)
            self.form_states.clear()

    def start(self):
        if self.running or self.worker.busy:
            return
        allowed, reason = self.page.can_edit()
        if not allowed:
            messagebox.showwarning(self.title(), reason, parent=self)
            return
        try:
            targets = [RegistrationTarget(key, int(length.get()))
                       for key, selected, length in self.targets if selected.get()]
            passes = [key for key, selected in self.passes if selected.get()]
            maximum = int(self.maximum.get())
            device = self.device.get()
            if not targets or not passes:
                raise ValueError(self._t("請選取片段及目標模型。", "Select passes and target models."))
            if not 1 <= maximum <= 100 or any(not 1 <= t.clip_len <= 300 for t in targets):
                raise ValueError("T: 1..300; maximum embeddings per pass: 1..100")
        except ValueError as exc:
            messagebox.showerror(self.title(), str(exc), parent=self)
            return
        if not messagebox.askyesno(self.title(), self._t(
                "以選取來源重新計算各模型特徵？已有特徵的片段（含停用）會跳過。"
                "已永久刪除的片段可從來源重建為新特徵。取消或失敗不新增任何特徵。",
                "Encode these passes for each selected model? Registered passes (including inactive ones) "
                "are skipped. Permanently deleted passes may be recreated as new embeddings. "
                "Cancellation or failure adds no embeddings."), parent=self):
            return
        self.page.pause()
        self.page.prepare_encoding()
        self.cancel_event.clear()
        self.running = True
        self._lock_form(True)
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.bar.start(12)
        self.status.set(self._t("準備註冊，尚未儲存特徵", "Preparing enrollment; no embeddings saved yet"))
        self.result_tree.delete(*self.result_tree.get_children())
        for target in targets:
            self.result_tree.insert("", "end", values=(target.bundle_id, target.clip_len,
                                    self._t("待處理", "Pending"), 0))
        self._append(self._t("開始重新編碼...", "Starting re-encoding..."))
        self.worker.start(lambda progress: self.controller.register_from_source(
            self.source_id, targets, passes, device=device, max_windows_per_pass=maximum,
            cancel=self.cancel_event, progress=progress))
        self.after_id = self.after(100, self.poll)

    def cancel(self):
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status.set(self._t("取消中，等待目前推論完成", "Cancelling; waiting for current inference"))
        self._append(self._t("取消中，等待目前推論完成...", "Cancelling after the current inference call..."))

    def poll(self):
        self.after_id = None
        for message in self.worker.drain():
            if message.kind == "progress":
                if not self.cancel_event.is_set():
                    self.status.set(str(message.payload))
                self._append(message.payload)
                continue
            self.running = False
            self.bar.stop()
            self._lock_form(False)
            self.start_button.configure(state="normal" if self.owner_available else "disabled")
            self.cancel_button.configure(state="disabled")
            if message.kind == "error":
                self.status.set(self._t("工作失敗，未新增特徵", "Job failed; no embeddings added"))
                for row in self.result_tree.get_children():
                    self.result_tree.set(row, "state", self._t("未儲存", "Not saved"))
                self._append(self._t("未新增任何 Gallery 特徵：", "No Gallery embeddings added: ") + message.payload["message"])
            else:
                report = message.payload
                states = {
                    "completed": self._t("完成", "Completed"), "registered": self._t("已註冊", "Registered"),
                    "cancelled": self._t("已取消", "Cancelled"), "failed": self._t("失敗", "Failed"),
                    "skipped": self._t("略過，未新增", "Skipped; no additions"),
                    "not_saved": self._t("未儲存", "Not saved"), "pending": self._t("未執行", "Not run"),
                }
                self.status.set(f'{states.get(report["status"], report["status"])} | '
                                + self._t("新增特徵：", "Embeddings added: ") + str(report["embeddings_added"]))
                self.result_tree.delete(*self.result_tree.get_children())
                for target in report["targets"]:
                    self.result_tree.insert("", "end", values=(target["bundle_id"], target["clip_len"],
                        states.get(target["status"], target["status"]), target["embeddings_added"]))
                self._append(f'{report["status"]}: {report["embeddings_added"]} embeddings | job {report["job_id"]}')
                for target in report["targets"]:
                    self._append(f'{target["bundle_id"]}, T={target["clip_len"]}: {target["status"]}, '
                                 f'{target["embeddings_added"]} embeddings')
                    for item in target["passes"]:
                        self._append(f'  {item["pass_id"]}: {item["status"]} ({item["windows"]} windows)')
                self.page.on_gallery_changed()
            self.page.refresh()
            return
        self.after_id = self.after(100, self.poll)

    def close(self):
        if self.running or self.worker.busy:
            self.cancel()
            return
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.destroy()
