"""Modal foreground-source transfer with cancellable I/O and identity mapping."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..source_transfer import SourceTransfer
from .workers import BackgroundWorker


class SourceTransferDialog(tk.Toplevel):
    def __init__(self, page, operation, source_ids=()):
        super().__init__(page)
        self.page, self.locale = page, page.i18n.locale
        storage = getattr(page.controller, "config", {}).get("realtime", {}).get("foreground_storage", {})
        self.storage_options = {key: storage[key] for key in ("library_limit_bytes", "min_free_bytes") if key in storage}
        self.worker = BackgroundWorker()
        self.cancel_event = threading.Event()
        self.running = False
        self.close_requested = False
        self.operation = operation
        self.path = ""
        self.source_ids = tuple(source_ids)
        self.preview = None
        self.people, self.person_map = {}, {}
        self.title(self.tr("Source transfer", "來源點雲匯出／匯入"))
        self.geometry("940x640")
        self.minsize(680, 480)
        self.transient(page.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.status = tk.StringVar(value=self.tr("Preparing...", "準備中..."))
        self.notes = tk.StringVar(value=self.tr(
            "Personal foreground XYZ and identity metadata. NOT encrypted; share only with authorized recipients.",
            "包含個人前景 XYZ 與人物資訊，資料包未加密，僅提供給獲得授權的接收者。"))
        for row, variable in ((0, self.status), (3, self.notes)):
            ttk.Label(self, textvariable=variable, wraplength=880).grid(
                row=row, column=0, sticky="ew", padx=12, pady=8)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=12)
        notebook = ttk.Notebook(self)
        notebook.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        self.person_tree = self._tree(notebook, self.tr("People", "人物"), (
            ("id", "Person ID", "人物 ID"), ("name", "Name", "姓名"),
            ("status", "Status", "狀態"), ("target", "Target ID", "目的 ID")))
        self.source_tree = self._tree(notebook, self.tr("Recordings", "來源片段"), (
            ("person", "Person ID", "人物 ID"), ("source", "Source UUID", "來源 UUID"),
            ("passes", "Passes", "片段數"), ("frames", "Frames", "幀數"),
            ("size", "MiB", "MiB"), ("profile", "Preprocessing", "前處理"),
            ("status", "Status", "狀態")))
        self.person_tree.bind("<<TreeviewSelect>>", lambda _e: self._actions())
        self.restore_var = tk.BooleanVar(value=False)
        self.restore_check = ttk.Checkbutton(self, variable=self.restore_var,
            text=self.tr("Restore previously deleted owners", "還原先前刪除的來源人物"),
            command=self._restore, state="disabled")
        self.restore_check.grid(row=4, column=0, sticky="w", padx=12)
        actions = ttk.Frame(self)
        actions.grid(row=5, column=0, sticky="ew", padx=12, pady=8)
        self.edit_buttons = []
        for col, (en, zh, callback) in enumerate((
            ("Include", "納入人物", self._include), ("New ID", "另建 ID", self._new_id),
            ("Merge into...", "合併至...", self._merge), ("Skip", "略過人物", self._skip),
        )):
            button = ttk.Button(actions, text=self.tr(en, zh), command=callback, state="disabled")
            button.grid(row=0, column=col, padx=(0, 5))
            self.edit_buttons.append(button)
        actions.columnconfigure(4, weight=1)
        self.apply_button = ttk.Button(actions, text=self.tr("Import", "確認匯入"),
                                       command=self._import, state="disabled")
        self.apply_button.grid(row=0, column=5, padx=5)
        self.close_button = ttk.Button(actions, text=self.tr("Close", "關閉"), command=self.close)
        self.close_button.grid(row=0, column=6)
        if operation == "export":
            self.restore_check.grid_remove()
            for button in self.edit_buttons + [self.apply_button]:
                button.grid_remove()
        self.bind("<Configure>", self._resize)
        self.grab_set()
        self._poll_id = self.after(100, self._poll)
        self._choose_id = self.after(50, lambda: self._choose(operation))

    def tr(self, en, zh):
        return zh if self.locale == "zh_TW" else en

    def _service(self):
        return SourceTransfer(self.page.library, **self.storage_options)

    def _resize(self, event):
        if event.widget is self:
            for widget in self.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.configure(wraplength=max(300, event.width - 30))

    def _tree(self, notebook, title, columns):
        host = ttk.Frame(notebook)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        tree = ttk.Treeview(host, columns=[c[0] for c in columns], show="headings", selectmode="browse")
        for key, en, zh in columns:
            tree.heading(key, text=self.tr(en, zh))
            tree.column(key, width=230 if key == "source" else 125, minwidth=70, stretch=True)
        vertical = ttk.Scrollbar(host, command=tree.yview)
        horizontal = ttk.Scrollbar(host, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        notebook.add(host, text=title)
        return tree

    def _choose(self, operation):
        options = dict(parent=self, filetypes=[("MPH sources", "*.mphsources")])
        if operation == "export":
            path = filedialog.asksaveasfilename(**options, defaultextension=".mphsources",
                                               initialfile="enrollment-sources.mphsources")
            if path and not messagebox.askyesno(self.title(), self.tr(
                    f"Export {len(self.source_ids)} selected recordings, including personal foreground points and names? The package is not encrypted.",
                    f"匯出選取的 {len(self.source_ids)} 筆來源（包含個人點雲與姓名）？資料包未加密。"), parent=self):
                path = ""
            if path:
                ids = self.source_ids
                self._start("export", lambda progress: self._service().export(
                    path, ids, cancel=self.cancel_event, progress=progress))
        else:
            path = filedialog.askopenfilename(**options)
            if path:
                self._start("preview", lambda progress: self._service().preview(
                    path, cancel=self.cancel_event, progress=progress))
        self.path = path
        if not path:
            self.close()

    def _start(self, operation, task):
        self.operation, self.running = operation, True
        self.cancel_event.clear()
        self.apply_button.configure(state="disabled")
        self.restore_check.configure(state="disabled")
        for button in self.edit_buttons:
            button.configure(state="disabled")
        self.close_button.configure(text=self.tr("Cancel", "取消"))
        labels = {
            "preview": self.tr("Verifying package; no data imported", "正在驗證資料包，尚未匯入"),
            "export": self.tr("Packaging selected sources", "正在打包選取來源"),
            "import": self.tr("Importing sources; waiting for transaction completion", "正在匯入來源，等待交易完成"),
        }
        self.status.set(labels[operation])
        self.progress.start(15)
        self.worker.start(task)

    def _poll(self):
        for message in self.worker.drain():
            if message.kind == "progress":
                if not self.cancel_event.is_set():
                    self.status.set(self.tr("Processed frames: ", "已處理幀數：") + str(message.payload))
                continue
            if message.kind not in {"result", "error"}:
                continue
            self.running = False
            self.progress.stop()
            self.close_button.configure(text=self.tr("Close", "關閉"))
            if message.kind == "error":
                self.status.set(self.tr("Stopped; close and preview again.", "作業已停止，請關閉後重新預覽。"))
                if not self.close_requested:
                    messagebox.showerror(self.title(), message.payload["message"], parent=self)
            elif self.operation == "preview":
                if not self.close_requested:
                    self._show_preview(message.payload)
            elif self.operation == "export":
                result = message.payload
                self.status.set(self.tr("Export complete: ", "匯出完成：") +
                                f"{result['sources']} / {result['frames']} " + self.tr("sources / frames", "來源／幀"))
                self.notes.set(result["path"] + "\nSHA256: " + result["archive_sha256"])
            else:
                result = message.payload
                self.status.set(self.tr("Added / duplicates / skipped: ", "新增／重複略過／人物略過：") +
                                f"{result['inserted']} / {result['duplicates']} / {result['skipped']}")
                self.notes.set(self.tr("Pre-import database backup: ", "匯入前資料庫備份：") + result["backup"] +
                               ("\n" + result["cleanup_warning"] if result.get("cleanup_warning") else ""))
                self.page.refresh()
                self.page.on_gallery_changed()
            if self.close_requested:
                self.close()
                return
        self._poll_id = self.after(100, self._poll)

    def _show_preview(self, preview):
        self.preview = preview
        statuses = {"new": "新增", "linked": "已對應", "conflict": "衝突", "deleted": "已刪除",
                    "duplicate": "已存在"}
        for person in preview["persons"]:
            uid = person["uid"]
            self.people[uid] = person
            target = person["target_id"] if person["status"] in {"new", "linked"} else None
            if person["status"] == "new" and target in self.person_map.values():
                target = None
            self.person_map[uid] = target
            self.person_tree.insert("", "end", iid=uid, values=(person["person_id"], person["display_name"],
                self.tr(person["status"], statuses[person["status"]]), target or "-"))
        for source in preview["sources"]:
            self.source_tree.insert("", "end", values=(self.people[source["person_uid"]]["person_id"],
                source["source_id"], source["pass_count"], source["frame_count"],
                f"{source['size_bytes']/2**20:.1f}", source["preprocessing_profile_id"],
                self.tr(source["status"], statuses[source["status"]])))
        self.status.set(self.tr("Verified sources / frames: ", "已驗證來源／幀數：") +
                        f"{len(preview['sources'])} / {preview['frames']}")
        self.notes.set(self.tr(
            "ID conflicts are skipped until explicitly mapped. Source UUID/content conflicts block import for that person. No model features or weights are imported.",
            "人物 ID 衝突預設略過，須明確指定對應。來源 UUID／內容衝突會阻止該人物匯入，不會覆寫既有來源。不匯入模型特徵或權重。"))
        self.restore_check.configure(state="normal" if any(p["status"] == "deleted" for p in self.people.values()) else "disabled")
        self._actions()

    def _selected(self):
        selected = self.person_tree.selection()
        return selected[0] if selected else None

    def _actions(self):
        enabled = self.preview is not None and self.operation == "preview" and not self.running
        uid = self._selected()
        person = self.people.get(uid)
        allowed = enabled and person is not None
        status = person["status"] if person else None
        restoring = self.restore_var.get()
        rules = (status in {"new", "linked"} or status == "deleted" and restoring,
                 status not in {"linked"} and (status != "deleted" or restoring),
                 status not in {"linked", "deleted"}, True)
        for button, rule in zip(self.edit_buttons, rules):
            button.configure(state="normal" if allowed and rule else "disabled")
        conflicts = any(s["status"] == "conflict" and self.person_map[s["person_uid"]] is not None
                        for s in (self.preview or {}).get("sources", []))
        self.apply_button.configure(state="normal" if enabled and any(self.person_map.values()) and not conflicts else "disabled")

    def _set_target(self, uid, target):
        self.person_map[uid] = target
        self.person_tree.set(uid, "target", target or "-")
        self._actions()

    def _include(self):
        uid = self._selected()
        if uid is None:
            return
        person = self.people[uid]
        target = person["target_id"]
        if person["status"] == "deleted" and (not self.restore_var.get() or person["target_exists"]):
            return
        if person["status"] != "linked" and target in [v for k, v in self.person_map.items() if k != uid]:
            return
        self._set_target(uid, target)

    def _new_id(self):
        uid = self._selected()
        if uid is None:
            return
        target = simpledialog.askstring(self.title(), self.tr("New person ID", "新的人物 ID"), parent=self)
        if not target or not target.strip():
            return
        target = target.strip()
        if self.page.library.repository.get_person(target) or target in self.person_map.values():
            messagebox.showerror(self.title(), self.tr("Choose an unused person ID.", "請指定未使用的人物 ID。"), parent=self)
            return
        self._set_target(uid, target)

    def _merge(self):
        uid = self._selected()
        if uid is None:
            return
        target = simpledialog.askstring(self.title(), self.tr("Existing person ID", "既有人物 ID"), parent=self)
        if not target:
            return
        person = self.page.library.repository.get_person(target.strip())
        if not person:
            messagebox.showerror(self.title(), self.tr("Person does not exist.", "人物不存在。"), parent=self)
            return
        if messagebox.askyesno(self.title(), self.tr(
                f"Confirm these are the SAME person: {self.people[uid]['display_name']} and {person['display_name']} ({target.strip()})?",
                f"確認這是同一人：{self.people[uid]['display_name']} 與 {person['display_name']}（{target.strip()}）？"), parent=self):
            self._set_target(uid, target.strip())

    def _skip(self):
        uid = self._selected()
        if uid is not None:
            self._set_target(uid, None)

    def _restore(self):
        for uid, person in self.people.items():
            if person["status"] == "deleted":
                self._set_target(uid, None)
        self._actions()

    def _import(self):
        if not self.preview or self.running:
            return
        if not messagebox.askyesno(self.title(), self.tr(
                "Import mapped source recordings? Existing Gallery features and person names/status remain unchanged.",
                "匯入已對應的來源點雲？既有 Gallery 特徵、人物名稱與啟用狀態不會覆寫。"), parent=self):
            return
        # Snapshot Tk state on the UI thread before starting filesystem/database work.
        path, preview, mapping = self.path, self.preview, dict(self.person_map)
        restore = self.restore_var.get()
        self._start("import", lambda progress: self._service().import_archive(
            path, preview, mapping, restore_deleted=restore, cancel=self.cancel_event, progress=progress))

    def close(self):
        if self.running:
            self.close_requested = True
            self.cancel_event.set()
            self.status.set(self.tr("Cancelling; waiting for cleanup...", "取消中，正在等待清理完成..."))
            return
        self.after_cancel(self._poll_id)
        self.after_cancel(self._choose_id)
        self.grab_release()
        self.destroy()
