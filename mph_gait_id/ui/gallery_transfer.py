"""Modal Gallery transfer with preview and explicit person conflict resolution."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Callable

from .workers import BackgroundWorker


class GalleryTransferDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, controller: Any, operation: str,
                 locale: str, on_changed: Callable[[], None] | None = None):
        super().__init__(parent)
        self.controller, self.locale, self.on_changed = controller, locale, on_changed
        self.worker = BackgroundWorker()
        self.preview: dict[str, Any] | None = None
        self.person_map: dict[str, str | None] = {}
        self.people: dict[str, dict] = {}
        self.path = ""
        self.title(self.tr("Gallery transfer", "Gallery 匯出／匯入"))
        self.geometry("880x580")
        self.minsize(640, 420)
        self.transient(parent.winfo_toplevel())
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self.status = tk.StringVar(value=self.tr("Preparing...", "準備中…"))
        ttk.Label(self, textvariable=self.status, wraplength=820).grid(
            row=0, column=0, sticky="ew", padx=12, pady=10)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.grid(row=1, column=0, sticky="ew", padx=12)
        notebook = ttk.Notebook(self)
        notebook.grid(row=2, column=0, sticky="nsew", padx=12, pady=10)
        self.person_tree = self._tree(notebook, self.tr("People", "人物"),
                                      ("id", "name", "status", "target"),
                                      ("Person ID", "Name", "Status", "Target ID"))
        self.model_tree = self._tree(notebook, self.tr("Models", "模型"),
                                     ("model", "status", "target"),
                                     ("Model", "Compatibility", "Local bundle"))
        self.person_tree.bind("<<TreeviewSelect>>", lambda _e: self._actions())
        self.notes = tk.StringVar(value=self.tr(
            "Contains personal biometric features. Archives are NOT encrypted. No weights or source point clouds are included.",
            "包含個人辨識特徵，匯出檔未加密。不包含模型權重或原始點雲。"))
        ttk.Label(self, textvariable=self.notes, wraplength=820).grid(
            row=3, column=0, sticky="ew", padx=12, pady=6)
        actions = ttk.Frame(self)
        actions.grid(row=4, column=0, sticky="ew", padx=12, pady=8)
        self.edit_buttons = []
        for column, (label, callback) in enumerate((
            (self.tr("New ID", "另建人物 ID"), self._new_id),
            (self.tr("Merge into...", "合併至既有人物…"), self._merge),
            (self.tr("Skip", "略過人物"), self._skip),
        )):
            button = ttk.Button(actions, text=label, command=callback, state="disabled")
            button.grid(row=0, column=column, padx=(0, 6))
            self.edit_buttons.append(button)
        actions.columnconfigure(3, weight=1)
        self.apply_button = ttk.Button(actions, text=self.tr("Import", "確認匯入"),
                                       state="disabled", command=self._import)
        self.apply_button.grid(row=0, column=4, padx=6)
        ttk.Button(actions, text=self.tr("Close", "關閉"), command=self.close).grid(row=0, column=5)
        self.bind("<Configure>", self._resize)
        self.grab_set()
        self.after(50, lambda: self._choose(operation))
        self._poll_id = self.after(100, self._poll)

    def tr(self, en: str, zh: str) -> str:
        return zh if self.locale == "zh_TW" else en

    def _resize(self, event: tk.Event) -> None:
        if event.widget is self:
            for widget in self.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.configure(wraplength=max(300, event.width - 30))

    def _tree(self, notebook: ttk.Notebook, label: str, columns: tuple,
              headings: tuple) -> ttk.Treeview:
        host = ttk.Frame(notebook)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        tree = ttk.Treeview(host, columns=columns, show="headings", selectmode="browse")
        translations = {"Person ID": "人物 ID", "Name": "姓名", "Status": "狀態", "Target ID": "目的 ID",
                        "Model": "模型", "Compatibility": "相容性", "Local bundle": "本機 Bundle"}
        for name, title in zip(columns, headings):
            tree.heading(name, text=self.tr(title, translations[title]))
            tree.column(name, width=190, minwidth=110, stretch=True)
        vertical = ttk.Scrollbar(host, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(host, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        notebook.add(host, text=label)
        return tree

    def _choose(self, operation: str) -> None:
        options = {"parent": self, "filetypes": [("MPH Gallery", "*.mphgallery")]}
        if operation == "export":
            self.path = filedialog.asksaveasfilename(
                **options, defaultextension=".mphgallery", initialfile="gallery.mphgallery")
            if self.path:
                self._start("export", lambda: self.controller.export_gallery(self.path))
        else:
            self.path = filedialog.askopenfilename(**options)
            if self.path:
                self._start("preview", lambda: self.controller.preview_gallery_import(self.path))
        if not self.path:
            self.close()

    def _start(self, operation: str, task: Callable) -> None:
        self.operation = operation
        self.status.set(self.tr("Checking / processing Gallery...", "正在檢查／處理 Gallery…"))
        self.apply_button.configure(state="disabled")
        for button in self.edit_buttons:
            button.configure(state="disabled")
        self.progress.start(15)
        self.worker.start(lambda _progress: task())

    def _poll(self) -> None:
        for message in self.worker.drain():
            if message.kind == "error":
                self.progress.stop()
                self.status.set(self.tr("Transfer failed. Close and preview again.", "處理失敗，請關閉後重新預覽。"))
                messagebox.showerror(self.title(), message.payload["message"], parent=self)
            elif message.kind == "result":
                self.progress.stop()
                result = message.payload
                if self.operation == "preview":
                    self._show_preview(result)
                elif self.operation == "export":
                    self.status.set(self.tr("Export complete", "匯出完成") +
                                    f": {result['persons']} / {result['embeddings']}\n{result['path']}")
                else:
                    self.status.set(self.tr("Import complete", "匯入完成") + "\n" +
                                    self.tr("Added / duplicates / skipped", "新增／重複／略過") +
                                    f": {result['inserted']} / {result['duplicates']} / {result['skipped']}")
                    self.notes.set(self.tr("Backup", "備份") + f": {result['backup']}\n" +
                                   self.tr("Retained package", "保留匯入包") + f": {result['retained_archive']}")
                    if self.on_changed:
                        self.on_changed()
        self._poll_id = self.after(100, self._poll)

    def _show_preview(self, preview: dict) -> None:
        self.preview = preview
        for person in preview["persons"]:
            uid = person["uid"]
            self.people[uid] = person
            self.person_map[uid] = None if person["status"] == "conflict" else person["target_id"]
            self.person_tree.insert("", "end", iid=uid, values=(
                person["person_id"], person["display_name"],
                self.tr(person["status"], {"new": "新增", "linked": "已對應", "conflict": "ID 衝突"}[person["status"]]),
                self.person_map[uid] or "-"))
        for key, model in preview["models"].items():
            self.model_tree.insert("", "end", iid=key, values=(
                model["display_name"], self.tr(model["status"], "相容" if model["target"] else "無可用相容模型"),
                model["target"]["bundle_id"] if model["target"] else "-"))
        self.status.set(self.tr("Available / unavailable / previously imported embeddings", "可用／不相容／已匯入過的特徵") +
                        f": {preview['compatible_embeddings']} / {preview['unavailable_embeddings']} / {preview['known_embedding_ids']}")
        self.notes.set(self.tr(
            "ID conflicts default to Skip. Missing/incompatible models remain in the retained package; install matching bundles and import again. Bundle settings describe the exporter, not historical enrollment settings. Confirm they have not changed since enrollment.",
            "ID 衝突預設略過。不相容模型的特徵保留在匯入包，安裝相符模型後可重新匯入。Bundle 設定為匯出當下的版本，請確認與原註冊時一致。"))
        self.apply_button.configure(state="normal")

    def _selected(self) -> str | None:
        selected = self.person_tree.selection()
        return selected[0] if selected else None

    def _actions(self) -> None:
        enabled = self._selected() is not None and not self.worker.busy and self.operation == "preview"
        for button in self.edit_buttons:
            button.configure(state="normal" if enabled else "disabled")

    def _set_target(self, uid: str, target: str | None) -> None:
        self.person_map[uid] = target
        self.person_tree.set(uid, "target", target or "-")

    def _new_id(self) -> None:
        uid = self._selected()
        if uid is None:
            return
        if self.people[uid]["status"] == "linked":
            messagebox.showinfo(self.title(), self.tr("Previously linked identities cannot be remapped.", "已匯入的人物不可重新對應 ID。"), parent=self)
            return
        target = simpledialog.askstring(self.title(), self.tr("New person ID", "新的人物 ID"), parent=self)
        if not target:
            return
        target = target.strip()
        if self.controller.get_person(target) or target in self.person_map.values():
            messagebox.showerror(self.title(), self.tr("ID already exists. Use explicit merge instead.", "ID 已存在，請使用合併功能。"), parent=self)
            return
        self._set_target(uid, target)

    def _merge(self) -> None:
        uid = self._selected()
        if uid is None:
            return
        if self.people[uid]["status"] == "linked":
            return
        target = simpledialog.askstring(self.title(), self.tr("Existing local person ID", "本機既有人物 ID"), parent=self)
        if not target:
            return
        target = target.strip()
        person = self.controller.get_person(target)
        if not person:
            messagebox.showerror(self.title(), self.tr("Person not found", "找不到該人物"), parent=self)
            return
        incoming = self.people[uid]
        if messagebox.askyesno(self.title(), self.tr("Confirm these are the SAME person:", "請確認以下為同一個人：") +
                              f"\n{incoming['person_id']} / {incoming['display_name']}\n{target} / {person['display_name']}", parent=self):
            self._set_target(uid, target)

    def _skip(self) -> None:
        uid = self._selected()
        if uid:
            self._set_target(uid, None)

    def _import(self) -> None:
        if self.preview is None:
            return
        if not messagebox.askyesno(self.title(), self.tr(
            "Import the selected identities? Existing inactive entries remain inactive. Only use archives from a trusted source.",
            "匯入選定人物？既有停用資料將保持停用。請只使用可信來源的匯入包。"), parent=self):
            return
        mapping = dict(self.person_map)
        self._start("import", lambda: self.controller.import_gallery(self.path, self.preview, mapping))

    def close(self) -> None:
        if self.worker.busy:
            messagebox.showinfo(self.title(), self.tr("Wait for the operation to finish.", "請等待作業完成後再關閉。"), parent=self)
            return
        self.grab_release()
        self.after_cancel(self._poll_id)
        self.destroy()
