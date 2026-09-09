from __future__ import annotations

from typing import Any, Callable

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from ..controller import GaitApplicationController
from ..i18n import I18n
from .gallery_transfer import GalleryTransferDialog


class GalleryManagerPage(ttk.Frame):
    """Point-cloud processing-version-scoped Gallery browser."""

    def __init__(
        self,
        master: tk.Misc,
        controller: GaitApplicationController,
        on_changed: Callable[[], None] | None = None,
        transfer_available: Callable[[], tuple[bool, str]] | None = None,
        can_edit: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(master, padding=14)
        self.controller = controller
        shared_i18n = getattr(self.winfo_toplevel(), "_gait_i18n", None)
        self.i18n: I18n = shared_i18n if isinstance(shared_i18n, I18n) else I18n("en")
        self.on_changed = on_changed
        self.transfer_available = transfer_available
        self.can_edit = can_edit or transfer_available
        self.show_all_people = tk.BooleanVar(value=False)
        self.transfer_dialog: GalleryTransferDialog | None = None
        self.bundle_var = tk.StringVar()
        self.clip_len_var = tk.IntVar(value=15)
        self.processing_version_var = tk.StringVar()
        self.summary_var = tk.StringVar(value="Gallery has not been loaded")
        self.person_detail_var = tk.StringVar(value="Select a registered identity")
        self.bundle_labels: dict[str, str] = {}
        self._pass_rows: dict[str, dict[str, Any]] = {}
        self._build()
        self.refresh_bundles()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.export_button = ttk.Button(header, text="Export all Gallery",
                                        command=lambda: self._transfer("export"))
        self.export_button.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.import_button = ttk.Button(header, text="Import Gallery",
                                        command=lambda: self._transfer("import"))
        self.import_button.grid(row=2, column=1, sticky="e", pady=(8, 0))
        ttk.Label(header, text="Gallery Manager", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="gallery.description",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))

        filters = ttk.LabelFrame(self, text="Gallery scope", padding=10)
        filters.grid(row=1, column=0, sticky="ew", pady=(12, 10))
        filters.columnconfigure(1, weight=1)
        ttk.Label(filters, text="Model bundle").grid(row=0, column=0, sticky="w")
        self.bundle_combo = ttk.Combobox(
            filters,
            textvariable=self.bundle_var,
            state="readonly",
        )
        self.bundle_combo.grid(row=0, column=1, sticky="ew", padx=(8, 12))
        self.bundle_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Label(filters, text="Frame length T").grid(row=0, column=2, sticky="w")
        clip_combo = ttk.Combobox(
            filters,
            textvariable=self.clip_len_var,
            values=[15, 30],
            state="readonly",
            width=7,
        )
        clip_combo.grid(row=0, column=3, padx=(8, 12))
        clip_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh())
        ttk.Button(filters, text="Refresh", command=self.refresh).grid(row=0, column=4)
        ttk.Label(filters, text="點雲處理版本").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.processing_version_combo = ttk.Combobox(
            filters,
            textvariable=self.processing_version_var,
            state="readonly",
            values=[self.controller.processing_version_name(self.i18n.locale)],
        )
        self.processing_version_combo.grid(
            row=1, column=1, sticky="ew", padx=(8, 12), pady=(8, 0)
        )
        self.processing_version_var.set(
            self.controller.processing_version_name(self.i18n.locale)
        )
        ttk.Label(filters, textvariable=self.summary_var).grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(8, 0)
        )

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=2, column=0, sticky="nsew")
        persons_group = ttk.LabelFrame(paned, text="Registered identities", padding=8)
        passes_group = ttk.LabelFrame(paned, text="Registration sessions and passes", padding=8)
        paned.add(persons_group, weight=2)
        paned.add(passes_group, weight=3)

        persons_group.columnconfigure(0, weight=1)
        persons_group.rowconfigure(0, weight=1)
        self.person_tree = ttk.Treeview(
            persons_group,
            columns=("id", "name", "embeddings", "models", "note"),
            show="headings",
            selectmode="browse",
        )
        for column, heading, width in (
            ("id", "Person ID", 105),
            ("name", "Display name", 150),
            ("embeddings", "Active emb.", 85),
            ("models", "Models", 65),
            ("note", "Note", 180),
        ):
            self.person_tree.heading(column, text=heading)
            self.person_tree.column(column, width=width, minwidth=width, stretch=False, anchor="w")
        person_scroll = ttk.Scrollbar(persons_group, command=self.person_tree.yview)
        person_hscroll = ttk.Scrollbar(persons_group, orient="horizontal", command=self.person_tree.xview)
        self.person_tree.configure(yscrollcommand=person_scroll.set, xscrollcommand=person_hscroll.set)
        self.person_tree.grid(row=0, column=0, sticky="nsew")
        person_scroll.grid(row=0, column=1, sticky="ns")
        person_hscroll.grid(row=1, column=0, sticky="ew")
        self.person_tree.bind("<<TreeviewSelect>>", lambda _event: self._load_passes())
        ttk.Checkbutton(persons_group, text="gallery.all_people", variable=self.show_all_people,
                        command=self.refresh).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(persons_group, text="gallery.rename_person",
                   command=lambda: self._edit_person("rename")).grid(
                       row=3, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(persons_group, text="gallery.delete_person",
                   command=lambda: self._edit_person("delete_person")).grid(
                       row=4, column=0, columnspan=2, sticky="ew")

        passes_group.columnconfigure(0, weight=1)
        passes_group.rowconfigure(1, weight=1)
        ttk.Label(passes_group, textvariable=self.person_detail_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 7)
        )
        self.pass_tree = ttk.Treeview(
            passes_group,
            columns=("session", "pass", "source", "direction", "active", "quality", "created", "model"),
            show="headings",
            selectmode="browse",
        )
        for column, heading, width in (
            ("session", "Session", 155),
            ("pass", "Pass", 80),
            ("source", "Enrollment source", 220),
            ("direction", "Direction", 125),
            ("active", "Active / total", 95),
            ("quality", "Quality", 70),
            ("created", "Created", 145),
            ("model", "Model key", 180),
        ):
            self.pass_tree.heading(column, text=heading)
            self.pass_tree.column(column, width=width, minwidth=width, stretch=False, anchor="w")
        pass_scroll = ttk.Scrollbar(passes_group, command=self.pass_tree.yview)
        pass_hscroll = ttk.Scrollbar(passes_group, orient="horizontal", command=self.pass_tree.xview)
        self.pass_tree.configure(yscrollcommand=pass_scroll.set, xscrollcommand=pass_hscroll.set)
        self.pass_tree.grid(row=1, column=0, sticky="nsew")
        pass_scroll.grid(row=1, column=1, sticky="ns")
        pass_hscroll.grid(row=2, column=0, sticky="ew")

        actions = ttk.Frame(passes_group)
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(9, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        ttk.Label(
            actions,
            text="gallery.delete_scope_hint",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(
            actions,
            text="Deactivate selected pass",
            command=lambda: self._set_selected_pass_active(False),
        ).grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(
            actions,
            text="Reactivate selected pass",
            command=lambda: self._set_selected_pass_active(True),
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Button(actions, text="gallery.delete_fragment", command=self._delete_fragment).grid(
            row=2, column=0, columnspan=2, sticky="ew")

    def _text(self, zh: str, en: str) -> str:
        return zh if self.i18n.locale == "zh_TW" else en

    def _edit_allowed(self) -> bool:
        if self.can_edit:
            allowed, reason = self.can_edit()
            if not allowed:
                messagebox.showwarning(self._text("暫時無法修改", "Edit unavailable"), reason, parent=self)
                return False
        return True

    def _scope_text(self, preview: dict) -> str:
        return (
            f"{preview['person_id']} / {preview['display_name']}\n"
            + self._text("啟用／停用／總特徵數", "Active / inactive / total embeddings")
            + f": {preview['active']} / {preview['inactive']} / {preview['total']}\n"
            + self._text("模型版本數", "Model versions") + f": {len(preview['models'])}\n"
            + "\n".join(preview["models"])
        )

    def _apply_edit(self, preview: dict, action: str, new_name: str = "") -> None:
        if not self._edit_allowed():
            return
        result = self.controller.apply_person_edit(preview, action, new_name)
        if action == "delete_fragment":
            self.show_all_people.set(True)
        self.refresh()
        if self.on_changed:
            self.on_changed()
        detail = self._text("已修改人物名稱。", "Person renamed.") if action == "rename" else (
            self._text("已刪除特徵數", "Deleted embeddings") + f": {result['deleted_embeddings']}"
        )
        messagebox.showinfo(self._text("Gallery 已更新", "Gallery updated"),
            detail + "\n\n" + self._text("修改前資料庫備份：", "Pre-edit database backup:")
            + "\n" + result["backup_path"], parent=self)

    def _edit_person(self, action: str) -> None:
        person_id = self._selected_person_id()
        if not person_id or not self._edit_allowed():
            return
        try:
            preview = self.controller.preview_person_edit(person_id)
            if action == "rename":
                name = simpledialog.askstring(
                    self._text("修改人物名稱", "Rename person"),
                    self._scope_text(preview) + "\n\n" + self._text(
                        "名稱套用至所有模型；ID 與特徵保留。新名稱：",
                        "Name applies to all models; ID and embeddings remain. New name:"),
                    initialvalue=preview["display_name"], parent=self,
                )
                if name is not None and name.strip() != preview["display_name"]:
                    self._apply_edit(preview, action, name)
            elif action == "delete_person":
                answer = simpledialog.askstring(
                    self._text("刪除人物及全部模型特徵", "Delete person and ALL model embeddings"),
                    self._scope_text(preview) + "\n\n" + self._text(
                        "跨所有模型刪除此人物及啟用／停用特徵，釋放 ID。\n"
                        "來源、報告與權重保留；先備份。舊匯入包不會恢復此人物。\n"
                        "請輸入完整 Person ID 確認：",
                        "Deletes this person and active/inactive embeddings across ALL models; frees the ID.\n"
                        "Sources, reports and weights remain; a backup is made first. Old archives cannot restore this identity.\n"
                        "Type the exact Person ID to confirm:"), parent=self,
                )
                if answer == person_id:
                    self._apply_edit(preview, action)
                elif answer is not None:
                    messagebox.showinfo("Gallery", self._text(
                        "ID 不符，已取消刪除。", "ID mismatch; deletion cancelled."), parent=self)
        except Exception as exc:
            messagebox.showerror("Gallery", str(exc), parent=self)

    def _delete_fragment(self) -> None:
        selection = self.pass_tree.selection()
        person_id, bundle_id = self._selected_person_id(), self._bundle_id()
        if not selection or not person_id or not bundle_id or not self._edit_allowed():
            return
        item = self._pass_rows.get(str(selection[0]))
        if item is None:
            return
        try:
            preview = self.controller.preview_fragment_delete(
                bundle_id, person_id, item, int(self.clip_len_var.get()))
            prompt = self._scope_text(preview) + (
                f"\nSession: {item['session_id']}\nPass: {item['pass_id']}\n"
                f"Source: {item['source_path']}\n\n"
            ) + self._text(
                "永久刪除此列的全部特徵（含停用）？\n"
                "保留人物、其他片段與來源檔案。先備份；舊匯入包中的已刪特徵會略過。",
                "Permanently delete all active and inactive embeddings in this row?\n"
                "Person, other fragments and source files remain. A backup is made first; deleted archive entries will be skipped.")
            if messagebox.askyesno(self._text("永久刪除片段", "Permanently delete fragment"), prompt, parent=self):
                self._apply_edit(preview, "delete_fragment")
        except Exception as exc:
            messagebox.showerror("Gallery", str(exc), parent=self)

    def _bundle_id(self) -> str | None:
        return self.bundle_labels.get(self.bundle_var.get())

    def _transfer(self, operation: str) -> None:
        if self.transfer_dialog is not None and self.transfer_dialog.winfo_exists():
            self.transfer_dialog.lift()
            return
        if self.transfer_available:
            allowed, reason = self.transfer_available()
            if not allowed:
                messagebox.showwarning("Gallery", reason, parent=self)
                return
        def changed() -> None:
            self.refresh_bundles()
            if self.on_changed:
                self.on_changed()
        self.transfer_dialog = GalleryTransferDialog(
            self, self.controller, operation, self.i18n.locale, changed,
        )

    def set_locale(self, locale_name: str) -> None:
        self.export_button.configure(text="匯出全部 Gallery" if locale_name == "zh_TW" else "Export all Gallery")
        self.import_button.configure(text="匯入 Gallery" if locale_name == "zh_TW" else "Import Gallery")
        display = self.controller.processing_version_name(locale_name)
        self.processing_version_combo.configure(values=[display])
        self.processing_version_var.set(display)
        self.refresh()

    def refresh_bundles(self) -> None:
        current = self._bundle_id()
        bundles = self.controller.available_bundles(refresh=True)
        self.bundle_labels = {
            f"{bundle.display_name} | {bundle.checkpoint_sha256[:8]}": bundle_id
            for bundle_id, bundle in bundles.items()
        }
        labels = list(self.bundle_labels)
        self.bundle_combo.configure(values=labels)
        preferred = str(
            self.controller.config.get("runtime", {}).get(
                "bundle",
                "mph_gait_fixed_special5_seed0_split0",
            )
        )
        selected = next(
            (label for label, bundle_id in self.bundle_labels.items() if bundle_id == current),
            next(
                (
                    label
                    for label, bundle_id in self.bundle_labels.items()
                    if bundle_id == preferred
                ),
                labels[0] if labels else "",
            ),
        )
        self.bundle_var.set(selected)
        self.refresh()

    def refresh(self) -> None:
        bundle_id = self._bundle_id()
        if not bundle_id:
            return
        selected_person = self._selected_person_id()
        clip_len = int(self.clip_len_var.get())
        summary = self.controller.database_summary(bundle_id, clip_len=clip_len)
        self.summary_var.set(
            f"{self.processing_version_var.get()} | T={clip_len}: "
            f"{summary.get('persons', 0)} identities / "
            f"{summary.get('active_embeddings', 0)} active embeddings"
        )
        children = self.person_tree.get_children()
        if children:
            self.person_tree.delete(*children)
        for person in self.controller.list_persons(
            None if self.show_all_people.get() else bundle_id,
            clip_len=None if self.show_all_people.get() else clip_len,
            include_inactive=True,
        ):
            person_id = str(person.get("person_id", ""))
            self.person_tree.insert(
                "",
                "end",
                iid=person_id,
                values=(
                    person_id,
                    person.get("display_name", ""),
                    person.get("embedding_count", 0),
                    person.get("model_count", 0),
                    person.get("note", ""),
                ),
            )
        if selected_person and self.person_tree.exists(selected_person):
            self.person_tree.selection_set(selected_person)
            self.person_tree.focus(selected_person)
        elif self.person_tree.get_children():
            first = self.person_tree.get_children()[0]
            self.person_tree.selection_set(first)
            self.person_tree.focus(first)
        self._load_passes()

    def _selected_person_id(self) -> str | None:
        selection = self.person_tree.selection()
        return str(selection[0]) if selection else None

    def _load_passes(self) -> None:
        children = self.pass_tree.get_children()
        if children:
            self.pass_tree.delete(*children)
        self._pass_rows = {}
        person_id = self._selected_person_id()
        bundle_id = self._bundle_id()
        if not person_id or not bundle_id:
            self.person_detail_var.set("Select a registered identity")
            return
        person = self.controller.get_person(person_id) or {}
        self.person_detail_var.set(
            f"{person_id} — {person.get('display_name', '')} | "
            "inactive passes remain stored but are excluded from recognition"
        )
        rows = self.controller.list_gallery_passes(
            bundle_id,
            person_id,
            clip_len=int(self.clip_len_var.get()),
            include_inactive=True,
        )
        for index, item in enumerate(rows):
            iid = f"pass-row-{index}"
            self._pass_rows[iid] = item
            active = int(item.get("active_embeddings") or 0)
            total = int(item.get("total_embeddings") or 0)
            quality = item.get("mean_quality")
            self.pass_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    item.get("session_id") or "legacy",
                    item.get("pass_id") or "legacy source",
                    item.get("source_path", ""),
                    str(item.get("direction") or "unknown").replace("_", " "),
                    f"{active} / {total}",
                    "-" if quality is None else f"{float(quality):.3f}",
                    item.get("created_at", ""),
                    item.get("model_key", ""),
                ),
            )

    def _set_selected_pass_active(self, active: bool) -> None:
        if not self._edit_allowed():
            return
        selection = self.pass_tree.selection()
        if not selection:
            messagebox.showinfo("Select a pass", "Select one pass first.", parent=self)
            return
        item = self._pass_rows.get(str(selection[0]))
        person_id = self._selected_person_id()
        bundle_id = self._bundle_id()
        if item is None or not person_id or not bundle_id:
            return
        session_id = str(item.get("session_id") or "")
        pass_id = str(item.get("pass_id") or "")
        if not session_id or not pass_id:
            messagebox.showinfo(
                "Legacy Gallery row",
                self._text(
                    "此片段沒有 session／pass ID，可永久刪除選取片段，或到離線頁停用來源。",
                    "This row has no session/pass IDs. Delete the selected fragment here, or deactivate its source on the Offline page."),
                parent=self,
            )
            return
        verb = "reactivate" if active else "deactivate"
        if not messagebox.askyesno(
            f"{verb.title()} pass",
            f"{verb.title()} {person_id} / {session_id} / {pass_id}?",
            parent=self,
        ):
            return
        try:
            count = self.controller.set_gallery_pass_active(
                bundle_id=bundle_id,
                person_id=person_id,
                model_key=str(item.get("model_key", "")),
                session_id=session_id,
                pass_id=pass_id,
                active=active,
            )
        except Exception as exc:
            messagebox.showerror("Gallery update failed", str(exc), parent=self)
            return
        self.refresh()
        if self.on_changed is not None:
            self.on_changed()
        messagebox.showinfo(
            "Gallery updated",
            f"{count} embeddings were {'reactivated' if active else 'deactivated'}.",
            parent=self,
        )
