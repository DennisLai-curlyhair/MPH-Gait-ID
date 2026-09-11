from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from mph_gait_id.controller import GaitApplicationController, OperationOutcome
from mph_gait_id.identity import suggest_registration_identity
from mph_gait_id.i18n import I18n, LANGUAGE_CHOICES

from .player import PreviewPlayer
from .gallery_manager import GalleryManagerPage
from .enrollment_sources import EnrollmentSourcesPage
from .performance_benchmark import PerformanceBenchmarkPage
from .scrollable import ScrollableFrame
from .workers import BackgroundWorker, WorkerMessage


COLORS = {
    "app_bg": "#eef1f4",
    "panel": "#ffffff",
    "border": "#cfd6dd",
    "text": "#1f2933",
    "muted": "#66717d",
    "accent": "#167a78",
    "accent_active": "#116361",
    "stable": "#16794b",
    "unknown": "#b33f3f",
    "warning": "#a56a12",
    "info": "#246b91",
}


MODEL_FAMILY_LABELS = {
    "Point Cloud / 點雲": "pointcloud",
}

METHOD_DISPLAY_NAMES = {
    "pointcloud_tuned": "TemporalPointNet",
    "pc_v1": "PointNet-TMax",
    "mph_gait": "MPH-Gait ID",
    "lidargaitpp": "LidarGait++",
}


class GaitIdentityWindow:
    def __init__(self, root: tk.Tk, controller: GaitApplicationController) -> None:
        self.root = root
        self.controller = controller
        configured_language = str(
            controller.config.get("ui", {}).get("language", "auto")
        )
        self.i18n = I18n(configured_language)
        setattr(self.root, "_gait_i18n", self.i18n)
        self.language_var = tk.StringVar(value=self.i18n.language_choice())
        self.worker = BackgroundWorker()
        self.current_outcome: OperationOutcome | None = None
        self.outcome_history: dict[str, OperationOutcome] = {}
        self.history_limit = 50
        self._active_source_signature: tuple[str, str] | None = None
        self.source_paths: list[Path] = []
        self.bundles = controller.available_bundles()
        if not self.bundles:
            raise RuntimeError("No model bundles were found in mph_gait_id/model_bundles")

        preferred = str(
            controller.config.get("runtime", {}).get(
                "bundle",
                "mph_gait_fixed_special5_seed0_split0",
            )
        )
        if preferred not in self.bundles:
            preferred = next(iter(self.bundles))
        self.bundle_labels: dict[str, str] = {}
        self.bundle_ids_to_labels: dict[str, str] = {}
        self.family_labels = dict(MODEL_FAMILY_LABELS)
        self.family_ids_to_labels = {
            family_id: label for label, family_id in self.family_labels.items()
        }
        self.method_labels: dict[str, str] = {}
        self.method_keys_to_labels: dict[str, str] = {}
        self._rebuild_bundle_labels()

        self.family_var = tk.StringVar()
        self.method_var = tk.StringVar()
        self.bundle_var = tk.StringVar()
        self._set_model_selection(preferred)
        self.model_info_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.device_var = tk.StringVar(value="auto")
        self.source_kind_var = tk.StringVar(value="尚未選擇來源")
        self.identity_suggestion_var = tk.StringVar(value="尚未從來源偵測人物")
        self.threshold_var = tk.DoubleVar(value=controller.provisional_threshold(preferred))
        self.margin_var = tk.DoubleVar(value=0.03)
        self.person_id_var = tk.StringVar()
        self.person_name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="系統就緒")
        self.database_var = tk.StringVar(value="Gallery 尚未載入")
        self.result_identity_var = tk.StringVar(value="尚無辨識結果")
        self.result_detail_var = tk.StringVar(value="-")
        self.result_state_var = tk.StringVar(value="IDLE")

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self.i18n.apply(self.root)
        self._on_bundle_changed()
        self._refresh_database()
        self.root.after(100, self._poll_worker)

    def _configure_window(self) -> None:
        geometry = str(self.controller.config.get("ui", {}).get("geometry", "1380x860"))
        self.root.title(self.i18n.tr("app.title"))
        self.root.geometry(geometry)
        self.root.minsize(980, 640)
        self.root.configure(background=COLORS["app_bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["app_bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Preview.TFrame", background=COLORS["panel"])
        style.configure("PreviewImage.TLabel", background="#181b1f", foreground="#dce3e8", padding=8)
        style.configure("TLabel", background=COLORS["app_bg"], foreground=COLORS["text"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Muted.Panel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure(
            "Header.TLabel",
            background=COLORS["app_bg"],
            foreground=COLORS["text"],
            font=("TkDefaultFont", 17, "bold"),
        )
        style.configure(
            "Result.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("TkDefaultFont", 16, "bold"),
        )
        for name, color in (
            ("Stable", COLORS["stable"]),
            ("Unknown", COLORS["unknown"]),
            ("Warning", COLORS["warning"]),
            ("Info", COLORS["info"]),
        ):
            style.configure(
                f"{name}.TLabel",
                background=COLORS["panel"],
                foreground=color,
                font=("TkDefaultFont", 10, "bold"),
            )
        style.configure(
            "TLabelframe",
            background=COLORS["panel"],
            bordercolor=COLORS["border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("TkDefaultFont", 10, "bold"),
        )
        style.configure(
            "TButton",
            padding=(10, 7),
            background="#e8edf1",
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
        )
        style.map("TButton", background=[("active", "#dce3e8")])
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#ffffff",
            bordercolor=COLORS["accent"],
            font=("TkDefaultFont", 10, "bold"),
            padding=(12, 9),
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent_active"]), ("disabled", "#9eb6b5")],
        )
        style.configure("Treeview", rowheight=27, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))
        style.configure("TNotebook", background=COLORS["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 7))
        style.configure("Horizontal.TProgressbar", background=COLORS["accent"])

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        app_bar = ttk.Frame(self.root, padding=(16, 8, 16, 0))
        app_bar.grid(row=0, column=0, sticky="ew")
        app_bar.columnconfigure(0, weight=1)
        ttk.Label(app_bar, text="app.title", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(app_bar, text="language").grid(
            row=0, column=1, sticky="e", padx=(10, 6)
        )
        self.language_combo = ttk.Combobox(
            app_bar,
            textvariable=self.language_var,
            values=list(LANGUAGE_CHOICES),
            state="readonly",
            width=12,
        )
        self.language_combo.grid(row=0, column=2, sticky="e")
        self.language_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._on_language_changed(),
        )

        self.mode_notebook = ttk.Notebook(self.root)
        self.mode_notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.offline_page = ttk.Frame(self.mode_notebook)
        self.realtime_page_host = ttk.Frame(self.mode_notebook, padding=10)
        self.gallery_page_host = ttk.Frame(self.mode_notebook, padding=10)
        self.performance_page_host = ttk.Frame(self.mode_notebook, padding=10)
        self.mode_notebook.add(self.offline_page, text="Offline registration / recognition")
        self.mode_notebook.add(self.realtime_page_host, text="Real-time Azure Kinect")
        self.mode_notebook.add(self.gallery_page_host, text="Gallery Manager")
        self.mode_notebook.add(self.performance_page_host, text="Live Performance Benchmark")
        self._build_offline_layout(self.offline_page)

        from mph_gait_id.realtime.ui_page import RealtimePage

        self.realtime_page_host.columnconfigure(0, weight=1)
        self.realtime_page_host.rowconfigure(0, weight=1)
        self.gallery_page_host.columnconfigure(0, weight=1)
        self.gallery_page_host.rowconfigure(0, weight=1)
        self.performance_page_host.columnconfigure(0, weight=1)
        self.performance_page_host.rowconfigure(0, weight=1)
        self.gallery_page = GalleryManagerPage(
            self.gallery_page_host,
            self.controller,
            on_changed=self._gallery_changed,
            transfer_available=self._gallery_transfer_available,
            can_edit=self._gallery_transfer_available,
        )
        self.gallery_page.grid(row=0, column=0, sticky="nsew")
        self.realtime_page = RealtimePage(
            self.realtime_page_host,
            self.controller,
            open_offline=lambda: self.mode_notebook.select(self.offline_page),
            open_gallery=self._open_gallery_from_realtime,
            on_gallery_changed=self._gallery_changed,
            camera_available=self._camera_available_for_realtime,
        )
        self.realtime_page.grid(row=0, column=0, sticky="nsew")
        self.performance_page = PerformanceBenchmarkPage(
            self.performance_page_host,
            self.controller,
            camera_available=self._camera_available_for_benchmark,
        )
        self.performance_page.grid(row=0, column=0, sticky="nsew")
        self.sources_page = EnrollmentSourcesPage(
            self.mode_notebook, self.controller, can_edit=self._source_edit_available,
            on_gallery_changed=self._gallery_changed, prepare_encoding=self._prepare_source_encoding)
        self.mode_notebook.add(self.sources_page, text="nav.sources")
        self.mode_notebook.bind(
            "<<NotebookTabChanged>>",
            lambda _event: self._on_mode_changed(),
        )

    def _on_language_changed(self) -> None:
        locale_name = LANGUAGE_CHOICES.get(self.language_var.get())
        if locale_name is None:
            return
        self.i18n.set_locale(locale_name)
        self.root.title(self.i18n.tr("app.title"))
        if hasattr(self, "realtime_page"):
            self.realtime_page.set_locale(locale_name)
        if hasattr(self, "gallery_page"):
            self.gallery_page.set_locale(locale_name)
        if hasattr(self, "performance_page"):
            self.performance_page.set_locale(locale_name)
        if hasattr(self, "sources_page"):
            self.sources_page.set_locale(locale_name)
        self.i18n.apply(self.root)

    def _on_mode_changed(self) -> None:
        if not hasattr(self, "realtime_page"):
            return
        if self.mode_notebook.select() == str(self.realtime_page_host):
            self.realtime_page.refresh_bundles()
        elif self.mode_notebook.select() == str(self.gallery_page_host):
            self.gallery_page.refresh_bundles()
        elif self.mode_notebook.select() == str(self.performance_page_host):
            self.performance_page.refresh_bundles()
        if hasattr(self, "sources_page"):
            self.sources_page.pause()
            if self.mode_notebook.select() == str(self.sources_page):
                self.sources_page.refresh()

    def _source_edit_available(self) -> tuple[bool, str]:
        dialog = getattr(getattr(self, "gallery_page", None), "transfer_dialog", None)
        if dialog is not None and dialog.winfo_exists():
            return False, ("請先關閉 Gallery 匯出／匯入視窗。" if self.i18n.locale == "zh_TW"
                           else "Close the Gallery export/import dialog first.")
        return self._gallery_transfer_available()

    def _source_operation_running(self) -> bool:
        return bool(getattr(getattr(self, "sources_page", None), "busy", False))

    def _prepare_source_encoding(self) -> None:
        for name in ("realtime_page", "performance_page"):
            pipeline = getattr(getattr(self, name, None), "pipeline", None)
            if pipeline is not None and not pipeline.running:
                pipeline._runtime = None

    def _camera_available_for_realtime(self) -> tuple[bool, str]:
        if self._source_operation_running():
            return False, "Enrollment source operation in progress."
        benchmark = getattr(self, "performance_page", None)
        pipeline = getattr(benchmark, "pipeline", None)
        if pipeline is not None and pipeline.running:
            return False, "請先停止第四頁的即時效率測試，再啟動即時辨識或註冊。"
        return True, ""

    def _camera_available_for_benchmark(self) -> tuple[bool, str]:
        if self._source_operation_running():
            return False, "Enrollment source operation in progress."
        realtime = getattr(self, "realtime_page", None)
        pipeline = getattr(realtime, "pipeline", None)
        if pipeline is not None and pipeline.running:
            return False, "請先停止即時辨識／註冊，再開始效率測試。"
        return True, ""

    def _gallery_transfer_available(self) -> tuple[bool, str]:
        running = self.worker.busy or self._source_operation_running()
        dialog = getattr(getattr(self, "gallery_page", None), "transfer_dialog", None)
        running = running or (dialog is not None and dialog.winfo_exists() and dialog.worker.busy)
        for name in ("realtime_page", "performance_page"):
            pipeline = getattr(getattr(self, name, None), "pipeline", None)
            running = running or (pipeline is not None and pipeline.running)
        pending = getattr(getattr(self, "realtime_page", None), "_pending_review_result", None)
        if pending is not None:
            return False, self.i18n.tr("gallery.finish_before_edit")
        if running:
            return False, ("請先停止辨識、註冊與效能測試。" if self.i18n.locale == "zh_TW"
                           else "Stop recognition, enrollment and benchmarking first.")
        return True, ""

    def _gallery_changed(self) -> None:
        self._refresh_database()
        if hasattr(self, "realtime_page"):
            self.realtime_page._refresh_gallery_summary()

    def _open_gallery_from_realtime(self) -> None:
        self.gallery_page.refresh_bundles()
        bundle_id = self.realtime_page._bundle_id(required=False)
        if bundle_id is not None:
            label = next(
                (
                    value
                    for value, candidate in self.gallery_page.bundle_labels.items()
                    if candidate == bundle_id
                ),
                None,
            )
            if label is not None:
                self.gallery_page.bundle_var.set(label)
        self.gallery_page.clip_len_var.set(int(self.realtime_page.clip_len_var.get()))
        self.gallery_page.refresh()
        self.mode_notebook.select(self.gallery_page_host)

    def _build_offline_layout(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        header = ttk.Frame(parent, padding=(18, 14, 18, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="步態身分辨識系統", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var, foreground=COLORS["muted"]).grid(
            row=0, column=1, sticky="e"
        )

        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        self.sidebar_scroll = ScrollableFrame(
            paned,
            width=390,
            canvas_background=COLORS["panel"],
            frame_style="Panel.TFrame",
        )
        content = ttk.Frame(paned, style="Panel.TFrame", padding=14)
        paned.add(self.sidebar_scroll, weight=0)
        paned.add(content, weight=1)
        self._build_sidebar(self.sidebar_scroll.content)
        self._build_content(content)

        status_bar = ttk.Frame(parent, padding=(16, 4, 16, 8))
        status_bar.grid(row=2, column=0, sticky="ew")
        status_bar.columnconfigure(0, weight=1)
        ttk.Label(status_bar, textvariable=self.database_var, foreground=COLORS["muted"]).grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(status_bar, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, sticky="e")

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        model_group = ttk.LabelFrame(parent, text="模型管理", padding=12)
        model_group.grid(row=0, column=0, sticky="ew")
        model_group.columnconfigure(0, weight=1)
        model_group.columnconfigure(1, weight=1)
        ttk.Label(model_group, text="輸入類型", style="Panel.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.family_combo = ttk.Combobox(
            model_group,
            textvariable=self.family_var,
            values=list(self.family_labels),
            state="readonly",
        )
        self.family_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 7))
        self.family_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_family_changed())

        ttk.Label(model_group, text="模型方法", style="Panel.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w"
        )
        self.method_combo = ttk.Combobox(
            model_group,
            textvariable=self.method_var,
            values=list(self.method_labels),
            state="readonly",
        )
        self.method_combo.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 7))
        self.method_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_method_changed())

        ttk.Label(model_group, text="權重 / Bundle", style="Panel.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w"
        )
        self.bundle_combo = ttk.Combobox(
            model_group,
            textvariable=self.bundle_var,
            state="readonly",
        )
        self.bundle_combo.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        self.bundle_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_bundle_changed())
        self._set_model_selection(self._bundle_id())

        ttk.Label(
            model_group,
            textvariable=self.model_info_var,
            style="Muted.Panel.TLabel",
            wraplength=340,
        ).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(
            model_group,
            textvariable=self.model_path_var,
            style="Muted.Panel.TLabel",
            wraplength=340,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 8))
        self.import_button = ttk.Button(model_group, text="匯入權重", command=self._import_checkpoint)
        self.import_button.grid(row=8, column=0, sticky="ew")
        self.rescan_button = ttk.Button(model_group, text="重新掃描", command=self._rescan_bundles)
        self.rescan_button.grid(row=8, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(model_group, text="Device", style="Panel.TLabel").grid(
            row=9, column=0, sticky="w", pady=(9, 0)
        )
        self.device_combo = ttk.Combobox(
            model_group,
            textvariable=self.device_var,
            values=["auto", "cuda", "cpu"],
            state="readonly",
            width=10,
        )
        self.device_combo.grid(row=9, column=1, sticky="e", pady=(9, 0))

        source_group = ttk.LabelFrame(parent, text="資料來源", padding=12)
        source_group.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        source_group.columnconfigure(0, weight=1)
        source_group.columnconfigure(1, weight=1)
        ttk.Label(
            source_group,
            text="註冊可加入多筆；辨識時請選取其中一筆",
            style="Muted.Panel.TLabel",
            wraplength=340,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self.source_list = tk.Listbox(
            source_group,
            height=4,
            selectmode=tk.EXTENDED,
            exportselection=False,
            background="#ffffff",
            foreground=COLORS["text"],
            selectbackground=COLORS["accent"],
            selectforeground="#ffffff",
            relief="solid",
            borderwidth=1,
        )
        source_vscroll = ttk.Scrollbar(source_group, orient="vertical", command=self.source_list.yview)
        source_hscroll = ttk.Scrollbar(source_group, orient="horizontal", command=self.source_list.xview)
        self.source_list.configure(
            yscrollcommand=source_vscroll.set,
            xscrollcommand=source_hscroll.set,
        )
        self.source_list.grid(row=1, column=0, columnspan=2, sticky="nsew")
        source_vscroll.grid(row=1, column=2, sticky="ns")
        source_hscroll.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.folder_button = ttk.Button(
            source_group,
            text="加入序列／人物資料夾",
            command=self._browse_folder,
        )
        self.folder_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        self.remove_source_button = ttk.Button(
            source_group, text="移除選取", command=self._remove_selected_sources
        )
        self.remove_source_button.grid(row=4, column=0, sticky="ew", pady=(7, 0))
        self.clear_source_button = ttk.Button(source_group, text="清空", command=self._clear_sources)
        self.clear_source_button.grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))
        ttk.Label(
            source_group,
            textvariable=self.source_kind_var,
            style="Muted.Panel.TLabel",
            wraplength=340,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        threshold_group = ttk.LabelFrame(parent, text="Open-set 設定", padding=12)
        threshold_group.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        threshold_group.columnconfigure(0, weight=1)
        ttk.Label(threshold_group, text="Unknown threshold", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.threshold_spin = ttk.Spinbox(
            threshold_group,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.threshold_var,
            width=9,
            format="%.2f",
        )
        self.threshold_spin.grid(row=0, column=1, sticky="e")
        ttk.Label(threshold_group, text="第一／二名最小差距", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", pady=(9, 0)
        )
        self.margin_spin = ttk.Spinbox(
            threshold_group,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.margin_var,
            width=9,
            format="%.2f",
        )
        self.margin_spin.grid(row=1, column=1, sticky="e", pady=(9, 0))
        ttk.Label(
            threshold_group,
            text="threshold.provisional",
            style="Muted.Panel.TLabel",
            wraplength=330,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(9, 0))
        ttk.Label(
            threshold_group,
            text="判定依據：整部來源所有 windows 的平均身分分數；單一段過線不會直接通過。",
            style="Muted.Panel.TLabel",
            wraplength=330,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))

        actions = ttk.Notebook(parent)
        actions.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        enroll_tab = ttk.Frame(actions, style="Panel.TFrame", padding=12)
        recognize_tab = ttk.Frame(actions, style="Panel.TFrame", padding=12)
        actions.add(enroll_tab, text="人物註冊")
        actions.add(recognize_tab, text="身分辨識")
        self._build_enroll_tab(enroll_tab)
        self._build_recognize_tab(recognize_tab)

        database_group = ttk.LabelFrame(parent, text="Gallery 資料庫", padding=10)
        database_group.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        database_group.columnconfigure(0, weight=1)
        database_group.rowconfigure(0, weight=1)
        self.person_tree = ttk.Treeview(
            database_group,
            columns=("id", "name", "embeddings"),
            show="headings",
            height=6,
        )
        self.person_tree.heading("id", text="ID")
        self.person_tree.heading("name", text="姓名")
        self.person_tree.heading("embeddings", text="Clip 特徵")
        self.person_tree.column("id", width=70, anchor="w")
        self.person_tree.column("name", width=130, anchor="w")
        self.person_tree.column("embeddings", width=78, anchor="e")
        gallery_scroll = ttk.Scrollbar(database_group, orient="vertical", command=self.person_tree.yview)
        self.person_tree.configure(yscrollcommand=gallery_scroll.set)
        self.person_tree.grid(row=0, column=0, sticky="nsew")
        gallery_scroll.grid(row=0, column=1, sticky="ns")
        self.person_tree.bind(
            "<Double-1>",
            lambda _event: self._manage_selected_gallery(),
        )
        gallery_actions = ttk.Frame(database_group, style="Panel.TFrame")
        gallery_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        gallery_actions.columnconfigure(0, weight=1)
        self.gallery_manage_button = ttk.Button(
            gallery_actions,
            text="管理選取人物",
            command=self._manage_selected_gallery,
        )
        self.gallery_manage_button.grid(row=0, column=0, sticky="w")
        ttk.Button(
            gallery_actions,
            text="重新整理",
            command=self._refresh_database,
        ).grid(row=0, column=1, sticky="e")

    def _build_enroll_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=0)
        ttk.Label(parent, text="Person ID", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.autofill_identity_button = ttk.Button(
            parent,
            text="從來源自動填入",
            command=lambda: self._autofill_identity_from_sources(
                show_feedback=True,
                force=True,
            ),
        )
        self.autofill_identity_button.grid(row=0, column=1, sticky="e")
        ttk.Entry(parent, textvariable=self.person_id_var).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(4, 5)
        )
        ttk.Label(
            parent,
            textvariable=self.identity_suggestion_var,
            style="Muted.Panel.TLabel",
            wraplength=310,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(parent, text="人物顯示名稱", style="Panel.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w"
        )
        ttk.Entry(parent, textvariable=self.person_name_var).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(4, 8)
        )
        ttk.Label(
            parent,
            text="人物備註（服裝與拍攝條件由來源自動記錄）",
            style="Panel.TLabel",
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        self.note_text = tk.Text(
            parent,
            height=2,
            wrap="word",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
        )
        self.note_text.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        self.enroll_button = ttk.Button(
            parent,
            text="開始註冊",
            style="Accent.TButton",
            command=self._start_enroll,
        )
        self.enroll_button.grid(row=7, column=0, columnspan=2, sticky="ew")

    def _build_recognize_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        self.recognize_button = ttk.Button(
            parent,
            text="開始辨識",
            style="Accent.TButton",
            command=self._start_recognize,
        )
        self.recognize_button.grid(row=0, column=0, sticky="ew")

    def _build_content(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        result_bar = ttk.Frame(parent, style="Panel.TFrame", padding=(4, 0, 4, 10))
        result_bar.grid(row=0, column=0, sticky="ew")
        result_bar.columnconfigure(0, weight=1)
        ttk.Label(result_bar, textvariable=self.result_identity_var, style="Result.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.state_label = ttk.Label(result_bar, textvariable=self.result_state_var, style="Info.TLabel")
        self.state_label.grid(row=0, column=1, sticky="e")
        ttk.Label(result_bar, textvariable=self.result_detail_var, style="Muted.Panel.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(3, 0)
        )

        self.content_paned = ttk.Panedwindow(parent, orient=tk.VERTICAL)
        self.content_paned.grid(row=1, column=0, sticky="nsew")
        preview_host = ttk.Frame(self.content_paned, style="Panel.TFrame")
        details_host = ttk.Frame(self.content_paned, style="Panel.TFrame")
        preview_host.columnconfigure(0, weight=1)
        preview_host.rowconfigure(0, weight=1)
        details_host.columnconfigure(0, weight=1)
        details_host.rowconfigure(0, weight=1)

        playback_fps = float(self.controller.config.get("ui", {}).get("playback_fps", 10.0))
        self.preview = PreviewPlayer(preview_host, playback_fps=playback_fps)
        self.preview.grid(row=0, column=0, sticky="nsew")

        details = ttk.Notebook(details_host)
        details.grid(row=0, column=0, sticky="nsew", pady=(10, 0))
        candidates_tab = ttk.Frame(details, style="Panel.TFrame", padding=8)
        windows_tab = ttk.Frame(details, style="Panel.TFrame", padding=8)
        history_tab = ttk.Frame(details, style="Panel.TFrame", padding=8)
        log_tab = ttk.Frame(details, style="Panel.TFrame", padding=8)
        details.add(candidates_tab, text="候選人物")
        details.add(windows_tab, text="逐視窗結果")
        details.add(history_tab, text="工作階段歷史")
        details.add(log_tab, text="執行紀錄")
        self.candidate_tree = self._tree(
            candidates_tab,
            columns=("rank", "id", "name", "score", "gallery"),
            headings=("排名", "Person ID", "名稱", "Similarity", "來源數 / Clip 特徵數"),
            widths=(55, 100, 150, 100, 155),
        )
        self.window_tree = self._tree(
            windows_tab,
            columns=("window", "frames", "id", "score"),
            headings=("Window", "Frame range", "Candidate", "Similarity"),
            widths=(70, 150, 150, 100),
        )
        self.history_tree = self._tree(
            history_tab,
            columns=("time", "operation", "identity", "source", "bundle"),
            headings=("時間", "操作", "人物", "來源", "模型 Bundle"),
            widths=(80, 80, 130, 240, 210),
        )
        self.history_tree.bind(
            "<Double-1>",
            lambda _event: self._restore_selected_history(),
        )
        ttk.Button(
            history_tab,
            text="載入選取結果",
            command=self._restore_selected_history,
        ).grid(row=1, column=0, columnspan=2, sticky="e", pady=(8, 0))
        log_tab.columnconfigure(0, weight=1)
        log_tab.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_tab,
            wrap="word",
            state="disabled",
            background="#f8fafb",
            foreground=COLORS["text"],
            relief="solid",
            borderwidth=1,
        )
        log_scroll = ttk.Scrollbar(log_tab, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        self.content_paned.add(preview_host, weight=3)
        self.content_paned.add(details_host, weight=2)

    def _tree(
        self,
        parent: ttk.Frame,
        columns: tuple[str, ...],
        headings: tuple[str, ...],
        widths: tuple[int, ...],
    ) -> ttk.Treeview:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=7)
        for column, heading, width in zip(columns, headings, widths):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="w")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        return tree

    def _rebuild_bundle_labels(self) -> None:
        self.bundle_labels = {
            (
                f"{bundle.display_name} | fold {bundle.fold} | "
                f"{bundle.checkpoint_sha256[:8]} | {bundle_id}"
            ): bundle_id
            for bundle_id, bundle in self.bundles.items()
        }
        self.bundle_ids_to_labels = {
            bundle_id: label for label, bundle_id in self.bundle_labels.items()
        }

    def _method_display_name(self, bundle: Any) -> str:
        friendly = METHOD_DISPLAY_NAMES.get(bundle.method_key, bundle.architecture)
        return f"{friendly} [{bundle.method_key}]"

    def _set_model_selection(self, bundle_id: str) -> None:
        if bundle_id not in self.bundles:
            bundle_id = next(iter(self.bundles))
        bundle = self.bundles[bundle_id]
        self.family_var.set(self.family_ids_to_labels[bundle.input_type])

        method_bundles: dict[str, Any] = {}
        for candidate in self.bundles.values():
            if candidate.input_type == bundle.input_type:
                method_bundles.setdefault(candidate.method_key, candidate)
        self.method_labels = {
            self._method_display_name(candidate): method_key
            for method_key, candidate in sorted(method_bundles.items())
        }
        self.method_keys_to_labels = {
            method_key: label for label, method_key in self.method_labels.items()
        }
        self.method_var.set(self.method_keys_to_labels[bundle.method_key])

        compatible_labels = [
            self.bundle_ids_to_labels[candidate_id]
            for candidate_id, candidate in self.bundles.items()
            if candidate.input_type == bundle.input_type
            and candidate.method_key == bundle.method_key
        ]
        self.bundle_var.set(self.bundle_ids_to_labels[bundle_id])
        if hasattr(self, "family_combo"):
            self.family_combo.configure(values=list(self.family_labels))
            self.method_combo.configure(values=list(self.method_labels))
            self.bundle_combo.configure(values=compatible_labels)

    def _on_family_changed(self) -> None:
        family_id = self.family_labels.get(self.family_var.get())
        candidates = [
            bundle for bundle in self.bundles.values() if bundle.input_type == family_id
        ]
        if not candidates:
            messagebox.showwarning("沒有模型", "目前輸入類型沒有可用的模型 Bundle。")
            return
        selected = sorted(candidates, key=lambda item: (item.method_key, item.display_name))[0]
        self._set_model_selection(selected.bundle_id)
        self._on_bundle_changed()

    def _on_method_changed(self) -> None:
        family_id = self.family_labels.get(self.family_var.get())
        method_key = self.method_labels.get(self.method_var.get())
        candidates = [
            bundle
            for bundle in self.bundles.values()
            if bundle.input_type == family_id and bundle.method_key == method_key
        ]
        if not candidates:
            messagebox.showwarning("沒有權重", "目前模型方法沒有可用的權重 Bundle。")
            return
        selected = sorted(candidates, key=lambda item: item.display_name)[0]
        self._set_model_selection(selected.bundle_id)
        self._on_bundle_changed()

    def _rescan_bundles(self, select_bundle_id: str | None = None) -> None:
        current = select_bundle_id or self._bundle_id()
        try:
            self.bundles = self.controller.available_bundles(refresh=True)
            self._rebuild_bundle_labels()
            if current not in self.bundles:
                current = next(iter(self.bundles))
            self._set_model_selection(current)
            self._on_bundle_changed()
            self.status_var.set(f"已載入 {len(self.bundles)} 個模型 Bundle")
        except Exception as exc:
            messagebox.showerror("模型掃描失敗", str(exc))

    def _import_checkpoint(self) -> None:
        if self._source_operation_running():
            messagebox.showwarning("Please wait", "Enrollment source operation in progress.")
            return
        selected = filedialog.askopenfilename(
            title="選擇可信任的 PyTorch checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not selected:
            return
        if not messagebox.askyesno(
            "確認權重來源",
            "PyTorch checkpoint 只能匯入可信任來源。是否繼續？",
        ):
            return
        display_name = simpledialog.askstring(
            "模型名稱",
            "請輸入此權重在介面中顯示的名稱：",
            initialvalue=Path(selected).stem,
            parent=self.root,
        )
        if not display_name or not display_name.strip():
            return
        self.status_var.set("正在驗證並複製權重到系統資料夾")
        self.progress.start(12)
        self.root.update_idletasks()
        try:
            bundle = self.controller.import_checkpoint(
                checkpoint_path=selected,
                display_name=display_name.strip(),
                template_bundle_id=self._bundle_id(),
            )
            self._rescan_bundles(select_bundle_id=bundle.bundle_id)
            messagebox.showinfo(
                "匯入完成",
                f"模型已複製到：\n{bundle.bundle_dir}\n\n後續載入不再依賴原始檔案。",
            )
        except Exception as exc:
            self.status_var.set("權重匯入失敗")
            messagebox.showerror("權重匯入失敗", str(exc))
        finally:
            self.progress.stop()

    def _browse_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="加入單一序列，或選擇人物資料夾以自動加入其下所有序列"
        )
        if not selected:
            return
        try:
            discovered = self.controller.discover_sequence_sources(
                root=selected,
                bundle_id=self._bundle_id(),
            )
        except Exception as exc:
            messagebox.showerror("資料夾掃描失敗", str(exc))
            return
        if not discovered:
            bundle = self.bundles[self._bundle_id()]
            expected = "clear_data_*.npy"
            messagebox.showwarning(
                "找不到相容序列",
                f"所選資料夾及其子資料夾中找不到 {expected}。",
            )
            return
        try:
            suggestion = suggest_registration_identity(discovered)
        except ValueError as exc:
            messagebox.showwarning(
                "人物資料夾包含不同人物",
                f"{exc}\n\n請改選單一人物（例如 P001）所在的資料夾。",
            )
            return
        self._add_sources(discovered)
        if len(discovered) > 1:
            person = suggestion.person_id if suggestion is not None else "未命名人物"
            self.status_var.set(
                f"已從 {person} 資料夾加入 {len(discovered)} 個相容序列"
            )

    def _add_sources(self, paths: Iterable[str | Path]) -> None:
        existing = {str(path) for path in self.source_paths}
        added = 0
        for value in paths:
            path = Path(value).expanduser().resolve()
            if not path.exists():
                messagebox.showwarning("找不到來源", f"來源不存在：\n{path}")
                continue
            if str(path) in existing:
                continue
            self.source_paths.append(path)
            self.source_list.insert("end", str(path))
            existing.add(str(path))
            added += 1
        if added:
            last = len(self.source_paths) - 1
            self.source_list.selection_clear(0, "end")
            self.source_list.selection_set(last)
            self.source_list.see(last)
        self._update_source_summary()
        if added:
            self._autofill_identity_from_sources(show_feedback=False)

    def _remove_selected_sources(self) -> None:
        selected = list(self.source_list.curselection())
        if not selected:
            messagebox.showinfo("未選取來源", "請先在來源清單選取要移除的項目。")
            return
        for index in reversed(selected):
            del self.source_paths[index]
            self.source_list.delete(index)
        self._update_source_summary()
        self._autofill_identity_from_sources(show_feedback=False)

    def _clear_sources(self) -> None:
        self.source_paths.clear()
        self.source_list.delete(0, "end")
        self.identity_suggestion_var.set("尚未從來源偵測人物")
        self._update_source_summary()

    def _autofill_identity_from_sources(
        self,
        show_feedback: bool = True,
        force: bool = False,
    ) -> None:
        if not self.source_paths:
            self.identity_suggestion_var.set("尚未從來源偵測人物")
            if show_feedback:
                messagebox.showinfo("尚無來源", "請先加入一個或多個註冊來源。")
            return
        try:
            suggestion = suggest_registration_identity(self.source_paths)
        except ValueError as exc:
            self.identity_suggestion_var.set(str(exc))
            if show_feedback:
                messagebox.showwarning("人物偵測失敗", str(exc))
            return
        if suggestion is None:
            self.identity_suggestion_var.set(
                "來源名稱未包含 Person ID；請手動輸入人物資料"
            )
            if show_feedback:
                messagebox.showinfo(
                    "無法自動偵測",
                    "來源路徑未包含 P001 這類 Person ID，請手動輸入。",
                )
            return

        existing = self.controller.get_person(suggestion.person_id)
        canonical_name = (
            str(existing["display_name"])
            if existing is not None
            else suggestion.default_display_name
        )
        self.person_id_var.set(suggestion.person_id)
        current_name = self.person_name_var.get().strip()
        if existing is not None or force or not current_name or current_name.upper().startswith("P"):
            self.person_name_var.set(canonical_name)

        detail = suggestion.detail
        if existing is not None:
            detail += f" | 已登記名稱：{canonical_name}"
        else:
            detail += " | 新人物，可將顯示名稱改為姓名"
        self.identity_suggestion_var.set(detail)

    def _on_bundle_changed(self) -> None:
        bundle = self.bundles[self._bundle_id()]
        signature = (bundle.input_type, bundle.mode)
        previous_signature = self._active_source_signature
        if (
            previous_signature is not None
            and previous_signature != signature
            and self.source_paths
        ):
            removed = len(self.source_paths)
            self.source_paths.clear()
            self.source_list.delete(0, "end")
            self.identity_suggestion_var.set("尚未從來源偵測人物")
            message = (
                f"模型輸入由 {previous_signature[0]}/{previous_signature[1]} 切換為 "
                f"{signature[0]}/{signature[1]}，已清空 {removed} 個舊來源"
            )
            self.status_var.set(message)
            self._append_log(message)
        self._active_source_signature = signature
        self.threshold_var.set(self.controller.provisional_threshold(bundle.bundle_id))
        clip_len = int(bundle.data.get("clip_len", 15))
        drop_first = int(bundle.data.get("drop_first_frames", 0))
        representation_detail = (
            f"{int(bundle.data.get('num_points', 0))} points/frame | "
            f"normalization {bundle.model.get('point_normalization', 'none')}"
        )
        configuration_source = str(
            bundle.training.get("configuration_source", "bundle.yaml")
        )
        self.model_info_var.set(
            f"{bundle.architecture}\n"
            f"{representation_detail}\n"
            f"clip {clip_len} | drop first {drop_first} | fold {bundle.fold}\n"
            f"參數來源 {configuration_source} | SHA {bundle.checkpoint_sha256[:12]}"
        )
        self.model_path_var.set(f"權重：{bundle.checkpoint}")
        self._update_source_summary()
        self._refresh_database()

    def _update_source_summary(self) -> None:
        requirement = "點雲：可選 sequence 或人物資料夾（自動尋找 clear_data_*.npy）"
        count = len(self.source_paths)
        selection = len(self.source_list.curselection()) if hasattr(self, "source_list") else 0
        if count:
            self.source_kind_var.set(
                f"已加入 {count} 個來源，選取 {selection} 個。{requirement}"
            )
        else:
            self.source_kind_var.set(f"尚未加入來源。{requirement}")

    def _start_enroll(self) -> None:
        sources = self._validated_sources()
        if not sources:
            return
        person_id = self.person_id_var.get().strip()
        display_name = self.person_name_var.get().strip()
        if not person_id or not display_name:
            messagebox.showwarning("缺少人物資料", "Person ID 與顯示名稱不可留空。")
            return
        try:
            suggestion = suggest_registration_identity(sources)
        except ValueError as exc:
            messagebox.showwarning("來源人物不一致", str(exc))
            return
        if suggestion is not None and person_id != suggestion.person_id:
            messagebox.showwarning(
                "Person ID 不一致",
                f"來源偵測為 {suggestion.person_id}，目前輸入為 {person_id}。"
                "請確認資料來源，或使用「從來源自動填入」。",
            )
            return
        existing_person = self.controller.get_person(person_id)
        if (
            existing_person is not None
            and str(existing_person["display_name"]) != display_name
        ):
            messagebox.showwarning(
                "人物名稱不一致",
                f"{person_id} 已登記為「{existing_person['display_name']}」。"
                "服裝條件請勿加入人物名稱。",
            )
            return
        note = self.note_text.get("1.0", "end").strip()
        bundle_id = self._bundle_id()

        def task(progress: Callable[[str], None]) -> OperationOutcome:
            return self.controller.enroll_many(
                sources=sources,
                bundle_id=bundle_id,
                person_id=person_id,
                display_name=display_name,
                note=note,
                device=self.device_var.get(),
                progress=progress,
            )

        self._start_task("人物註冊", task)

    def _start_recognize(self) -> None:
        source = self._validated_recognition_source()
        if source is None:
            return
        try:
            threshold = float(self.threshold_var.get())
            margin = float(self.margin_var.get())
        except (TypeError, ValueError, tk.TclError):
            messagebox.showwarning("設定錯誤", "Threshold 與 margin 必須是數值。")
            return
        if not 0.0 <= threshold <= 1.0 or not 0.0 <= margin <= 1.0:
            messagebox.showwarning("設定錯誤", "Threshold 與 margin 必須介於 0 和 1。")
            return
        bundle_id = self._bundle_id()

        def task(progress: Callable[[str], None]) -> OperationOutcome:
            return self.controller.recognize(
                source=source,
                bundle_id=bundle_id,
                threshold=threshold,
                min_margin=margin,
                device=self.device_var.get(),
                progress=progress,
            )

        self._start_task("身分辨識", task)

    def _start_task(
        self,
        label: str,
        task: Callable[[Callable[[str], None]], OperationOutcome],
    ) -> None:
        if self.worker.busy or self._source_operation_running():
            messagebox.showinfo("系統忙碌", "請等待目前操作完成。")
            return
        self._set_busy(True)
        self.status_var.set(f"{label}執行中")
        self._append_log(f"[{label}] 開始")
        self.worker.start(task)

    def _poll_worker(self) -> None:
        for message in self.worker.drain():
            self._handle_worker_message(message)
        self.root.after(100, self._poll_worker)

    def _handle_worker_message(self, message: WorkerMessage) -> None:
        if message.kind == "progress":
            self.status_var.set(str(message.payload))
            self._append_log(str(message.payload))
        elif message.kind == "result":
            self._set_busy(False)
            self._show_outcome(message.payload)
        elif message.kind == "error":
            self._set_busy(False)
            payload = dict(message.payload)
            self.status_var.set("操作失敗")
            self._append_log(payload.get("traceback", payload.get("message", "Unknown error")))
            messagebox.showerror("操作失敗", payload.get("message", "Unknown error"))

    def _show_outcome(
        self,
        outcome: OperationOutcome,
        add_to_history: bool = True,
        append_log: bool = True,
    ) -> None:
        self.current_outcome = outcome
        result = outcome.result
        if result.get("report_warning") and append_log:
            self._append_log(str(result["report_warning"]))
            messagebox.showwarning("Report warning", str(result["report_warning"]))
        if add_to_history:
            self._add_outcome_history(outcome)
        self.preview.load(outcome)
        self._clear_tree(self.candidate_tree)
        self._clear_tree(self.window_tree)

        if outcome.operation == "enroll":
            person = str(result.get("display_name", result.get("person_id", "-")))
            coherence = float(result.get("embedding_coherence", 0.0))
            source_count = int(result.get("source_count", 1))
            self.result_identity_var.set(f"已註冊：{person}")
            self.result_state_var.set("ENROLLED")
            self.state_label.configure(style="Stable.TLabel")
            self.result_detail_var.set(
                f"{source_count} 個來源 | 保存 {result.get('stored_embeddings', 0)} embeddings | "
                f"coherence {coherence:.3f} | {outcome.elapsed_seconds:.2f}s"
            )
            self.status_var.set("人物註冊完成")
            self._refresh_database()
        else:
            state = str(result.get("state", "unknown"))
            identity = str(result.get("display_name", "Unknown"))
            score = float(result.get("similarity", 0.0))
            margin = result.get("similarity_margin")
            margin_text = "-" if margin is None else f"{float(margin):.3f}"
            self.result_identity_var.set(identity)
            self.result_state_var.set(state.upper())
            self.result_detail_var.set(
                f"similarity {score:.3f} | margin {margin_text} | "
                f"threshold {float(result.get('threshold', 0.0)):.2f} | {outcome.elapsed_seconds:.2f}s"
            )
            self._set_state_style(state)
            self.status_var.set("身分辨識完成")
            for rank, item in enumerate(result.get("top_candidates", []), start=1):
                self.candidate_tree.insert(
                    "",
                    "end",
                    values=(
                        rank,
                        item.get("person_id", "-"),
                        item.get("display_name", "-"),
                        f"{float(item.get('similarity', 0.0)):.4f}",
                        (
                            f"{item.get('gallery_sources', '-')}/"
                            f"{item.get('gallery_embeddings', '-')}"
                        ),
                    ),
                )
            for item in result.get("window_results", []):
                start = item.get("start_frame_number", item.get("start_index", "-"))
                end = item.get("end_frame_number", item.get("end_index", "-"))
                self.window_tree.insert(
                    "",
                    "end",
                    values=(
                        item.get("window_index", "-"),
                        f"{start} - {end}",
                        item.get("display_name", item.get("person_id", "-")),
                        f"{float(item.get('similarity', 0.0)):.4f}",
                    ),
                )
        if append_log:
            self._append_log(json.dumps(result, ensure_ascii=False, indent=2))

    def _add_outcome_history(self, outcome: OperationOutcome) -> None:
        result = outcome.result
        operation_label = "註冊" if outcome.operation == "enroll" else "辨識"
        identity = str(
            result.get(
                "display_name",
                result.get("person_id", "Unknown"),
            )
        )
        source_value = result.get(
            "source",
            outcome.prepared_source.original_path,
        )
        source_name = Path(str(source_value)).name
        bundle_name = str(
            result.get(
                "bundle_display_name",
                result.get("bundle_id", result.get("model_key", "-")),
            )
        )
        item_id = self.history_tree.insert(
            "",
            0,
            values=(
                datetime.now().strftime("%H:%M:%S"),
                operation_label,
                identity,
                source_name,
                bundle_name,
            ),
        )
        self.outcome_history[item_id] = outcome
        children = self.history_tree.get_children()
        for expired_id in children[self.history_limit :]:
            self.history_tree.delete(expired_id)
            self.outcome_history.pop(str(expired_id), None)

    def _restore_selected_history(self) -> None:
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showinfo(
                "未選取歷史結果",
                "請先選取一筆歷史紀錄。",
            )
            return
        outcome = self.outcome_history.get(str(selected[0]))
        if outcome is None:
            messagebox.showwarning(
                "歷史結果不存在",
                "此結果已不在目前工作階段的記憶體中。",
            )
            return
        self._show_outcome(
            outcome,
            add_to_history=False,
            append_log=False,
        )
        self.status_var.set("已載入工作階段歷史結果；目前模型選擇未變更")

    def _set_state_style(self, state: str) -> None:
        style = {
            "stable": "Stable.TLabel",
            "unknown": "Unknown.TLabel",
            "low_confidence": "Warning.TLabel",
        }.get(state, "Info.TLabel")
        self.state_label.configure(style=style)

    def _set_busy(self, busy: bool) -> None:
        state = ["disabled"] if busy else ["!disabled"]
        self.enroll_button.state(state)
        self.recognize_button.state(state)
        self.autofill_identity_button.state(state)
        selector_state = "disabled" if busy else "readonly"
        self.family_combo.configure(state=selector_state)
        self.method_combo.configure(state=selector_state)
        self.bundle_combo.configure(state=selector_state)
        self.device_combo.configure(state=selector_state)
        self.import_button.state(state)
        self.rescan_button.state(state)
        self.folder_button.state(state)
        self.remove_source_button.state(state)
        self.clear_source_button.state(state)
        self.gallery_manage_button.state(state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _refresh_database(self) -> None:
        bundle_id = self._bundle_id()
        summary = self.controller.database_summary(bundle_id=bundle_id)
        overall = self.controller.database_summary()
        self.database_var.set(
            f"目前模型 Gallery: {summary['persons']} 人 / "
            f"{summary['active_embeddings']} 個 clip 特徵 | "
            f"全部模型: {overall['persons']} 人 / "
            f"{overall['active_embeddings']} 個 clip 特徵"
        )
        self._clear_tree(self.person_tree)
        for item in self.controller.list_persons(bundle_id=bundle_id):
            self.person_tree.insert(
                "",
                "end",
                values=(item["person_id"], item["display_name"], item["embedding_count"]),
            )

    def _manage_selected_gallery(self) -> None:
        allowed, reason = self._gallery_transfer_available()
        if not allowed:
            messagebox.showwarning("Gallery", reason, parent=self.root)
            return
        selected = self.person_tree.selection()
        if not selected:
            messagebox.showinfo(
                "未選取人物",
                "請先在 Gallery 表格選取一位人物。",
            )
            return
        values = self.person_tree.item(selected[0], "values")
        if not values:
            return
        person_id = str(values[0])
        display_name = str(values[1])
        bundle_id = self._bundle_id()
        bundle = self.bundles[bundle_id]

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Gallery 管理 - {display_name} ({person_id})")
        dialog.geometry("1040x430")
        dialog.minsize(720, 320)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)

        ttk.Label(
            dialog,
            text=(
                f"{person_id} / {display_name}\n"
                f"目前權重：{bundle.display_name}\n"
                "停用只影響目前權重，不會刪除其他影像或點雲模型的 Gallery。\n"
                "內容指紋用於阻止同一序列改名或複製後重複註冊。"
            ),
            padding=(12, 10),
        ).grid(row=0, column=0, sticky="ew")

        table_host = ttk.Frame(dialog, padding=(12, 0, 12, 0))
        table_host.grid(row=1, column=0, sticky="nsew")
        table_host.columnconfigure(0, weight=1)
        table_host.rowconfigure(0, weight=1)
        source_tree = ttk.Treeview(
            table_host,
            columns=("source", "fingerprint", "embeddings", "quality", "model"),
            show="headings",
        )
        source_tree.heading("source", text="註冊來源")
        source_tree.heading("fingerprint", text="內容指紋")
        source_tree.heading("embeddings", text="Clip 特徵數")
        source_tree.heading("quality", text="Coherence")
        source_tree.heading("model", text="Model key")
        source_tree.column("source", width=500, anchor="w")
        source_tree.column("fingerprint", width=120, anchor="w")
        source_tree.column("embeddings", width=90, anchor="e")
        source_tree.column("quality", width=90, anchor="e")
        source_tree.column("model", width=120, anchor="w")
        source_vscroll = ttk.Scrollbar(
            table_host,
            orient="vertical",
            command=source_tree.yview,
        )
        source_hscroll = ttk.Scrollbar(
            table_host,
            orient="horizontal",
            command=source_tree.xview,
        )
        source_tree.configure(
            yscrollcommand=source_vscroll.set,
            xscrollcommand=source_hscroll.set,
        )
        source_tree.grid(row=0, column=0, sticky="nsew")
        source_vscroll.grid(row=0, column=1, sticky="ns")
        source_hscroll.grid(row=1, column=0, sticky="ew")

        status = tk.StringVar()
        source_rows: dict[str, dict[str, Any]] = {}

        def reload_sources() -> None:
            self._clear_tree(source_tree)
            source_rows.clear()
            rows = self.controller.list_gallery_sources(
                bundle_id=bundle_id,
                person_id=person_id,
            )
            for row in rows:
                quality = row.get("mean_quality")
                quality_label = "-" if quality is None else f"{float(quality):.3f}"
                fingerprint = str(row.get("source_fingerprint") or "-")
                if fingerprint.startswith("sha256:"):
                    fingerprint = fingerprint[len("sha256:") :]
                item_id = source_tree.insert(
                    "",
                    "end",
                    values=(
                        row["source_path"],
                        fingerprint[:12],
                        row["embedding_count"],
                        quality_label,
                        str(row["model_key"])[:12],
                    ),
                )
                source_rows[item_id] = row
            status.set(
                f"{len(rows)} 個註冊來源 / "
                f"{sum(int(row['embedding_count']) for row in rows)} 個 clip 特徵"
            )

        def deactivate_source() -> None:
            source_selection = source_tree.selection()
            if not source_selection:
                messagebox.showinfo(
                    "未選取來源",
                    "請先選取一個註冊來源。",
                    parent=dialog,
                )
                return
            row = source_rows[str(source_selection[0])]
            if not messagebox.askyesno(
                "停用註冊來源",
                "此來源的 clip 特徵將不再參與辨識，但資料列會保留供稽核，"
                "之後也可重新註冊。是否繼續？\n\n"
                f"{row['source_path']}",
                parent=dialog,
            ):
                return
            count = self.controller.deactivate_gallery_source(
                bundle_id=bundle_id,
                person_id=person_id,
                model_key=str(row["model_key"]),
                source_path=str(row["source_path"]),
            )
            reload_sources()
            self._refresh_database()
            self.status_var.set(f"已停用 {count} 筆 Gallery clip 特徵")

        def deactivate_person() -> None:
            if not messagebox.askyesno(
                "停用目前權重下的人物特徵",
                f"將停用 {display_name} 在「{bundle.display_name}」下的全部特徵。"
                "其他模型的 Gallery 不受影響。是否繼續？",
                parent=dialog,
            ):
                return
            count = self.controller.deactivate_person_gallery(
                bundle_id=bundle_id,
                person_id=person_id,
            )
            reload_sources()
            self._refresh_database()
            self.status_var.set(f"已停用 {count} 筆 Gallery embeddings")

        actions = ttk.Frame(dialog, padding=12)
        actions.grid(row=2, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        ttk.Label(actions, textvariable=status).grid(row=0, column=0, sticky="w")
        ttk.Button(
            actions,
            text="停用選取來源",
            command=deactivate_source,
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            actions,
            text=self.i18n.tr("gallery.clear_model"),
            command=deactivate_person,
        ).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(
            actions,
            text="關閉",
            command=dialog.destroy,
        ).grid(row=0, column=3, padx=(8, 0))
        reload_sources()

    def _validated_sources(self) -> list[Path]:
        if not self.source_paths:
            messagebox.showwarning("缺少輸入", "請先加入點雲資料夾。")
            return []
        return self._validate_source_paths(self.source_paths)

    def _validate_source_paths(self, paths: Iterable[Path]) -> list[Path]:
        candidates = list(paths)
        missing = [path for path in candidates if not path.exists()]
        if missing:
            messagebox.showwarning(
                "找不到來源",
                "下列來源已不存在：\n" + "\n".join(str(path) for path in missing),
            )
            return []

        bundle = self.bundles[self._bundle_id()]
        incompatible = []
        for path in candidates:
            compatible = path.is_dir() and any(path.glob("clear_data_*.npy"))
            if not compatible:
                incompatible.append(path)

        if incompatible:
            expected = "包含 clear_data_*.npy 的資料夾"
            messagebox.showwarning(
                "來源與模型不相容",
                f"目前模型：{bundle.display_name}\n需要：{expected}\n\n"
                "不相容來源：\n" + "\n".join(str(path) for path in incompatible),
            )
            return []
        return candidates

    def _validated_recognition_source(self) -> Path | None:
        if not self.source_paths:
            messagebox.showwarning("缺少輸入", "請先加入點雲資料夾。")
            return None
        if len(self.source_paths) == 1:
            candidate = self.source_paths[0]
        else:
            selected = list(self.source_list.curselection())
            if len(selected) != 1:
                messagebox.showwarning(
                    "辨識來源不明確",
                    "來源清單有多筆時，請只選取一筆作為本次辨識來源。",
                )
                return None
            candidate = self.source_paths[selected[0]]
        validated = self._validate_source_paths([candidate])
        return validated[0] if validated else None

    def _bundle_id(self) -> str:
        label = self.bundle_var.get()
        if label not in self.bundle_labels:
            raise KeyError(f"Unknown model bundle selection: {label}")
        return self.bundle_labels[label]

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", str(message).rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        children = tree.get_children()
        if children:
            tree.delete(*children)

    def _close(self) -> None:
        if (getattr(getattr(self, "realtime_page", None), "commit_pending", False)
                or self._source_operation_running()):
            messagebox.showwarning("Please wait", self.i18n.tr("sources.committing"), parent=self.root)
            return
        if hasattr(self, "sources_page"):
            self.sources_page.pause()
        dialog = getattr(getattr(self, "gallery_page", None), "transfer_dialog", None)
        if dialog is not None and dialog.winfo_exists() and dialog.worker.busy:
            dialog.close()
            return
        if hasattr(self, "realtime_page"):
            self.realtime_page.close()
        if hasattr(self, "performance_page"):
            self.performance_page.close()
        self.preview.stop()
        self.root.destroy()
