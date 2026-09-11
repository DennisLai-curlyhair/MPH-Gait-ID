from __future__ import annotations
from copy import deepcopy
import math
import time

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..config import nested
from ..controller import GaitApplicationController
from ..identity import suggest_registration_identity
from ..i18n import I18n
from ..ui.preview import render_image
from ..ui.scrollable import ScrollableFrame
from ..ui.layout import SplitPane, WrappedLabel
from ..ui.workflow import ControlLock, form_controls, snapshot_feedback
from ..ui.workers import BackgroundWorker
from .detection import (
    DEFAULT_SAM_CHECKPOINT,
    DEFAULT_SAM_MODEL_TYPE,
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_SEG_MODEL,
    resolve_sam_checkpoint_path,
    resolve_yolo_weights_reference,
)
from .devices import probe_azure_kinect
from .pipeline import RealtimeConfig, RealtimePipeline
from .types import PipelineSnapshot


class RealtimePage(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        controller: GaitApplicationController,
        open_offline: Callable[[], None] | None = None,
        open_gallery: Callable[[], None] | None = None,
        on_gallery_changed: Callable[[], None] | None = None,
        camera_available: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(master)
        self.controller = controller
        shared_i18n = getattr(self.winfo_toplevel(), "_gait_i18n", None)
        self.i18n: I18n = shared_i18n if isinstance(shared_i18n, I18n) else I18n("en")
        self._build_option_labels()
        self.open_offline = open_offline
        self.open_gallery = open_gallery
        self.on_gallery_changed = on_gallery_changed
        self.camera_available = camera_available
        self.pipeline: RealtimePipeline | None = None
        self.commit_worker = BackgroundWorker()
        self.commit_pending = False
        self.device_worker = BackgroundWorker()
        self._device_check_pending = False
        self.stop_worker = BackgroundWorker()
        self._stop_pending = False
        self._session_active = False
        self._capture_ready = False
        self._pass_action = None
        self._pass_requested_at = 0.0
        self._pass_result = {}
        self._readiness_after = None
        self._gallery_counts = {}
        self.save_foreground_var = tk.BooleanVar(value=False)
        self.device_status = None
        self._rgb_photo: ImageTk.PhotoImage | None = None
        self._cloud_photo: ImageTk.PhotoImage | None = None
        self._last_snapshot: PipelineSnapshot | None = None
        self._pending_review_result: dict[str, Any] | None = None
        self._review_window: tk.Toplevel | None = None
        self._review_selection_vars: dict[str, tk.BooleanVar] = {}
        self._review_choices: dict[str, bool] = {}

        self.operation_var = tk.StringVar(
            value=self._option_label(self.operation_labels, "recognize")
        )
        self.source_var = tk.StringVar(
            value=self._option_label(self.source_labels, "replay")
        )
        self.replay_path_var = tk.StringVar(
            value=str(
                controller.model_store.root.parent
                / "data"
                / "replay"
                / "P3_C4_V001"
            )
        )
        self.detector_var = tk.StringVar(
            value=self._option_label(self.detector_labels, "replay_depth")
        )
        self.default_yolo_weights = str(
            nested(
                controller.config,
                "realtime",
                "default_yolo_model",
                DEFAULT_YOLO_MODEL,
            )
        )
        self.default_yolo_seg_weights = str(
            nested(
                controller.config,
                "realtime",
                "default_yolo_seg_model",
                DEFAULT_YOLO_SEG_MODEL,
            )
        )
        self.yolo_weights_var = tk.StringVar(value=self.default_yolo_weights)
        self.sam_checkpoint_var = tk.StringVar(
            value=str(
                nested(
                    controller.config,
                    "realtime",
                    "default_sam_checkpoint",
                    DEFAULT_SAM_CHECKPOINT,
                )
            )
        )
        self.sam_refresh_interval_var = tk.IntVar(
            value=int(
                nested(
                    controller.config,
                    "realtime",
                    "sam_refresh_interval",
                    10,
                )
            )
        )
        self.bundle_var = tk.StringVar()
        self.device_var = tk.StringVar(value="auto")
        self.clip_len_var = tk.IntVar(value=15)
        self.stride_var = tk.IntVar(value=5)
        self.threshold_var = tk.DoubleVar(value=0.55)
        self.margin_var = tk.DoubleVar(value=0.03)
        self.replay_fps_var = tk.DoubleVar(value=10.0)
        self.detector_confidence_var = tk.DoubleVar(value=0.35)
        self.loop_var = tk.BooleanVar(value=True)
        self.allow_multi_enrollment_var = tk.BooleanVar(
            value=bool(
                nested(
                    controller.config,
                    "realtime",
                    "allow_multiple_people_enrollment",
                    True,
                )
            )
        )
        self.show_rgb_pointcloud_var = tk.BooleanVar(
            value=bool(
                nested(
                    controller.config,
                    "ui",
                    "show_rgb_pointcloud_overlay",
                    True,
                )
            )
        )
        self.cloud_zoom_var = tk.DoubleVar(
            value=float(
                nested(
                    controller.config,
                    "ui",
                    "realtime_cloud_zoom",
                    1.0,
                )
            )
        )
        self.cloud_zoom_text_var = tk.StringVar()
        self.enrollment_person_id_var = tk.StringVar()
        self.enrollment_name_var = tk.StringVar()
        self.enrollment_note_var = tk.StringVar()
        self.pass_direction_var = tk.StringVar(
            value=self._option_label(self.pass_direction_labels, "front_facing")
        )
        self.pass_status_var = tk.StringVar(value="Start the session, then record one-way passes")
        self.enrollment_duration_var = tk.DoubleVar(
            value=float(
                nested(
                    controller.config,
                    "registration",
                    "realtime_duration_seconds",
                    20,
                )
            )
        )
        self.enrollment_warmup_var = tk.DoubleVar(
            value=float(
                nested(
                    controller.config,
                    "registration",
                    "realtime_warmup_seconds",
                    3,
                )
            )
        )
        self.enrollment_min_var = tk.IntVar(
            value=int(nested(controller.config, "registration", "min_clips", 5))
        )
        self.enrollment_max_var = tk.IntVar(
            value=int(
                nested(controller.config, "registration", "max_embeddings", 10)
            )
        )
        self.status_var = tk.StringVar(value="Ready")
        self.device_var_text = tk.StringVar(value="Device has not been checked")
        self.result_var = tk.StringVar(value=self.i18n.tr("No realtime result"))
        self.result_detail_var = tk.StringVar(value=self.i18n.tr("Select a source and start the pipeline"))
        self.fps_var = tk.StringVar(value="0.0")
        self.capture_fps_var = tk.StringVar(value="0.0")
        self.sampling_fps_var = tk.StringVar(value="0.0")
        self.window_span_var = tk.StringVar(value="0.00 s")
        self.frame_gap_var = tk.StringVar(value="0 / 0 ms")
        self.buffer_var = tk.StringVar(value="0 / 15")
        self.points_var = tk.StringVar(value="0")
        self.detection_latency_var = tk.StringVar(value="0.0 ms")
        self.latency_var = tk.StringVar(value="0.0 ms")
        self.enrollment_progress_var = tk.StringVar(value="-")
        self.gallery_var = tk.StringVar(value="Gallery: 0 identities")
        self.i18n.register_ui_variables(self.status_var, self.device_var_text, self.pass_status_var)
        self.bundle_labels: dict[str, str] = {}
        self._update_cloud_zoom_text()

        self._build()
        self.refresh_bundles()
        self._source_changed()
        self._operation_changed()
        for variable in (self.operation_var, self.source_var, self.detector_var,
                         self.replay_path_var, self.yolo_weights_var, self.sam_checkpoint_var,
                         self.bundle_var, self.clip_len_var, self.stride_var,
                         self.enrollment_person_id_var, self.enrollment_name_var,
                         self.enrollment_duration_var, self.enrollment_warmup_var,
                         self.enrollment_min_var, self.enrollment_max_var,
                         self.threshold_var, self.margin_var):
            variable.trace_add("write", self._schedule_readiness)
        self._refresh_readiness()
        self.after(50, self._poll)

    @staticmethod
    def _option_label(options: dict[str, str], option_id: str) -> str:
        return next(label for label, value in options.items() if value == option_id)

    def _detector_values_for_source(self, source_mode: str) -> list[str]:
        if source_mode != "azure_kinect":
            return list(self.detector_labels)
        return [
            self._option_label(self.detector_labels, detector_id)
            for detector_id in ("yolo_seg", "yolo", "yolo_sam")
        ]

    def _build_option_labels(self) -> None:
        self.source_labels = {
            self.i18n.tr("source.azure"): "azure_kinect",
            self.i18n.tr("source.replay"): "replay",
        }
        self.detector_labels = {
            self.i18n.tr("detector.yolo"): "yolo",
            self.i18n.tr("detector.yolo_seg"): "yolo_seg",
            self.i18n.tr("detector.yolo_sam"): "yolo_sam",
            self.i18n.tr("detector.replay"): "replay_depth",
        }
        self.operation_labels = {
            self.i18n.tr("realtime.recognition"): "recognize",
            self.i18n.tr("realtime.enrollment"): "enroll",
        }
        self.pass_direction_labels = {
            self.i18n.tr("direction.front_facing"): "front_facing",
        }
        self.pass_direction_names = {
            value: label for label, value in self.pass_direction_labels.items()
        }

    def set_locale(self, _locale_name: str) -> None:
        operation = self.operation_labels.get(self.operation_var.get(), "recognize")
        source = self.source_labels.get(self.source_var.get(), "replay")
        detector = self.detector_labels.get(self.detector_var.get(), "replay_depth")
        direction = self.pass_direction_labels.get(
            self.pass_direction_var.get(),
            "front_facing",
        )
        self._build_option_labels()
        self.operation_var.set(self._option_label(self.operation_labels, operation))
        self.source_var.set(self._option_label(self.source_labels, source))
        self.detector_var.set(self._option_label(self.detector_labels, detector))
        self.pass_direction_var.set(
            self._option_label(self.pass_direction_labels, direction)
        )
        self.source_combo.configure(values=list(self.source_labels))
        self.detector_combo.configure(values=self._detector_values_for_source(source))
        self.pass_direction_combo.configure(values=list(self.pass_direction_labels))
        self.recognize_radio.configure(
            text=self.i18n.tr("recognition"),
            value=self._option_label(self.operation_labels, "recognize"),
        )
        self.enroll_radio.configure(
            text=self.i18n.tr("enrollment"),
            value=self._option_label(self.operation_labels, "enroll"),
        )
        self.allow_multi_enrollment_check.configure(
            text=self.i18n.tr("realtime.allow_multi_enrollment")
        )
        self.save_foreground_check.configure(text=self.i18n.tr("sources.opt_in"))
        self.rgb_pointcloud_check.configure(
            text=self.i18n.tr("realtime.rgb_pointcloud")
        )
        self.start_button.configure(text=self.i18n.tr(
            "Start guided enrollment session" if operation == "enroll" else "Start realtime recognition"))
        if getattr(self, "_last_snapshot", None) is None:
            for name in ("result_var", "result_detail_var"):
                variable = getattr(self, name, None)
                if variable is not None:
                    variable.set(self.i18n.tr(variable.get()))
        if "readiness_vars" in self.__dict__:
            self._refresh_readiness()
            self._update_pass_controls()
            self._update_review_controls()
            if self._stop_pending:
                self.result_var.set(self.i18n.tr("workflow.stopping"))
            elif self._last_snapshot is not None and self._session_active:
                self._render_feedback(self._last_snapshot)

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        panes.grid(row=0, column=0, sticky="nsew")
        sidebar = ScrollableFrame(
            panes,
            width=390,
            canvas_background="#ffffff",
            frame_style="Panel.TFrame",
        )
        content = ttk.Frame(panes, style="Panel.TFrame", padding=12)
        panes.add(sidebar, weight=0)
        panes.add(content, weight=1)
        self._build_sidebar(sidebar.content)
        self._build_content(content)
        self._config_lock = ControlLock(form_controls(sidebar.content, exclude=(
            self.start_button, self.stop_button, self.start_pass_button,
            self.end_pass_button, self.discard_pass_button, self.finish_review_button,
            self.pass_direction_combo)))

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        operation = ttk.LabelFrame(parent, text="Realtime operation", padding=12)
        operation.grid(row=0, column=0, sticky="ew")
        operation.columnconfigure(0, weight=1)
        operation.columnconfigure(1, weight=1)
        self.recognize_radio = ttk.Radiobutton(
            operation,
            text="Recognition",
            value=self._option_label(self.operation_labels, "recognize"),
            variable=self.operation_var,
            command=self._operation_changed,
        )
        self.recognize_radio.grid(row=0, column=0, sticky="w")
        self.enroll_radio = ttk.Radiobutton(
            operation,
            text="Enrollment",
            value=self._option_label(self.operation_labels, "enroll"),
            variable=self.operation_var,
            command=self._operation_changed,
        )
        self.enroll_radio.grid(row=0, column=1, sticky="w")

        self.enrollment_frame = ttk.Frame(operation, style="Panel.TFrame")
        self.enrollment_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0)
        )
        self.enrollment_frame.columnconfigure(1, weight=1)
        enrollment_fields = [
            ("Person ID", self.enrollment_person_id_var),
            ("Display name", self.enrollment_name_var),
            ("Note", self.enrollment_note_var),
        ]
        self.enrollment_entries: list[ttk.Entry] = []
        for row, (label, variable) in enumerate(enrollment_fields):
            ttk.Label(
                self.enrollment_frame,
                text=label,
                style="Panel.TLabel",
            ).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 6, 0))
            entry = ttk.Entry(self.enrollment_frame, textvariable=variable)
            entry.grid(
                row=row,
                column=1,
                sticky="ew",
                padx=(8, 0),
                pady=(0 if row == 0 else 6, 0),
            )
            self.enrollment_entries.append(entry)
        ttk.Label(
            self.enrollment_frame,
            text="Max session / pass warm-up (s)",
            style="Panel.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(6, 0))
        timing = ttk.Frame(self.enrollment_frame, style="Panel.TFrame")
        timing.grid(row=3, column=1, sticky="e", padx=(8, 0), pady=(6, 0))
        self.enrollment_duration_spin = ttk.Spinbox(
            timing,
            from_=5,
            to=180,
            increment=1,
            textvariable=self.enrollment_duration_var,
            width=6,
        )
        self.enrollment_duration_spin.grid(row=0, column=0)
        ttk.Label(timing, text="/", style="Muted.Panel.TLabel").grid(
            row=0, column=1, padx=4
        )
        self.enrollment_warmup_spin = ttk.Spinbox(
            timing,
            from_=0,
            to=10,
            increment=1,
            textvariable=self.enrollment_warmup_var,
            width=6,
        )
        self.enrollment_warmup_spin.grid(row=0, column=2)
        ttk.Label(
            self.enrollment_frame,
            text="Min. / stored embeddings",
            style="Panel.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=(6, 0))
        counts = ttk.Frame(self.enrollment_frame, style="Panel.TFrame")
        counts.grid(row=4, column=1, sticky="e", padx=(8, 0), pady=(6, 0))
        self.enrollment_min_spin = ttk.Spinbox(
            counts,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.enrollment_min_var,
            width=6,
        )
        self.enrollment_min_spin.grid(row=0, column=0)
        ttk.Label(counts, text="/", style="Muted.Panel.TLabel").grid(
            row=0, column=1, padx=4
        )
        self.enrollment_max_spin = ttk.Spinbox(
            counts,
            from_=1,
            to=50,
            increment=1,
            textvariable=self.enrollment_max_var,
            width=6,
        )
        self.enrollment_max_spin.grid(row=0, column=2)
        self.autofill_button = ttk.Button(
            self.enrollment_frame,
            text="Autofill from replay name",
            command=self._autofill_replay_identity,
        )
        self.autofill_button.grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        self.allow_multi_enrollment_check = ttk.Checkbutton(
            self.enrollment_frame,
            text=self.i18n.tr("realtime.allow_multi_enrollment"),
            variable=self.allow_multi_enrollment_var,
        )
        self.allow_multi_enrollment_check.grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(8, 0),
        )

        self.save_foreground_check = ttk.Checkbutton(
            self.enrollment_frame, text=self.i18n.tr("sources.opt_in"),
            variable=self.save_foreground_var)
        self.save_foreground_check.grid(row=8, column=0, columnspan=2, sticky="w", pady=8)

        pass_controls = ttk.LabelFrame(
            self.enrollment_frame,
            text="Guided one-way passes",
            padding=8,
        )
        pass_controls.grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )
        pass_controls.columnconfigure(0, weight=1)
        self.pass_direction_combo = ttk.Combobox(
            pass_controls,
            textvariable=self.pass_direction_var,
            values=list(self.pass_direction_labels),
            state="readonly",
        )
        self.pass_direction_combo.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
        )
        self.start_pass_button = ttk.Button(
            pass_controls,
            text="Start pass",
            command=self._start_enrollment_pass,
            state="disabled",
        )
        self.start_pass_button.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.end_pass_button = ttk.Button(
            pass_controls,
            text="End pass",
            command=lambda: self._end_enrollment_pass(False),
            state="disabled",
        )
        self.end_pass_button.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=(6, 0),
        )
        self.discard_pass_button = ttk.Button(
            pass_controls,
            text="Discard pass",
            command=lambda: self._end_enrollment_pass(True),
            state="disabled",
        )
        self.discard_pass_button.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.finish_review_button = ttk.Button(
            pass_controls,
            text="Finish & review",
            command=self._finish_enrollment_for_review,
            state="disabled",
        )
        self.finish_review_button.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(6, 0),
            pady=(6, 0),
        )
        ttk.Label(
            pass_controls,
            textvariable=self.pass_status_var,
            style="Muted.Panel.TLabel",
            wraplength=300,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))
        self.pass_tree = ttk.Treeview(
            pass_controls,
            columns=("pass", "direction", "frames", "emb", "state"),
            show="headings",
            height=4,
        )
        for column, heading, width in (
            ("pass", "Pass", 65),
            ("direction", "Direction", 105),
            ("frames", "Frames", 58),
            ("emb", "Emb.", 48),
            ("state", "State", 70),
        ):
            self.pass_tree.heading(column, text=heading)
            self.pass_tree.column(column, width=width, anchor="w")
        self.pass_tree.grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(7, 0),
        )

        source = ttk.LabelFrame(parent, text="Realtime source", padding=12)
        source.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        source.columnconfigure(0, weight=1)
        ttk.Label(source, text="Input source", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.source_combo = ttk.Combobox(
            source,
            textvariable=self.source_var,
            values=list(self.source_labels),
            state="readonly",
        )
        self.source_combo.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.source_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._source_changed(),
        )

        replay_row = ttk.Frame(source, style="Panel.TFrame")
        replay_row.grid(row=2, column=0, sticky="ew")
        replay_row.columnconfigure(0, weight=1)
        self.replay_entry = ttk.Entry(replay_row, textvariable=self.replay_path_var)
        self.replay_entry.grid(row=0, column=0, sticky="ew")
        self.replay_button = ttk.Button(
            replay_row,
            text="Browse",
            command=self._browse_replay,
            width=8,
        )
        self.replay_button.grid(row=0, column=1, padx=(7, 0))

        device_actions = ttk.Frame(source, style="Panel.TFrame")
        device_actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        device_actions.columnconfigure(1, weight=1)
        self.check_device_button = ttk.Button(
            device_actions,
            text="Check Azure Kinect",
            command=self._check_device,
        )
        self.check_device_button.grid(row=0, column=0, sticky="w")
        ttk.Label(
            device_actions,
            textvariable=self.device_var_text,
            style="Muted.Panel.TLabel",
            wraplength=185,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(source, text="Person detector", style="Panel.TLabel").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        self.detector_combo = ttk.Combobox(
            source,
            textvariable=self.detector_var,
            state="readonly",
        )
        self.detector_combo.grid(row=5, column=0, sticky="ew", pady=(4, 7))
        self.detector_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._detector_changed(),
        )
        yolo_row = ttk.Frame(source, style="Panel.TFrame")
        yolo_row.grid(row=6, column=0, sticky="ew")
        yolo_row.columnconfigure(0, weight=1)
        self.yolo_entry = ttk.Entry(yolo_row, textvariable=self.yolo_weights_var)
        self.yolo_entry.grid(row=0, column=0, sticky="ew")
        self.yolo_button = ttk.Button(
            yolo_row,
            text="Local weights",
            command=self._browse_yolo,
        )
        self.yolo_button.grid(row=0, column=1, padx=(7, 0))

        sam_row = ttk.Frame(source, style="Panel.TFrame")
        sam_row.grid(row=7, column=0, sticky="ew", pady=(7, 0))
        sam_row.columnconfigure(0, weight=1)
        self.sam_entry = ttk.Entry(
            sam_row,
            textvariable=self.sam_checkpoint_var,
        )
        self.sam_entry.grid(row=0, column=0, sticky="ew")
        self.sam_button = ttk.Button(
            sam_row,
            text="SAM weights",
            command=self._browse_sam,
        )
        self.sam_button.grid(row=0, column=1, padx=(7, 0))
        sam_refresh_row = ttk.Frame(source, style="Panel.TFrame")
        sam_refresh_row.grid(row=8, column=0, sticky="ew", pady=(7, 0))
        sam_refresh_row.columnconfigure(0, weight=1)
        self.sam_refresh_label = ttk.Label(
            sam_refresh_row,
            text="SAM refresh interval (frames)",
            style="Muted.Panel.TLabel",
        )
        self.sam_refresh_label.grid(row=0, column=0, sticky="w")
        self.sam_refresh_spin = ttk.Spinbox(
            sam_refresh_row,
            from_=1,
            to=30,
            increment=1,
            textvariable=self.sam_refresh_interval_var,
            width=6,
        )
        self.sam_refresh_spin.grid(row=0, column=1, sticky="e")

        model = ttk.LabelFrame(parent, text="Model and checkpoint", padding=12)
        model.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        model.columnconfigure(0, weight=1)
        self.bundle_combo = ttk.Combobox(
            model,
            textvariable=self.bundle_var,
            state="readonly",
        )
        self.bundle_combo.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.bundle_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._bundle_changed(),
        )
        ttk.Button(model, text="Rescan", command=self.refresh_bundles).grid(
            row=1, column=0, sticky="ew", pady=(7, 0)
        )
        ttk.Button(model, text="Import checkpoint", command=self._import_checkpoint).grid(
            row=1, column=1, sticky="ew", padx=(7, 0), pady=(7, 0)
        )
        ttk.Label(model, textvariable=self.gallery_var, style="Muted.Panel.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.realtime_gallery_tree = ttk.Treeview(
            model,
            columns=("id", "name", "embeddings"),
            show="headings",
            height=4,
        )
        for column, heading, width in (
            ("id", "Person ID", 85),
            ("name", "Registered name", 140),
            ("embeddings", "Emb.", 48),
        ):
            self.realtime_gallery_tree.heading(column, text=heading)
            self.realtime_gallery_tree.column(column, width=width, anchor="w")
        self.realtime_gallery_tree.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(7, 0),
        )
        if self.open_offline is not None:
            ttk.Button(
                model,
                text="Open Offline batch",
                command=self.open_offline,
            ).grid(row=4, column=0, sticky="ew", pady=(7, 0))
        if self.open_gallery is not None:
            ttk.Button(
                model,
                text="Manage Gallery",
                command=self.open_gallery,
            ).grid(row=4, column=1, sticky="ew", padx=(7, 0), pady=(7, 0))

        settings = ttk.LabelFrame(parent, text="Inference settings", padding=12)
        settings.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        settings.columnconfigure(1, weight=1)
        rows = [
            ("Frame length", self.clip_len_var, [15, 30]),
            ("Inference stride", self.stride_var, [1, 3, 5, 10]),
            ("Device", self.device_var, ["auto", "cuda", "cpu"]),
        ]
        for index, (label, variable, values) in enumerate(rows):
            ttk.Label(settings, text=label, style="Panel.TLabel").grid(
                row=index, column=0, sticky="w", pady=(0 if index == 0 else 7, 0)
            )
            selector = ttk.Combobox(
                settings,
                textvariable=variable,
                values=values,
                state="readonly",
                width=10,
            )
            selector.grid(
                row=index,
                column=1,
                sticky="e",
                pady=(0 if index == 0 else 7, 0),
            )
            if label == "Frame length":
                self.clip_len_combo = selector
                selector.bind(
                    "<<ComboboxSelected>>",
                    lambda _event: self._clip_len_changed(),
                )
        ttk.Label(settings, text="Unknown threshold", style="Panel.TLabel").grid(
            row=3, column=0, sticky="w", pady=(7, 0)
        )
        self.threshold_spin = ttk.Spinbox(
            settings,
            from_=-1.0,
            to=1.0,
            increment=0.01,
            textvariable=self.threshold_var,
            width=9,
            format="%.2f",
        )
        self.threshold_spin.grid(row=3, column=1, sticky="e", pady=(7, 0))
        ttk.Label(settings, text="Min. top-2 margin", style="Panel.TLabel").grid(
            row=4, column=0, sticky="w", pady=(7, 0)
        )
        self.margin_spin = ttk.Spinbox(
            settings,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.margin_var,
            width=9,
            format="%.2f",
        )
        self.margin_spin.grid(row=4, column=1, sticky="e", pady=(7, 0))
        ttk.Label(settings, text="Replay FPS", style="Panel.TLabel").grid(
            row=5, column=0, sticky="w", pady=(7, 0)
        )
        ttk.Spinbox(
            settings,
            from_=0.5,
            to=60.0,
            increment=0.5,
            textvariable=self.replay_fps_var,
            width=9,
        ).grid(row=5, column=1, sticky="e", pady=(7, 0))
        ttk.Label(settings, text="YOLO confidence", style="Panel.TLabel").grid(
            row=6, column=0, sticky="w", pady=(7, 0)
        )
        ttk.Spinbox(
            settings,
            from_=0.05,
            to=0.95,
            increment=0.05,
            textvariable=self.detector_confidence_var,
            width=9,
            format="%.2f",
        ).grid(row=6, column=1, sticky="e", pady=(7, 0))
        self.loop_check = ttk.Checkbutton(
            settings,
            text="Loop replay",
            variable=self.loop_var,
        )
        self.loop_check.grid(row=7, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(
            settings,
            text="threshold.provisional",
            style="Muted.Panel.TLabel",
            wraplength=300,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(7, 0))

        actions = ttk.Frame(parent, style="Panel.TFrame")
        actions.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(
            actions,
            text="Start realtime recognition",
            style="Accent.TButton",
            command=self.start,
        )
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self.stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        WrappedLabel(
            parent,
            textvariable=self.status_var,
            style="Muted.Panel.TLabel",
        ).grid(row=5, column=0, sticky="ew", pady=(10, 0))
        readiness = ttk.LabelFrame(parent, text="workflow.readiness", padding=8)
        readiness.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        readiness.columnconfigure(1, weight=1)
        self.readiness_vars = {}
        for row, key in enumerate(("source", "model", "detector", "gallery", "identity", "settings")):
            variable = tk.StringVar()
            self.readiness_vars[key] = variable
            ttk.Label(readiness, text="workflow." + key, style="Muted.Panel.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 8), pady=3)
            WrappedLabel(readiness, textvariable=variable, style="Panel.TLabel").grid(
                row=row, column=1, sticky="ew", pady=3)

    def _build_content(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        result = ttk.Frame(parent, style="Panel.TFrame")
        result.grid(row=0, column=0, sticky="ew")
        result.columnconfigure(0, weight=1)
        WrappedLabel(result, textvariable=self.result_var, style="Result.TLabel").grid(
            row=0, column=0, sticky="ew"
        )
        WrappedLabel(result, textvariable=self.result_detail_var, style="Muted.Panel.TLabel").grid(
            row=1, column=0, sticky="ew", pady=(3, 0)
        )
        self.enrollment_stages = ttk.Frame(result, style="Panel.TFrame")
        self.enrollment_stages.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        self.stage_labels = []
        for column, stage in enumerate(("prepare", "capture", "review", "save")):
            self.enrollment_stages.columnconfigure(column, weight=1, uniform="stage")
            label = WrappedLabel(self.enrollment_stages, text="workflow.stage." + stage,
                                 style="Muted.Panel.TLabel")
            label.grid(row=0, column=column, sticky="ew")
            self.stage_labels.append(label)
        self.buffer_progress = ttk.Progressbar(result, mode="determinate", maximum=15)
        self.buffer_progress.grid(row=3, column=0, sticky="ew", pady=(5, 0))

        telemetry = ttk.Frame(parent, style="Panel.TFrame")
        telemetry.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        telemetry_items = [
                ("Pipeline FPS", self.fps_var),
                ("Frame-read FPS", self.capture_fps_var),
                ("Valid sampling FPS", self.sampling_fps_var),
                ("Window span", self.window_span_var),
                ("Mean / max gap", self.frame_gap_var),
                ("Frame buffer", self.buffer_var),
                ("Person points", self.points_var),
                ("Detection / SAM", self.detection_latency_var),
                ("Gait inference", self.latency_var),
                ("Enrollment", self.enrollment_progress_var),
            ]
        for index, (title, variable) in enumerate(
            telemetry_items
        ):
            row = index // 5
            column = index % 5
            telemetry.columnconfigure(column, weight=1, uniform="metric")
            item = ttk.Frame(telemetry, style="Panel.TFrame", padding=(8, 5))
            item.grid(
                row=row,
                column=column,
                sticky="ew",
                padx=(0 if column == 0 else 5, 0),
                pady=(0 if row == 0 else 5, 0),
            )
            item.columnconfigure(0, weight=1)
            WrappedLabel(item, text=title, style="Muted.Panel.TLabel").grid(
                row=0, column=0, sticky="ew"
            )
            WrappedLabel(item, textvariable=variable, style="Panel.TLabel").grid(
                row=1, column=0, sticky="ew"
            )

        paned = SplitPane(parent, orient=tk.VERTICAL, fraction=0.76, minimum=(220, 80))
        paned.grid(row=2, column=0, sticky="nsew")
        previews = ttk.Frame(paned, style="Panel.TFrame")
        details = ttk.Frame(paned, style="Panel.TFrame")
        paned.add(previews, weight=4)
        paned.add(details, weight=1)
        previews.columnconfigure(0, weight=1, minsize=0, uniform="preview")
        previews.columnconfigure(1, weight=1, minsize=0, uniform="preview")
        previews.rowconfigure(0, weight=1, minsize=100)

        rgb_group = ttk.LabelFrame(previews, text="RGB detection", padding=5)
        cloud_group = ttk.LabelFrame(previews, text="Filtered person point cloud", padding=5)
        rgb_group.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        cloud_group.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        for group in (rgb_group, cloud_group):
            group.columnconfigure(0, weight=1)
            group.rowconfigure(1, weight=1)
        self.preview_container = previews
        self.rgb_group = rgb_group
        self.cloud_group = cloud_group
        self._preview_layout_mode = ""
        previews.bind("<Configure>", self._update_preview_layout, add="+")

        rgb_controls = ttk.Frame(rgb_group, style="Panel.TFrame")
        rgb_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.rgb_pointcloud_check = ttk.Checkbutton(
            rgb_controls,
            text=self.i18n.tr("realtime.rgb_pointcloud"),
            variable=self.show_rgb_pointcloud_var,
            command=self._refresh_last_preview,
        )
        self.rgb_pointcloud_check.grid(row=0, column=0, sticky="w")

        cloud_controls = ttk.Frame(cloud_group, style="Panel.TFrame")
        cloud_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        cloud_controls.columnconfigure(0, weight=1)
        self.cloud_scale_label = WrappedLabel(
            cloud_controls,
            text="Fixed 2.4 m view",
            style="Muted.Panel.TLabel",
        )
        self.cloud_scale_label.grid(row=0, column=0, sticky="ew")
        ttk.Button(
            cloud_controls,
            text="−",
            width=3,
            command=lambda: self._adjust_cloud_zoom(1.0 / 1.2),
        ).grid(row=0, column=1, padx=(5, 0))
        ttk.Label(
            cloud_controls,
            textvariable=self.cloud_zoom_text_var,
            style="Panel.TLabel",
            width=7,
            anchor="center",
        ).grid(row=0, column=2, padx=4)
        ttk.Button(
            cloud_controls,
            text="+",
            width=3,
            command=lambda: self._adjust_cloud_zoom(1.2),
        ).grid(row=0, column=3)
        ttk.Button(
            cloud_controls,
            text="Reset",
            width=6,
            command=self._reset_cloud_zoom,
        ).grid(row=0, column=4, padx=(5, 0))

        self.rgb_label = ttk.Label(
            rgb_group,
            text="Waiting for source",
            anchor="center",
            style="PreviewImage.TLabel",
        )
        self.cloud_label = ttk.Label(
            cloud_group,
            text="Foreground point cloud",
            anchor="center",
            style="PreviewImage.TLabel",
        )
        self.rgb_label.grid(row=1, column=0, sticky="nsew")
        self.cloud_label.grid(row=1, column=0, sticky="nsew")

        details.columnconfigure(0, weight=1)
        details.rowconfigure(0, weight=1)
        self.candidates = ttk.Treeview(
            details,
            columns=("rank", "id", "name", "score", "gallery"),
            show="headings",
            height=5,
        )
        headings = ("Rank", "Person ID", "Name", "Similarity", "Gallery clips")
        widths = (55, 100, 160, 100, 110)
        for column, heading, width in zip(
            self.candidates["columns"], headings, widths
        ):
            self.candidates.heading(column, text=heading)
            self.candidates.column(column, width=width, anchor="w")
        scroll = ttk.Scrollbar(details, command=self.candidates.yview)
        horizontal = ttk.Scrollbar(details, orient="horizontal", command=self.candidates.xview)
        self.candidates.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        self.candidates.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

    def _update_preview_layout(self, event: tk.Event) -> None:
        width = max(1, int(getattr(event, "width", 1)))
        height = max(1, int(getattr(event, "height", 1)))
        threshold = 700 if self._preview_layout_mode == "wide" else 740
        tall = height >= (280 if self._preview_layout_mode == "stacked" else 320)
        mode = "stacked" if width < threshold and tall else "wide"
        if mode == "wide" and width < 740:
            self.cloud_scale_label.grid_remove()
        else:
            self.cloud_scale_label.grid()
        if mode == self._preview_layout_mode:
            return
        self._preview_layout_mode = mode
        for group in (self.rgb_group, self.cloud_group):
            group.grid_forget()
        previews = self.preview_container
        if mode == "wide":
            previews.columnconfigure(0, weight=1, minsize=0, uniform="preview")
            previews.columnconfigure(1, weight=1, minsize=0, uniform="preview")
            previews.rowconfigure(0, weight=1, minsize=100)
            previews.rowconfigure(1, weight=0, minsize=0)
            self.rgb_group.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
            self.cloud_group.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        else:
            previews.columnconfigure(0, weight=1, minsize=0, uniform="")
            previews.columnconfigure(1, weight=0, minsize=0, uniform="")
            previews.rowconfigure(0, weight=1, minsize=100)
            previews.rowconfigure(1, weight=1, minsize=100)
            self.rgb_group.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
            self.cloud_group.grid(row=1, column=0, sticky="nsew", pady=(4, 0))

    def refresh_bundles(self) -> None:
        bundles = self.controller.available_bundles(refresh=True)
        current_id = self._bundle_id(required=False)
        old_revision = getattr(self, "_bundle_revisions", {}).get(current_id)
        self._bundle_revisions = {
            key: (item.checkpoint_sha256, deepcopy(item.data), deepcopy(item.model))
            for key, item in bundles.items()
        }
        self.bundle_labels = {
            f"{item.display_name} | {item.checkpoint_sha256[:8]}": bundle_id
            for bundle_id, item in bundles.items()
        }
        labels = list(self.bundle_labels)
        self.bundle_combo.configure(values=labels)
        selected = next(
            (
                label
                for label, bundle_id in self.bundle_labels.items()
                if bundle_id == current_id
            ),
            None,
        )
        if selected is None and labels:
            preferred = str(
                nested(
                    self.controller.config,
                    "runtime",
                    "bundle",
                    "mph_gait_fixed_special5_seed0_split0",
                )
            )
            selected = next(
                (
                    label
                    for label, bundle_id in self.bundle_labels.items()
                    if bundle_id == preferred
                ),
                labels[0],
            )
        if selected:
            self.bundle_var.set(selected)
            selected_id = self.bundle_labels[selected]
            changed = old_revision is not None and old_revision != self._bundle_revisions[selected_id]
            if selected_id != current_id or changed:
                self._bundle_changed()
            else:
                self._refresh_gallery_summary()

    def _bundle_id(self, required: bool = True) -> str | None:
        value = self.bundle_labels.get(self.bundle_var.get())
        if value is None and required:
            raise ValueError("Select a model bundle")
        return value

    def _operation(self) -> str:
        return self.operation_labels[self.operation_var.get()]

    @staticmethod
    def _should_reject_multiple_people(
        operation: str,
        allow_multiple_people_enrollment: bool,
        reject_multiple_people_recognition: bool,
    ) -> bool:
        if operation == "enroll":
            return not bool(allow_multiple_people_enrollment)
        return bool(reject_multiple_people_recognition)

    def _operation_changed(self, reset_result: bool = True) -> None:
        enrolling = self._operation() == "enroll"
        if enrolling:
            self.enrollment_frame.grid()
            self.start_button.configure(text="Start guided enrollment session")
            self.loop_var.set(False)
            if reset_result:
                self.result_var.set(self.i18n.tr("Realtime enrollment ready"))
                self.result_detail_var.set(self.i18n.tr("Enter an identity and begin walking"))
            self._autofill_replay_identity(silent=True)
        else:
            self.enrollment_frame.grid_remove()
            self.start_button.configure(text="Start realtime recognition")
            if reset_result:
                self.result_var.set(self.i18n.tr("No realtime result"))
                self.result_detail_var.set(self.i18n.tr("Select a source and start the pipeline"))
        recognition_state = "disabled" if enrolling else "normal"
        self.threshold_spin.configure(state=recognition_state)
        self.margin_spin.configure(state=recognition_state)
        self._update_loop_control()
        self._update_pass_controls()

    def _clip_len_changed(self) -> None:
        bundle_id = self._bundle_id(required=False)
        if bundle_id:
            self.threshold_var.set(
                self.controller.provisional_threshold(
                    bundle_id,
                    clip_len=int(self.clip_len_var.get()),
                )
            )
        self._refresh_gallery_summary()

    def _refresh_gallery_summary(self) -> None:
        bundle_id = self._bundle_id(required=False)
        if not bundle_id:
            return
        try:
            clip_len = int(self.clip_len_var.get())
        except (ValueError, tk.TclError):
            self._schedule_readiness()
            return
        summary = self.controller.database_summary(
            bundle_id,
            clip_len=clip_len,
        )
        self._gallery_counts = summary
        self._schedule_readiness()
        self.gallery_var.set(
            f"Gallery T={clip_len}: "
            f"{summary.get('persons', 0)} identities / "
            f"{summary.get('active_embeddings', 0)} clip embeddings"
        )
        children = self.realtime_gallery_tree.get_children()
        if children:
            self.realtime_gallery_tree.delete(*children)
        for person in self.controller.list_persons(bundle_id, clip_len=clip_len):
            self.realtime_gallery_tree.insert(
                "",
                "end",
                values=(
                    person.get("person_id", ""),
                    person.get("display_name", ""),
                    person.get("embedding_count", 0),
                ),
            )

    def _schedule_readiness(self, *_args) -> None:
        if self._readiness_after is not None:
            self.after_cancel(self._readiness_after)
        self._readiness_after = self.after_idle(self._refresh_readiness)

    def _refresh_readiness(self) -> None:
        if self._readiness_after is not None:
            self.after_cancel(self._readiness_after)
            self._readiness_after = None
        tr = self.i18n.tr
        enrolling = self._operation() == "enroll"
        blocker = ""
        source = self.source_labels.get(self.source_var.get())
        if source == "replay":
            source_ok = Path(self.replay_path_var.get()).expanduser().is_dir()
            source_text = tr("workflow.ready" if source_ok else "workflow.source_missing")
            if not source_ok:
                blocker = source_text
        else:
            source_text = (self.device_status.message if self.device_status is not None
                           else tr("workflow.not_checked"))
        self.readiness_vars["source"].set(source_text)
        bundle_id = self._bundle_id(required=False)
        try:
            bundle = self.controller.model_store.get(bundle_id) if bundle_id else None
            model_ok = bundle is not None and bundle.checkpoint.is_file()
        except (KeyError, ValueError, FileNotFoundError):
            model_ok = False
        self.readiness_vars["model"].set(tr("workflow.model_pending" if model_ok else "workflow.model_missing"))
        if not model_ok:
            blocker = blocker or tr("workflow.model_missing")
        detector = self.detector_labels.get(self.detector_var.get())
        detector_text = tr("workflow.ready")
        try:
            if detector in {"yolo", "yolo_seg", "yolo_sam"}:
                resolve_yolo_weights_reference(self.yolo_weights_var.get().strip() or DEFAULT_YOLO_MODEL)
                detector_text = tr("workflow.detector_pending")
            if detector == "yolo_sam":
                resolve_sam_checkpoint_path(self.sam_checkpoint_var.get().strip())
        except (FileNotFoundError, ValueError) as exc:
            detector_text = str(exc)
            blocker = blocker or detector_text
        self.readiness_vars["detector"].set(detector_text)
        try:
            length = int(self.clip_len_var.get())
            valid = length > 0 and int(self.stride_var.get()) > 0
            valid = valid and all(math.isfinite(float(var.get())) for var in (
                self.threshold_var, self.margin_var, self.enrollment_duration_var,
                self.enrollment_warmup_var))
            if enrolling:
                valid = valid and (float(self.enrollment_duration_var.get()) > 0
                    and float(self.enrollment_warmup_var.get()) >= 0
                    and 0 < int(self.enrollment_min_var.get()) <= int(self.enrollment_max_var.get()))
        except (ValueError, tk.TclError):
            valid, length = False, 0
        self.readiness_vars["settings"].set(tr("workflow.ready" if valid else "workflow.invalid_settings"))
        if not valid:
            blocker = blocker or tr("workflow.invalid_settings")
        gallery = self._gallery_counts
        if gallery.get("bundle_id") != bundle_id or gallery.get("clip_len") != length:
            gallery = {}
        self.readiness_vars["gallery"].set(tr("workflow.gallery_optional") if enrolling else
            (tr("workflow.gallery_count", persons=gallery.get("persons", 0),
                count=gallery.get("active_embeddings", 0), length=length)
             if gallery.get("active_embeddings", 0) else tr("workflow.no_gallery.detail")))
        identity_text = tr("workflow.not_required")
        if enrolling:
            person_id = self.enrollment_person_id_var.get().strip()
            name = self.enrollment_name_var.get().strip()
            identity_text = tr("workflow.ready")
            if not person_id or not name:
                identity_text = tr("workflow.identity_missing")
                blocker = blocker or identity_text
            else:
                person = self.controller.get_person(person_id)
                if person is not None and str(person["display_name"]) != name:
                    identity_text = tr("workflow.identity_conflict", name=person["display_name"])
                    blocker = blocker or identity_text
        self.readiness_vars["identity"].set(identity_text)
        self._readiness_blocker = blocker
        busy = (self._session_active or self._stop_pending or self._device_check_pending
                or self.commit_pending or self._pending_review_result is not None)
        self.start_button.configure(state="disabled" if busy or blocker else "normal")

    def _render_feedback(self, snapshot: PipelineSnapshot) -> None:
        title, detail = snapshot_feedback(snapshot, self.i18n.tr)
        self.result_var.set(title)
        self.result_detail_var.set(detail)
        self.buffer_progress.configure(maximum=max(1, snapshot.clip_len),
            value=min(snapshot.buffer_size, max(1, snapshot.clip_len)))

    def _update_pass_tree(self, passes: list[dict[str, Any]] | None = None) -> None:
        if passes is None and self.pipeline is not None:
            passes = self.pipeline.enrollment_passes()
        passes = passes or []
        children = self.pass_tree.get_children()
        if children:
            self.pass_tree.delete(*children)
        for item in passes:
            requested = str(item.get("requested_direction") or "unknown")
            observed = str(item.get("observed_direction") or "unknown")
            direction = self.pass_direction_names.get(
                requested,
                requested.replace("_", " "),
            )
            if observed not in {"", "unknown"} and observed != requested:
                direction = f"{direction} / {observed.replace('_', ' ')}"
            state = "discarded" if item.get("discarded") else (
                "recording" if item.get("ended_at") is None else "kept"
            )
            self.pass_tree.insert(
                "",
                "end",
                values=(
                    item.get("pass_id", ""),
                    direction,
                    item.get("frame_count", 0),
                    item.get("embedding_count", 0),
                    state,
                ),
            )

    def _update_pass_controls(self, result: dict[str, Any] | None = None) -> None:
        acknowledged = result is not None
        if result is not None:
            self._pass_result = dict(result)
        result = self._pass_result
        enrolling = self._operation() == "enroll"
        running = bool(self.pipeline is not None and self.pipeline.running)
        reviewing = self._pending_review_result is not None
        state = str(result.get("state", ""))
        active = bool(result.get("active_pass_id")) or state == "enrolling"
        requested = state == "enrollment_countdown"
        if acknowledged and self._pass_action == "start" and (active or requested):
            self._pass_action = None
        elif acknowledged and self._pass_action == "end" and state == "enrollment_paused" and not active:
            self._pass_action = None
        pending = self._pass_action
        available = (enrolling and running and self._capture_ready and not reviewing
                     and not self.commit_pending and not self._stop_pending)
        self.pass_direction_combo.configure(
            state="readonly" if available and not active and not requested and not pending else "disabled")
        self.start_pass_button.configure(
            state="normal" if available and not active and not requested and not pending else "disabled")
        self.end_pass_button.configure(
            state="normal" if available and (active or requested or pending == "start")
            and pending not in {"end", "finish"} else "disabled")
        self.discard_pass_button.configure(state=self.end_pass_button.cget("state"))
        self.finish_review_button.configure(
            state="normal" if not self.commit_pending and not self._stop_pending
            and (reviewing or available and not pending) else "disabled",
            text=self.i18n.tr("Open review" if reviewing else "Finish & review"))
        if enrolling:
            self.enrollment_stages.grid()
        else:
            self.enrollment_stages.grid_remove()
        stage = 3 if self.commit_pending or state == "enrolled" else 2 if reviewing else 1 if available else 0
        for index, label in enumerate(self.stage_labels):
            label.configure(style="Panel.TLabel" if index == stage else "Muted.Panel.TLabel")
        if self.commit_pending:
            text = self.i18n.tr("sources.committing")
        elif self._stop_pending:
            text = self.i18n.tr("workflow.stopping")
        elif reviewing:
            text = self.i18n.tr("workflow.enrollment_review.detail",
                               passes=self._pending_review_result.get("usable_passes", 0),
                               count=self._pending_review_result.get("captured_embeddings", 0))
        elif pending:
            text = self.i18n.tr("workflow.request." + pending)
        elif running and not self._capture_ready:
            text = self.i18n.tr("workflow.source_ready.detail")
        elif active and available:
            text = self.i18n.tr("Recording this one-way pass; press End at the finish line.")
        elif requested and available:
            text = self.i18n.tr("workflow.request.start")
        elif available:
            text = self.i18n.tr("Session ready. Choose a direction and press Start pass.")
        else:
            text = self.i18n.tr("Start the session, then record one-way passes.")
        self.pass_status_var.set(text)

    def _start_enrollment_pass(self) -> None:
        if self.start_pass_button.instate(["disabled"]):
            return
        try:
            direction = self.pass_direction_labels[self.pass_direction_var.get()]
            self.pipeline.start_enrollment_pass(direction)
            self._pass_action = "start"
            self._pass_requested_at = time.time()
            self._update_pass_controls()
        except Exception as exc:
            messagebox.showerror("Cannot start pass", str(exc), parent=self)

    def _end_enrollment_pass(self, discard: bool) -> None:
        if self.end_pass_button.instate(["disabled"]):
            return
        try:
            self.pipeline.end_enrollment_pass(discard=discard)
            self._pass_action = "end"
            self._pass_requested_at = time.time()
            self._update_pass_controls()
        except Exception as exc:
            messagebox.showerror("Cannot end pass", str(exc), parent=self)

    def _finish_enrollment_for_review(self) -> None:
        if self.finish_review_button.instate(["disabled"]):
            return
        if self._pending_review_result is not None:
            self._open_enrollment_review()
            return
        try:
            self.pipeline.finish_enrollment_for_review()
            self._pass_action = "finish"
            self._pass_requested_at = time.time()
            self._update_pass_controls()
        except Exception as exc:
            messagebox.showerror("Cannot finish enrollment", str(exc), parent=self)

    def _update_loop_control(self) -> None:
        replay = self.source_labels[self.source_var.get()] == "replay"
        recognition = self._operation() == "recognize"
        self.loop_check.configure(
            state="normal" if replay and recognition else "disabled"
        )

    def _autofill_replay_identity(self, silent: bool = False) -> None:
        if self._operation() != "enroll":
            return
        if self.source_labels[self.source_var.get()] != "replay":
            return
        replay = Path(self.replay_path_var.get()).expanduser()
        if not replay.exists():
            return
        try:
            suggestion = suggest_registration_identity([replay])
        except Exception as exc:
            if not silent:
                messagebox.showerror("Identity detection failed", str(exc))
            return
        if suggestion is None:
            if not silent:
                messagebox.showinfo(
                    "No identity found",
                    "The replay folder name does not contain a recognizable person ID.",
                )
            return
        self.enrollment_person_id_var.set(suggestion.person_id)
        existing = self.controller.get_person(suggestion.person_id)
        self.enrollment_name_var.set(
            str(existing["display_name"])
            if existing is not None
            else suggestion.default_display_name
        )

    def _source_changed(self) -> None:
        source_mode = self.source_labels[self.source_var.get()]
        azure = source_mode == "azure_kinect"
        state = "disabled" if azure else "normal"
        self.replay_entry.configure(state=state)
        self.replay_button.configure(state=state)
        detector_values = self._detector_values_for_source(source_mode)
        self.detector_combo.configure(values=detector_values)
        if self.detector_var.get() not in detector_values:
            self.detector_var.set(detector_values[0])
        self._detector_changed()
        self._update_loop_control()
        self.autofill_button.configure(state="normal" if not azure else "disabled")

    def _detector_changed(self) -> None:
        detector = self.detector_labels.get(self.detector_var.get())
        current_weights = self.yolo_weights_var.get().strip()
        if detector == "yolo_seg" and current_weights in {
            "",
            self.default_yolo_weights,
            DEFAULT_YOLO_MODEL,
        }:
            self.yolo_weights_var.set(self.default_yolo_seg_weights)
        elif detector in {"yolo", "yolo_sam"} and current_weights in {
            "",
            self.default_yolo_seg_weights,
            DEFAULT_YOLO_SEG_MODEL,
        }:
            self.yolo_weights_var.set(self.default_yolo_weights)
        yolo_state = (
            "normal" if detector in {"yolo", "yolo_seg", "yolo_sam"} else "disabled"
        )
        sam_state = "normal" if detector == "yolo_sam" else "disabled"
        self.yolo_entry.configure(state=yolo_state)
        self.yolo_button.configure(state=yolo_state)
        self.sam_entry.configure(state=sam_state)
        self.sam_button.configure(state=sam_state)
        self.sam_refresh_spin.configure(state=sam_state)

    def _bundle_changed(self) -> None:
        bundle_id = self._bundle_id(required=False)
        if not bundle_id:
            return
        bundle = self.controller.model_store.get(bundle_id)
        self.clip_len_var.set(int(bundle.data.get("clip_len", 15)))
        self.threshold_var.set(
            self.controller.provisional_threshold(
                bundle_id,
                clip_len=int(self.clip_len_var.get()),
            )
        )
        self._refresh_gallery_summary()

    def _browse_replay(self) -> None:
        selected = filedialog.askdirectory(title="Select synchronized replay folder")
        if selected:
            self.replay_path_var.set(selected)
            self._autofill_replay_identity(silent=True)

    def _browse_yolo(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select local Ultralytics YOLO checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if selected:
            self.yolo_weights_var.set(selected)

    def _browse_sam(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select official Meta SAM checkpoint",
            filetypes=[("SAM checkpoint", "*.pth"), ("All files", "*.*")],
        )
        if selected:
            self.sam_checkpoint_var.set(selected)

    def _check_device(self, check_access: bool = True) -> None:
        if self._session_active or self._device_check_pending or self._pending_review_result is not None:
            return
        if self.camera_available is not None:
            available, reason = self.camera_available()
            if not available:
                messagebox.showwarning("Kinect is busy", reason, parent=self)
                return
        self._device_check_pending = True
        self.device_status = None
        self.device_var_text.set(self.i18n.tr("workflow.starting.detail"))
        self._set_running(False)
        self.device_worker.start(lambda _progress: probe_azure_kinect(check_access=check_access))

    def _import_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a trusted checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not selected:
            return
        if not messagebox.askyesno(
            "Trusted checkpoint",
            "Only import PyTorch checkpoints from a trusted source. Continue?",
        ):
            return
        display_name = simpledialog.askstring(
            "Display name",
            "Name shown in the model list:",
            initialvalue=Path(selected).stem,
            parent=self,
        )
        if not display_name:
            return
        try:
            bundle = self.controller.import_checkpoint(
                selected,
                display_name.strip(),
                template_bundle_id=self._bundle_id(),
            )
            self.refresh_bundles()
            target = next(
                label
                for label, bundle_id in self.bundle_labels.items()
                if bundle_id == bundle.bundle_id
            )
            self.bundle_var.set(target)
            self._bundle_changed()
            messagebox.showinfo(
                "Import complete",
                f"Checkpoint copied into the application:\n{bundle.bundle_dir}",
            )
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))

    def start(self) -> None:
        """Start from a Tk callback without allowing silent UI failures."""

        if self._session_active or self._stop_pending or self.commit_pending or self._device_check_pending:
            return
        if self.camera_available is not None:
            available, reason = self.camera_available()
            if not available:
                messagebox.showwarning("Kinect is busy", reason, parent=self)
                return
        try:
            self._start_impl()
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if self.pipeline is not None and self.pipeline.running:
                self.stop()
            self.status_var.set("Unable to start the realtime pipeline")
            self.result_var.set("Realtime startup failed")
            self.result_detail_var.set(detail)
            self._set_running(False)
            messagebox.showerror("Realtime startup failed", detail, parent=self)

    def _reject_start(self, title: str, detail: str) -> bool:
        self.status_var.set(title)
        self.result_var.set("Realtime startup blocked")
        self.result_detail_var.set(detail)
        messagebox.showerror(title, detail, parent=self)
        return False

    def _start_impl(self) -> bool:
        if self._pending_review_result is not None:
            self._open_enrollment_review()
            return False
        if self.pipeline is not None and self.pipeline.running:
            self.status_var.set("A realtime pipeline is already running")
            self.result_detail_var.set(
                "Stop the current pipeline before starting another session"
            )
            messagebox.showinfo(
                "Realtime pipeline already running",
                "A capture worker is still active. Stop it before starting "
                "another guided enrollment session.",
                parent=self,
            )
            return False
        self._refresh_readiness()
        if self._readiness_blocker:
            return self._reject_start(self.i18n.tr("workflow.readiness"), self._readiness_blocker)
        operation = self._operation()
        source_mode = self.source_labels[self.source_var.get()]
        self.status_var.set(
            "Checking enrollment settings"
            if operation == "enroll"
            else "Checking recognition settings"
        )
        self.result_var.set(
            "Starting guided enrollment"
            if operation == "enroll"
            else "Starting realtime recognition"
        )
        self.result_detail_var.set("Validating source, detector, and model bundle")
        self.update_idletasks()
        if source_mode == "azure_kinect":
            # connected_device_count() is a native SDK call and can block on a
            # busy Windows camera backend. The worker probes and opens the
            # device before loading any model, keeping the Tk event loop live.
            self.device_status = None
            self.device_var_text.set(
                "Checking and opening Azure Kinect DK in the background"
            )
        detector = self.detector_labels[self.detector_var.get()]
        yolo_weights = self.yolo_weights_var.get().strip() or DEFAULT_YOLO_MODEL
        if detector in {"yolo", "yolo_seg", "yolo_sam"}:
            try:
                yolo_weights = resolve_yolo_weights_reference(yolo_weights)
            except (FileNotFoundError, ValueError) as exc:
                return self._reject_start("Invalid YOLO model", str(exc))
        sam_checkpoint = self.sam_checkpoint_var.get().strip()
        if detector == "yolo_sam":
            try:
                sam_checkpoint = resolve_sam_checkpoint_path(sam_checkpoint)
            except (FileNotFoundError, ValueError) as exc:
                return self._reject_start("Invalid SAM model", str(exc))
        replay = Path(self.replay_path_var.get()).expanduser()
        if source_mode == "replay" and not replay.is_dir():
            return self._reject_start(
                "Replay unavailable",
                f"Folder not found:\n{replay}",
            )
        bundle_id = self._bundle_id()
        assert bundle_id is not None
        bundle = self.controller.model_store.get(bundle_id)
        num_points = int(bundle.data.get("num_points") or 1024)
        person_id = self.enrollment_person_id_var.get().strip()
        display_name = self.enrollment_name_var.get().strip()
        min_embeddings = int(self.enrollment_min_var.get())
        max_embeddings = int(self.enrollment_max_var.get())
        if operation == "enroll":
            if not person_id or not display_name:
                return self._reject_start(
                    "Enrollment identity required",
                    "Enter both Person ID and Display name before enrollment.",
                )
            if max_embeddings < min_embeddings:
                return self._reject_start(
                    "Invalid enrollment settings",
                    "Stored embeddings must be greater than or equal to the minimum.",
                )
            existing = self.controller.get_person(person_id)
            if (
                existing is not None
                and str(existing["display_name"]) != display_name
            ):
                return self._reject_start(
                    "Person ID conflict",
                    f"{person_id} is already registered as "
                    f"{existing['display_name']}. "
                    "同一人改名：請到 Gallery 管理 → 修改人物名稱，再重新註冊。 "
                    "For the same person, rename via Gallery Manager before enrolling again.",
                )
        config = RealtimeConfig(
            bundle_id=bundle_id,
            operation=operation,
            source_mode=source_mode,
            replay_path=str(replay),
            replay_fps=float(self.replay_fps_var.get()),
            replay_loop=(
                bool(self.loop_var.get()) if operation == "recognize" else False
            ),
            detector=detector,
            yolo_weights=yolo_weights,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=DEFAULT_SAM_MODEL_TYPE,
            sam_refresh_interval=max(1, int(self.sam_refresh_interval_var.get())),
            detector_confidence=float(self.detector_confidence_var.get()),
            device=self.device_var.get(),
            clip_len=int(self.clip_len_var.get()),
            inference_stride=int(self.stride_var.get()),
            num_points=num_points,
            threshold=float(self.threshold_var.get()),
            min_margin=float(self.margin_var.get()),
            enrollment_person_id=person_id,
            enrollment_display_name=display_name,
            enrollment_note=self.enrollment_note_var.get().strip(),
            enrollment_duration_s=float(self.enrollment_duration_var.get()),
            enrollment_warmup_s=float(self.enrollment_warmup_var.get()),
            enrollment_min_embeddings=min_embeddings,
            enrollment_max_embeddings=max_embeddings,
            enrollment_stride=int(self.clip_len_var.get()),
            save_enrollment_foreground=bool(self.save_foreground_var.get()),
            guided_passes=(operation == "enroll"),
            preprocessing_profile_id=self.controller.processing_version_id(),
            reject_multiple_people=self._should_reject_multiple_people(
                operation,
                bool(self.allow_multi_enrollment_var.get()),
                bool(
                    nested(
                        self.controller.config,
                        "realtime",
                        "reject_multiple_people_recognition",
                        False,
                    )
                ),
            ),
            multi_person_policy=str(
                nested(
                    self.controller.config,
                    "realtime",
                    "multi_person_policy",
                    "primary_salience",
                )
            ),
            max_valid_frame_gap_s=float(
                nested(
                    self.controller.config,
                    "realtime",
                    "max_valid_frame_gap_seconds",
                    1.0,
                )
            ),
        )
        self._pending_review_result = None
        self._pass_result = {}
        self._review_choices = {}
        self._pass_action = None
        self._capture_ready = False
        self._last_snapshot = None
        self.buffer_progress.configure(value=0, maximum=config.clip_len)
        self._fill_candidates(None)
        self._update_pass_tree([])
        self.pipeline = RealtimePipeline(self.controller, config)
        self.pipeline.start()
        self._session_active = True
        self._set_running(True)
        self.status_var.set(
            "Enrollment session starting; wait for ready, then start a one-way pass"
            if operation == "enroll"
            else "Starting realtime recognition"
        )
        if operation == "enroll":
            self.result_var.set("Guided enrollment session started")
            self.result_detail_var.set("Gallery is unchanged until selected passes are committed")
            self._update_pass_controls()
        return True

    def stop(self) -> None:
        if self._stop_pending or self.commit_pending:
            return
        if self.pipeline is None or not self.pipeline.running:
            return
        self._stop_pending = True
        self._capture_ready = False
        self.status_var.set(self.i18n.tr("workflow.stopping"))
        self.result_var.set(self.i18n.tr("workflow.stopping"))
        self.stop_button.configure(state="disabled")
        self._update_pass_controls()
        pipeline = self.pipeline
        self.stop_worker.start(lambda _progress: pipeline.stop())

    def close(self) -> None:
        if self._readiness_after is not None:
            self.after_cancel(self._readiness_after)
            self._readiness_after = None
        if self.pipeline is not None:
            self.pipeline.stop()
        if (self.pipeline is not None and not self.pipeline.running
                and self._pending_review_result is not None and not self.commit_pending):
            self.pipeline.abandon_enrollment()
            self._pending_review_result = None

    def _poll(self) -> None:
        for message in self.device_worker.drain():
            self._device_check_pending = False
            if message.kind == "result":
                self.device_status = message.payload
                self.device_var_text.set(message.payload.message)
            elif message.kind == "error":
                self.device_var_text.set(message.payload["message"])
            self._set_running(self._session_active)
            self._refresh_readiness()
        for message in self.stop_worker.drain():
            if message.kind == "error":
                self.status_var.set(message.payload["message"])
        if self.pipeline is not None:
            snapshot = self.pipeline.poll_latest()
            if snapshot is not None and not self._stop_pending:
                self._show_snapshot(snapshot)
                if snapshot.state in {
                    "error", "complete", "enrolled", "enrollment_failed", "enrollment_review",
                }:
                    self._capture_ready = False
                    self._pass_action = None
                    if snapshot.state == "enrollment_review":
                        self._pending_review_result = dict(snapshot.result or {})
                    self._update_pass_controls(snapshot.result)
            # A terminal snapshot can precede the worker's source cleanup.
            if self._session_active and not self.pipeline.running and not self.stop_worker.busy:
                stopped = self._stop_pending
                self._stop_pending = False
                self._session_active = False
                self._capture_ready = False
                self._pass_action = None
                if stopped:
                    self._pass_result = {}
                    self._update_pass_tree([])
                self._set_running(False)
                if stopped:
                    self.status_var.set(self.i18n.tr("workflow.stopped"))
                    self.result_var.set(self.i18n.tr("workflow.stopped"))
                    self.result_detail_var.set(self.i18n.tr("No Gallery embeddings were written")
                                               if self.pipeline.operation == "enroll" else "")
                if self._pending_review_result is not None:
                    self._update_pass_tree(list(self._pending_review_result.get("passes", [])))
                    self._update_pass_controls(self._pending_review_result)
                    self.after(20, self._open_enrollment_review)
                self._refresh_gallery_summary()
        self.after(50, self._poll)

    def _set_running(self, running: bool) -> None:
        busy = (running or self._session_active or self._stop_pending or self.commit_pending
                or self._device_check_pending or self._pending_review_result is not None)
        self._config_lock.set_locked(busy)
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(
            state="normal" if running and not self._stop_pending else "disabled")
        self._update_pass_controls()
        self._schedule_readiness()

    def _show_snapshot(self, snapshot: PipelineSnapshot) -> None:
        if snapshot.device_status is not None:
            changed = self.device_status != snapshot.device_status
            self.device_status = snapshot.device_status
            self.device_var_text.set(snapshot.device_status.message)
            if changed:
                self._schedule_readiness()
        elif (
            snapshot.state == "error"
            and self.source_labels.get(self.source_var.get()) == "azure_kinect"
        ):
            self.device_var_text.set(snapshot.message)
        self.status_var.set(snapshot.message)
        self.fps_var.set(f"{snapshot.fps:.1f}")
        self.capture_fps_var.set(f"{snapshot.capture_fps:.1f}")
        self.sampling_fps_var.set(f"{snapshot.effective_sampling_fps:.1f}")
        self.window_span_var.set(f"{snapshot.window_duration_s:.2f} s")
        self.frame_gap_var.set(
            f"{snapshot.mean_frame_gap_ms:.0f} / {snapshot.max_frame_gap_ms:.0f} ms"
        )
        self.buffer_var.set(f"{snapshot.buffer_size} / {snapshot.clip_len}")
        self.points_var.set(f"{snapshot.person_point_count:,}")
        self.detection_latency_var.set(f"{snapshot.detection_ms:.1f} ms")
        self.latency_var.set(f"{snapshot.inference_ms:.1f} ms")
        self.enrollment_progress_var.set(
            (
                f"{snapshot.enrollment_embeddings}/"
                f"{snapshot.enrollment_min_embeddings} | "
                f"{snapshot.enrollment_elapsed_s:.1f}/"
                f"{snapshot.enrollment_duration_s:.1f}s"
            )
            if snapshot.operation == "enroll"
            else "-"
        )
        if snapshot.frame_index >= 0 and snapshot.state not in {
                "error", "complete", "enrolled", "enrollment_failed", "enrollment_review"}:
            self._capture_ready = True
        self._render_feedback(snapshot)
        if snapshot.operation == "enroll":
            passes = list((snapshot.result or {}).get("passes", []))
            if passes:
                self._update_pass_tree(passes)
            result = snapshot.result if snapshot.timestamp >= self._pass_requested_at else None
            self._update_pass_controls(result)
        visible_result = snapshot.result if snapshot.state not in {
            "starting", "source_ready", "waiting_person", "multiple_people",
            "insufficient_points", "error", "complete"} else None
        self._fill_candidates(visible_result)
        self._show_images(snapshot)

    def _open_enrollment_review(self) -> None:
        result = self._pending_review_result
        if result is None:
            return
        if self._review_window is not None and self._review_window.winfo_exists():
            self._review_window.deiconify()
            self._review_window.lift()
            self._review_window.focus_force()
            return
        window = tk.Toplevel(self)
        self._review_window = window
        window.title("Review guided enrollment passes")
        window.transient(self.winfo_toplevel())
        window.geometry("980x520")
        window.minsize(820, 420)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)
        window.protocol("WM_DELETE_WINDOW", self._close_enrollment_review)

        name = str(result.get("display_name", ""))
        self._review_title_var = tk.StringVar(value=self.i18n.tr("workflow.review_title", name=name))
        WrappedLabel(
            window,
            textvariable=self._review_title_var,
            style="Result.TLabel",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        ttk.Label(
            window,
            text=(
                "Choose which captured passes to include. Automatic movement direction "
                "is diagnostic only; front-facing orientation must be controlled by the operator."
            ),
            wraplength=900,
        ).grid(row=1, column=0, sticky="w", padx=16, pady=(0, 10))

        pass_scroll = ScrollableFrame(window, width=930)
        pass_scroll.grid(row=2, column=0, sticky="nsew", padx=16)
        table = pass_scroll.content
        table.columnconfigure(2, weight=1)
        headings = (
            self.i18n.tr("review.include"),
            "Pass",
            "Capture type / observed movement",
            "Frames",
            "Embeddings",
            "Mean points",
            "State",
        )
        for column, heading in enumerate(headings):
            ttk.Label(table, text=heading, style="Panel.TLabel").grid(
                row=0, column=column, sticky="w", padx=(0, 12), pady=(0, 6)
            )
        self._review_selection_vars = {}
        passes = list(result.get("passes", []))
        for row, item in enumerate(passes, start=1):
            pass_id = str(item.get("pass_id", ""))
            recommended = bool(item.get("quality_accepted"))
            selectable = bool(
                item.get("selectable", not item.get("discarded") and int(item.get("embedding_count", 0)) > 0)
            )
            variable = tk.BooleanVar(value=selectable and self._review_choices.get(pass_id, recommended))
            variable.trace_add("write", lambda *_args: self._update_review_controls())
            self._review_selection_vars[pass_id] = variable
            ttk.Checkbutton(
                table,
                variable=variable,
                state="normal" if selectable else "disabled",
            ).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
            requested = str(item.get("requested_direction") or "unknown")
            observed = str(item.get("observed_direction") or "unknown")
            values = (
                pass_id,
                f"{self.pass_direction_names.get(requested, requested)} / "
                f"{observed.replace('_', ' ')}",
                str(item.get("frame_count", 0)),
                str(item.get("embedding_count", 0)),
                f"{float(item.get('mean_person_points', 0.0)):,.0f}",
                (
                    "recommended"
                    if recommended
                    else (
                        f"manual selection allowed — {item.get('quality_message', 'review required')}"
                        if selectable
                        else str(item.get("quality_message") or "not selectable")
                    )
                ),
            )
            for column, value in enumerate(values, start=1):
                ttk.Label(table, text=value).grid(
                    row=row, column=column, sticky="w", padx=(0, 12), pady=4
                )
        if not passes:
            ttk.Label(table, text="No pass was captured.").grid(
                row=1, column=0, columnspan=7, sticky="w", pady=8
            )

        self._review_status_var = tk.StringVar()
        summary = ttk.Frame(window, padding=(16, 4))
        summary.grid(row=3, column=0, sticky="ew")
        summary.columnconfigure(0, weight=1)
        WrappedLabel(summary, textvariable=self._review_status_var).grid(row=0, column=0, sticky="ew")
        self._review_progress = ttk.Progressbar(summary, mode="indeterminate")
        self._review_progress.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        actions = ttk.Frame(window, padding=(16, 12))
        actions.grid(row=4, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        ttk.Button(
            actions,
            text="Continue adding passes",
            command=self._resume_enrollment_review,
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(
            actions,
            text="Abandon session",
            command=self._abandon_enrollment_review,
        ).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(
            actions,
            text="Close (reopen in this app)",
            command=self._close_enrollment_review,
        ).grid(row=0, column=3, padx=(8, 0))
        self._review_commit_button = ttk.Button(
            actions,
            text="Commit selected passes",
            style="Accent.TButton",
            command=self._commit_enrollment_review,
        )
        self._review_commit_button.grid(row=0, column=4, padx=(8, 0))
        self._review_lock = ControlLock(form_controls(window))
        self._update_review_controls()
        self.i18n.apply(window)

    def _update_review_controls(self) -> None:
        if self._review_window is None or not self._review_window.winfo_exists():
            return
        selected = {key for key, variable in self._review_selection_vars.items() if variable.get()}
        count = sum(int(item.get("embedding_count", 0))
                    for item in (self._pending_review_result or {}).get("passes", [])
                    if item.get("pass_id") in selected)
        self._review_lock.set_locked(self.commit_pending)
        self._review_commit_button.configure(state="normal" if selected and not self.commit_pending else "disabled")
        self._review_status_var.set(self.i18n.tr("sources.committing") if self.commit_pending else
            self.i18n.tr("workflow.review_selection", passes=len(selected), count=count))
        self._review_title_var.set(self.i18n.tr("workflow.review_title",
            name=(self._pending_review_result or {}).get("display_name", "")))

    def _close_enrollment_review(self) -> None:
        if self.commit_pending:
            return
        if self._review_window is not None:
            self._review_choices = {key: variable.get() for key, variable in self._review_selection_vars.items()}
            self._review_window.destroy()
        self._review_window = None
        self._review_selection_vars = {}

    def _resume_enrollment_review(self) -> None:
        if self.commit_pending:
            return
        if self.pipeline is None:
            return
        try:
            self.pipeline.resume_enrollment_capture()
        except Exception as exc:
            messagebox.showerror(
                "Cannot continue enrollment",
                str(exc),
                parent=self._review_window,
            )
            return
        self._pending_review_result = None
        self._close_enrollment_review()
        self._session_active = True
        self._capture_ready = False
        self._pass_action = None
        self._pass_result = {}
        self._last_snapshot = None
        self._set_running(True)
        self.status_var.set(
            "Enrollment session resumed; existing passes and embeddings were kept"
        )
        self.result_var.set("Ready to add another pass")
        self.result_detail_var.set(
            "Choose a direction and press Start pass; Gallery is still unchanged"
        )
        self._update_pass_controls()

    def _commit_enrollment_review(self) -> None:
        if self.commit_pending:
            return
        if self.pipeline is None:
            return
        selected = [
            pass_id
            for pass_id, variable in self._review_selection_vars.items()
            if variable.get()
        ]
        if not selected:
            messagebox.showwarning("Gallery", self.i18n.tr("workflow.select_pass"), parent=self._review_window)
            return
        passes_by_id = {
            str(item.get("pass_id", "")): item
            for item in list((self._pending_review_result or {}).get("passes", []))
        }
        manual_overrides = [
            pass_id
            for pass_id in selected
            if not bool(passes_by_id.get(pass_id, {}).get("quality_accepted"))
        ]
        if manual_overrides and not messagebox.askyesno(
            "Confirm manually reviewed passes",
            "The following passes have an automatic movement warning but contain "
            "embeddings and were selected manually:\n\n"
            f"{', '.join(manual_overrides)}\n\n"
            "Confirm that the person was facing the camera and include them?",
            parent=self._review_window,
        ):
            return
        pipeline = self.pipeline
        self.commit_pending = True
        self.status_var.set(self.i18n.tr("sources.committing"))
        self._update_review_controls()
        self._review_progress.start(12)
        self._set_running(False)
        self.commit_worker.start(lambda _progress: pipeline.commit_enrollment(selected))
        self.after(100, self._poll_commit)

    def _poll_commit(self) -> None:
        for message in self.commit_worker.drain():
            if message.kind == "error":
                self.commit_pending = False
                self._review_progress.stop()
                self._update_review_controls()
                self._update_pass_controls()
                self.status_var.set(message.payload["message"])
                messagebox.showerror("Gallery commit failed", message.payload["message"],
                                     parent=self._review_window)
                return
            if message.kind == "result":
                self.commit_pending = False
                self._review_progress.stop()
                self._enrollment_committed(message.payload)
                return
        self.after(100, self._poll_commit)

    def _enrollment_committed(self, result: dict[str, Any]) -> None:
        self._pending_review_result = None
        self._close_enrollment_review()
        self._review_choices = {}
        self._refresh_gallery_summary()
        if self.on_gallery_changed is not None:
            self.on_gallery_changed()
        self.status_var.set("Enrollment committed successfully; Gallery is now updated")
        self.result_var.set(f"Enrollment complete: {result.get('display_name', '')}")
        self.result_detail_var.set(
            f"stored={result.get('stored_embeddings', 0)} | "
            f"selected passes={len(result.get('selected_pass_ids', []))} | "
            f"session={result.get('session_id', '')}"
        )
        self._set_running(False)
        self._update_pass_controls(result)
        messagebox.showinfo(
            "Enrollment successful",
            "Selected passes were written to Gallery successfully.\n\n"
            f"Person: {result.get('person_id', '')} — {result.get('display_name', '')}\n"
            f"Stored embeddings: {result.get('stored_embeddings', 0)}\n"
            f"Session: {result.get('session_id', '')}",
            parent=self,
        )

    def _abandon_enrollment_review(self) -> None:
        if self.commit_pending:
            return
        if not messagebox.askyesno(
            "Abandon enrollment",
            "Discard this review session without changing Gallery?",
            parent=self._review_window,
        ):
            return
        try:
            if self.pipeline is not None:
                self.pipeline.abandon_enrollment()
        except Exception as exc:
            messagebox.showerror("Cannot abandon session", str(exc), parent=self._review_window)
            return
        self._pending_review_result = None
        self._close_enrollment_review()
        self._review_choices = {}
        self.status_var.set("Enrollment session abandoned; Gallery was not modified")
        self.result_var.set("Enrollment abandoned")
        self.result_detail_var.set("No embeddings were written to Gallery")
        self._set_running(False)

    def _fill_candidates(self, result: dict[str, Any] | None) -> None:
        children = self.candidates.get_children()
        if children:
            self.candidates.delete(*children)
        if result is None:
            return
        for rank, item in enumerate(result.get("top_candidates", []), start=1):
            self.candidates.insert(
                "",
                "end",
                values=(
                    rank,
                    item.get("person_id"),
                    item.get("display_name"),
                    f"{float(item.get('similarity', 0.0)):.4f}",
                    item.get("gallery_embeddings", 0),
                ),
            )

    def _update_cloud_zoom_text(self) -> None:
        zoom = max(0.25, min(4.0, float(self.cloud_zoom_var.get())))
        self.cloud_zoom_text_var.set(f"{zoom:.2f}×")

    def _adjust_cloud_zoom(self, factor: float) -> None:
        current = float(self.cloud_zoom_var.get())
        self.cloud_zoom_var.set(max(0.25, min(4.0, current * float(factor))))
        self._update_cloud_zoom_text()
        self._refresh_last_preview()

    def _reset_cloud_zoom(self) -> None:
        self.cloud_zoom_var.set(1.0)
        self._update_cloud_zoom_text()
        self._refresh_last_preview()

    def _refresh_last_preview(self) -> None:
        if self._last_snapshot is not None:
            self._show_images(self._last_snapshot)

    def _show_images(self, snapshot: PipelineSnapshot) -> None:
        self._last_snapshot = snapshot
        if snapshot.color_bgr is not None:
            color = snapshot.color_bgr.copy()
            if (
                snapshot.person_mask_rgb is not None
                and bool(self.show_rgb_pointcloud_var.get())
            ):
                mask = np.asarray(snapshot.person_mask_rgb, dtype=bool)
                if mask.shape != color.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (color.shape[1], color.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                overlay = color.copy()
                overlay[mask] = (
                    0.35 * overlay[mask] + 0.65 * np.array([80, 190, 60])
                ).astype(np.uint8)
                color = overlay
            if snapshot.detection is not None:
                detection = snapshot.detection
                x1, y1, x2, y2 = detection.bbox_xyxy
                cv2.rectangle(color, (x1, y1), (x2, y2), (32, 210, 120), 3)
                cv2.putText(
                    color,
                    f"person {detection.confidence:.2f} | {detection.detector}",
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (32, 210, 120),
                    2,
                    cv2.LINE_AA,
                )
            image = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            fitted = render_image(
                image,
                (max(320, self.rgb_label.winfo_width()), max(240, self.rgb_label.winfo_height())),
            )
            self._rgb_photo = ImageTk.PhotoImage(fitted)
            self.rgb_label.configure(image=self._rgb_photo, text="")
        if snapshot.person_points_mm is not None:
            image = self._render_points(
                snapshot.person_points_mm,
                (max(260, self.cloud_label.winfo_width()), max(180, self.cloud_label.winfo_height())),
                zoom=float(self.cloud_zoom_var.get()),
            )
            self._cloud_photo = ImageTk.PhotoImage(image)
            self.cloud_label.configure(image=self._cloud_photo, text="")

    @staticmethod
    def _render_points(
        points: np.ndarray,
        size: tuple[int, int],
        zoom: float = 1.0,
    ) -> Image.Image:
        width, height = size
        canvas = np.full((height, width, 3), (24, 27, 31), dtype=np.uint8)
        values = np.asarray(points, dtype=np.float32)
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) > 9000:
            indices = np.linspace(0, len(values) - 1, 9000).astype(np.int64)
            values = values[indices]
        if not len(values):
            return Image.fromarray(canvas)
        x = values[:, 0] - np.median(values[:, 0])
        y = values[:, 1] - np.median(values[:, 1])
        # Keep a fixed physical 2.4 m square instead of fitting the current
        # percentiles. Point count and partial masks can no longer make the
        # person jump in size; the user-controlled zoom is the only scale change.
        physical_span_mm = 2400.0
        scale = min(
            (width * 0.84) / physical_span_mm,
            (height * 0.84) / physical_span_mm,
        )
        scale *= max(0.25, min(4.0, float(zoom)))
        px = np.rint(width * 0.5 + x * scale).astype(np.int32)
        py = np.rint(height * 0.5 + y * scale).astype(np.int32)
        visible = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        px, py = px[visible], py[visible]
        depth = values[visible, 2]
        low, high = np.percentile(depth, [2, 98])
        ratio = np.clip((depth - low) / max(float(high - low), 1.0), 0.0, 1.0)
        colors = np.stack(
            (50 + 180 * ratio, 190 - 75 * ratio, 230 - 120 * ratio),
            axis=1,
        ).astype(np.uint8)
        canvas[py, px] = colors
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text(
            (10, height - 20),
            f"{len(values):,} displayed points | fixed 2.4 m | {float(zoom):.2f}x",
            fill=(210, 218, 225),
        )
        return image
