from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..config import nested
from ..controller import GaitApplicationController
from ..i18n import I18n
from ..realtime.benchmark import LiveBenchmarkRecorder
from ..realtime.detection import (
    DEFAULT_SAM_CHECKPOINT,
    DEFAULT_SAM_MODEL_TYPE,
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_SEG_MODEL,
    resolve_sam_checkpoint,
    resolve_yolo_weights_reference,
)
from ..realtime.pipeline import RealtimeConfig, RealtimePipeline
from ..realtime.types import PipelineSnapshot


def _number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, name))
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        output[prefix] = value
    return output


class PerformanceBenchmarkPage(ttk.Frame):
    """Fourth-page live benchmark that persists scalar telemetry only."""

    def __init__(
        self,
        master: tk.Misc,
        controller: GaitApplicationController,
        camera_available: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        super().__init__(master)
        self.controller = controller
        shared_i18n = getattr(self.winfo_toplevel(), "_gait_i18n", None)
        self.i18n: I18n = (
            shared_i18n if isinstance(shared_i18n, I18n) else I18n("en")
        )
        self.camera_available = camera_available
        self.pipeline: RealtimePipeline | None = None
        self.recorder: LiveBenchmarkRecorder | None = None
        self._finishing = False
        self._reports: dict[str, dict[str, Any]] = {}
        self._report_paths: dict[str, Path] = {}
        self.bundle_labels: dict[str, str] = {}

        self.bundle_var = tk.StringVar()
        detector_id = str(
            nested(controller.config, "benchmark", "default_detector", "yolo_seg")
        )
        detector_label = {
            "yolo_seg": "YOLO person segmentation",
            "yolo": "YOLO person bbox",
            "yolo_sam": "YOLO + SAM",
        }.get(detector_id, "YOLO person segmentation")
        self.detector_var = tk.StringVar(value=detector_label)
        self.device_var = tk.StringVar(value="auto")
        self.clip_len_var = tk.IntVar(value=15)
        self.stride_var = tk.IntVar(
            value=int(nested(controller.config, "benchmark", "inference_stride", 5))
        )
        self.warmup_var = tk.DoubleVar(
            value=float(nested(controller.config, "benchmark", "warmup_seconds", 10))
        )
        self.duration_var = tk.DoubleVar(
            value=float(nested(controller.config, "benchmark", "duration_seconds", 60))
        )
        self.posture_var = tk.StringVar(
            value=str(nested(controller.config, "benchmark", "posture", "stationary"))
        )
        self.expected_person_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Ready. No RGB, depth, or point-cloud frames will be saved."
        )
        self.phase_var = tk.StringVar(value="Idle")
        self.progress_text_var = tk.StringVar(value="0 / 60 s")
        self.gallery_var = tk.StringVar(value="Current model Gallery: -")
        self.current_values: dict[str, tk.StringVar] = {
            "pipeline": tk.StringVar(value="0.0 FPS"),
            "valid": tk.StringVar(value="0.0 FPS"),
            "read": tk.StringVar(value="0.0 ms"),
            "detect": tk.StringVar(value="0.0 ms"),
            "preprocess": tk.StringVar(value="0.0 ms"),
            "model": tk.StringVar(value="0.0 ms"),
            "gallery": tk.StringVar(value="0.0 ms"),
            "points": tk.StringVar(value="0"),
        }

        self._build()
        self.refresh_bundles()
        self.after(50, self._poll)

    def _tr(self, zh: str, en: str) -> str:
        return zh if self.i18n.locale == "zh_TW" else en

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(8, 4, 8, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.title_label = ttk.Label(
            header,
            text=self._tr("即時效率測試", "Live Performance Benchmark"),
            style="Header.TLabel",
        )
        self.title_label.grid(row=0, column=0, sticky="w")
        self.privacy_label = ttk.Label(
            header,
            text=self._tr(
                "只保存數值統計，不保存 RGB、深度、點雲或特徵",
                "Numeric telemetry only; RGB, depth, point clouds, and embeddings are not saved",
            ),
        )
        self.privacy_label.grid(row=1, column=0, sticky="w", pady=(3, 0))

        controls = ttk.LabelFrame(
            self,
            text=self._tr("測試設定", "Benchmark settings"),
            padding=12,
        )
        controls.grid(row=1, column=0, sticky="ew", padx=8)
        for column in range(8):
            controls.columnconfigure(column, weight=1 if column in {1, 3, 5} else 0)

        ttk.Label(controls, text=self._tr("點雲模型", "Point-cloud model")).grid(
            row=0, column=0, sticky="w"
        )
        self.bundle_combo = ttk.Combobox(
            controls, textvariable=self.bundle_var, state="readonly", width=42
        )
        self.bundle_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(6, 14))
        self.bundle_combo.bind("<<ComboboxSelected>>", lambda _event: self._bundle_changed())

        ttk.Label(controls, text=self._tr("人體偵測", "Person detector")).grid(
            row=0, column=4, sticky="w"
        )
        self.detector_combo = ttk.Combobox(
            controls,
            textvariable=self.detector_var,
            values=(
                "YOLO person segmentation",
                "YOLO person bbox",
                "YOLO + SAM",
            ),
            state="readonly",
            width=25,
        )
        self.detector_combo.grid(row=0, column=5, sticky="ew", padx=(6, 14))

        ttk.Label(controls, text="Device").grid(row=0, column=6, sticky="w")
        self.device_combo = ttk.Combobox(
            controls,
            textvariable=self.device_var,
            values=("auto", "cuda", "cpu"),
            state="readonly",
            width=8,
        )
        self.device_combo.grid(row=0, column=7, sticky="ew", padx=(6, 0))

        ttk.Label(controls, text="Clip length T").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.clip_label = ttk.Label(controls, textvariable=self.clip_len_var)
        self.clip_label.grid(row=1, column=1, sticky="w", padx=(6, 14), pady=(10, 0))
        ttk.Label(controls, text="Inference stride").grid(row=1, column=2, sticky="w", pady=(10, 0))
        self.stride_combo = ttk.Combobox(
            controls,
            textvariable=self.stride_var,
            values=(1, 3, 5, 10),
            state="readonly",
            width=7,
        )
        self.stride_combo.grid(row=1, column=3, sticky="w", padx=(6, 14), pady=(10, 0))
        ttk.Label(controls, text=self._tr("暖機秒數", "Warm-up seconds")).grid(
            row=1, column=4, sticky="w", pady=(10, 0)
        )
        self.warmup_spin = ttk.Spinbox(
            controls, textvariable=self.warmup_var, from_=0, to=60, increment=5, width=8
        )
        self.warmup_spin.grid(row=1, column=5, sticky="w", padx=(6, 14), pady=(10, 0))
        ttk.Label(controls, text=self._tr("測量秒數", "Measure seconds")).grid(
            row=1, column=6, sticky="w", pady=(10, 0)
        )
        self.duration_spin = ttk.Spinbox(
            controls, textvariable=self.duration_var, from_=10, to=600, increment=10, width=8
        )
        self.duration_spin.grid(row=1, column=7, sticky="w", padx=(6, 0), pady=(10, 0))

        ttk.Label(controls, text=self._tr("受測狀態", "Subject state")).grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        posture = ttk.Frame(controls)
        posture.grid(row=2, column=1, columnspan=3, sticky="w", pady=(10, 0))
        self.stationary_radio = ttk.Radiobutton(
            posture,
            text=self._tr("靜坐（只測效率）", "Stationary (efficiency only)"),
            variable=self.posture_var,
            value="stationary",
        )
        self.stationary_radio.grid(row=0, column=0, sticky="w")
        self.walking_radio = ttk.Radiobutton(
            posture,
            text=self._tr("行走（觀察結果行為）", "Walking (observe result behavior)"),
            variable=self.posture_var,
            value="walking",
        )
        self.walking_radio.grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(controls, text=self._tr("預期 Person ID（選填）", "Expected Person ID (optional)")).grid(
            row=2, column=4, sticky="w", pady=(10, 0)
        )
        self.expected_entry = ttk.Entry(controls, textvariable=self.expected_person_var)
        self.expected_entry.grid(row=2, column=5, sticky="ew", padx=(6, 14), pady=(10, 0))
        self.gallery_label = ttk.Label(controls, textvariable=self.gallery_var)
        self.gallery_label.grid(row=2, column=6, columnspan=2, sticky="e", pady=(10, 0))

        actions = ttk.Frame(controls)
        actions.grid(row=3, column=0, columnspan=8, sticky="ew", pady=(12, 0))
        actions.columnconfigure(3, weight=1)
        self.start_button = ttk.Button(
            actions,
            text=self._tr("開始效率測試", "Start benchmark"),
            style="Accent.TButton",
            command=self.start,
        )
        self.start_button.grid(row=0, column=0, sticky="w")
        self.stop_button = ttk.Button(
            actions,
            text=self._tr("停止並保留部分結果", "Stop and keep partial result"),
            command=self.stop,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.export_button = ttk.Button(
            actions,
            text=self._tr("匯出選取報告", "Export selected report"),
            command=self._export_selected,
            state="disabled",
        )
        self.export_button.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=3, sticky="e")

        content = ttk.Panedwindow(self, orient=tk.VERTICAL)
        content.grid(row=2, column=0, sticky="nsew", padx=8, pady=(8, 8))
        live = ttk.LabelFrame(
            content,
            text=self._tr("即時監看（不顯示影像）", "Live telemetry (no images)"),
            padding=10,
        )
        history = ttk.LabelFrame(
            content,
            text=self._tr("本次程式的測試結果", "Benchmark results in this app session"),
            padding=10,
        )
        content.add(live, weight=0)
        content.add(history, weight=1)

        live.columnconfigure(1, weight=1)
        ttk.Label(live, textvariable=self.phase_var, style="Result.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(live, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=1, sticky="ew", padx=(14, 8))
        ttk.Label(live, textvariable=self.progress_text_var).grid(row=0, column=2, sticky="e")
        items = (
            ("End-to-end", "pipeline"),
            ("Valid point cloud", "valid"),
            ("Kinect read", "read"),
            ("Detection", "detect"),
            ("Point preprocess", "preprocess"),
            ("Model encode", "model"),
            ("Gallery match", "gallery"),
            ("Person points", "points"),
        )
        metrics = ttk.Frame(live)
        metrics.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        for index, (title, key) in enumerate(items):
            metrics.columnconfigure(index, weight=1)
            cell = ttk.Frame(metrics, padding=(6, 2))
            cell.grid(row=0, column=index, sticky="ew")
            ttk.Label(cell, text=title).grid(row=0, column=0)
            ttk.Label(cell, textvariable=self.current_values[key], style="Info.TLabel").grid(
                row=1, column=0, pady=(3, 0)
            )

        history.columnconfigure(0, weight=1)
        history.rowconfigure(0, weight=1)
        columns = (
            "time",
            "model",
            "detector",
            "mode",
            "seconds",
            "fps",
            "valid_fps",
            "model_p50",
            "model_p95",
            "gallery_p50",
            "valid_pct",
        )
        self.tree = ttk.Treeview(history, columns=columns, show="headings", height=8)
        headings = (
            "Time",
            "Model",
            "Detector",
            "Mode",
            "Seconds",
            "E2E FPS",
            "Valid FPS",
            "Model P50",
            "Model P95",
            "Gallery P50",
            "Valid %",
        )
        widths = (125, 230, 130, 85, 70, 75, 75, 90, 90, 95, 70)
        for column, heading, width in zip(columns, headings, widths):
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, minwidth=55, anchor="center")
        self.tree.column("model", anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(history, orient="vertical", command=self.tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scroll.set)
        self.detail_var = tk.StringVar(
            value=self._tr(
                "完成測試後，這裡會顯示報告摘要與自動保存位置。",
                "A report summary and automatic save path appear here after completion.",
            )
        )
        ttk.Label(history, textvariable=self.detail_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._show_selected())

    def set_locale(self, _locale_name: str) -> None:
        # Dynamic result text remains intact; static labels are rebuilt by the
        # application-level I18n traversal where message aliases exist.
        self.title_label.configure(text=self._tr("即時效率測試", "Live Performance Benchmark"))
        self.privacy_label.configure(
            text=self._tr(
                "只保存數值統計，不保存 RGB、深度、點雲或特徵",
                "Numeric telemetry only; RGB, depth, point clouds, and embeddings are not saved",
            )
        )

    def refresh_bundles(self) -> None:
        current = self.bundle_labels.get(self.bundle_var.get())
        bundles = self.controller.available_bundles(refresh=True)
        self.bundle_labels = {
            f"{bundle.display_name} | {bundle.checkpoint_sha256[:8]}": bundle_id
            for bundle_id, bundle in bundles.items()
            if bundle.input_type == "pointcloud"
        }
        labels = list(self.bundle_labels)
        self.bundle_combo.configure(values=labels)
        selected = next(
            (label for label, bundle_id in self.bundle_labels.items() if bundle_id == current),
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
                (label for label, bundle_id in self.bundle_labels.items() if bundle_id == preferred),
                labels[0],
            )
        if selected:
            self.bundle_var.set(selected)
            self._bundle_changed()

    def _bundle_id(self) -> str:
        bundle_id = self.bundle_labels.get(self.bundle_var.get())
        if bundle_id is None:
            raise ValueError("Select a point-cloud model")
        return bundle_id

    def _bundle_changed(self) -> None:
        try:
            bundle_id = self._bundle_id()
        except ValueError:
            return
        bundle = self.controller.model_store.get(bundle_id)
        clip_len = int(bundle.data.get("clip_len") or 15)
        self.clip_len_var.set(clip_len)
        summary = self.controller.database_summary(bundle_id, clip_len=clip_len)
        self.gallery_var.set(
            f"Gallery: {summary.get('persons', 0)} people / "
            f"{summary.get('active_embeddings', 0)} embeddings"
        )

    def _detector_settings(self) -> tuple[str, str, str]:
        label = self.detector_var.get()
        default_yolo = str(
            nested(
                self.controller.config,
                "realtime",
                "default_yolo_model",
                DEFAULT_YOLO_MODEL,
            )
        )
        default_seg = str(
            nested(
                self.controller.config,
                "realtime",
                "default_yolo_seg_model",
                DEFAULT_YOLO_SEG_MODEL,
            )
        )
        if label == "YOLO person segmentation":
            return "yolo_seg", resolve_yolo_weights_reference(default_seg), ""
        if label == "YOLO person bbox":
            return "yolo", resolve_yolo_weights_reference(default_yolo), ""
        sam = str(
            nested(
                self.controller.config,
                "realtime",
                "default_sam_checkpoint",
                DEFAULT_SAM_CHECKPOINT,
            )
        )
        return (
            "yolo_sam",
            resolve_yolo_weights_reference(default_yolo),
            resolve_sam_checkpoint(sam),
        )

    def start(self) -> None:
        if self.pipeline is not None and self.pipeline.running:
            return
        if self.camera_available is not None:
            available, reason = self.camera_available()
            if not available:
                messagebox.showwarning("Kinect is busy", reason, parent=self)
                return
        try:
            duration = float(self.duration_var.get())
            warmup = float(self.warmup_var.get())
            if duration <= 0 or warmup < 0:
                raise ValueError("Duration must be positive and warm-up must not be negative")
            bundle_id = self._bundle_id()
            bundle = self.controller.model_store.get(bundle_id)
            detector, yolo_weights, sam_checkpoint = self._detector_settings()
            clip_len = int(bundle.data.get("clip_len") or 15)
            stride = int(self.stride_var.get())
            if stride <= 0:
                raise ValueError("Inference stride must be positive")
        except Exception as exc:
            messagebox.showerror("Invalid benchmark settings", str(exc), parent=self)
            return

        gallery = self.controller.database_summary(bundle_id, clip_len=clip_len)
        metadata = {
            "bundle_id": bundle_id,
            "model_display_name": bundle.display_name,
            "architecture": bundle.architecture,
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "detector": detector,
            "detector_label": self.detector_var.get(),
            "yolo_weights": yolo_weights,
            "sam_checkpoint": sam_checkpoint or None,
            "device_requested": self.device_var.get(),
            "clip_len": clip_len,
            "num_points": int(bundle.data.get("num_points") or 1024),
            "inference_stride": stride,
            "processing_version_id": self.controller.processing_version_id(),
            "gallery_persons": int(gallery.get("persons", 0)),
            "gallery_embeddings": int(gallery.get("active_embeddings", 0)),
            "source": "azure_kinect_live",
        }
        self.recorder = LiveBenchmarkRecorder(
            warmup_seconds=warmup,
            duration_seconds=duration,
            metadata=metadata,
            expected_person_id=self.expected_person_var.get(),
            posture=self.posture_var.get(),
        )
        config = RealtimeConfig(
            bundle_id=bundle_id,
            operation="recognize",
            source_mode="azure_kinect",
            replay_loop=False,
            detector=detector,
            yolo_weights=yolo_weights,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=DEFAULT_SAM_MODEL_TYPE,
            sam_refresh_interval=max(
                1,
                int(
                    nested(
                        self.controller.config,
                        "realtime",
                        "sam_refresh_interval",
                        10,
                    )
                ),
            ),
            detector_confidence=0.35,
            device=self.device_var.get(),
            clip_len=clip_len,
            inference_stride=stride,
            num_points=int(bundle.data.get("num_points") or 1024),
            threshold=self.controller.provisional_threshold(bundle_id, clip_len=clip_len),
            min_margin=float(
                nested(self.controller.config, "recognition", "min_margin", 0.03)
            ),
            preprocessing_profile_id=self.controller.processing_version_id(),
            reject_multiple_people=False,
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
        self.pipeline = RealtimePipeline(
            self.controller,
            config,
            metric_sink=self.recorder.accept,
        )
        self._finishing = False
        self.pipeline.start()
        self._set_running(True)
        self.phase_var.set(self._tr("正在開啟 Kinect 與載入模型", "Opening Kinect and loading model"))
        self.status_var.set(
            self._tr(
                "偵測到有效人體點雲後才會開始暖機計時",
                "Warm-up starts after the first valid person point cloud",
            )
        )
        self.progress.configure(value=0)
        self.progress_text_var.set(f"0 / {duration:.0f} s")

    def stop(self) -> None:
        if self.pipeline is None:
            return
        self._finish(manual=True)

    def close(self) -> None:
        if self.pipeline is not None:
            if self.recorder is not None:
                self.recorder.cancel()
            self.pipeline.stop()
        self.pipeline = None
        self.recorder = None

    def _poll(self) -> None:
        if self.pipeline is not None:
            snapshot = self.pipeline.poll_latest()
            if snapshot is not None:
                self._show_snapshot(snapshot)
                if snapshot.state in {"error", "complete"} and not self._finishing:
                    self.status_var.set(snapshot.message)
                    self._finish(manual=True)
            if self.recorder is not None:
                progress = self.recorder.progress()
                if progress.phase == "waiting_person":
                    self.phase_var.set(self._tr("等待有效人體點雲", "Waiting for a valid person point cloud"))
                    self.progress.configure(value=0)
                elif progress.phase == "warmup":
                    self.phase_var.set(self._tr("暖機中", "Warm-up"))
                    self.progress.configure(value=progress.progress * 100.0)
                    self.progress_text_var.set(
                        f"warm-up {progress.elapsed_s:.1f} / {float(self.warmup_var.get()):.0f} s"
                    )
                else:
                    self.phase_var.set(
                        self._tr("測量完成", "Complete")
                        if progress.phase == "complete"
                        else self._tr("正式測量中", "Measuring")
                    )
                    self.progress.configure(value=progress.progress * 100.0)
                    self.progress_text_var.set(
                        f"{progress.elapsed_s:.1f} / {float(self.duration_var.get()):.0f} s | "
                        f"valid {progress.valid_frame_count}/{progress.frame_count}"
                    )
                if self.recorder.completed and not self._finishing:
                    self.after_idle(lambda: self._finish(manual=False))
        self.after(50, self._poll)

    def _show_snapshot(self, snapshot: PipelineSnapshot) -> None:
        self.status_var.set(snapshot.message)
        self.current_values["pipeline"].set(f"{snapshot.capture_fps:.1f} FPS")
        self.current_values["valid"].set(f"{snapshot.effective_sampling_fps:.1f} FPS")
        self.current_values["read"].set(f"{snapshot.frame_read_ms:.1f} ms")
        self.current_values["detect"].set(f"{snapshot.detection_ms:.1f} ms")
        self.current_values["preprocess"].set(f"{snapshot.preprocessing_ms:.1f} ms")
        self.current_values["model"].set(f"{snapshot.model_encode_ms:.1f} ms")
        self.current_values["gallery"].set(f"{snapshot.gallery_match_ms:.1f} ms")
        self.current_values["points"].set(f"{snapshot.person_point_count:,}")

    def _finish(self, manual: bool) -> None:
        if self._finishing:
            return
        self._finishing = True
        recorder = self.recorder
        pipeline = self.pipeline
        if manual and recorder is not None:
            recorder.cancel()
        if pipeline is not None:
            pipeline.stop()
        report = recorder.report() if recorder is not None else None
        self.pipeline = None
        self.recorder = None
        self._set_running(False)
        if report is None:
            self.phase_var.set(self._tr("已停止，沒有可保存的正式測量資料", "Stopped; no measured samples to save"))
            self.status_var.set(
                self._tr(
                    "測量開始前即停止，因此沒有建立報告",
                    "Stopped before measurement began; no report was created",
                )
            )
        else:
            path = self._save_report(report)
            test_id = str(report["test_id"])
            self._reports[test_id] = report
            self._report_paths[test_id] = path
            self._insert_report(report)
            self.tree.selection_set(test_id)
            self.tree.focus(test_id)
            self._show_selected()
            self.export_button.configure(state="normal")
            self.phase_var.set(
                self._tr("部分結果已保存", "Partial result saved")
                if manual
                else self._tr("測量完成", "Benchmark complete")
            )
            self.status_var.set(f"Telemetry report: {path}")
        self._finishing = False

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "readonly"
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        for widget in (
            self.bundle_combo,
            self.detector_combo,
            self.device_combo,
            self.stride_combo,
        ):
            widget.configure(state=state)
        entry_state = "disabled" if running else "normal"
        for widget in (
            self.warmup_spin,
            self.duration_spin,
            self.expected_entry,
            self.stationary_radio,
            self.walking_radio,
        ):
            widget.configure(state=entry_state)

    def _save_report(self, report: dict[str, Any]) -> Path:
        root = self.controller.output_root / "performance_benchmarks"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{report['test_id']}.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _insert_report(self, report: dict[str, Any]) -> None:
        summary = report["summary"]
        config = report["configuration"]
        latency = report["latency_ms_and_rates"]
        test_id = str(report["test_id"])
        self.tree.insert(
            "",
            "end",
            iid=test_id,
            values=(
                str(report["created_at"])[11:19],
                config.get("model_display_name", config.get("bundle_id", "-")),
                config.get("detector_label", config.get("detector", "-")),
                "seated" if config.get("posture") == "stationary" else "walking",
                _number(summary.get("measured_seconds"), 1),
                _number(summary.get("end_to_end_fps"), 1),
                _number(summary.get("valid_pointcloud_fps"), 1),
                f"{_number(latency['model_encode_ms'].get('p50'), 1)} ms",
                f"{_number(latency['model_encode_ms'].get('p95'), 1)} ms",
                f"{_number(latency['gallery_match_ms'].get('p50'), 1)} ms",
                _number(summary.get("valid_frame_percent"), 1),
            ),
        )

    def _show_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        test_id = str(selection[0])
        report = self._reports.get(test_id)
        if report is None:
            return
        summary = report["summary"]
        latency = report["latency_ms_and_rates"]
        gpu_peak = report.get("gpu_memory", {}).get("peak_allocated_mib")
        path = self._report_paths.get(test_id)
        self.detail_var.set(
            f"model P50/P95={_number(latency['model_encode_ms']['p50'])}/"
            f"{_number(latency['model_encode_ms']['p95'])} ms | "
            f"detection P50={_number(latency['detection_ms']['p50'])} ms | "
            f"updates={_number(summary['recognition_updates_per_second'], 2)}/s | "
            f"valid={_number(summary['valid_frame_percent'])}% | "
            f"GPU peak={_number(gpu_peak)} MiB | {path or '-'}"
        )

    def _export_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Export report", "Select a benchmark row first.", parent=self)
            return
        report = self._reports.get(str(selection[0]))
        if report is None:
            return
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Export numeric benchmark report",
            defaultextension=".json",
            initialfile=f"{report['test_id']}.json",
            filetypes=(("JSON report", "*.json"), ("CSV row", "*.csv")),
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.lower() == ".csv":
            row = _flatten(report)
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
        else:
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self.status_var.set(f"Exported numeric report: {path}")
