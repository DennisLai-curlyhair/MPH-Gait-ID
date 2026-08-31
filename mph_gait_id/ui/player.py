from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk

from PIL import ImageTk

from mph_gait_id.controller import OperationOutcome

from .preview import (
    PreviewViewOptions,
    active_window_result,
    frame_number,
    render_preview_frame,
)


class PreviewPlayer(ttk.Frame):
    def __init__(self, master: tk.Misc, playback_fps: float = 10.0) -> None:
        super().__init__(master, style="Preview.TFrame")
        self.default_fps = max(1.0, float(playback_fps))
        self.frames: tuple[Path, ...] = ()
        self.input_type = "pointcloud"
        self.result: dict[str, Any] = {}
        self.current_index = 0
        self.playing = False
        self._view_options_by_input = {
            "pointcloud": PreviewViewOptions(),
        }
        self.view_options = self._view_options_by_input[self.input_type]
        self._after_id: str | None = None
        self._resize_after_id: str | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._cache: OrderedDict[tuple[Any, ...], ImageTk.PhotoImage] = OrderedDict()

        self.image_label = ttk.Label(
            self,
            text="選擇點雲資料夾後開始註冊／辨識",
            anchor="center",
            style="PreviewImage.TLabel",
        )
        self.image_label.grid(row=0, column=0, columnspan=5, sticky="nsew")
        self.image_label.bind("<Configure>", self._on_preview_resized)

        self.play_button = ttk.Button(self, text="播放", command=self.toggle_play, width=8)
        self.play_button.grid(row=1, column=0, padx=(0, 8), pady=(10, 0))
        self.previous_button = ttk.Button(self, text="上一格", command=self.previous, width=8)
        self.previous_button.grid(row=1, column=1, padx=(0, 8), pady=(10, 0))
        self.next_button = ttk.Button(self, text="下一格", command=self.next, width=8)
        self.next_button.grid(row=1, column=2, padx=(0, 10), pady=(10, 0))

        self.position = tk.DoubleVar(value=0)
        self.overlay_enabled = tk.BooleanVar(value=True)
        self.overlay_position_var = tk.StringVar(value="右上")
        self.overlay_positions = {
            "左上": "top_left",
            "右上": "top_right",
            "左下": "bottom_left",
            "右下": "bottom_right",
        }
        self.slider = ttk.Scale(self, from_=0, to=0, variable=self.position, command=self.seek)
        self.slider.grid(row=1, column=3, sticky="ew", pady=(10, 0))
        self.frame_label = ttk.Label(self, text="0 / 0", width=14, anchor="e")
        self.frame_label.grid(row=1, column=4, padx=(10, 0), pady=(10, 0))

        self.view_controls = ttk.Frame(self, style="Preview.TFrame")
        self.view_controls.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        self.view_controls.columnconfigure(6, weight=1)
        self.view_title_label = ttk.Label(
            self.view_controls,
            text="預覽視角",
            style="Panel.TLabel",
        )
        self.view_title_label.grid(row=0, column=0, sticky="w", padx=(0, 7))
        ttk.Button(
            self.view_controls, text="左轉", width=6, command=lambda: self._rotate(-90)
        ).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(
            self.view_controls, text="右轉", width=6, command=lambda: self._rotate(90)
        ).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(
            self.view_controls,
            text="水平鏡像",
            width=9,
            command=self._flip_horizontal,
        ).grid(row=0, column=3, padx=(0, 5))
        ttk.Button(
            self.view_controls,
            text="垂直翻轉",
            width=9,
            command=self._flip_vertical,
        ).grid(row=0, column=4, padx=(0, 5))
        ttk.Button(
            self.view_controls, text="重設", width=6, command=self._reset_view
        ).grid(row=0, column=5, padx=(0, 8))
        self.view_state_label = ttk.Label(
            self.view_controls,
            text="",
            style="Muted.Panel.TLabel",
            anchor="e",
        )
        self.view_state_label.grid(row=0, column=6, sticky="e")

        ttk.Label(self.view_controls, text="畫面位置", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 7), pady=(5, 0)
        )
        ttk.Button(
            self.view_controls, text="左", width=5, command=lambda: self._pan(-24, 0)
        ).grid(row=1, column=1, padx=(0, 5), pady=(5, 0))
        ttk.Button(
            self.view_controls, text="右", width=5, command=lambda: self._pan(24, 0)
        ).grid(row=1, column=2, padx=(0, 5), pady=(5, 0))
        ttk.Button(
            self.view_controls, text="上", width=5, command=lambda: self._pan(0, -24)
        ).grid(row=1, column=3, padx=(0, 5), pady=(5, 0))
        ttk.Button(
            self.view_controls, text="下", width=5, command=lambda: self._pan(0, 24)
        ).grid(row=1, column=4, padx=(0, 5), pady=(5, 0))
        ttk.Button(
            self.view_controls, text="放大", width=6, command=lambda: self._zoom(1.2)
        ).grid(row=1, column=5, padx=(0, 5), pady=(5, 0))
        ttk.Button(
            self.view_controls, text="縮小", width=6, command=lambda: self._zoom(1 / 1.2)
        ).grid(row=1, column=6, sticky="w", pady=(5, 0))

        ttk.Checkbutton(
            self.view_controls,
            text="顯示畫面標註",
            variable=self.overlay_enabled,
            command=self._overlay_changed,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(
            self.view_controls,
            text="標註位置",
            style="Panel.TLabel",
        ).grid(row=2, column=3, sticky="e", pady=(6, 0))
        self.overlay_position_combo = ttk.Combobox(
            self.view_controls,
            textvariable=self.overlay_position_var,
            values=list(self.overlay_positions),
            state="readonly",
            width=7,
        )
        self.overlay_position_combo.grid(
            row=2,
            column=4,
            columnspan=2,
            sticky="w",
            pady=(6, 0),
        )
        self.overlay_position_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._overlay_changed(),
        )
        self.view_controls.grid_remove()
        self._update_view_state_label()

        self.columnconfigure(3, weight=1)
        self.rowconfigure(0, weight=1)

    def load(self, outcome: OperationOutcome) -> None:
        self.stop()
        self.frames = outcome.prepared_source.preview_frames
        self.input_type = outcome.prepared_source.input_type
        self.view_options = self._view_options_by_input.setdefault(
            self.input_type,
            PreviewViewOptions(),
        )
        self._update_view_state_label()
        self.result = outcome.result
        self.current_index = 0
        self._cache.clear()
        maximum = max(0, len(self.frames) - 1)
        self.slider.configure(to=maximum)
        self.position.set(0)
        self.view_title_label.configure(text="點雲視角")
        self.view_controls.grid()
        self.show_index(0)

    def clear(self) -> None:
        self.stop()
        self.frames = ()
        self.result = {}
        self._cache.clear()
        self.image_label.configure(image="", text="尚無可視化結果")
        self.frame_label.configure(text="0 / 0")
        self.view_controls.grid_remove()

    def toggle_play(self) -> None:
        if not self.frames:
            return
        if self.playing:
            self.stop()
        else:
            if self.current_index >= len(self.frames) - 1:
                self.current_index = 0
            self.playing = True
            self.play_button.configure(text="暫停")
            self._schedule_next()

    def stop(self) -> None:
        self.playing = False
        self.play_button.configure(text="播放")
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def previous(self) -> None:
        self.stop()
        self.show_index(max(0, self.current_index - 1))

    def next(self) -> None:
        self.stop()
        self.show_index(min(max(0, len(self.frames) - 1), self.current_index + 1))

    def seek(self, value: str) -> None:
        if not self.frames:
            return
        self.stop()
        self.show_index(int(round(float(value))))

    def show_index(self, index: int) -> None:
        if not self.frames:
            return
        self.current_index = max(0, min(int(index), len(self.frames) - 1))
        self.position.set(self.current_index)
        path = self.frames[self.current_index]
        number = frame_number(path)
        overlay = self._overlay(number) if self.overlay_enabled.get() else None
        overlay_position = self.overlay_positions.get(
            self.overlay_position_var.get(),
            "top_right",
        )
        if overlay is not None:
            overlay = {**overlay, "position": overlay_position}
        width = max(320, self.image_label.winfo_width())
        height = max(240, self.image_label.winfo_height())
        view_key = self.view_options.cache_key()
        cache_key = (
            self.current_index,
            width,
            height,
            *view_key,
            bool(self.overlay_enabled.get()),
            overlay_position,
        )
        photo = self._cache.get(cache_key)
        if photo is None:
            image = render_preview_frame(
                path=path,
                input_type=self.input_type,
                target_size=(width, height),
                overlay=overlay,
                pointcloud_view=self.view_options,
            )
            photo = ImageTk.PhotoImage(image)
            self._cache[cache_key] = photo
            while len(self._cache) > 24:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(cache_key)
        self._photo = photo
        self.image_label.configure(image=photo, text="")
        suffix = f"frame {number}" if number is not None else path.name
        self.frame_label.configure(
            text=f"{self.current_index + 1} / {len(self.frames)}  {suffix}"
        )

    def _overlay(self, number: int | None) -> dict[str, Any]:
        if self.result.get("operation") == "enroll":
            coherence = float(self.result.get("embedding_coherence", 0.0))
            source_count = int(self.result.get("source_count", 1))
            return {
                "state": "enroll",
                "lines": [
                    "Enrollment preview",
                    f"ID: {self.result.get('person_id', '-')}",
                    f"Sources: {source_count} | coherence: {coherence:.3f}",
                ],
            }

        windows = list(self.result.get("window_results", []))
        active = active_window_result(windows, number, self.current_index)
        if active is None:
            return {
                "state": "idle",
                "lines": [
                    "Collecting frames",
                    "Prediction starts after the first complete clip",
                ],
            }
        return {
            "state": str(self.result.get("state", "closed_set")),
            "lines": [
                f"Window {active.get('window_index', '-')}",
                f"Candidate: {active.get('display_name', active.get('person_id', '-'))}",
                f"Similarity: {float(active.get('similarity', 0.0)):.3f}",
            ],
        }

    def _overlay_changed(self) -> None:
        self._cache.clear()
        if self.frames:
            self.show_index(self.current_index)

    def _schedule_next(self) -> None:
        if not self.playing or not self.frames:
            return
        if self.current_index >= len(self.frames) - 1:
            self.stop()
            return
        self.show_index(self.current_index + 1)
        fps = self.default_fps
        source_fps = self.result.get("source_adapter", {}).get("fps")
        if source_fps:
            fps = max(1.0, min(30.0, float(source_fps)))
        delay_ms = max(20, int(round(1000.0 / fps)))
        self._after_id = self.after(delay_ms, self._schedule_next)

    def _on_preview_resized(self, _event: tk.Event[tk.Misc]) -> None:
        if not self.frames:
            return
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(140, self._render_after_resize)

    def _render_after_resize(self) -> None:
        self._resize_after_id = None
        if self.frames:
            self.show_index(self.current_index)

    def _rotate(self, amount: int) -> None:
        self._set_view(rotation=(self.view_options.rotation + amount) % 360)

    def _flip_horizontal(self) -> None:
        self._set_view(flip_horizontal=not self.view_options.flip_horizontal)

    def _flip_vertical(self) -> None:
        self._set_view(flip_vertical=not self.view_options.flip_vertical)

    def _pan(self, x: int, y: int) -> None:
        self._set_view(
            pan_x=self.view_options.pan_x + int(x),
            pan_y=self.view_options.pan_y + int(y),
        )

    def _zoom(self, factor: float) -> None:
        zoom = max(0.25, min(4.0, self.view_options.zoom * float(factor)))
        self._set_view(zoom=zoom)

    def _reset_view(self) -> None:
        self.view_options = PreviewViewOptions()
        self._view_options_by_input[self.input_type] = self.view_options
        self._refresh_view()

    def _set_view(self, **changes: Any) -> None:
        self.view_options = replace(self.view_options, **changes)
        self._view_options_by_input[self.input_type] = self.view_options
        self._refresh_view()

    def _refresh_view(self) -> None:
        self._cache.clear()
        self._update_view_state_label()
        if self.frames:
            self.show_index(self.current_index)

    def _update_view_state_label(self) -> None:
        horizontal = "鏡像" if self.view_options.flip_horizontal else "原向"
        vertical = "翻轉" if self.view_options.flip_vertical else "原向"
        self.view_state_label.configure(
            text=(
                f"{self.view_options.rotation % 360}° | H {horizontal} | "
                f"V {vertical} | {self.view_options.zoom:.2f}x"
            )
        )


