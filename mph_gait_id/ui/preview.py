from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class PreviewViewOptions:
    """Display-only controls; model preprocessing is intentionally unaffected."""

    rotation: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    pan_x: int = 0
    pan_y: int = 0
    zoom: float = 1.0

    def cache_key(self) -> tuple[int, bool, bool, int, int, float]:
        return (
            int(self.rotation) % 360,
            bool(self.flip_horizontal),
            bool(self.flip_vertical),
            int(self.pan_x),
            int(self.pan_y),
            round(float(self.zoom), 3),
        )


# Backward-compatible name for callers created before image controls were added.
PointCloudViewOptions = PreviewViewOptions


def frame_number(path: str | Path) -> int | None:
    match = NUMBER_PATTERN.search(Path(path).stem)
    return int(match.group(1)) if match else None


def active_window_result(
    window_results: list[dict[str, Any]],
    current_frame_number: int | None,
    current_index: int,
) -> dict[str, Any] | None:
    completed = []
    for item in window_results:
        end_number = item.get("end_frame_number")
        end_index = item.get("end_index")
        if current_frame_number is not None and end_number is not None:
            if int(end_number) <= current_frame_number:
                completed.append(item)
        elif end_index is not None and int(end_index) <= current_index:
            completed.append(item)
    return completed[-1] if completed else None


def render_preview_frame(
    path: str | Path,
    input_type: str,
    target_size: tuple[int, int],
    overlay: dict[str, Any] | None = None,
    pointcloud_view: PreviewViewOptions | None = None,
) -> Image.Image:
    source = Path(path)
    if input_type != "pointcloud":
        raise ValueError(f"Preview only supports point-cloud input, got: {input_type}")
    image = render_pointcloud(source, target_size, pointcloud_view)
    if overlay:
        draw_overlay(image, overlay)
    return image


def render_pointcloud(
    path: Path,
    target_size: tuple[int, int],
    view: PreviewViewOptions | None = None,
) -> Image.Image:
    view = view or PreviewViewOptions()
    width, height = max(320, target_size[0]), max(240, target_size[1])
    canvas = np.full((height, width, 3), (24, 27, 31), dtype=np.uint8)
    points = np.asarray(np.load(path), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Invalid point-cloud frame: {path}")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return Image.fromarray(canvas, mode="RGB")

    x_values = points[:, 0].copy()
    y_values = points[:, 1].copy()
    z_values = points[:, 2]
    x_values -= float(np.median(x_values))
    y_values -= float(np.median(y_values))
    if view.flip_horizontal:
        x_values *= -1.0
    if view.flip_vertical:
        y_values *= -1.0

    rotation = int(view.rotation) % 360
    if rotation == 90:
        x_values, y_values = -y_values.copy(), x_values.copy()
    elif rotation == 180:
        x_values, y_values = -x_values, -y_values
    elif rotation == 270:
        x_values, y_values = y_values.copy(), -x_values.copy()
    elif rotation != 0:
        raise ValueError("Point-cloud preview rotation must be a multiple of 90 degrees")

    x_low, x_high = np.percentile(x_values, [1, 99])
    y_low, y_high = np.percentile(y_values, [1, 99])
    if x_high <= x_low:
        x_high = x_low + 1.0
    if y_high <= y_low:
        y_high = y_low + 1.0

    margin_x = max(24, int(width * 0.08))
    margin_y = max(24, int(height * 0.08))
    available_width = max(1, width - 2 * margin_x)
    available_height = max(1, height - 2 * margin_y)
    scale = min(
        available_width / max(float(x_high - x_low), 1e-6),
        available_height / max(float(y_high - y_low), 1e-6),
    )
    scale *= max(0.25, min(4.0, float(view.zoom)))
    visible_x_center = (float(x_low) + float(x_high)) * 0.5
    visible_y_center = (float(y_low) + float(y_high)) * 0.5
    px_float = width * 0.5 + (x_values - visible_x_center) * scale + int(view.pan_x)
    # Azure-style point clouds use positive Y toward the bottom of the image.
    py_float = height * 0.5 + (y_values - visible_y_center) * scale + int(view.pan_y)
    visible = (
        (px_float >= 0)
        & (px_float < width)
        & (py_float >= 0)
        & (py_float < height)
    )
    px = np.clip(np.rint(px_float[visible]), 0, width - 1).astype(np.int32)
    py = np.clip(np.rint(py_float[visible]), 0, height - 1).astype(np.int32)
    z_values = z_values[visible]
    if not len(px):
        return Image.fromarray(canvas, mode="RGB")

    z_low, z_high = np.percentile(z_values, [2, 98])
    depth = (z_values - z_low) / max(1e-6, z_high - z_low)
    depth = np.clip(depth, 0.0, 1.0)
    colors = np.stack(
        [
            50 + 180 * depth,
            190 - 75 * depth,
            230 - 120 * depth,
        ],
        axis=1,
    ).astype(np.uint8)
    canvas[py, px] = colors
    occupied = np.zeros((height, width), dtype=np.uint8)
    occupied[py, px] = 255
    expanded = cv2.dilate(occupied, np.ones((3, 3), np.uint8), iterations=1)
    edge = (expanded > 0) & (occupied == 0)
    canvas[edge] = np.maximum(canvas[edge], np.array([55, 90, 100], dtype=np.uint8))

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    orientation = (
        f"Front view | {len(points)} points | {rotation} deg | "
        f"zoom {float(view.zoom):.2f}x"
    )
    draw.text((14, height - 24), orientation, fill=(205, 213, 221))
    return image


def render_image(
    image: Image.Image,
    target_size: tuple[int, int],
    view: PreviewViewOptions | None = None,
) -> Image.Image:
    view = view or PreviewViewOptions()
    width, height = max(320, target_size[0]), max(240, target_size[1])
    source = image.convert("RGB")

    rotation = int(view.rotation) % 360
    if rotation == 90:
        source = source.transpose(Image.Transpose.ROTATE_270)
    elif rotation == 180:
        source = source.transpose(Image.Transpose.ROTATE_180)
    elif rotation == 270:
        source = source.transpose(Image.Transpose.ROTATE_90)
    elif rotation != 0:
        raise ValueError("Image preview rotation must be a multiple of 90 degrees")

    if view.flip_horizontal:
        source = ImageOps.mirror(source)
    if view.flip_vertical:
        source = ImageOps.flip(source)

    base_scale = min(width / max(1, source.width), height / max(1, source.height))
    scale = base_scale * max(0.25, min(4.0, float(view.zoom)))
    resized = source.resize(
        (
            max(1, int(round(source.width * scale))),
            max(1, int(round(source.height * scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), (24, 27, 31))
    left = (width - resized.width) // 2 + int(view.pan_x)
    top = (height - resized.height) // 2 + int(view.pan_y)
    canvas.paste(resized, (left, top))
    return canvas


def letterbox(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    return render_image(image, target_size)


def draw_overlay(image: Image.Image, overlay: dict[str, Any]) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    lines = [str(value) for value in overlay.get("lines", []) if value]
    if not lines:
        return

    max_text_width = max(80, min(360, image.width - 52))

    def fit_text(value: str) -> str:
        if draw.textbbox((0, 0), value, font=font)[2] <= max_text_width:
            return value
        suffix = "..."
        candidate = value
        while candidate and draw.textbbox(
            (0, 0),
            candidate + suffix,
            font=font,
        )[2] > max_text_width:
            candidate = candidate[:-1]
        return candidate.rstrip() + suffix

    lines = [fit_text(value) for value in lines]
    state = str(overlay.get("state", "idle"))
    accent = {
        "stable": (36, 150, 95, 255),
        "closed_set": (34, 132, 190, 255),
        "low_confidence": (210, 139, 24, 255),
        "unknown": (190, 62, 62, 255),
        "enroll": (83, 94, 112, 255),
    }.get(state, (83, 94, 112, 255))

    text_width = max(
        draw.textbbox((0, 0), value, font=font)[2]
        for value in lines
    )
    line_height = 17
    box_width = min(image.width - 24, text_width + 28)
    box_height = 18 + line_height * len(lines)
    margin = 12
    position = str(overlay.get("position", "top_right"))
    left = margin if position.endswith("left") else image.width - margin - box_width
    top = margin if position.startswith("top") else image.height - margin - box_height
    right = left + box_width
    bottom = top + box_height

    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=5,
        fill=(12, 15, 18, 168),
        outline=accent,
        width=2,
    )
    for index, line in enumerate(lines):
        draw.text(
            (left + 14, top + 10 + index * line_height),
            line,
            fill=(245, 247, 249, 255),
            font=font,
        )
