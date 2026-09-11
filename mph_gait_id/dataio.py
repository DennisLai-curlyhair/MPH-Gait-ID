from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


POINT_FRAME_RE = re.compile(r"clear_data_(\d+)\.npy", re.IGNORECASE)


def frame_number(path: Path) -> int | float:
    match = POINT_FRAME_RE.fullmatch(path.name)
    return int(match.group(1)) if match else math.inf


def collect_pointcloud_frames(source: str | Path) -> list[Path]:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise ValueError(f"Point-cloud input must be an NPY sequence directory: {source_path}")
    frames = sorted(
        (path for path in source_path.glob("clear_data_*.npy")
         if POINT_FRAME_RE.fullmatch(path.name)), key=frame_number
    )
    if not frames:
        raise FileNotFoundError(f"No clear_data_*.npy frames found in {source_path}")
    return frames


@dataclass(frozen=True)
class FrameWindow:
    window_index: int
    start_index: int
    end_index: int
    start_frame_number: int | None
    end_frame_number: int | None
    frame_paths: tuple[Path, ...]


def build_windows(
    frame_paths: list[Path],
    window_size: int,
    stride: int,
    drop_first_frames: int,
    include_last: bool = True,
) -> list[FrameWindow]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    usable = list(frame_paths[max(0, int(drop_first_frames)) :])
    if not usable:
        raise ValueError("No frames remain after drop_first_frames")

    if len(usable) < window_size:
        raise ValueError(
            f"Insufficient frames after drop_first_frames: got {len(usable)}, "
            f"need {window_size}. Short sequences are not padded for enrollment or identification."
        )
    else:
        starts = list(range(0, len(usable) - window_size + 1, stride))
        last_start = len(usable) - window_size
        if include_last and starts[-1] != last_start:
            starts.append(last_start)
        selections = [usable[start : start + window_size] for start in starts]

    windows: list[FrameWindow] = []
    for index, (start, selected) in enumerate(zip(starts, selections)):
        first_number = frame_number(selected[0])
        last_number = frame_number(selected[-1])
        windows.append(
            FrameWindow(
                window_index=index,
                start_index=start,
                end_index=min(start + window_size - 1, len(usable) - 1),
                start_frame_number=None if not np.isfinite(first_number) else int(first_number),
                end_frame_number=None if not np.isfinite(last_number) else int(last_number),
                frame_paths=tuple(selected),
            )
        )
    return windows


class _WindowDataset(Dataset):
    def window_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "window_index": window.window_index,
                "start_index": window.start_index,
                "end_index": window.end_index,
                "start_frame_number": window.start_frame_number,
                "end_frame_number": window.end_frame_number,
                "first_frame": str(window.frame_paths[0]),
                "last_frame": str(window.frame_paths[-1]),
            }
            for window in self.windows
        ]


class PointCloudWindowDataset(_WindowDataset):
    def __init__(
        self,
        source: str | Path,
        num_points: int,
        window_size: int,
        stride: int,
        drop_first_frames: int,
    ) -> None:
        self.source = Path(source).expanduser().resolve()
        self.num_points = int(num_points)
        if self.num_points < 2:
            raise ValueError("num_points must be at least 2")
        self.windows = build_windows(
            collect_pointcloud_frames(self.source),
            window_size=int(window_size),
            stride=int(stride),
            drop_first_frames=int(drop_first_frames),
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _load_points(self, path: Path) -> torch.Tensor:
        raw = np.load(path, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] < 3 or raw.dtype.kind not in "fiu":
            raise ValueError(f"Expected a numeric [N,C>=3] point array: {path}")
        points = np.asarray(raw[:, :3], dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 2 or not np.any(np.ptp(points, axis=0) > 0):
            raise ValueError(f"Frame requires at least two distinct finite XYZ points: {path}")
        if len(points) >= self.num_points:
            indices = np.linspace(0, len(points) - 1, self.num_points).round().astype(np.int64)
        else:
            repeats = int(math.ceil(self.num_points / len(points)))
            indices = np.tile(np.arange(len(points), dtype=np.int64), repeats)[: self.num_points]
        return torch.from_numpy(points[indices].copy()).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        return {
            "points": torch.stack([self._load_points(path) for path in window.frame_paths]),
            "path": str(self.source),
        }


def build_window_dataset(
    source: str | Path,
    input_type: str,
    mode: str,
    channels: int,
    data_config: dict[str, Any],
    window_size: int,
    stride: int,
    drop_first_frames: int,
) -> Dataset:
    if input_type == "pointcloud":
        return PointCloudWindowDataset(
            source=source,
            num_points=int(data_config.get("num_points", 1024)),
            window_size=window_size,
            stride=stride,
            drop_first_frames=drop_first_frames,
        )
    raise ValueError(f"Unsupported input type: {input_type}")
