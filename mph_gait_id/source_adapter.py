from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataio import collect_pointcloud_frames
from .model_store import ModelBundle
from .source_fingerprint import fingerprint_files


def discover_sequence_folders(
    root: str | Path,
    bundle: ModelBundle,
) -> list[Path]:
    """Find direct or nested sequence folders compatible with a model bundle."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ValueError(f"Sequence root must be a directory: {root_path}")

    if bundle.input_type != "pointcloud":
        raise ValueError("Only point-cloud sequence folders are supported")
    folders = {
        path.parent
        for path in root_path.rglob("clear_data_*.npy")
        if path.is_file()
    }
    return sorted(folders, key=lambda path: str(path).lower())


@dataclass(frozen=True)
class PreparedSource:
    original_path: Path
    inference_path: Path
    preview_frames: tuple[Path, ...]
    source_kind: str
    input_type: str
    mode: str
    fps: float
    source_fingerprint: str

    def metadata(self) -> dict[str, Any]:
        return {
            "original_source": str(self.original_path),
            "inference_source": str(self.inference_path),
            "source_kind": self.source_kind,
            "input_type": self.input_type,
            "mode": self.mode,
            "fps": self.fps,
            "preview_frame_count": len(self.preview_frames),
            "source_fingerprint": self.source_fingerprint,
        }


class SourcePreparer:
    """Prepare a point-cloud folder without changing model preprocessing."""

    def __init__(self, folder_fps: float = 10.0) -> None:
        self.folder_fps = max(1.0, float(folder_fps))

    def prepare(self, source: str | Path, bundle: ModelBundle) -> PreparedSource:
        source_path = Path(source).expanduser().resolve()
        if source_path.is_dir():
            return self._prepare_folder(source_path, bundle)
        if source_path.is_file():
            raise ValueError("Only organized point-cloud folders are accepted")
        raise FileNotFoundError(f"Input source does not exist: {source_path}")

    def _prepare_folder(self, source: Path, bundle: ModelBundle) -> PreparedSource:
        if bundle.input_type != "pointcloud":
            raise ValueError("Only point-cloud model bundles are supported")
        frames = collect_pointcloud_frames(source)
        return PreparedSource(
            original_path=source,
            inference_path=source,
            preview_frames=tuple(frames),
            source_kind="folder",
            input_type=bundle.input_type,
            mode=bundle.mode,
            fps=self.folder_fps,
            source_fingerprint=fingerprint_files(frames, "pointcloud-frames"),
        )
