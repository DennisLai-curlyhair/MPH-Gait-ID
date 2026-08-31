from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from .dataio import collect_pointcloud_frames


FINGERPRINT_ALGORITHM = "gait-source-sha256-v1"
READ_CHUNK_SIZE = 1024 * 1024


def _update_file_digest(digest: Any, path: Path) -> None:
    size = path.stat().st_size
    digest.update(int(size).to_bytes(8, byteorder="big", signed=False))
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_SIZE):
            digest.update(chunk)


def fingerprint_files(paths: Iterable[str | Path], source_kind: str) -> str:
    """Hash ordered file content without including absolute paths or file names."""

    files = [Path(path).expanduser().resolve() for path in paths]
    if not files:
        raise ValueError("Cannot fingerprint an empty source")

    digest = hashlib.sha256()
    digest.update(FINGERPRINT_ALGORITHM.encode("ascii"))
    digest.update(b"\0")
    digest.update(source_kind.encode("ascii"))
    digest.update(b"\0")
    digest.update(len(files).to_bytes(8, byteorder="big", signed=False))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Fingerprint input file does not exist: {path}")
        _update_file_digest(digest, path)
    return f"sha256:{digest.hexdigest()}"


def compute_source_fingerprint(
    source: str | Path,
    input_type: str,
    mode: str,
) -> str:
    """Create a content identity for a model-compatible point-cloud folder."""

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"Fingerprint source does not exist: {source_path}")

    if input_type == "pointcloud":
        frames = collect_pointcloud_frames(source_path)
        return fingerprint_files(frames, source_kind="pointcloud-frames")
    raise ValueError(f"Unsupported fingerprint input type: {input_type}")
