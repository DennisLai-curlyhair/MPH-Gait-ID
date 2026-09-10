"""Opt-in, model-independent foreground recordings with transactional Gallery links."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import threading
from typing import Any
import uuid

import numpy as np

from .database import GalleryRepository, _utc_now
from .gallery_provenance import initialize_provenance, portable_uid
from .model_store import sha256_file as _sha


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


class _Lease:
    """Single writer across app instances; OS releases the lock after a crash."""

    def __init__(self, root: Path) -> None:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.handle = (root / ".writer.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0, 2)
                if not self.handle.tell():
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError("Enrollment source library is busy in another session") from exc

    def close(self) -> None:
        if not self.handle.closed:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class EnrollmentSourceLibrary:
    def __init__(self, repository: GalleryRepository) -> None:
        self.repository = repository
        self.root = repository.path.parent / (repository.path.name + "_enrollment_sources")

    def initialize(self) -> None:
        self.repository.initialize()
        initialize_provenance(self.repository)
        with self.repository.connect() as con:
            # Source owners survive Gallery deletion. Never rebind by a reused Person ID.
            con.execute("""CREATE TABLE IF NOT EXISTS enrollment_sources (
                source_id TEXT PRIMARY KEY, person_uid TEXT NOT NULL,
                captured_person_id TEXT NOT NULL, captured_name TEXT NOT NULL,
                session_id TEXT NOT NULL, created_at TEXT NOT NULL,
                frame_count INTEGER NOT NULL, pass_count INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL, manifest_sha256 TEXT NOT NULL)""")

    def list_sources(self) -> list[dict[str, Any]]:
        self.initialize()
        with self.repository.connect() as con:
            return [dict(row) for row in con.execute("""
                SELECT s.*, p.person_id AS current_person_id, p.display_name AS current_name
                FROM enrollment_sources s
                LEFT JOIN gallery_transfer_aliases a ON a.kind='person' AND a.uid=s.person_uid
                LEFT JOIN persons p ON p.person_id=a.local_id
                ORDER BY s.created_at DESC, s.source_id
            """)]

    def record_dir(self, source_id: str) -> Path:
        if str(uuid.UUID(source_id)) != source_id:
            raise ValueError("Invalid enrollment source ID")
        path = self.root / "records" / source_id
        if path.is_symlink() or path.resolve().parent != (self.root / "records").resolve():
            raise ValueError("Unsafe enrollment source path")
        return path

    def manifest(self, source_id: str) -> dict[str, Any]:
        with self.repository.connect() as con:
            row = con.execute("SELECT manifest_sha256 FROM enrollment_sources WHERE source_id=?",
                              (source_id,)).fetchone()
        if row is None:
            raise ValueError("Enrollment source was deleted or is not committed")
        path = self.record_dir(source_id) / "manifest.json"
        if path.is_symlink() or _sha(path) != row[0]:
            raise ValueError("Enrollment source manifest failed SHA256 verification")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "mph-foreground-source-v1" or data.get("source_id") != source_id:
            raise ValueError("Unsupported enrollment source manifest")
        return data

    def load_frame(self, source_id: str, frame: dict[str, Any]) -> np.ndarray:
        name = str(frame["file"])
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("Invalid source frame path")
        path = self.record_dir(source_id) / name
        if path.is_symlink() or _sha(path) != frame["sha256"]:
            raise ValueError("Source frame failed SHA256 verification")
        points = np.load(path, allow_pickle=False)
        if (points.dtype != np.dtype("float32") or points.shape != (int(frame["points"]), 3)
                or not np.isfinite(points).all()):
            raise ValueError("Invalid foreground XYZ array")
        return points

    def delete(self, source_id: str) -> None:
        """Delete only source media. Gallery embeddings and transfer history remain."""
        with _Lease(self.root):
            directory = self.record_dir(source_id)
            trash = self.root / "trash" / source_id
            trash.parent.mkdir(parents=True, exist_ok=True)
            moved = False
            try:
                with self.repository.connect() as con:
                    con.execute("BEGIN IMMEDIATE")
                    if con.execute("SELECT 1 FROM enrollment_sources WHERE source_id=?",
                                   (source_id,)).fetchone() is None:
                        raise ValueError("Enrollment source no longer exists")
                    if directory.exists():
                        directory.rename(trash)
                        moved = True
                    con.execute("DELETE FROM enrollment_sources WHERE source_id=?", (source_id,))
            except Exception:
                if moved:
                    trash.rename(directory)
                raise
            if moved:
                shutil.rmtree(trash)

    def usage(self) -> dict[str, int]:
        rows = self.list_sources()
        committed = sum(int(row["size_bytes"]) for row in rows)
        total = _size(self.root) if self.root.exists() else 0
        return dict(committed_bytes=committed, disk_bytes=total,
                    uncommitted_bytes=max(0, total - committed), source_count=len(rows))

    def cleanup_uncommitted(self) -> None:
        """Explicit crash cleanup, never run automatically while review may resume."""
        self.initialize()
        with _Lease(self.root):
            known = {row["source_id"] for row in self.list_sources()}
            # Recover deletion interrupted before its SQLite commit.
            for path in (self.root / "trash").glob("*"):
                if path.name in known and not self.record_dir(path.name).exists():
                    path.rename(self.record_dir(path.name))
                elif path.name not in known and path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
            for path in (self.root / "staging").glob("*"):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
            for path in (self.root / "records").glob("*"):
                if path.name not in known and path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)


class ForegroundCapture:
    """Bounded asynchronous writer. Capture errors invalidate source-backed enrollment."""

    def __init__(self, library: EnrollmentSourceLibrary, *, session_limit_bytes: int = 2 << 30,
                 library_limit_bytes: int = 20 << 30, min_free_bytes: int = 512 << 20,
                 queue_size: int = 8, max_frame_points: int = 2_000_000,
                 max_frames: int = 9000) -> None:
        if min(session_limit_bytes, library_limit_bytes, queue_size, max_frame_points, max_frames) <= 0:
            raise ValueError("Source storage limits must be positive")
        if min_free_bytes < 0:
            raise ValueError("Minimum free space cannot be negative")
        library.initialize()
        self.library = library
        self.lease = _Lease(library.root)
        self.source_id = str(uuid.uuid4())
        self.stage = library.root / "staging" / self.source_id
        try:
            self.stage.mkdir(mode=0o700, parents=True)
            self.initial_bytes = _size(library.root)
        except Exception:
            self.lease.close()
            raise
        self.session_limit = int(session_limit_bytes)
        self.library_limit = int(library_limit_bytes)
        self.min_free = int(min_free_bytes)
        self.max_frame_points = int(max_frame_points)
        self.max_frames = int(max_frames)
        self.bytes_written = 0
        self.next_index = 0
        self.last_packet: tuple[str, int, float] | None = None
        self.frames: list[dict[str, Any]] = []
        self.queue: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self.thread: threading.Thread | None = None
        self.error: Exception | None = None
        self.closed = False

    def check(self) -> None:
        if self.closed:
            raise RuntimeError("Foreground capture is closed")
        if self.error is not None:
            raise RuntimeError(f"Foreground recording failed; enrollment not saved: {self.error}") from self.error

    def append(self, pass_id: str, points: np.ndarray, *, frame_index: int,
               timestamp: float, segment_start: bool, metadata: dict[str, Any]) -> None:
        self.check()
        values = np.asarray(points)
        if (values.ndim != 2 or values.shape[1] != 3 or not 0 < len(values) <= self.max_frame_points
                or not np.isfinite(values).all() or not np.isfinite(timestamp)):
            raise ValueError("Foreground recording requires finite, nonempty XYZ and timestamp")
        if self.next_index >= self.max_frames:
            self.error = ValueError("Foreground capture frame limit reached")
            self.check()
        previous = self.last_packet
        segment_start = (segment_start or previous is None or previous[0] != pass_id
                         or frame_index <= previous[1] or timestamp <= previous[2])
        record = dict(file=f"frame_{self.next_index:06d}.npy", pass_id=pass_id,
                      sensor_frame_index=int(frame_index), timestamp=float(timestamp),
                      segment_start=bool(segment_start), points=len(values), metadata=metadata)
        record = json.loads(_json(record))
        values = np.array(values, dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Foreground coordinates exceed FP32 range")
        if self.thread is None:
            self.thread = threading.Thread(target=self._write_loop, name="foreground-source-writer", daemon=True)
            self.thread.start()
        try:
            self.queue.put_nowait((record, values))
        except queue.Full:
            self.error = RuntimeError("Source writer queue is full; disk cannot keep up")
            self.check()
        self.next_index += 1
        self.last_packet = (pass_id, int(frame_index), float(timestamp))

    def _write_loop(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is not None:
                    continue
                record, points = item
                reserve = points.nbytes + len(_json(record)) + 4096
                if self.bytes_written + reserve > self.session_limit:
                    raise OSError("Source recording exceeds the per-session storage limit")
                if self.initial_bytes + self.bytes_written + reserve > self.library_limit:
                    raise OSError("Source library storage limit reached")
                if shutil.disk_usage(self.stage).free < self.min_free + reserve:
                    raise OSError("Insufficient free disk space for source recording")
                path = self.stage / record["file"]
                with path.open("xb") as handle:
                    np.save(handle, points, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                record.update(sha256=_sha(path), size_bytes=path.stat().st_size)
                line = _json(record) + b"\n"
                with (self.stage / "frames.jsonl").open("ab") as handle:
                    handle.write(line)
                self.bytes_written += int(record["size_bytes"]) + len(line)
                self.frames.append(record)
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def pause(self) -> None:
        if self.thread is not None:
            self.queue.put(None)
            self.thread.join()
            self.thread = None
        self.check()

    def discard_pass(self, pass_id: str) -> None:
        self.pause()
        for frame in self.frames:
            if frame["pass_id"] == pass_id:
                (self.stage / frame["file"]).unlink(missing_ok=True)
        self.frames = [f for f in self.frames if f["pass_id"] != pass_id]
        (self.stage / "frames.jsonl").write_bytes(b"".join(_json(f) + b"\n" for f in self.frames))
        self.bytes_written = _size(self.stage)

    def prepare(self, selected_pass_ids: set[str], metadata: dict[str, Any]) -> PreparedSource:
        self.pause()
        frames = [f for f in self.frames if f["pass_id"] in selected_pass_ids]
        if {f["pass_id"] for f in frames} != selected_pass_ids:
            raise ValueError("A selected pass has no preserved foreground frames")
        destination = self.library.record_dir(self.source_id)
        destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        try:
            for frame in frames:
                # Same filesystem: staging remains available for a failed commit/retry.
                source = self.stage / frame["file"]
                if _sha(source) != frame["sha256"]:
                    raise ValueError("Staged source frame failed SHA256 verification")
                try:
                    os.link(source, destination / frame["file"])
                except OSError:
                    if shutil.disk_usage(destination).free < self.min_free + source.stat().st_size:
                        raise OSError("Insufficient disk space to commit source recording")
                    shutil.copyfile(source, destination / frame["file"])
            manifest = dict(schema="mph-foreground-source-v1", source_id=self.source_id,
                            created_at=_utc_now(), coordinate_convention="camera_X_right_Y_down_Z_forward",
                            units="mm", dtype="float32", storage_stage="foreground_before_sampling",
                            centered=False, scale_normalized=False, raw_sensor_frames_saved=False,
                            frames=frames, metadata=metadata)
            return PreparedSource(self, destination, manifest)
        except Exception:
            shutil.rmtree(destination)
            raise

    def abandon(self) -> None:
        if self.closed:
            return
        try:
            try:
                self.pause()
            except RuntimeError:
                pass
            finally:
                shutil.rmtree(self.stage, ignore_errors=False)
        finally:
            self.closed = True
            self.lease.close()


class PreparedSource:
    def __init__(self, capture: ForegroundCapture, path: Path, manifest: dict[str, Any]) -> None:
        self.capture, self.path, self.manifest = capture, path, manifest
        self.source_id = capture.source_id

    def attach(self, con: sqlite3.Connection, person_id: str, display_name: str,
               embedding_ids: list[int]) -> None:
        uid = portable_uid(con, "person", person_id)
        self.manifest.update(person_uid=uid, captured_person_id=person_id,
                             captured_name=display_name, original_embedding_ids=embedding_ids)
        data = _json(self.manifest)
        projected = self.capture.bytes_written + len(data)
        if (projected > self.capture.session_limit
                or self.capture.initial_bytes + projected > self.capture.library_limit):
            raise OSError("Source manifest exceeds storage limits")
        if shutil.disk_usage(self.path).free < self.capture.min_free + len(data):
            raise OSError("Insufficient disk space for source manifest")
        with (self.path / "manifest.json").open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        frames = self.manifest["frames"]
        con.execute("INSERT INTO enrollment_sources VALUES (?,?,?,?,?,?,?,?,?,?)", (
            self.source_id, uid, person_id, display_name,
            self.manifest["metadata"]["session_id"], self.manifest["created_at"],
            len(frames), len({f["pass_id"] for f in frames}),
            sum(f["size_bytes"] for f in frames) + len(data), hashlib.sha256(data).hexdigest()))

    def rollback(self) -> None:
        # A committed Gallery transaction owns its files, even if later reporting fails.
        with self.capture.library.repository.connect() as con:
            committed = con.execute("SELECT 1 FROM enrollment_sources WHERE source_id=?",
                                    (self.source_id,)).fetchone()
        if committed is None:
            shutil.rmtree(self.path)

    def finish(self) -> None:
        self.capture.abandon()
