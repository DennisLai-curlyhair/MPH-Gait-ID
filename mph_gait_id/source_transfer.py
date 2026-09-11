"""Portable, data-only foreground source archives; no models or embeddings."""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import uuid
import zipfile

import numpy as np

from .database import _utc_now
from .enrollment_sources import EnrollmentSourceLibrary, _Lease, _json, _size
from .gallery_provenance import is_deleted, restore_record


SCHEMA = "mph-source-transfer-v1"
MAX_MANIFEST = 64 << 20
MAX_ARCHIVE = 32 << 30
MAX_MEMBERS = 100001
MAX_SOURCES = 1000
MAX_POINTS = 2_000_000
CAPTURE_FIELDS = (
    "preprocessing_profile_id", "preprocessing_source_sha256", "filter_parameters",
    "detector_confidence", "yolo_image_size", "source_type", "foreground_detector",
)
FILTER_FIELDS = ("near_mm", "far_mm", "depth_margin_mm", "min_person_points", "max_valid_frame_gap_s")
PASS_FIELDS = ("pass_id", "requested_direction", "observed_direction", "started_at", "ended_at",
               "start_frame", "end_frame", "frame_count", "mean_person_points",
               "direction_displacement_mm", "direction_monotonic_ratio", "direction_match",
               "quality_accepted", "discarded")
FRAME_FIELDS = ("file", "pass_id", "sensor_frame_index", "timestamp", "segment_start",
                "points", "sha256", "size_bytes")
SOURCE_FIELDS = ("schema", "source_id", "created_at", "coordinate_convention", "units", "dtype",
                 "storage_stage", "centered", "scale_normalized", "raw_sensor_frames_saved",
                 "person_uid", "captured_person_id", "captured_name")


class SourceTransferCancelled(RuntimeError):
    pass


def _cancel(event):
    if event is not None and event.is_set():
        raise SourceTransferCancelled("Source transfer cancelled; no new sources were committed")


def _text(value, label, maximum=512):
    if not isinstance(value, str) or not value or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {label}")
    return value


def _integer(value, label, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {label}")
    return value


def _uuid(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("Invalid source UUID")
    return value


def _digest(stream, cancel=None):
    stream.seek(0)
    digest = hashlib.sha256()
    while True:
        _cancel(cancel)
        block = stream.read(1 << 20)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _portable(manifest):
    """Whitelist geometry provenance; discard local paths, model config and job IDs."""
    source = {key: manifest[key] for key in SOURCE_FIELDS}
    metadata = manifest["metadata"]
    capture = {key: metadata.get("capture", {})[key] for key in CAPTURE_FIELDS
               if key in metadata.get("capture", {})}
    if "filter_parameters" in capture:
        capture["filter_parameters"] = {key: value for key, value in capture["filter_parameters"].items()
                                        if key in FILTER_FIELDS}
    source["metadata"] = {
        "session_id": metadata["session_id"], "capture": capture,
        "passes": [{key: value for key, value in item.items() if key in PASS_FIELDS}
                   for item in metadata.get("passes", [])],
    }
    for key in ("consent", "selected_pass_ids"):
        if key in metadata:
            source["metadata"][key] = metadata[key]
    source["frames"] = [{**{key: item[key] for key in FRAME_FIELDS}, "metadata": {}}
                        for item in manifest["frames"]]
    return source


def _fingerprint(source):
    # Paths, source UUID, session ID and human-readable names are not content identity.
    data = {key: source[key] for key in ("coordinate_convention", "units", "dtype",
                                        "storage_stage", "centered", "scale_normalized")}
    data["frames"] = [{key: item[key] for key in FRAME_FIELDS if key != "file"}
                      for item in source["frames"]]
    data["capture"] = source["metadata"]["capture"]
    return hashlib.sha256(_json(data)).hexdigest()


def _validate_source(source):
    if not isinstance(source, dict):
        raise ValueError("Invalid source manifest")
    sid = _uuid(source.get("source_id"))
    if (source.get("schema") != "mph-foreground-source-v1"
            or source.get("coordinate_convention") != "camera_X_right_Y_down_Z_forward"
            or source.get("units") != "mm" or source.get("dtype") != "float32"
            or source.get("storage_stage") != "foreground_before_sampling"
            or source.get("centered") is not False or source.get("scale_normalized") is not False
            or source.get("raw_sensor_frames_saved") is not False):
        raise ValueError("Expected uncentered, unsampled foreground camera XYZ in mm")
    for key in ("person_uid", "captured_person_id", "captured_name", "created_at"):
        _text(source.get(key), key)
    metadata = source.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("capture"), dict):
        raise ValueError("Missing source capture provenance")
    _text(metadata.get("session_id"), "session ID")
    capture = metadata["capture"]
    _text(capture.get("preprocessing_profile_id"), "preprocessing profile")
    if "filter_parameters" in capture:
        if not isinstance(capture["filter_parameters"], dict):
            raise ValueError("Invalid filter parameters")
        for value in capture["filter_parameters"].values():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Invalid numeric filter parameter")
    if not isinstance(metadata.get("passes", []), list):
        raise ValueError("Invalid pass metadata")
    for item in metadata.get("passes", []):
        if not isinstance(item, dict) or set(item) - set(PASS_FIELDS):
            raise ValueError("Invalid pass metadata fields")
        _text(item.get("pass_id"), "pass metadata ID")
        for value in item.values():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("Invalid pass metadata value")
    for key in ("preprocessing_source_sha256", "source_type", "foreground_detector"):
        if key in capture:
            _text(capture[key], key)
    for key in ("detector_confidence", "yolo_image_size"):
        if key in capture and (type(capture[key]) not in (int, float) or not math.isfinite(capture[key])):
            raise ValueError("Invalid detector parameter")
    if "consent" in metadata:
        _text(metadata["consent"], "capture consent provenance")
    if "selected_pass_ids" in metadata:
        if not isinstance(metadata["selected_pass_ids"], list):
            raise ValueError("Invalid selected pass IDs")
        for pid in metadata["selected_pass_ids"]:
            _text(pid, "selected pass ID")
    frames = source.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) < MAX_MEMBERS:
        raise ValueError("Invalid source frame count")
    names = set()
    previous = None
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid frame metadata")
        name = frame.get("file")
        if not isinstance(name, str) or not re.fullmatch(r"frame_[0-9]{6}\.npy", name) or name in names:
            raise ValueError("Unsafe or duplicate source frame name")
        names.add(name)
        _text(frame.get("pass_id"), "pass ID")
        _integer(frame.get("points"), "point count", 1, MAX_POINTS)
        _integer(frame.get("sensor_frame_index"), "frame index", 0, 2**63 - 1)
        _integer(frame.get("size_bytes"), "frame size", 1, MAX_POINTS * 12 + 10000)
        digest = frame.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("Invalid frame SHA256")
        timestamp = frame.get("timestamp")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError("Invalid frame timestamp")
        if type(frame.get("segment_start")) is not bool:
            raise ValueError("Missing segment boundary")
        if (previous is None or frame["pass_id"] != previous["pass_id"]
                or timestamp <= previous["timestamp"]
                or frame["sensor_frame_index"] <= previous["sensor_frame_index"]):
            if not frame["segment_start"]:
                raise ValueError("Missing boundary at source discontinuity")
        previous = frame
    # Equality rejects undocumented fields, including workstation paths and model metadata.
    if _portable(source) != source:
        raise ValueError("Unsupported source metadata fields")
    _json(source)  # Reject NaN/Infinity in optional provenance as well.
    return sid


def _validate_points(payload, frame):
    if len(payload) != frame["size_bytes"] or hashlib.sha256(payload).hexdigest() != frame["sha256"]:
        raise ValueError("Source frame failed size/SHA256 verification")
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("Unsupported NPY format")
    offset = stream.tell()
    if (shape != (frame["points"], 3) or dtype != np.dtype("float32")
            or len(payload) - offset != frame["points"] * 12):
        raise ValueError("Invalid foreground XYZ array header")
    if not np.isfinite(np.frombuffer(payload, dtype=dtype, offset=offset)).all():
        raise ValueError("Non-finite foreground XYZ")


def _read_member(archive, name, size, cancel):
    chunks = []
    remaining = size
    with archive.open(name) as stream:
        while remaining:
            _cancel(cancel)
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("Truncated archive member")
            chunks.append(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("Oversized archive member")
    return b"".join(chunks)


@contextmanager
def _archive(path, cancel=None):
    with Path(path).open("rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_ARCHIVE:
            raise ValueError("Archive exceeds size limit")
        digest = _digest(stream, cancel)
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS or len({item.filename for item in infos}) != len(infos):
                raise ValueError("Too many or duplicate ZIP members")
            if sum(item.file_size for item in infos) > MAX_ARCHIVE:
                raise ValueError("Archive expanded size exceeds limit")
            for item in infos:
                mode = (item.external_attr >> 16) & 0xFFFF
                if (item.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                        or item.flag_bits & 1 or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise ValueError("Unsupported or unsafe ZIP member")
            info = archive.getinfo("manifest.json")
            if info.file_size > MAX_MANIFEST:
                raise ValueError("Archive manifest exceeds size limit")
            package = json.loads(_read_member(archive, info.filename, info.file_size, cancel),
                                 object_pairs_hook=_unique_json)
            if (not isinstance(package, dict) or package.get("schema") != SCHEMA
                    or set(package) != {"schema", "created_at", "persons", "sources"}):
                raise ValueError("Unsupported source archive schema")
            _text(package["created_at"], "archive creation time")
            sources, people = package["sources"], package["persons"]
            if (not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES
                    or not isinstance(people, list) or not 1 <= len(people) <= MAX_SOURCES):
                raise ValueError("Invalid archive source/person count")
            persons = {}
            for person in people:
                if not isinstance(person, dict) or set(person) != {"uid", "person_id", "display_name", "status"}:
                    raise ValueError("Invalid portable person record")
                for key in person:
                    _text(person[key], key)
                if person["status"] not in {"active", "inactive"} or person["uid"] in persons:
                    raise ValueError("Invalid or duplicate person")
                persons[person["uid"]] = person
            expected, ids = {"manifest.json"}, set()
            for source in sources:
                sid = _validate_source(source)
                if sid in ids or source["person_uid"] not in persons:
                    raise ValueError("Duplicate source UUID or unknown owner")
                ids.add(sid)
                for frame in source["frames"]:
                    name = f"sources/{sid}/{frame['file']}"
                    expected.add(name)
                    if archive.getinfo(name).file_size != frame["size_bytes"]:
                        raise ValueError("Incorrect frame size in ZIP directory")
            if expected != {item.filename for item in infos}:
                raise ValueError("Unexpected or missing archive members")
            if set(persons) != {s["person_uid"] for s in sources}:
                raise ValueError("Person without source recording")
            yield archive, package, digest, stream


class SourceTransfer:
    def __init__(self, library: EnrollmentSourceLibrary, *, library_limit_bytes=20 << 30,
                 min_free_bytes=512 << 20):
        self.library = library
        self.repository = library.repository
        self.library_limit = int(library_limit_bytes)
        self.min_free = int(min_free_bytes)
        if self.library_limit <= 0 or self.min_free < 0:
            raise ValueError("Invalid transfer storage limits")
        library.initialize()
        with self.repository.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS enrollment_source_transfers (
                transfer_id TEXT PRIMARY KEY, archive_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL, report_json TEXT NOT NULL)""")

    @staticmethod
    def _state(con):
        snapshot = {}
        for table in ("persons", "enrollment_sources", "gallery_transfer_aliases", "gallery_transfer_tombstones"):
            snapshot[table] = sorted((dict(row) for row in con.execute(f"SELECT * FROM {table}")),
                                     key=lambda row: _json(row))
        return hashlib.sha256(_json(snapshot)).hexdigest()

    def export(self, path, source_ids, *, cancel=None, progress=None):
        progress = progress or (lambda _message: None)
        path = Path(path).expanduser().absolute()
        root = self.library.root.resolve()
        if (path.resolve().is_relative_to(root) or path.resolve() == self.repository.path.resolve()
                or path.is_symlink() or path.suffix.lower() != ".mphsources"):
            raise ValueError("Choose a .mphsources file outside the source library")
        if not source_ids or len(set(source_ids)) != len(source_ids) or len(source_ids) > MAX_SOURCES:
            raise ValueError("Select unique committed source recordings")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with _Lease(root):
            try:
                sources, people = [], {}
                with self.repository.connect() as con:
                    con.execute("BEGIN")
                    state = self._state(con)
                    for sid in source_ids:
                        _cancel(cancel)
                        source = _portable(self.library.manifest(sid))
                        _validate_source(source)
                        uid = source["person_uid"]
                        person = con.execute("""SELECT p.* FROM persons p JOIN gallery_transfer_aliases a
                            ON a.kind='person' AND a.local_id=p.person_id WHERE a.uid=?""", (uid,)).fetchone()
                        if person is None or is_deleted(con, "person", uid):
                            raise ValueError("Source owner was deleted; restore its original person before exporting")
                        people[uid] = dict(uid=uid, person_id=person["person_id"],
                                           display_name=person["display_name"], status=person["status"])
                        sources.append(source)
                package = dict(schema=SCHEMA, created_at=_utc_now(), persons=list(people.values()), sources=sources)
                payload = _json(package)
                total = sum(len(s["frames"]) for s in sources)
                size = len(payload) + sum(f["size_bytes"] for s in sources for f in s["frames"])
                if len(payload) > MAX_MANIFEST or total + 1 > MAX_MEMBERS or size > MAX_ARCHIVE:
                    raise ValueError("Selection exceeds archive limits; export fewer sources")
                zip_overhead = (1 << 20) + total * 1024 + size // 100
                if shutil.disk_usage(path.parent).free < size + self.min_free + zip_overhead:
                    raise OSError("Insufficient free space for source export")
                with temporary.open("xb") as output:
                    try:
                        temporary.chmod(0o600)
                    except OSError:
                        pass
                    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                        archive.writestr("manifest.json", payload)
                        done = 0
                        for source in sources:
                            for frame in source["frames"]:
                                _cancel(cancel)
                                file = self.library.record_dir(source["source_id"]) / frame["file"]
                                if file.is_symlink() or file.stat().st_size != frame["size_bytes"]:
                                    raise ValueError("Source file changed or is unsafe")
                                data = file.read_bytes()
                                _validate_points(data, frame)
                                archive.writestr(f"sources/{source['source_id']}/{frame['file']}", data)
                                done += 1
                                progress(f"{done}/{total}")
                    output.flush()
                    os.fsync(output.fileno())
                with self.repository.connect() as con:
                    if self._state(con) != state:
                        raise ValueError("Source library or identities changed during export")
                _cancel(cancel)
                if temporary.stat().st_size > MAX_ARCHIVE:
                    raise ValueError("Compressed archive exceeds size limit; export fewer sources")
                with temporary.open("rb") as handle:
                    digest = _digest(handle, cancel)
                os.replace(temporary, path)
                return dict(path=str(path), sources=len(sources), persons=len(people), frames=total,
                            archive_sha256=digest, size_bytes=path.stat().st_size)
            finally:
                temporary.unlink(missing_ok=True)

    def _describe(self, con, package, cancel=None):
        people = []
        for person in package["persons"]:
            uid = person["uid"]
            alias = con.execute("SELECT local_id FROM gallery_transfer_aliases WHERE kind='person' AND uid=?",
                                (uid,)).fetchone()
            pid = alias[0] if alias else person["person_id"]
            local = con.execute("SELECT * FROM persons WHERE person_id=?", (pid,)).fetchone()
            status = ("deleted" if is_deleted(con, "person", uid)
                      else "linked" if alias and local else "conflict" if local or alias else "new")
            people.append({**person, "target_id": pid, "status": status,
                           "archived_status": person["status"], "local_name": local["display_name"] if local else None,
                           "target_exists": bool(local)})
        existing = {}
        for row in con.execute("SELECT * FROM enrollment_sources"):
            _cancel(cancel)
            manifest = _portable(self.library.manifest(row["source_id"]))
            existing[row["source_id"]] = (row, _fingerprint(manifest), manifest)
        descriptions = []
        for source in package["sources"]:
            _cancel(cancel)
            sid, fingerprint = source["source_id"], _fingerprint(source)
            row = existing.get(sid)
            collision = next((key for key, value in existing.items() if value[1] == fingerprint and key != sid), None)
            status = "new"
            if row:
                status = "duplicate" if row[2] == source else "conflict"
                if status == "duplicate":
                    for frame in row[2]["frames"]:
                        _cancel(cancel)
                        self.library.load_frame(sid, frame)
            elif collision:
                status = "conflict"
            descriptions.append(dict(source_id=sid, person_uid=source["person_uid"], status=status,
                                     frame_count=len(source["frames"]),
                                     pass_count=len({f["pass_id"] for f in source["frames"]}),
                                     size_bytes=sum(f["size_bytes"] for f in source["frames"]),
                                     preprocessing_profile_id=source["metadata"]["capture"]["preprocessing_profile_id"],
                                     conflicting_source_id=collision))
        return people, descriptions

    def preview(self, path, *, cancel=None, progress=None):
        progress = progress or (lambda _message: None)
        with _Lease(self.library.root), _archive(path, cancel) as (archive, package, digest, stream):
            total = sum(len(s["frames"]) for s in package["sources"])
            done = 0
            for source in package["sources"]:
                for frame in source["frames"]:
                    data = _read_member(archive, f"sources/{source['source_id']}/{frame['file']}", frame["size_bytes"], cancel)
                    _validate_points(data, frame)
                    done += 1
                    progress(f"{done}/{total}")
            if _digest(stream, cancel) != digest:
                raise ValueError("Archive changed while validating")
            with self.repository.connect() as con:
                con.execute("BEGIN")
                state = self._state(con)
                people, sources = self._describe(con, package, cancel)
            return dict(archive_sha256=digest, database_state=state, persons=people, sources=sources,
                        frames=total, size_bytes=sum(s["size_bytes"] for s in sources))

    def _backup(self, expected_state):
        backup = self.repository.path.parent / "source_transfer_backups" / f"gallery-{uuid.uuid4().hex}.sqlite3"
        backup.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with self.repository.connect() as source, closing(sqlite3.connect(backup)) as destination:
                source.execute("BEGIN")
                if self._state(source) != expected_state:
                    raise ValueError("Source library or identities changed; preview again")
                source.backup(destination)
        except Exception:
            backup.unlink(missing_ok=True)
            raise
        try:
            backup.chmod(0o600)
        except OSError:
            pass
        return backup

    def import_archive(self, path, preview, person_map, *, restore_deleted=False, cancel=None, progress=None):
        progress = progress or (lambda _message: None)
        if type(restore_deleted) is not bool:
            raise ValueError("restore_deleted must be boolean")
        stage = self.library.root / "staging" / f"transfer-{uuid.uuid4().hex}"
        moved = []
        committed = False
        with _Lease(self.library.root), _archive(path, cancel) as (archive, package, digest, stream):
            if digest != preview["archive_sha256"]:
                raise ValueError("Archive changed after preview; preview again")
            if set(person_map) != {p["uid"] for p in package["persons"]}:
                raise ValueError("Choose an action for every person")
            for target in person_map.values():
                if target is not None:
                    _text(target, "target person ID")
            with self.repository.connect() as con:
                con.execute("BEGIN")
                if self._state(con) != preview["database_state"]:
                    raise ValueError("Source library or identities changed; preview again")
                people, descriptions = self._describe(con, package, cancel)
                for target in set(person_map.values()) - {None}:
                    if (list(person_map.values()).count(target) > 1
                            and con.execute("SELECT 1 FROM persons WHERE person_id=?", (target,)).fetchone() is None):
                        raise ValueError("Different imported identities require distinct new IDs")
            if people != preview["persons"] or descriptions != preview["sources"]:
                raise ValueError("Import plan changed; preview again")
            selected = [s for s in package["sources"] if person_map[s["person_uid"]] is not None]
            if not selected:
                raise ValueError("No sources selected for import")
            statuses = {s["source_id"]: s["status"] for s in descriptions}
            if any(statuses[s["source_id"]] == "conflict" for s in selected):
                raise ValueError("Source UUID/content collision; do not relabel an existing recording")
            incoming = [s for s in selected if statuses[s["source_id"]] == "new"]
            fingerprints = [_fingerprint(s) for s in selected]
            if len(set(fingerprints)) != len(fingerprints):
                raise ValueError("Repeated source contents under different UUIDs")
            required = sum(len(_json(s)) + sum(f["size_bytes"] for f in s["frames"]) for s in incoming)
            if required and _size(self.library.root) + required > self.library_limit:
                raise OSError("Source library storage limit reached")
            if shutil.disk_usage(self.library.root).free < required + self.min_free + self.repository.path.stat().st_size:
                raise OSError("Insufficient free space for source import and database backup")
            report = dict(transfer_id=str(uuid.uuid4()), archive_sha256=digest, created_at=_utc_now(),
                          inserted=len(incoming), duplicates=len(selected)-len(incoming),
                          skipped=len(package["sources"])-len(selected), restored_persons=0,
                          sources=[s["source_id"] for s in selected])
            try:
                stage.mkdir(parents=True, mode=0o700)
                done = 0
                total = sum(len(s["frames"]) for s in incoming)
                for source in incoming:
                    directory = stage / source["source_id"]
                    directory.mkdir(mode=0o700)
                    for frame in source["frames"]:
                        _cancel(cancel)
                        if shutil.disk_usage(stage).free < frame["size_bytes"] + self.min_free:
                            raise OSError("Insufficient free disk space during source import")
                        data = _read_member(archive, f"sources/{source['source_id']}/{frame['file']}", frame["size_bytes"], cancel)
                        _validate_points(data, frame)
                        with (directory / frame["file"]).open("xb") as handle:
                            handle.write(data)
                            handle.flush()
                            os.fsync(handle.fileno())
                        done += 1
                        progress(f"{done}/{total}")
                    with (directory / "manifest.json").open("xb") as handle:
                        handle.write(_json(source))
                        handle.flush()
                        os.fsync(handle.fileno())
                if _digest(stream, cancel) != digest:
                    raise ValueError("Archive changed during extraction")
                report["backup"] = str(self._backup(preview["database_state"]))
                _cancel(cancel)
                with self.repository.connect() as con:
                    con.execute("BEGIN IMMEDIATE")
                    if self._state(con) != preview["database_state"]:
                        raise ValueError("Source library or identities changed; preview again")
                    for person in people:
                        uid, target = person["uid"], person_map[person["uid"]]
                        if target is None:
                            continue
                        local = con.execute("SELECT * FROM persons WHERE person_id=?", (target,)).fetchone()
                        alias = con.execute("SELECT local_id FROM gallery_transfer_aliases WHERE kind='person' AND uid=?", (uid,)).fetchone()
                        if alias and (alias[0] != target or local is None):
                            raise ValueError("Already linked identities cannot be reassigned")
                        if person["status"] == "deleted":
                            if not restore_deleted or local or list(person_map.values()).count(target) != 1:
                                raise ValueError("Deleted person requires explicit restore to an unused distinct ID")
                        if local is None:
                            con.execute("INSERT INTO persons VALUES (?,?,?,?,?,?)",
                                        (target, person["display_name"], "", person["archived_status"], _utc_now(), _utc_now()))
                        con.execute("INSERT OR IGNORE INTO gallery_transfer_aliases VALUES ('person',?,?)", (uid, target))
                        if person["status"] == "deleted":
                            report["restored_persons"] += restore_record(con, "person", uid, target, digest)
                    for source in incoming:
                        _cancel(cancel)
                        sid = source["source_id"]
                        destination = self.library.record_dir(sid)
                        if destination.exists():
                            raise ValueError("Uncommitted source directory exists; inspect and clean temporary files first")
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        (stage / sid).rename(destination)
                        moved.append(destination)
                        payload = _json(source)
                        con.execute("INSERT INTO enrollment_sources VALUES (?,?,?,?,?,?,?,?,?,?)", (
                            sid, source["person_uid"], source["captured_person_id"], source["captured_name"],
                            source["metadata"]["session_id"], source["created_at"], len(source["frames"]),
                            len({f["pass_id"] for f in source["frames"]}),
                            len(payload) + sum(f["size_bytes"] for f in source["frames"]), hashlib.sha256(payload).hexdigest()))
                    _cancel(cancel)
                    con.execute("INSERT INTO enrollment_source_transfers VALUES (?,?,?,?)", (
                        report["transfer_id"], digest, report["created_at"], _json(report).decode()))
                committed = True
            finally:
                if not committed:
                    for directory in moved:
                        shutil.rmtree(directory)
                if stage.exists():
                    try:
                        shutil.rmtree(stage)
                    except OSError:
                        if not committed:
                            raise
                        report["cleanup_warning"] = "Import committed; clean uncommitted source files after closing transfer"
            return report
