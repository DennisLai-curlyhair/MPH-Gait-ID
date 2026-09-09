"""Versioned, data-only Gallery archives. Never loads weights or extracts ZIP paths."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import uuid
import zipfile

import numpy as np

from .database import GalleryRepository, _utc_now
from .gallery_provenance import initialize_provenance, is_deleted, portable_uid
from .model_record import build_model_record
from .model_store import ModelBundle, ModelStore


FORMAT = "mph-gait-gallery"
VERSION = 1
MAX_BYTES = 512 * 1024 * 1024
MAX_JSON = 16 * 1024 * 1024
MAX_ROWS = 100_000
MEMBERS = {"manifest.json", "records.json", "embeddings.bin"}


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _string(value: Any, label: str, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError(f"Invalid {label}")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {label}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".gallery-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _encoder_hash() -> str:
    root = Path(__file__).parent
    paths = [root / name for name in ("model_adapter.py", "dataio.py", "runtime.py", "model_record.py")]
    paths += sorted((root / "models").glob("*.py"))
    return _hash(b"".join(
        path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") for path in paths
    ))


def _bundle_spec(bundle: ModelBundle) -> dict[str, Any]:
    # Runtime T/drop settings are carried separately; batch size is not a feature definition.
    data = {key: value for key, value in bundle.data.items()
            if key not in {"batch_size", "clip_len", "drop_first_frames"}}
    return {"architecture": bundle.architecture, "channels": bundle.channels,
            "data": data, "model": bundle.model, "encoder_sha256": _encoder_hash()}


def _space(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model["compatibility"].items() if key != "bundle_id"}


def _window_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    # Whitelist portable timing and pass metadata, not source paths or sensor serials.
    keys = {"window_index", "start_frame", "end_frame", "start_frame_number",
            "end_frame_number", "frame_indices", "captured_at", "effective_sampling_fps",
            "window_duration_s", "mean_frame_gap_ms", "max_frame_gap_ms", "pass_id",
            "requested_direction", "observed_direction", "manual_quality_override"}
    window = metadata.get("window", {})
    return {key: value for key, value in window.items() if key in keys}


def read_archive(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Gallery archive exceeds the 512 MiB limit")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("Gallery archive exceeds the 512 MiB limit")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        if len(infos) != 3 or {item.filename for item in infos} != MEMBERS:
            raise ValueError("Unexpected or duplicate archive members")
        if sum(item.file_size for item in infos) > MAX_BYTES:
            raise ValueError("Uncompressed Gallery exceeds the 512 MiB limit")
        contents = {}
        for item in infos:
            limit = MAX_BYTES if item.filename == "embeddings.bin" else MAX_JSON
            if item.file_size > limit or item.flag_bits & 1:
                raise ValueError("Oversized or encrypted archive member")
            with archive.open(item) as handle:
                contents[item.filename] = handle.read(limit + 1)
            if len(contents[item.filename]) != item.file_size:
                raise ValueError("Archive member size mismatch")
    manifest = json.loads(contents["manifest.json"])
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("Unsupported Gallery archive format/version")
    for name in ("records.json", "embeddings.bin"):
        if manifest.get("sha256", {}).get(name) != _hash(contents[name]):
            raise ValueError(f"Checksum mismatch: {name}")
    records = json.loads(contents["records.json"])
    for name in ("persons", "models", "embeddings"):
        if not isinstance(records.get(name), list) or len(records[name]) > MAX_ROWS:
            raise ValueError(f"Invalid {name} collection")
    people, person_ids, models, uids = set(), set(), {}, set()
    for person in records["persons"]:
        uid = _string(person.get("uid"), "person uid")
        if uid in people:
            raise ValueError("Duplicate person uid")
        people.add(uid)
        _string(person.get("person_id"), "person ID")
        if person["person_id"] in person_ids:
            raise ValueError("Duplicate person ID")
        person_ids.add(person["person_id"])
        _string(person.get("display_name"), "person name")
        for key in ("note", "status", "created_at", "updated_at"):
            if not isinstance(person.get(key), str) or len(person[key]) > 4096:
                raise ValueError(f"Invalid person {key}")
    for model in records["models"]:
        key = _string(model.get("key"), "model key")
        if key in models or not isinstance(model.get("compatibility"), dict):
            raise ValueError("Invalid or duplicate model")
        contract = model["compatibility"]
        for field in ("bundle_id", "method_key", "architecture", "input_type", "input_mode",
                      "checkpoint_sha256", "preprocessing_profile_id"):
            _string(contract.get(field), field)
        for field, lo, hi in (("embedding_dim", 1, 65536), ("num_points", 1, 1_000_000),
                              ("clip_len", 1, 10000), ("drop_first_frames", 0, 100000)):
            _integer(contract.get(field), field, lo, hi)
        _string(model.get("display_name"), "model display name")
        if model.get("bundle_spec") is not None and not isinstance(model["bundle_spec"], dict):
            raise ValueError("Invalid model bundle specification")
        models[key] = model
    offset = 0
    features = contents["embeddings.bin"]
    for row in records["embeddings"]:
        uid = _string(row.get("uid"), "embedding uid")
        if uid in uids or row.get("person_uid") not in people or row.get("model") not in models:
            raise ValueError("Duplicate embedding or unresolved foreign key")
        uids.add(uid)
        dimension = models[row["model"]]["compatibility"]["embedding_dim"]
        if row.get("offset") != offset or type(row.get("offset")) is not int:
            raise ValueError("Invalid embedding offset")
        end = offset + dimension * 4
        if end > len(features):
            raise ValueError("Truncated embedding payload")
        vector = np.frombuffer(features[offset:end], dtype="<f4")
        if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 0.001:
            raise ValueError("Embedding must be finite and L2 normalized")
        _integer(row.get("active"), "active state", 0, 1)
        _string(row.get("source_fingerprint"), "source fingerprint")
        for field in ("session_id", "pass_id", "direction"):
            if row.get(field) is not None:
                _string(row[field], field)
        _string(row.get("source_type"), "source type")
        _string(row.get("created_at"), "embedding creation time")
        quality = row.get("quality_score")
        if quality is not None and (type(quality) not in (int, float) or not np.isfinite(quality)):
            raise ValueError("Invalid quality score")
        if not isinstance(row.get("window"), dict):
            raise ValueError("Invalid window metadata")
        row["vector"] = features[offset:end]
        offset = end
    if offset != len(features):
        raise ValueError("Unused embedding payload")
    _json(records["persons"])
    _json(records["models"])
    for row in records["embeddings"]:
        _json(row["window"])
    return {"records": records, "manifest": manifest, "sha256": _hash(raw), "raw": raw}


class GalleryTransfer:
    def __init__(self, repository: GalleryRepository, store: ModelStore, profile_id: str):
        self.repository, self.store, self.profile_id = repository, store, profile_id
        self.storage = repository.path.parent / "gallery_transfer"
        initialize_provenance(repository)

    @staticmethod
    def _state(con: sqlite3.Connection) -> str:
        digest = hashlib.sha256()
        for table in ("persons", "model_versions", "gallery_embeddings", "gallery_transfer_aliases", "gallery_transfer_model_specs", "gallery_transfer_tombstones"):
            for row in con.execute(f"SELECT * FROM {table} ORDER BY rowid"):
                for value in row:
                    data = value if isinstance(value, bytes) else _json(value)
                    digest.update(len(data).to_bytes(8, "little") + data)
        return digest.hexdigest()

    def export(self, path: str | Path) -> dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if target == self.repository.path or target.suffix != ".mphgallery":
            raise ValueError("Choose a .mphgallery file, not a database or model file")
        self.store.refresh()
        records: dict[str, list] = {"persons": [], "models": [], "embeddings": []}
        blobs = bytearray()
        with self.repository.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            def uid(kind: str, local_id: Any) -> str:
                return portable_uid(con, kind, local_id)
            for row in con.execute("SELECT * FROM persons ORDER BY person_id"):
                records["persons"].append({**dict(row), "uid": uid("person", row["person_id"])})
            for row in con.execute("SELECT * FROM model_versions ORDER BY model_key"):
                config = json.loads(row["config_json"])
                saved = con.execute("SELECT spec_json FROM gallery_transfer_model_specs WHERE model_key=?",
                                    (row["model_key"],)).fetchone()
                spec = json.loads(saved[0]) if saved else None
                bundle = self.store.bundles().get(config.get("bundle_id"))
                if spec is None and bundle and bundle.checkpoint_sha256 == row["checkpoint_sha256"]:
                    spec = _bundle_spec(bundle)
                records["models"].append({"key": row["model_key"], "display_name": row["display_name"],
                                          "compatibility": config.get("compatibility", {}), "bundle_spec": spec})
            for row in con.execute("SELECT * FROM gallery_embeddings ORDER BY embedding_id"):
                item = dict(row)
                metadata = json.loads(item["metadata_json"])
                fingerprint = item["source_fingerprint"] or "legacy:" + _hash(item["source_path"].encode())
                portable = {key: item[key] for key in ("source_type", "session_id", "pass_id", "direction",
                                                     "quality_score", "active", "created_at")}
                portable.update(uid=uid("embedding", item["embedding_id"]),
                                person_uid=uid("person", item["person_id"]), model=item["model_key"],
                                source_fingerprint=fingerprint, offset=len(blobs),
                                window=_window_metadata(metadata))
                vector = np.frombuffer(item["embedding"], dtype=np.float32).astype("<f4")
                blobs.extend(vector.tobytes())
                records["embeddings"].append(portable)
        payload = _json(records)
        manifest = {"format": FORMAT, "version": VERSION, "created_at": _utc_now(),
                    "sha256": {"records.json": _hash(payload), "embeddings.bin": _hash(blobs)}}
        if len(payload) > MAX_JSON or len(payload) + len(blobs) > MAX_BYTES:
            raise ValueError("Gallery is too large for this archive format")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", _json(manifest))
            archive.writestr("records.json", payload)
            archive.writestr("embeddings.bin", blobs)
        # Validate before replacing an existing export.
        fd, name = tempfile.mkstemp(suffix=".mphgallery")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(buffer.getvalue())
            read_archive(name)
        finally:
            Path(name).unlink(missing_ok=True)
        _atomic_write(target, buffer.getvalue())
        return {"path": str(target), "persons": len(records["persons"]),
                "embeddings": len(records["embeddings"]), "sha256": _hash(buffer.getvalue())}

    def _match_models(self, records: dict[str, Any]) -> dict[str, dict[str, Any]]:
        self.store.refresh()
        matches, verified = {}, {}
        for source in records["models"]:
            candidates = []
            for bundle in self.store.bundles().values():
                if source["compatibility"]["preprocessing_profile_id"] != self.profile_id:
                    continue
                spec = source["bundle_spec"]
                if spec is None or spec != _bundle_spec(bundle):
                    continue
                record = build_model_record(bundle, self.store, source["compatibility"]["clip_len"],
                                            source["compatibility"]["drop_first_frames"], self.profile_id)
                if _space(record) != _space(source):
                    continue
                if bundle.bundle_id not in verified:
                    verified[bundle.bundle_id] = self.store.verify(bundle.bundle_id)["valid"]
                if verified[bundle.bundle_id]:
                    candidates.append(record)
            candidates.sort(key=lambda r: (r["bundle_id"] != source["compatibility"]["bundle_id"], r["bundle_id"]))
            matches[source["key"]] = {
                "display_name": source["display_name"],
                "status": "compatible" if candidates else "unavailable",
                "target": candidates[0] if candidates else None,
            }
        return matches

    def preview(self, path: str | Path) -> dict[str, Any]:
        package = read_archive(path)
        records = package["records"]
        matches = self._match_models(records)
        people = []
        with self.repository.connect() as con:
            con.execute("BEGIN")
            state = self._state(con)
            for person in records["persons"]:
                alias = con.execute("SELECT local_id FROM gallery_transfer_aliases WHERE kind='person' AND uid=?",
                                    (person["uid"],)).fetchone()
                pid = alias[0] if alias else person["person_id"]
                local = con.execute("SELECT * FROM persons WHERE person_id=?", (pid,)).fetchone()
                status = ("deleted" if is_deleted(con, "person", person["uid"])
                          else "linked" if alias and local else "conflict" if local else "new")
                people.append({**person, "status": status, "target_id": pid,
                               "local_name": local["display_name"] if local else None})
            duplicate_count = sum(bool(con.execute(
                "SELECT 1 FROM gallery_transfer_aliases WHERE kind='embedding' AND uid=?", (row["uid"],)
            ).fetchone()) for row in records["embeddings"])
            deleted_count = sum(is_deleted(con, "embedding", row["uid"])
                                or is_deleted(con, "person", row["person_uid"])
                                for row in records["embeddings"])
        available = sum(matches[row["model"]]["status"] == "compatible" for row in records["embeddings"])
        return {"archive_sha256": package["sha256"], "database_state": state, "persons": people,
                "models": matches, "embeddings": len(records["embeddings"]),
                "compatible_embeddings": available, "unavailable_embeddings": len(records["embeddings"]) - available,
                "known_embedding_ids": duplicate_count, "deleted_embeddings": deleted_count}

    def import_archive(self, path: str | Path, preview: dict[str, Any],
                       person_map: dict[str, str | None]) -> dict[str, Any]:
        package = read_archive(path)
        if package["sha256"] != preview["archive_sha256"]:
            raise ValueError("Archive changed after preview; preview it again")
        records = package["records"]
        matches = self._match_models(records)
        if matches != preview["models"]:
            raise ValueError("Model availability changed after preview; preview it again")
        if set(person_map) != {p["uid"] for p in records["persons"]}:
            raise ValueError("Choose an action for every imported person")
        for target in person_map.values():
            if target is not None:
                _string(target, "target person ID")
        # Retain the validated data-only package so missing models can be imported later.
        self.storage.mkdir(parents=True, exist_ok=True)
        stored_package = self.storage / "incoming" / f"{package['sha256']}.mphgallery"
        _atomic_write(stored_package, package["raw"])
        backup = self.storage / "backups" / f"gallery-{uuid.uuid4().hex}.sqlite3"
        backup.parent.mkdir(parents=True, exist_ok=True)
        with self.repository.connect() as source, sqlite3.connect(backup) as destination:
            source.execute("BEGIN")
            if self._state(source) != preview["database_state"]:
                raise ValueError("Gallery changed after preview; preview it again")
            source.backup(destination)
        try:
            backup.chmod(0o600)
        except OSError:
            pass
        inserted = duplicates = skipped = deleted_skipped = 0
        with self.repository.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            if self._state(con) != preview["database_state"]:
                raise ValueError("Gallery changed after preview; preview it again")
            for person in records["persons"]:
                target = person_map[person["uid"]]
                if target is None:
                    continue
                if is_deleted(con, "person", person["uid"]):
                    raise ValueError("Previously deleted person cannot be restored or remapped from this archive")
                alias = con.execute("SELECT local_id FROM gallery_transfer_aliases WHERE kind='person' AND uid=?",
                                    (person["uid"],)).fetchone()
                if alias and alias[0] != target:
                    raise ValueError("An imported identity is already linked to a different local ID")
                if alias and not con.execute("SELECT 1 FROM persons WHERE person_id=?", (target,)).fetchone():
                    raise ValueError("Previously imported person was removed; automatic restoration is disabled")
                if not con.execute("SELECT 1 FROM persons WHERE person_id=?", (target,)).fetchone():
                    con.execute("INSERT INTO persons (person_id,display_name,note,status,created_at,updated_at)"
                                " VALUES (?,?,?,?,?,?)",
                                (target, person["display_name"], person["note"], person["status"],
                                 person["created_at"], person["updated_at"]))
                con.execute("INSERT OR IGNORE INTO gallery_transfer_aliases VALUES ('person',?,?)",
                            (person["uid"], target))
            for row in records["embeddings"]:
                target_person = person_map[row["person_uid"]]
                target_model = matches[row["model"]]["target"]
                if (is_deleted(con, "embedding", row["uid"])
                        or is_deleted(con, "person", row["person_uid"])):
                    skipped += 1
                    deleted_skipped += 1
                    continue
                if target_person is None or target_model is None:
                    skipped += 1
                    continue
                key = target_model["model_key"]
                native_vector = np.frombuffer(row["vector"], dtype="<f4").astype(np.float32).tobytes()
                existing_model = con.execute("SELECT config_json FROM model_versions WHERE model_key=?", (key,)).fetchone()
                if existing_model:
                    if _space(json.loads(existing_model[0])) != _space(target_model):
                        raise ValueError("Local model compatibility collision")
                else:
                    self.repository.upsert_model(target_model, connection=con)
                source_model = next(model for model in records["models"] if model["key"] == row["model"])
                spec_json = _json(source_model["bundle_spec"]).decode()
                saved = con.execute("SELECT spec_json FROM gallery_transfer_model_specs WHERE model_key=?", (key,)).fetchone()
                if saved and saved[0] != spec_json:
                    raise ValueError("Local encoding contract collision")
                con.execute("INSERT OR IGNORE INTO gallery_transfer_model_specs VALUES (?,?)", (key, spec_json))
                alias = con.execute("SELECT local_id FROM gallery_transfer_aliases WHERE kind='embedding' AND uid=?",
                                    (row["uid"],)).fetchone()
                existing = con.execute("SELECT * FROM gallery_embeddings WHERE embedding_id=?", (alias[0],)).fetchone() if alias else None
                if alias and existing is None:
                    raise ValueError("Previously imported embedding was removed; automatic restoration is disabled")
                if existing is None:
                    # Content fingerprints, not source paths, survive renamed folders/devices.
                    for candidate in con.execute(
                        "SELECT * FROM gallery_embeddings WHERE model_key=? AND source_fingerprint=?",
                        (key, row["source_fingerprint"]),
                    ):
                        if candidate["person_id"] != target_person:
                            raise ValueError("The same source is already assigned to another person")
                        if bytes(candidate["embedding"]) == native_vector and _window_metadata(json.loads(candidate["metadata_json"])) == row["window"]:
                            existing = candidate
                            break
                if existing is not None:
                    if (existing["person_id"] != target_person or existing["model_key"] != key
                            or bytes(existing["embedding"]) != native_vector
                            or _window_metadata(json.loads(existing["metadata_json"])) != row["window"]
                            or existing["source_fingerprint"] != row["source_fingerprint"]):
                        raise ValueError("Embedding identity/content collision")
                    local_id = existing["embedding_id"]
                    duplicates += 1
                else:
                    # No source media is present: use a portable identifier, never an A-PC path.
                    source_id = "gallery-source:" + _hash(row["source_fingerprint"].encode())
                    metadata = {"window": row["window"], "transfer": {"uid": row["uid"],
                                "archive_sha256": package["sha256"], "source_media_available": False}}
                    cursor = con.execute("""INSERT INTO gallery_embeddings
                        (person_id,model_key,embedding,embedding_dim,source_path,source_fingerprint,
                         source_type,session_id,pass_id,direction,quality_score,metadata_json,active,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (target_person, key, sqlite3.Binary(native_vector), target_model["embedding_dim"],
                         source_id, row["source_fingerprint"], row["source_type"], row["session_id"],
                         row["pass_id"], row["direction"], row["quality_score"], _json(metadata).decode(),
                         row["active"], row["created_at"]))
                    local_id = cursor.lastrowid
                    inserted += 1
                con.execute("INSERT OR IGNORE INTO gallery_transfer_aliases VALUES ('embedding',?,?)",
                            (row["uid"], str(local_id)))
        return {"inserted": inserted, "duplicates": duplicates, "skipped": skipped,
                "deleted_skipped": deleted_skipped,
                "backup": str(backup), "retained_archive": str(stored_package)}
