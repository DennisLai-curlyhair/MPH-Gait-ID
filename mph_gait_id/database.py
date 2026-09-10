from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .source_fingerprint import compute_source_fingerprint


SCHEMA_VERSION = 3


class _ClosingConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3's context manager and then release the file."""

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc, traceback))
        finally:
            self.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class GalleryRepository:
    """SQLite repository whose embeddings are bound to an exact model fingerprint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    schema_version INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_versions (
                    model_key TEXT PRIMARY KEY,
                    method_key TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    input_mode TEXT NOT NULL,
                    fold INTEGER NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    checkpoint_sha256 TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gallery_embeddings (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    source_path TEXT NOT NULL,
                    source_fingerprint TEXT,
                    source_type TEXT NOT NULL,
                    session_id TEXT,
                    pass_id TEXT,
                    direction TEXT,
                    quality_score REAL,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id),
                    FOREIGN KEY(model_key) REFERENCES model_versions(model_key)
                );

                CREATE INDEX IF NOT EXISTS idx_gallery_model
                    ON gallery_embeddings(model_key, active);
                CREATE INDEX IF NOT EXISTS idx_gallery_person
                    ON gallery_embeddings(person_id, model_key, active);
                CREATE INDEX IF NOT EXISTS idx_gallery_source
                    ON gallery_embeddings(source_path, model_key, active);
                """
            )
            row = connection.execute(
                "SELECT schema_version FROM schema_info LIMIT 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_info(schema_version) VALUES (?)",
                    (SCHEMA_VERSION,),
                )
            else:
                current_version = int(row["schema_version"])
                if current_version in {1, 2}:
                    columns = {
                        str(item["name"])
                        for item in connection.execute(
                            "PRAGMA table_info(gallery_embeddings)"
                        ).fetchall()
                    }
                    if "source_fingerprint" not in columns:
                        connection.execute(
                            "ALTER TABLE gallery_embeddings "
                            "ADD COLUMN source_fingerprint TEXT"
                        )
                    if "session_id" not in columns:
                        connection.execute(
                            "ALTER TABLE gallery_embeddings ADD COLUMN session_id TEXT"
                        )
                    if "pass_id" not in columns:
                        connection.execute(
                            "ALTER TABLE gallery_embeddings ADD COLUMN pass_id TEXT"
                        )
                    if "direction" not in columns:
                        connection.execute(
                            "ALTER TABLE gallery_embeddings ADD COLUMN direction TEXT"
                        )
                    connection.execute(
                        "UPDATE schema_info SET schema_version = ?",
                        (SCHEMA_VERSION,),
                    )
                elif current_version != SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Unsupported gallery schema {current_version}; "
                        f"expected {SCHEMA_VERSION}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gallery_fingerprint
                ON gallery_embeddings(source_fingerprint, model_key, active)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_gallery_pass
                ON gallery_embeddings(session_id, pass_id, model_key, active)
                """
            )
        self._backfill_source_fingerprints()

    def _backfill_source_fingerprints(self) -> None:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT
                    e.model_key,
                    e.source_path,
                    m.input_type,
                    m.input_mode
                FROM gallery_embeddings e
                JOIN model_versions m ON m.model_key = e.model_key
                WHERE
                    e.active = 1
                    AND (e.source_fingerprint IS NULL OR e.source_fingerprint = '')
                """
            ).fetchall()

        for row in rows:
            try:
                fingerprint = compute_source_fingerprint(
                    source=str(row["source_path"]),
                    input_type=str(row["input_type"]),
                    mode=str(row["input_mode"]),
                )
            except (FileNotFoundError, OSError, ValueError):
                continue
            with self.connect() as connection:
                connection.execute(
                    """
                    UPDATE gallery_embeddings
                    SET source_fingerprint = ?
                    WHERE
                        model_key = ?
                        AND source_path = ?
                        AND (source_fingerprint IS NULL OR source_fingerprint = '')
                    """,
                    (fingerprint, str(row["model_key"]), str(row["source_path"])),
                )

    def upsert_person(
        self,
        person_id: str,
        display_name: str,
        note: str = "",
        connection: sqlite3.Connection | None = None,
    ) -> None:
        person_id = person_id.strip()
        display_name = display_name.strip()
        if not person_id or not display_name:
            raise ValueError("person_id and display_name must not be empty")
        now = _utc_now()
        def write(target: sqlite3.Connection) -> None:
            target.execute(
                """
                INSERT INTO persons(person_id, display_name, note, status, created_at, updated_at)
                VALUES (?, ?, ?, 'active', ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    note = CASE
                        WHEN excluded.note = '' THEN persons.note
                        ELSE excluded.note
                    END,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (person_id, display_name, note, now, now),
            )
        if connection is not None:
            write(connection)
            return
        with self.connect() as owned_connection:
            write(owned_connection)

    def get_person(self, person_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT person_id, display_name, note, status, created_at, updated_at
                FROM persons
                WHERE person_id = ?
                """,
                (person_id.strip(),),
            ).fetchone()
        return None if row is None else dict(row)

    def upsert_model(
        self,
        model: dict[str, Any],
        connection: sqlite3.Connection | None = None,
    ) -> None:
        now = _utc_now()
        config_json = json.dumps(model, ensure_ascii=True, sort_keys=True)

        def write(target: sqlite3.Connection) -> None:
            existing = target.execute(
                "SELECT checkpoint_sha256, config_json FROM model_versions WHERE model_key = ?",
                (model["model_key"],),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["checkpoint_sha256"]) != str(model["checkpoint_sha256"])
                    or str(existing["config_json"]) != config_json
                ):
                    try:
                        previous_config = json.loads(str(existing["config_json"]))
                    except (TypeError, ValueError):
                        previous_config = None
                    current_config = dict(model)
                    if isinstance(previous_config, dict):
                        previous_comparable = dict(previous_config)
                        current_comparable = dict(current_config)
                        previous_comparable.pop("display_name", None)
                        current_comparable.pop("display_name", None)
                    else:
                        previous_comparable = None
                        current_comparable = None
                    if (
                        str(existing["checkpoint_sha256"])
                        == str(model["checkpoint_sha256"])
                        and previous_comparable == current_comparable
                    ):
                        # A public model rename is presentation metadata, not a
                        # new embedding space. Preserve model_key and Gallery.
                        target.execute(
                            """
                            UPDATE model_versions
                            SET display_name = ?, config_json = ?
                            WHERE model_key = ?
                            """,
                            (model["display_name"], config_json, model["model_key"]),
                        )
                        return
                    raise RuntimeError(
                        "A model_key collision was detected with different checkpoint/config metadata"
                    )
                return
            target.execute(
                """
                INSERT INTO model_versions(
                    model_key, method_key, display_name, input_type, input_mode, fold,
                    checkpoint_path, checkpoint_sha256, embedding_dim, config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model["model_key"],
                    model["method_key"],
                    model["display_name"],
                    model["input_type"],
                    model["input_mode"],
                    int(model["fold"]),
                    model["checkpoint_path"],
                    model["checkpoint_sha256"],
                    int(model["embedding_dim"]),
                    config_json,
                    now,
                ),
            )
        if connection is not None:
            write(connection)
            return
        with self.connect() as owned_connection:
            write(owned_connection)

    def synchronize_bundle_display_name(
        self,
        bundle_id: str,
        checkpoint_sha256: str,
        display_name: str,
    ) -> int:
        """Rename matching model records without changing compatibility keys."""

        updated = 0
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT model_key, display_name, config_json
                FROM model_versions
                WHERE checkpoint_sha256 = ?
                """,
                (str(checkpoint_sha256),),
            ).fetchall()
            for row in rows:
                try:
                    config = json.loads(str(row["config_json"]))
                except (TypeError, ValueError):
                    continue
                if str(config.get("bundle_id")) != str(bundle_id):
                    continue
                if (
                    str(row["display_name"]) == str(display_name)
                    and str(config.get("display_name")) == str(display_name)
                ):
                    continue
                config["display_name"] = str(display_name)
                connection.execute(
                    """
                    UPDATE model_versions
                    SET display_name = ?, config_json = ?
                    WHERE model_key = ?
                    """,
                    (
                        str(display_name),
                        json.dumps(config, ensure_ascii=True, sort_keys=True),
                        str(row["model_key"]),
                    ),
                )
                updated += 1
        return updated

    def find_enrolled_source(
        self,
        model_key: str,
        source_path: str | Path,
        source_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        source = str(Path(source_path).expanduser().resolve())
        fingerprint = str(source_fingerprint or "").strip()
        with self.connect() as connection:
            if fingerprint:
                row = connection.execute(
                    """
                    SELECT person_id, source_path, source_fingerprint
                    FROM gallery_embeddings
                    WHERE
                        model_key = ?
                        AND active = 1
                        AND (source_path = ? OR source_fingerprint = ?)
                    LIMIT 1
                    """,
                    (model_key, source, fingerprint),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT person_id, source_path, source_fingerprint
                    FROM gallery_embeddings
                    WHERE model_key = ? AND source_path = ? AND active = 1
                    LIMIT 1
                    """,
                    (model_key, source),
                ).fetchone()
        return None if row is None else dict(row)

    def source_is_enrolled(
        self,
        model_key: str,
        source_path: str | Path,
        source_fingerprint: str | None = None,
    ) -> bool:
        return self.find_enrolled_source(
            model_key=model_key,
            source_path=source_path,
            source_fingerprint=source_fingerprint,
        ) is not None

    def add_embeddings(
        self,
        person_id: str,
        model_key: str,
        embeddings: np.ndarray,
        source_path: str | Path,
        source_fingerprint: str,
        source_type: str,
        quality_score: float,
        metadata: Iterable[dict[str, Any]],
        connection: sqlite3.Connection | None = None,
    ) -> list[int]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embeddings must have shape [M, D] with M > 0")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings contain NaN or Inf")
        metadata_rows = list(metadata)
        if len(metadata_rows) != matrix.shape[0]:
            raise ValueError("metadata length must match the number of embeddings")
        source = str(Path(source_path).expanduser().resolve())
        fingerprint = str(source_fingerprint).strip()
        if not fingerprint:
            raise ValueError("source_fingerprint must not be empty")
        now = _utc_now()
        def write(target: sqlite3.Connection) -> list[int]:
            inserted: list[int] = []
            for vector, item in zip(matrix, metadata_rows):
                window = item.get("window", {}) if isinstance(item, dict) else {}
                source_metadata = (
                    item.get("source_metadata", {}) if isinstance(item, dict) else {}
                )
                realtime = (
                    source_metadata.get("realtime_enrollment", {})
                    if isinstance(source_metadata, dict)
                    else {}
                )
                session_id = str(realtime.get("session_id") or "").strip() or None
                pass_id = str(window.get("pass_id") or "").strip() or None
                requested_direction = str(
                    window.get("requested_direction") or ""
                ).strip()
                observed_direction = str(
                    window.get("observed_direction") or ""
                ).strip()
                direction = (
                    requested_direction
                    if requested_direction == "front_facing"
                    else observed_direction or requested_direction or None
                )
                cursor = target.execute(
                    """
                    INSERT INTO gallery_embeddings(
                        person_id, model_key, embedding, embedding_dim, source_path,
                        source_fingerprint, source_type, session_id, pass_id, direction,
                        quality_score, metadata_json, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        person_id,
                        model_key,
                        sqlite3.Binary(vector.tobytes(order="C")),
                        int(vector.size),
                        source,
                        fingerprint,
                        source_type,
                        session_id,
                        pass_id,
                        direction,
                        float(quality_score),
                        json.dumps(item, ensure_ascii=True, sort_keys=True),
                        now,
                    ),
                )
                inserted.append(int(cursor.lastrowid))
            return inserted

        if connection is not None:
            return write(connection)
        with self.connect() as owned_connection:
            return write(owned_connection)

    def load_gallery(self, model_key: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    e.embedding_id, e.person_id, p.display_name, e.embedding,
                    e.embedding_dim, e.source_path, e.source_fingerprint,
                    e.quality_score, e.metadata_json
                FROM gallery_embeddings e
                JOIN persons p ON p.person_id = e.person_id
                WHERE e.model_key = ? AND e.active = 1 AND p.status = 'active'
                ORDER BY e.person_id, e.embedding_id
                """,
                (model_key,),
            ).fetchall()
        gallery = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vector.size != int(row["embedding_dim"]):
                raise RuntimeError(f"Corrupt embedding row: {row['embedding_id']}")
            gallery.append(
                {
                    "embedding_id": int(row["embedding_id"]),
                    "person_id": str(row["person_id"]),
                    "display_name": str(row["display_name"]),
                    "embedding": vector,
                    "source_path": str(row["source_path"]),
                    "source_fingerprint": row["source_fingerprint"],
                    "quality_score": row["quality_score"],
                    "metadata": json.loads(row["metadata_json"]),
                }
            )
        return gallery

    def model_keys_for_bundle(
        self,
        bundle_id: str,
        clip_len: int | None = None,
        preprocessing_profile_id: str | None = None,
    ) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT model_key, config_json FROM model_versions ORDER BY model_key"
            ).fetchall()
        keys = []
        for row in rows:
            config = json.loads(row["config_json"])
            compatibility = config.get("compatibility", {})
            same_clip = (
                clip_len is None
                or int(compatibility.get("clip_len", -1)) == int(clip_len)
            )
            same_processing = (
                preprocessing_profile_id is None
                or str(compatibility.get("preprocessing_profile_id", ""))
                == str(preprocessing_profile_id)
            )
            if (
                str(config.get("bundle_id")) == str(bundle_id)
                and same_clip
                and same_processing
            ):
                keys.append(str(row["model_key"]))
        return keys

    def list_gallery_sources(
        self,
        person_id: str,
        model_keys: Sequence[str],
    ) -> list[dict[str, Any]]:
        selected_keys = list(model_keys)
        if not selected_keys:
            return []
        placeholders = ",".join("?" for _ in selected_keys)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    e.person_id,
                    e.model_key,
                    e.source_path,
                    e.source_fingerprint,
                    e.source_type,
                    COUNT(e.embedding_id) AS embedding_count,
                    AVG(e.quality_score) AS mean_quality,
                    MIN(e.created_at) AS first_created_at,
                    MAX(e.created_at) AS last_created_at
                FROM gallery_embeddings e
                WHERE
                    e.active = 1
                    AND e.person_id = ?
                    AND e.model_key IN ({placeholders})
                GROUP BY
                    e.person_id,
                    e.model_key,
                    e.source_path,
                    e.source_fingerprint,
                    e.source_type
                ORDER BY first_created_at, e.source_path
                """,
                [person_id.strip(), *selected_keys],
            ).fetchall()
        return [dict(row) for row in rows]

    def list_gallery_passes(
        self,
        person_id: str,
        model_keys: Sequence[str],
        include_inactive: bool = True,
    ) -> list[dict[str, Any]]:
        selected_keys = list(model_keys)
        if not selected_keys:
            return []
        placeholders = ",".join("?" for _ in selected_keys)
        active_filter = "" if include_inactive else "AND e.active = 1"
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    e.person_id,
                    e.model_key,
                    e.source_path,
                    COALESCE(e.session_id, '') AS session_id,
                    COALESCE(e.pass_id, '') AS pass_id,
                    COALESCE(e.direction, 'legacy_or_unknown') AS direction,
                    COUNT(e.embedding_id) AS total_embeddings,
                    SUM(CASE WHEN e.active = 1 THEN 1 ELSE 0 END) AS active_embeddings,
                    AVG(e.quality_score) AS mean_quality,
                    MIN(e.created_at) AS created_at
                FROM gallery_embeddings e
                WHERE
                    e.person_id = ?
                    AND e.model_key IN ({placeholders})
                    {active_filter}
                GROUP BY
                    e.person_id, e.model_key, e.source_path,
                    COALESCE(e.session_id, ''), COALESCE(e.pass_id, ''),
                    COALESCE(e.direction, 'legacy_or_unknown')
                ORDER BY created_at DESC, e.session_id, e.pass_id
                """,
                [person_id.strip(), *selected_keys],
            ).fetchall()
        return [dict(row) for row in rows]

    def set_pass_embeddings_active(
        self,
        person_id: str,
        model_key: str,
        session_id: str,
        pass_id: str,
        active: bool,
    ) -> int:
        if not session_id.strip() or not pass_id.strip():
            raise ValueError("Pass-level management requires session_id and pass_id")
        now = _utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE gallery_embeddings
                SET active = ?
                WHERE
                    person_id = ?
                    AND model_key = ?
                    AND session_id = ?
                    AND pass_id = ?
                    AND active != ?
                """,
                (
                    1 if active else 0,
                    person_id.strip(),
                    model_key,
                    session_id.strip(),
                    pass_id.strip(),
                    1 if active else 0,
                ),
            )
            count = int(cursor.rowcount)
            if count:
                connection.execute(
                    "UPDATE persons SET updated_at = ? WHERE person_id = ?",
                    (now, person_id.strip()),
                )
        return count

    def deactivate_source_embeddings(
        self,
        person_id: str,
        model_key: str,
        source_path: str | Path,
    ) -> int:
        source = str(source_path)
        now = _utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE gallery_embeddings
                SET active = 0
                WHERE
                    person_id = ?
                    AND model_key = ?
                    AND source_path = ?
                    AND active = 1
                """,
                (person_id.strip(), model_key, source),
            )
            count = int(cursor.rowcount)
            if count:
                connection.execute(
                    "UPDATE persons SET updated_at = ? WHERE person_id = ?",
                    (now, person_id.strip()),
                )
        return count

    def deactivate_person_embeddings(
        self,
        person_id: str,
        model_keys: Sequence[str],
    ) -> int:
        selected_keys = list(model_keys)
        if not selected_keys:
            return 0
        placeholders = ",".join("?" for _ in selected_keys)
        now = _utc_now()
        with self.connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE gallery_embeddings
                SET active = 0
                WHERE
                    person_id = ?
                    AND model_key IN ({placeholders})
                    AND active = 1
                """,
                [person_id.strip(), *selected_keys],
            )
            count = int(cursor.rowcount)
            if count:
                connection.execute(
                    "UPDATE persons SET updated_at = ? WHERE person_id = ?",
                    (now, person_id.strip()),
                )
        return count

    def list_persons(
        self,
        model_keys: Sequence[str] | None = None,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        selected_keys = list(model_keys) if model_keys is not None else None
        with self.connect() as connection:
            if selected_keys is None:
                rows = connection.execute(
                    """
                    SELECT
                        p.person_id, p.display_name, p.note, p.status, p.created_at,
                        COUNT(CASE WHEN e.active = 1 THEN 1 END) AS embedding_count,
                        COUNT(DISTINCT CASE WHEN e.active = 1 THEN e.model_key END) AS model_count
                    FROM persons p
                    LEFT JOIN gallery_embeddings e ON e.person_id = p.person_id
                    GROUP BY p.person_id
                    ORDER BY p.person_id
                    """
                ).fetchall()
            elif not selected_keys:
                rows = []
            else:
                placeholders = ",".join("?" for _ in selected_keys)
                if include_inactive:
                    rows = connection.execute(
                        f"""
                        SELECT
                            p.person_id, p.display_name, p.note, p.status, p.created_at,
                            COUNT(CASE WHEN e.active = 1 THEN 1 END) AS embedding_count,
                            COUNT(DISTINCT e.model_key) AS model_count
                        FROM persons p
                        JOIN gallery_embeddings e ON e.person_id = p.person_id
                        WHERE e.model_key IN ({placeholders})
                        GROUP BY p.person_id
                        ORDER BY p.person_id
                        """,
                        selected_keys,
                    ).fetchall()
                else:
                    rows = connection.execute(
                        f"""
                        SELECT
                            p.person_id, p.display_name, p.note, p.status, p.created_at,
                            COUNT(e.embedding_id) AS embedding_count,
                            COUNT(DISTINCT e.model_key) AS model_count
                        FROM persons p
                        JOIN gallery_embeddings e ON e.person_id = p.person_id
                        WHERE e.active = 1 AND e.model_key IN ({placeholders})
                        GROUP BY p.person_id
                        ORDER BY p.person_id
                        """,
                        selected_keys,
                    ).fetchall()
        return [dict(row) for row in rows]

    def summary(self, model_keys: Sequence[str] | None = None) -> dict[str, Any]:
        selected_keys = list(model_keys) if model_keys is not None else None
        with self.connect() as connection:
            if selected_keys is None:
                persons = int(
                    connection.execute(
                        """
                        SELECT COUNT(DISTINCT person_id)
                        FROM gallery_embeddings
                        WHERE active = 1
                        """
                    ).fetchone()[0]
                )
                models = int(connection.execute("SELECT COUNT(*) FROM model_versions").fetchone()[0])
                embeddings = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM gallery_embeddings WHERE active = 1"
                    ).fetchone()[0]
                )
            elif not selected_keys:
                persons = models = embeddings = 0
            else:
                placeholders = ",".join("?" for _ in selected_keys)
                persons = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(DISTINCT person_id) FROM gallery_embeddings
                        WHERE active = 1 AND model_key IN ({placeholders})
                        """,
                        selected_keys,
                    ).fetchone()[0]
                )
                models = len(selected_keys)
                embeddings = int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*) FROM gallery_embeddings
                        WHERE active = 1 AND model_key IN ({placeholders})
                        """,
                        selected_keys,
                    ).fetchone()[0]
                )
        return {
            "database": str(self.path),
            "schema_version": SCHEMA_VERSION,
            "persons": persons,
            "models": models,
            "active_embeddings": embeddings,
        }
