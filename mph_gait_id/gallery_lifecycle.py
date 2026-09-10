"""Scoped Gallery management with transactional edits and pre-edit backups."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from typing import Any

from .database import GalleryRepository, _utc_now
from .gallery_provenance import initialize_provenance, retire_records


class GalleryLifecycle:
    def __init__(self, repository: GalleryRepository) -> None:
        self.repository = repository
        initialize_provenance(repository)

    def _snapshot(self, con: sqlite3.Connection, person_id: str, fragment: dict | None) -> dict:
        person = con.execute("SELECT * FROM persons WHERE person_id=?", (person_id,)).fetchone()
        if person is None:
            raise ValueError("Person no longer exists; refresh Gallery Manager.")
        where, params = "person_id = ?", [person_id]
        if fragment is not None:
            # Use exactly the same grouping as list_gallery_passes, including legacy rows.
            where += """ AND model_key = ? AND source_path = ?
                AND COALESCE(session_id, '') = ? AND COALESCE(pass_id, '') = ?
                AND COALESCE(direction, 'legacy_or_unknown') = ?"""
            params += [fragment[key] for key in (
                "model_key", "source_path", "session_id", "pass_id", "direction"
            )]
        rows = [dict(row) for row in con.execute(
            "SELECT embedding_id, model_key, active, created_at FROM gallery_embeddings WHERE "
            + where + " ORDER BY embedding_id", params)]
        if fragment is not None and not rows:
            raise ValueError("Selected fragment no longer exists; refresh Gallery Manager.")
        token = hashlib.sha256(json.dumps(
            {"person": dict(person), "rows": rows}, sort_keys=True
        ).encode()).hexdigest()
        active = sum(row["active"] == 1 for row in rows)
        return {
            "person_id": person_id, "display_name": person["display_name"],
            "total": len(rows), "active": active, "inactive": len(rows) - active,
            "models": sorted({row["model_key"] for row in rows}),
            "fragment": fragment, "token": token,
            "embedding_ids": [row["embedding_id"] for row in rows],
        }

    def preview(self, person_id: str, fragment: dict | None = None) -> dict:
        with self.repository.connect() as con:
            con.execute("BEGIN")
            return self._snapshot(con, person_id.strip(), fragment)

    def _backup(self) -> str:
        root = self.repository.path.parent / "backups"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        fd, name = tempfile.mkstemp(prefix=f"gallery_before_edit_{stamp}_", suffix=".sqlite3", dir=root)
        os.close(fd)
        # The writer holds BEGIN IMMEDIATE but has not mutated rows yet. A separate
        # reader can back up the committed state, including WAL pages.
        with self.repository.connect() as source, sqlite3.connect(name) as destination:
            source.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Gallery backup verification failed; edit cancelled.")
        return name

    def apply(self, preview: dict[str, Any], action: str, new_name: str = "") -> dict:
        if action not in {"rename", "delete_fragment", "delete_person"}:
            raise ValueError("Unknown Gallery action")
        fragment = preview["fragment"]
        if (action == "delete_fragment") != (fragment is not None):
            raise ValueError("Gallery action and scope do not match")
        name = new_name.strip()
        if action == "rename" and (not name or len(name) > 1024 or "\x00" in name):
            raise ValueError("Display name must contain 1 to 1024 characters and no NUL character")
        with self.repository.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            current = self._snapshot(con, preview["person_id"], fragment)
            if current != preview:
                raise ValueError("Gallery changed after confirmation; refresh and try again.")
            backup = self._backup()
            person_id = current["person_id"]
            if action == "rename":
                con.execute("UPDATE persons SET display_name=?, updated_at=? WHERE person_id=?",
                            (name, _utc_now(), person_id))
                deleted = 0
            else:
                retire_records(con, "embedding", current["embedding_ids"])
                con.executemany("DELETE FROM gallery_embeddings WHERE person_id=? AND embedding_id=?",
                                [(person_id, value) for value in current["embedding_ids"]])
                deleted = current["total"]
                if action == "delete_person":
                    retire_records(con, "person", [person_id])
                    con.execute("DELETE FROM persons WHERE person_id=?", (person_id,))
                else:
                    con.execute("UPDATE persons SET updated_at=? WHERE person_id=?", (_utc_now(), person_id))
        return {"action": action, "person_id": person_id, "deleted_embeddings": deleted,
                "backup_path": backup}
