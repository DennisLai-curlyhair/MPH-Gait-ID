"""Persistent transfer identities and local deletion history."""
from __future__ import annotations

import sqlite3
from typing import Iterable
import uuid

from .database import GalleryRepository, _utc_now


def initialize_provenance(repository: GalleryRepository) -> None:
    with repository.connect() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS gallery_transfer_meta (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), origin TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS gallery_transfer_aliases (
                kind TEXT NOT NULL, uid TEXT NOT NULL, local_id TEXT NOT NULL,
                PRIMARY KEY(kind, uid));
            CREATE INDEX IF NOT EXISTS idx_gallery_transfer_local_id
                ON gallery_transfer_aliases(kind, local_id);
            CREATE TABLE IF NOT EXISTS gallery_transfer_model_specs (
                model_key TEXT PRIMARY KEY, spec_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS gallery_transfer_tombstones (
                kind TEXT NOT NULL, uid TEXT NOT NULL, deleted_at TEXT NOT NULL,
                PRIMARY KEY(kind, uid));
        """)
        con.execute("INSERT OR IGNORE INTO gallery_transfer_meta VALUES (1,?)", (str(uuid.uuid4()),))


def legacy_uid(con: sqlite3.Connection, kind: str, local_id: str | int) -> str:
    origin = con.execute("SELECT origin FROM gallery_transfer_meta WHERE singleton=1").fetchone()[0]
    return f"{origin}:{kind}:{local_id}"


def is_deleted(con: sqlite3.Connection, kind: str, uid: str) -> bool:
    return con.execute(
        "SELECT 1 FROM gallery_transfer_tombstones WHERE kind=? AND uid=?", (kind, uid)
    ).fetchone() is not None


def portable_uid(con: sqlite3.Connection, kind: str, local_id: str | int) -> str:
    row = con.execute(
        "SELECT uid FROM gallery_transfer_aliases WHERE kind=? AND local_id=? ORDER BY uid LIMIT 1",
        (kind, str(local_id)),
    ).fetchone()
    if row:
        if is_deleted(con, kind, row[0]):
            raise ValueError("Deleted Gallery identity is still linked; inspect database provenance")
        return str(row[0])
    uid = legacy_uid(con, kind, local_id)
    if is_deleted(con, kind, uid):
        # Human-readable Person IDs may be reused, but transfer identities must not be.
        uid = f"{uid}:generation:{uuid.uuid4()}"
    con.execute("INSERT INTO gallery_transfer_aliases VALUES (?,?,?)", (kind, uid, str(local_id)))
    return uid


def retire_records(con: sqlite3.Connection, kind: str, local_ids: Iterable[str | int]) -> None:
    now = _utc_now()
    for local_id in local_ids:
        uids = {row[0] for row in con.execute(
            "SELECT uid FROM gallery_transfer_aliases WHERE kind=? AND local_id=?",
            (kind, str(local_id)),
        )}
        # Older exports did not persist their generated UID in the aliases table.
        uids.add(legacy_uid(con, kind, local_id))
        con.executemany("INSERT OR IGNORE INTO gallery_transfer_tombstones VALUES (?,?,?)",
                        [(kind, uid, now) for uid in uids])
        con.execute("DELETE FROM gallery_transfer_aliases WHERE kind=? AND local_id=?",
                    (kind, str(local_id)))
