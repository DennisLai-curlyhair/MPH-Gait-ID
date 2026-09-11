"""Explicit, backed-up migration of v0.3.0 Gallery feature contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import uuid

from .database import GalleryRepository, _utc_now
from .embedding_contract import LEGACY_V030_ENCODER, bundle_spec, compatible_definition
from .model_record import build_model_record
from .model_store import ModelStore
from .scripts.download_assets import load_manifest


def plan_migration(con, store, confirmed_bundles=()):
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"model_versions", "gallery_embeddings"} <= tables:
        raise ValueError("Not an initialized MPH-Gait ID Gallery")
    assets = {a["sha256"] for a in load_manifest()["assets"].values()
              if a["provider"] == "project_release"}
    plan = []
    targets = set()
    for row in con.execute("SELECT * FROM model_versions ORDER BY model_key"):
        config = json.loads(row["config_json"])
        old = config.get("compatibility", {})
        if old.get("contract_version") == 2:
            continue
        item = {"old_key": row["model_key"], "status": "blocked", "reason": "",
                "old_config": config, "legacy_spec": None}
        plan.append(item)
        bundle = store.bundles().get(config.get("bundle_id"))
        if bundle is None or not store.verify(bundle.bundle_id)["valid"]:
            item["reason"] = "Matching verified model bundle is missing"
            continue
        saved = (con.execute("SELECT spec_json FROM gallery_transfer_model_specs WHERE model_key=?",
                             (row["model_key"],)).fetchone()
                 if "gallery_transfer_model_specs" in tables else None)
        spec = json.loads(saved[0]) if saved else config.get("inference_spec")
        if spec is None:
            if bundle.bundle_id not in confirmed_bundles or bundle.checkpoint_sha256 not in assets:
                item["reason"] = (
                    "No recorded coordinate/encoder provenance. Re-encode saved sources, or confirm "
                    "this bundled model was used with its unmodified v0.3.0 settings using "
                    f"--confirm-legacy-bundle {bundle.bundle_id}"
                )
                continue
            spec = {**bundle_spec(bundle), "encoder_sha256": LEGACY_V030_ENCODER}
            item["explicit_confirmation"] = True
        item["legacy_spec"] = spec
        try:
            record = build_model_record(bundle, store, old["clip_len"], old["drop_first_frames"],
                                        old["preprocessing_profile_id"])
        except (KeyError, ValueError, TypeError):
            item["reason"] = "Incomplete legacy compatibility metadata"
            continue
        if not compatible_definition(old, record["compatibility"], spec, record["inference_spec"]):
            item["reason"] = "Encoder, checkpoint, coordinate, input or preprocessing contract differs"
            continue
        key = record["model_key"]
        if key in targets or con.execute("SELECT 1 FROM model_versions WHERE model_key=?", (key,)).fetchone():
            item["reason"] = "Destination model already exists; automatic merging is disabled"
            continue
        targets.add(key)
        item.update(status="ready", target=record, embeddings=con.execute(
            "SELECT COUNT(*) FROM gallery_embeddings WHERE model_key=?", (row["model_key"],)
        ).fetchone()[0])
    return plan


def migrate_gallery(path: Path, store: ModelStore, *, apply=False, confirmed_bundles=()):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gallery does not exist: {path}")
    # Preview is read-only, including for older database schemas.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        plan = plan_migration(con, store, confirmed_bundles)
    report = {"applied": False, "models": plan}
    if not apply or not plan:
        return report
    if any(item["status"] != "ready" for item in plan):
        raise ValueError("Migration has blocked models. Resolve them before applying; no rows changed")
    repository = GalleryRepository(path)
    backup = path.with_name(f"{path.name}.before-contract-v2-{uuid.uuid4().hex}.sqlite3")
    with repository.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        if plan_migration(con, store, confirmed_bundles) != plan:
            raise ValueError("Gallery or model definitions changed; preview again")
        # The writer lock prevents concurrent edits; backup reads the committed snapshot.
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(backup) as target:
                source.backup(target)
        try:
            backup.chmod(0o600)
        except OSError:
            pass
        con.execute("CREATE TABLE IF NOT EXISTS gallery_transfer_model_specs "
                    "(model_key TEXT PRIMARY KEY, spec_json TEXT NOT NULL)")
        con.execute("CREATE TABLE IF NOT EXISTS gallery_contract_migrations "
                    "(old_key TEXT PRIMARY KEY, new_key TEXT NOT NULL, audit_json TEXT NOT NULL)")
        for item in plan:
            record, old_key = item["target"], item["old_key"]
            repository.upsert_model(record, connection=con)
            for row in con.execute("SELECT embedding_id,metadata_json FROM gallery_embeddings WHERE model_key=?",
                                   (old_key,)).fetchall():
                metadata = json.loads(row["metadata_json"])
                metadata["model_key"] = record["model_key"]
                metadata["previous_model_key"] = old_key
                con.execute("UPDATE gallery_embeddings SET model_key=?,metadata_json=? WHERE embedding_id=?",
                            (record["model_key"], json.dumps(metadata), row["embedding_id"]))
            con.execute("INSERT INTO gallery_transfer_model_specs VALUES (?,?)",
                        (record["model_key"], json.dumps(record["inference_spec"], sort_keys=True)))
            con.execute("INSERT INTO gallery_contract_migrations VALUES (?,?,?)",
                        (old_key, record["model_key"], json.dumps(
                            {**item, "backup": str(backup), "migrated_at": _utc_now()}, sort_keys=True)))
            con.execute("DELETE FROM gallery_transfer_model_specs WHERE model_key=?", (old_key,))
            con.execute("DELETE FROM model_versions WHERE model_key=?", (old_key,))
        if con.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("Foreign-key validation failed; migration rolled back")
    return {**report, "applied": True, "backup": str(backup)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply after creating a SQLite backup; default is preview")
    parser.add_argument("--confirm-legacy-bundle", action="append", default=[],
                        help="Confirm this bundled model used unchanged v0.3.0 settings when no provenance was saved")
    args = parser.parse_args()
    store = ModelStore(args.bundle_root) if args.bundle_root else ModelStore()
    print(json.dumps(migrate_gallery(args.db, store, apply=args.apply,
                                    confirmed_bundles=args.confirm_legacy_bundle), indent=2))


if __name__ == "__main__":
    main()
