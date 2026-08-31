#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "mph_gait_id"
FORBIDDEN_TEXT = (
    "/home/yenting/",
    "Point_Cloud_Gait_Recognition_V2/",
    "gait_identity_system_realtime_v7",
    "gallery_v7.sqlite3",
    "outputs_v7",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    return yaml.safe_load(
        (PACKAGE_ROOT / "assets/manifest.yaml").read_text(encoding="utf-8")
    ) or {}


def validate_assets() -> dict[str, Any]:
    manifest = load_manifest()
    assets = manifest["assets"]
    required = set(manifest["profiles"]["realtime-yolo"])
    records: dict[str, Any] = {}
    for name, metadata in assets.items():
        target = PACKAGE_ROOT / metadata["target"]
        present = target.is_file()
        expected = metadata["sha256"]
        valid = present and sha256(target) == expected
        records[name] = {
            "present": present,
            "valid": valid,
            "size_bytes": target.stat().st_size if present else 0,
            "required_for_default": name in required,
        }
        if name in required and not valid:
            raise RuntimeError(f"Missing or invalid required asset: {name}")
    if records["sam_vit_b"]["present"]:
        raise RuntimeError("Optional SAM checkpoint must not be committed")
    return records


def validate_clean_runtime_dirs() -> None:
    for directory in (ROOT / "data", ROOT / "outputs"):
        unexpected = [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        if unexpected:
            names = ", ".join(str(path.relative_to(ROOT)) for path in unexpected)
            raise RuntimeError(f"Generated runtime data is included: {names}")


def validate_text() -> list[str]:
    failures: list[str] = []
    suffixes = {".py", ".sh", ".ps1", ".md", ".yaml", ".yml", ".toml", ".txt"}
    current = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == current:
            continue
        if path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in FORBIDDEN_TEXT:
            if marker in text:
                failures.append(f"{path.relative_to(ROOT)}: {marker}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MPH-Gait ID release")
    parser.add_argument(
        "--skip-package-import",
        action="store_true",
        help="Skip importing ModelStore; useful before dependencies are installed.",
    )
    args = parser.parse_args()

    required = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        PACKAGE_ROOT / "configs/system.yaml",
        PACKAGE_ROOT / "assets/manifest.yaml",
        PACKAGE_ROOT / "ui_app.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required release files: {missing}")

    validate_clean_runtime_dirs()
    text_failures = validate_text()
    if text_failures:
        raise RuntimeError(
            "Workspace-specific references remain:\n" + "\n".join(text_failures)
        )
    assets = validate_assets()

    bundles: list[str] | str = "package import skipped"
    if not args.skip_package_import:
        sys.path.insert(0, str(ROOT))
        from mph_gait_id.model_store import ModelStore

        bundles = sorted(ModelStore().bundles())

    print(
        json.dumps(
            {
                "status": "ready",
                "root": str(ROOT),
                "bundles": bundles,
                "assets": assets,
                "gallery_database_included": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
