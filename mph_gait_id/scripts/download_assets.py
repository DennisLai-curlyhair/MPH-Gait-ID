#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = SYSTEM_ROOT / "assets" / "manifest.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    value = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("Asset manifest must be a mapping")
    return value


def valid(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def download_url(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != expected_sha256:
            raise RuntimeError("Downloaded file failed SHA256 verification; existing file retained")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_ultralytics(model_id: str, target: Path, expected_sha256: str) -> None:
    if model_id not in {"yolov8n.pt", "yolov8n-seg.pt"}:
        raise ValueError("Only the pinned YOLOv8n detector and segmenter are supported")
    url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_id}"
    download_url(url, target, expected_sha256)


def asset_url(asset: dict[str, Any]) -> str | None:
    if asset.get("provider") == "url":
        return str(asset.get("url") or "") or None
    if asset.get("provider") == "project_release":
        base = os.environ.get("GAIT_IDENTITY_ASSET_BASE_URL", "").rstrip("/")
        name = str(asset.get("release_name") or "")
        return f"{base}/{name}" if base and name else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify application assets")
    parser.add_argument("--profile", default="offline-demo")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    profiles = manifest.get("profiles", {})
    assets = manifest.get("assets", {})
    names = profiles.get(args.profile)
    if not isinstance(names, list):
        parser.error(f"Unknown profile: {args.profile}")
    failures = 0
    for name in names:
        asset = assets[name]
        target = SYSTEM_ROOT / str(asset["target"])
        expected = str(asset["sha256"])
        if valid(target, expected):
            print(f"OK      {name}: {target.relative_to(SYSTEM_ROOT)}")
            continue
        if args.check:
            print(f"MISSING {name}: {target.relative_to(SYSTEM_ROOT)}")
            failures += 1
            continue
        provider = str(asset.get("provider"))
        try:
            if provider == "ultralytics":
                download_ultralytics(str(asset["model_id"]), target, expected)
            else:
                url = asset_url(asset)
                if not url:
                    raise RuntimeError(
                        "Set GAIT_IDENTITY_ASSET_BASE_URL to the project Release/Hugging "
                        "Face asset directory, or place this checkpoint manually"
                    )
                download_url(url, target, expected)
            if not valid(target, expected):
                raise RuntimeError("downloaded file failed SHA256 verification")
            print(f"DOWNLOADED {name}: {target.relative_to(SYSTEM_ROOT)}")
        except Exception as exc:
            print(f"FAILED  {name}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
