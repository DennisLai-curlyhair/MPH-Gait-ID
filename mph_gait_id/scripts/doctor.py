#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.config import load_config
from mph_gait_id.model_store import ModelStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the local application environment")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--kinect", action="store_true")
    parser.add_argument("--sam", action="store_true")
    args = parser.parse_args()
    checks: list[dict[str, object]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    record("python", sys.version_info >= (3, 10), platform.python_version())
    core_modules = (
        ("numpy", "numpy", ">=1.24,<2.0"),
        ("cv2", "opencv-python", ">=4.8,<4.11"),
        ("PIL", "Pillow", ">=10.0,<11.0"),
        ("yaml", "PyYAML", ">=6.0,<7.0"),
        ("torch", "torch", ">=2.6,<3.0"),
    )
    for module_name, distribution, specification in core_modules:
        try:
            module = importlib.import_module(module_name)
            installed = str(
                getattr(module, "__version__", "")
                or importlib.metadata.version(distribution)
            )
            compatible = Version(installed) in SpecifierSet(specification)
            record(
                module_name,
                compatible,
                f"{installed} | required {specification}",
            )
        except Exception as exc:
            record(module_name, False, f"{type(exc).__name__}: {exc}")
    try:
        config = load_config()
        record("config", int(config.get("schema_version", 0)) == 4, str(config["_config_path"]))
    except Exception as exc:
        record("config", False, f"{type(exc).__name__}: {exc}")
    try:
        bundles = ModelStore().bundles()
        record("model_bundles", bool(bundles), f"{len(bundles)} available")
    except Exception as exc:
        record("model_bundles", False, f"{type(exc).__name__}: {exc}")
    if args.realtime:
        realtime_modules = (
            ("ultralytics", "ultralytics", "==8.3.221"),
            ("pyk4a", "pyk4a", "==1.5.0"),
        )
        for module_name, distribution, specification in realtime_modules:
            try:
                module = importlib.import_module(module_name)
                installed = str(
                    getattr(module, "__version__", "")
                    or importlib.metadata.version(distribution)
                )
                compatible = (
                    True
                    if not specification
                    else Version(installed) in SpecifierSet(specification)
                )
                detail = installed if not specification else f"{installed} | required {specification}"
                record(module_name, compatible, detail)
            except Exception as exc:
                record(module_name, False, f"{type(exc).__name__}: {exc}")
    if args.sam:
        try:
            module = importlib.import_module("segment_anything")
            version = str(getattr(module, "__version__", "installed"))
            record("segment_anything", True, version)
        except Exception as exc:
            record("segment_anything", False, f"{type(exc).__name__}: {exc}")
    if args.kinect:
        try:
            from mph_gait_id.realtime.devices import probe_azure_kinect

            # An explicit --kinect check must verify camera ownership, not only
            # enumerate the USB device. Enumeration can succeed while Media
            # Foundation still rejects the color camera with E_ACCESSDENIED.
            status = probe_azure_kinect(check_access=True)
            record("azure_kinect", status.connected, status.message)
        except Exception as exc:
            record("azure_kinect", False, f"{type(exc).__name__}: {exc}")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(bool(item["ok"]) for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
