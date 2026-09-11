#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    nested,
    resolve_system_path,
)
from mph_gait_id.database import GalleryRepository  # noqa: E402
from mph_gait_id.model_store import ModelStore  # noqa: E402
from mph_gait_id.runtime import SystemModelRuntime  # noqa: E402
from mph_gait_id.services import RecognitionService, RegistrationService  # noqa: E402


LOGGER = logging.getLogger("mph_gait_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MPH-Gait ID."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--database", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db", help="Initialize the SQLite gallery database.")
    doctor = commands.add_parser("doctor", help="Verify local model bundles and database availability.")
    doctor.add_argument("--bundles", default="all")
    doctor.add_argument("--skip-hash", action="store_true")
    commands.add_parser("list-models", help="List model bundles stored inside this application.")
    commands.add_parser("list-persons", help="List enrolled identities.")

    enroll = commands.add_parser("enroll", help="Enroll one identity from a sequence folder.")
    _add_runtime_args(enroll)
    enroll.add_argument("--source", required=True)
    enroll.add_argument("--person-id", required=True)
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--note", default="")
    enroll.add_argument("--min-clips", type=int, default=None)
    enroll.add_argument("--max-embeddings", type=int, default=None)
    enroll.add_argument("--allow-duplicate-source", action="store_true")

    recognize = commands.add_parser("recognize", help="Recognize one sequence folder.")
    _add_runtime_args(recognize)
    recognize.add_argument("--source", required=True)
    recognize.add_argument("--threshold", type=float, default=None)
    recognize.add_argument("--min-margin", type=float, default=None)
    recognize.add_argument("--top-k-per-identity", type=int, default=None)
    recognize.add_argument("--allow-source-overlap", action="store_true")
    return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--drop-first-frames", type=int, default=None)


def _repository(args: argparse.Namespace, config: dict[str, Any]) -> GalleryRepository:
    configured = args.database or nested(
        config,
        "storage",
        "database",
        "../data/gallery.sqlite3",
    )
    repository = GalleryRepository(resolve_system_path(configured))
    repository.initialize()
    return repository


def _model_store(config: dict[str, Any]) -> ModelStore:
    root = nested(config, "storage", "model_bundles", "model_bundles")
    return ModelStore(resolve_system_path(root))


def _runtime(args: argparse.Namespace, config: dict[str, Any], store: ModelStore) -> SystemModelRuntime:
    bundle = args.bundle or nested(
        config,
        "runtime",
        "bundle",
        "mph_gait_fixed_special5_seed0_split0",
    )
    device = args.device or nested(config, "runtime", "device", "auto")
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(nested(config, "runtime", "batch_size", 16))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(nested(config, "runtime", "num_workers", 0))
    return SystemModelRuntime(
        bundle_id=bundle,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_store=store,
        runtime_clip_len=args.window_size,
        runtime_drop_first_frames=args.drop_first_frames,
        preprocessing_profile_id=str(
            nested(
                config,
                "realtime",
                "processing_version_id",
                "person_foreground_pointcloud_v1",
            )
        ),
    )


def _extract(runtime: SystemModelRuntime, args: argparse.Namespace, config: dict[str, Any]):
    stride = args.stride
    if stride is None:
        stride = int(nested(config, "folder_input", "stride", 5))
    return runtime.extract_folder(
        source=args.source,
        window_size=args.window_size,
        stride=stride,
        drop_first_frames=args.drop_first_frames,
    )


def _save_result(
    config: dict[str, Any],
    operation: str,
    result: dict[str, Any],
    output_root: str | None = None,
) -> Path:
    configured = output_root or nested(config, "storage", "output_root", "../outputs")
    root = resolve_system_path(configured)
    output_dir = root / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{operation}"
    from .report_io import write_result
    return write_result(output_dir, result)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    repository = _repository(args, config)
    store = _model_store(config)

    if args.command == "init-db":
        _print(repository.summary())
        return 0
    if args.command == "list-models":
        _print([bundle.public_record() for bundle in store.bundles().values()])
        return 0
    if args.command == "list-persons":
        _print({"database": str(repository.path), "persons": repository.list_persons()})
        return 0
    if args.command == "doctor":
        bundle_ids = (
            list(store.bundles())
            if args.bundles.strip().lower() == "all"
            else [value.strip() for value in args.bundles.split(",") if value.strip()]
        )
        reports = [store.verify(bundle_id, check_hash=not args.skip_hash) for bundle_id in bundle_ids]
        result = {
            "system_root": str(SYSTEM_ROOT),
            "database": repository.summary(),
            "models": reports,
            "passed": all(item["valid"] for item in reports),
        }
        _print(result)
        return 0 if result["passed"] else 2

    runtime = _runtime(args, config, store)
    batch = _extract(runtime, args, config)
    if args.command == "enroll":
        min_clips = args.min_clips
        if min_clips is None:
            min_clips = int(nested(config, "registration", "min_clips", 5))
        maximum = args.max_embeddings
        if maximum is None:
            maximum = int(nested(config, "registration", "max_embeddings", 10))
        result = RegistrationService(repository).enroll(
            batch=batch,
            person_id=args.person_id,
            display_name=args.name,
            note=args.note,
            min_embeddings=min_clips,
            max_embeddings=maximum,
            allow_duplicate_source=args.allow_duplicate_source,
        )
    elif args.command == "recognize":
        top_k = args.top_k_per_identity
        if top_k is None:
            top_k = int(nested(config, "recognition", "top_k_per_identity", 3))
        threshold = args.threshold
        if threshold is None:
            threshold = nested(config, "recognition", "unknown_threshold", None)
        min_margin = args.min_margin
        if min_margin is None:
            min_margin = float(nested(config, "recognition", "min_margin", 0.03))
        result = RecognitionService(repository).recognize(
            batch=batch,
            top_k_per_identity=top_k,
            threshold=threshold,
            min_margin=min_margin,
            allow_source_overlap=args.allow_source_overlap,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")

    try:
        _save_result(config, args.command, result, output_root=args.output_root)
    except (OSError, ValueError, TypeError) as exc:
        from .report_io import report_warning
        report_warning(result, exc)
    _print(result)
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        if args.log_level == "DEBUG":
            LOGGER.exception("Unhandled application error")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
