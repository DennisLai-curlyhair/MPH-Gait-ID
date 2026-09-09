"""Headless access to the same Gallery transfer service used by the desktop UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH
from .controller import GaitApplicationController


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transfer a model-compatible Gallery without loading model weights.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--database", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export every model and identity, including inactive entries.")
    export.add_argument("archive")
    preview = commands.add_parser("preview", help="Validate the archive and print the import preview as JSON.")
    preview.add_argument("archive")
    load = commands.add_parser("import", help="Import against a saved preview; conflicts default to skip.")
    load.add_argument("archive")
    load.add_argument("--preview-file", required=True)
    load.add_argument("--person-map", help="JSON object mapping each incoming person uid to a local ID or null.")
    load.add_argument("--apply", action="store_true", help="Confirm the merge and the source bundle/enrollment settings.")
    args = parser.parse_args(argv)
    if args.command == "import" and not args.apply:
        parser.error("Import requires --apply after reviewing compatibility and identity conflicts")
    controller = GaitApplicationController(config_path=args.config, database_path=args.database)
    if args.command == "export":
        result = controller.export_gallery(args.archive)
    elif args.command == "preview":
        result = controller.preview_gallery_import(args.archive)
    else:
        plan = json.loads(Path(args.preview_file).read_text(encoding="utf-8"))
        mapping = (
            json.loads(Path(args.person_map).read_text(encoding="utf-8")) if args.person_map
            else {person["uid"]: None if person["status"] in {"conflict", "deleted"} else person["target_id"]
                  for person in plan["persons"]}
        )
        result = controller.import_gallery(args.archive, plan, mapping)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
