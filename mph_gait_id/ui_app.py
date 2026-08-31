#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.config import DEFAULT_CONFIG_PATH  # noqa: E402
from mph_gait_id.controller import GaitApplicationController  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MPH-Gait ID desktop UI."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--database", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate this standalone package and print available model bundles.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = GaitApplicationController(
        config_path=args.config,
        database_path=args.database,
        output_root=args.output_root,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "system_root": str(SYSTEM_ROOT),
                    "database": controller.database_summary(),
                    "model_bundles": [
                        {
                            **bundle.public_record(),
                            "provisional_unknown_threshold": controller.provisional_threshold(key),
                        }
                        for key, bundle in controller.available_bundles().items()
                    ],
                    "status": "ready",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    import tkinter as tk

    from mph_gait_id.ui.main_window import GaitIdentityWindow

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(
            "Unable to open the desktop window. Run this command from a graphical session "
            "with DISPLAY available.\n"
            f"Tk error: {exc}",
            file=sys.stderr,
        )
        return 2
    GaitIdentityWindow(root, controller)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
