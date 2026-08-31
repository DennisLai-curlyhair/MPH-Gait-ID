#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.realtime.pipeline import (
    RealtimeConfig,
    RealtimePipeline,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Headless realtime enrollment smoke test with an isolated Gallery"
    )
    value.add_argument(
        "--replay",
        default=str(SYSTEM_ROOT / "data" / "replay" / "P3_C4_V001"),
    )
    value.add_argument(
        "--bundle",
        default="pointnet_tmax_fixed_special5_seed0_split0",
    )
    value.add_argument("--clip-len", type=int, default=15)
    value.add_argument("--fps", type=float, default=120.0)
    value.add_argument("--timeout", type=float, default=180.0)
    value.add_argument("--person-id", default="SMOKE_P003")
    value.add_argument("--name", default="Replay smoke P003")
    value.add_argument("--min-embeddings", type=int, default=5)
    value.add_argument("--max-embeddings", type=int, default=6)
    value.add_argument(
        "--cancel-after-embeddings",
        type=int,
        default=0,
        help="Cancel after this many embeddings and require an unchanged Gallery.",
    )
    value.add_argument(
        "--expect-failure",
        action="store_true",
        help="Require enrollment_failed and an unchanged Gallery.",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="mph_gait_id_enroll_") as temp:
        temp_root = Path(temp)
        controller = GaitApplicationController(
            database_path=temp_root / "gallery.sqlite3",
            output_root=temp_root / "outputs",
        )
        config = RealtimeConfig(
            bundle_id=args.bundle,
            operation="enroll",
            source_mode="replay",
            replay_path=args.replay,
            replay_fps=args.fps,
            replay_loop=False,
            detector="replay_depth",
            clip_len=args.clip_len,
            enrollment_person_id=args.person_id,
            enrollment_display_name=args.name,
            enrollment_note="headless replay enrollment smoke test",
            enrollment_duration_s=300.0,
            enrollment_warmup_s=0.0,
            enrollment_min_embeddings=args.min_embeddings,
            enrollment_max_embeddings=args.max_embeddings,
            enrollment_stride=args.clip_len,
            enrollment_session_root=str(temp_root / "sessions"),
            guided_passes=False,
        )
        pipeline = RealtimePipeline(controller, config)
        pipeline.start()
        deadline = time.monotonic() + args.timeout
        states: dict[str, int] = {}
        terminal = None
        cancelled = False
        while time.monotonic() < deadline:
            snapshot = pipeline.poll_latest()
            if snapshot is not None:
                states[snapshot.state] = states.get(snapshot.state, 0) + 1
                if snapshot.state in {"enrolled", "enrollment_failed", "error"}:
                    terminal = snapshot
                    break
                if (
                    args.cancel_after_embeddings > 0
                    and snapshot.enrollment_embeddings
                    >= args.cancel_after_embeddings
                ):
                    pipeline.stop()
                    cancelled = True
                    break
            if not pipeline.running and snapshot is None:
                break
            time.sleep(0.02)
        pipeline.stop()

        summary = controller.database_summary(
            args.bundle,
            clip_len=args.clip_len,
        )
        result = terminal.result if terminal is not None else None
        if args.expect_failure:
            passed = bool(
                terminal is not None
                and terminal.state == "enrollment_failed"
                and summary["persons"] == 0
                and summary["models"] == 0
                and summary["active_embeddings"] == 0
            )
        elif args.cancel_after_embeddings > 0:
            passed = bool(
                cancelled
                and summary["persons"] == 0
                and summary["models"] == 0
                and summary["active_embeddings"] == 0
            )
        else:
            passed = bool(
                terminal is not None
                and terminal.state == "enrolled"
                and summary["persons"] == 1
                and summary["models"] == 1
                and args.min_embeddings
                <= summary["active_embeddings"]
                <= args.max_embeddings
            )
        report = {
            "status": "ok" if passed else "failed",
            "bundle": args.bundle,
            "clip_len": args.clip_len,
            "replay": str(Path(args.replay).expanduser().resolve()),
            "terminal_state": (
                "cancelled_by_test"
                if cancelled
                else (terminal.state if terminal is not None else None)
            ),
            "terminal_message": terminal.message if terminal is not None else None,
            "cancelled": cancelled,
            "state_counts": states,
            "gallery_summary": summary,
            "result": result,
            "session_manifests": sorted(
                str(path.relative_to(temp_root))
                for path in (temp_root / "sessions").glob("*/*.json")
            ),
            "isolated_temp_gallery": True,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
