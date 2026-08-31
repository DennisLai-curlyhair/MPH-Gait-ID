#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.controller import GaitApplicationController  # noqa: E402
from mph_gait_id.realtime.detection import (  # noqa: E402
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_SEG_MODEL,
)
from mph_gait_id.realtime.pipeline import (  # noqa: E402
    RealtimeConfig,
    RealtimePipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Start pass -> End -> Review -> temporary Gallery commit."
    )
    parser.add_argument(
        "--source",
        choices=["replay", "azure_kinect"],
        default="replay",
    )
    parser.add_argument(
        "--replay",
        default=str(SYSTEM_ROOT / "data" / "replay" / "P3_C4_V001"),
    )
    parser.add_argument(
        "--detector",
        choices=["replay_depth", "yolo", "yolo_seg"],
        default="replay_depth",
    )
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--target-embeddings", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    states: Counter[str] = Counter()
    started = time.perf_counter()
    with TemporaryDirectory(prefix="mph-gait-id-guided-") as raw:
        root = Path(raw)
        controller = GaitApplicationController(
            database_path=root / "gallery.sqlite3",
            output_root=root / "outputs",
        )
        pipeline = RealtimePipeline(
            controller,
            RealtimeConfig(
                bundle_id="pointnet_tmax_fixed_special5_seed0_split0",
                operation="enroll",
                source_mode=args.source,
                replay_path=args.replay,
                replay_fps=float(args.fps),
                replay_loop=False,
                detector=args.detector,
                yolo_weights=(
                    DEFAULT_YOLO_SEG_MODEL
                    if args.detector == "yolo_seg"
                    else DEFAULT_YOLO_MODEL
                ),
                device="auto",
                clip_len=15,
                num_points=1024,
                enrollment_person_id="MPH_GAIT_ID_SELFTEST",
                enrollment_display_name="MPH-Gait ID Guided Self Test",
                enrollment_note="temporary validation only",
                enrollment_duration_s=float(args.timeout),
                enrollment_warmup_s=1.0,
                enrollment_min_embeddings=2,
                enrollment_max_embeddings=5,
                enrollment_stride=5,
                enrollment_session_root=str(root / "sessions"),
                guided_passes=True,
            ),
        )
        target = max(2, int(args.target_embeddings))
        error: str | None = None
        review = None
        pipeline.start()
        pipeline.start_enrollment_pass("left_to_right")
        pass_ended = False
        finish_requested = False
        try:
            while time.perf_counter() - started < float(args.timeout):
                snapshot = pipeline.poll_latest()
                if snapshot is None:
                    time.sleep(0.05)
                    continue
                states[snapshot.state] += 1
                if snapshot.state == "error":
                    error = snapshot.message
                    break
                if snapshot.enrollment_embeddings >= target and not pass_ended:
                    pipeline.end_enrollment_pass(False)
                    pass_ended = True
                passes = pipeline.enrollment_passes()
                if (
                    pass_ended
                    and passes
                    and passes[-1].get("ended_at") is not None
                    and not finish_requested
                ):
                    pipeline.finish_enrollment_for_review()
                    finish_requested = True
                if snapshot.state == "enrollment_review":
                    review = snapshot
                    break
        finally:
            if not finish_requested or review is None:
                pipeline.stop()
        if error or review is None:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": error or "Timed out before enrollment review",
                        "states": dict(states),
                        "passes": pipeline.enrollment_passes(),
                    },
                    indent=2,
                )
            )
            return 1
        if states["starting"] < 1 or states["source_ready"] < 1:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": "Missing background startup state snapshots",
                        "states": dict(states),
                    },
                    indent=2,
                )
            )
            return 1
        deadline = time.perf_counter() + 3.0
        while pipeline.running and time.perf_counter() < deadline:
            time.sleep(0.02)
        passes = pipeline.enrollment_passes()
        selected = [
            str(item["pass_id"])
            for item in passes
            if not item.get("discarded") and int(item.get("embedding_count", 0)) > 0
        ]
        result = pipeline.commit_enrollment(selected)
        db_passes = controller.list_gallery_passes(
            "pointnet_tmax_fixed_special5_seed0_split0",
            "MPH_GAIT_ID_SELFTEST",
            clip_len=15,
            include_inactive=True,
        )
        output = {
            "status": "ok",
            "source": args.source,
            "replay": (
                str(Path(args.replay).expanduser().resolve())
                if args.source == "replay"
                else None
            ),
            "detector": args.detector,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "states": dict(states),
            "passes": passes,
            "selected_pass_ids": selected,
            "stored_embeddings": result.get("stored_embeddings", 0),
            "database_pass_rows": len(db_passes),
            "formal_gallery_modified": False,
        }
        print(json.dumps(output, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
