#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.controller import (  # noqa: E402
    GaitApplicationController,
)
from mph_gait_id.realtime.pipeline import (  # noqa: E402
    RealtimeConfig,
    RealtimePipeline,
)
from mph_gait_id.realtime.detection import (  # noqa: E402
    DEFAULT_YOLO_MODEL,
    DEFAULT_YOLO_SEG_MODEL,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one live gait inference.")
    parser.add_argument(
        "--detector",
        choices=["yolo", "yolo_seg", "yolo_sam"],
        default="yolo",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sam-refresh-interval", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    controller = GaitApplicationController()
    pipeline = RealtimePipeline(
        controller,
        RealtimeConfig(
            bundle_id="pointnet_tmax_fixed_special5_seed0_split0",
            operation="recognize",
            source_mode="azure_kinect",
            detector=args.detector,
            yolo_weights=(
                DEFAULT_YOLO_SEG_MODEL
                if args.detector == "yolo_seg"
                else DEFAULT_YOLO_MODEL
            ),
            device="auto",
            clip_len=15,
            inference_stride=5,
            num_points=1024,
            sam_refresh_interval=max(1, int(args.sam_refresh_interval)),
        ),
    )
    states: Counter[str] = Counter()
    valid_snapshots = 0
    max_points = 0
    max_buffer = 0
    detection_times = []
    started = time.perf_counter()
    inference_snapshot = None
    error = None
    pipeline.start()
    try:
        while time.perf_counter() - started < float(args.timeout):
            snapshot = pipeline.poll_latest()
            if snapshot is None:
                time.sleep(0.05)
                continue
            states[snapshot.state] += 1
            if snapshot.person_point_count > 0:
                valid_snapshots += 1
                max_points = max(max_points, snapshot.person_point_count)
                max_buffer = max(max_buffer, snapshot.buffer_size)
                detection_times.append(snapshot.detection_ms)
            if snapshot.state == "error":
                error = snapshot.message
                break
            if snapshot.diagnostics.get("inference_ran"):
                inference_snapshot = snapshot
                break
    finally:
        pipeline.stop()
    result = {
        "status": "ok" if inference_snapshot is not None else "failed",
        "detector": args.detector,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "states": dict(states),
        "valid_snapshots": valid_snapshots,
        "max_person_points": max_points,
        "max_buffer": max_buffer,
        "mean_detection_ms": (
            round(sum(detection_times) / len(detection_times), 1)
            if detection_times
            else None
        ),
        "inference_ms": (
            round(inference_snapshot.inference_ms, 1)
            if inference_snapshot is not None
            else None
        ),
        "pipeline_fps": (
            round(inference_snapshot.fps, 2)
            if inference_snapshot is not None
            else None
        ),
        "result_state": (
            inference_snapshot.result.get("state")
            if inference_snapshot is not None and inference_snapshot.result
            else None
        ),
        "error": error,
    }
    print(json.dumps(result, indent=2))
    return 0 if inference_snapshot is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
