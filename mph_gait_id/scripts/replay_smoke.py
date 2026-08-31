#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.controller import GaitApplicationController
from mph_gait_id.realtime.pipeline import RealtimeConfig, RealtimePipeline


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Headless realtime replay smoke test")
    value.add_argument(
        "--replay",
        default=str(
            SYSTEM_ROOT
            / "data"
            / "replay"
            / "P3_C4_V001"
        ),
    )
    value.add_argument(
        "--bundle",
        default="pointnet_tmax_fixed_special5_seed0_split0",
    )
    value.add_argument("--clip-len", type=int, default=15)
    value.add_argument("--inference-stride", type=int, default=5)
    value.add_argument("--fps", type=float, default=120.0)
    value.add_argument("--timeout", type=float, default=120.0)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    controller = GaitApplicationController()
    config = RealtimeConfig(
        bundle_id=args.bundle,
        source_mode="replay",
        replay_path=args.replay,
        replay_fps=args.fps,
        replay_loop=False,
        detector="replay_depth",
        clip_len=args.clip_len,
        inference_stride=args.inference_stride,
        threshold=controller.provisional_threshold(
            args.bundle,
            clip_len=args.clip_len,
        ),
    )
    pipeline = RealtimePipeline(controller, config)
    pipeline.start()
    deadline = time.monotonic() + args.timeout
    counts: dict[str, int] = {}
    last = None
    last_activity = None
    inference_samples: list[float] = []
    point_samples: list[int] = []
    while time.monotonic() < deadline:
        snapshot = pipeline.poll_latest()
        if snapshot is not None:
            last = snapshot
            counts[snapshot.state] = counts.get(snapshot.state, 0) + 1
            if snapshot.person_point_count > 0:
                last_activity = snapshot
            if snapshot.diagnostics.get("inference_ran"):
                inference_samples.append(float(snapshot.inference_ms))
            if snapshot.person_point_count > 0:
                point_samples.append(int(snapshot.person_point_count))
            if snapshot.state in {"complete", "error"}:
                break
        if not pipeline.running and snapshot is None:
            break
        time.sleep(0.02)
    pipeline.stop()
    if last is None:
        print(json.dumps({"status": "failed", "error": "no snapshots"}, indent=2))
        return 1
    activity = last_activity or last
    report = {
        "status": "ok" if last.state != "error" else "failed",
        "bundle": args.bundle,
        "replay": str(Path(args.replay).resolve()),
        "state_counts": counts,
        "last_state": last.state,
        "last_message": last.message,
        "last_processed_frame": activity.frame_index,
        "buffer": [activity.buffer_size, activity.clip_len],
        "last_person_points": activity.person_point_count,
        "person_points_range": (
            [min(point_samples), max(point_samples)] if point_samples else []
        ),
        "last_pipeline_fps": activity.fps,
        "last_capture_fps": activity.capture_fps,
        "effective_sampling_fps": activity.effective_sampling_fps,
        "window_duration_s": activity.window_duration_s,
        "mean_frame_gap_ms": activity.mean_frame_gap_ms,
        "max_frame_gap_ms": activity.max_frame_gap_ms,
        "last_preprocessing_ms": activity.preprocessing_ms,
        "inference_count": len(inference_samples),
        "inference_ms_mean": (
            sum(inference_samples) / len(inference_samples)
            if inference_samples
            else 0.0
        ),
        "diagnostics": activity.diagnostics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
