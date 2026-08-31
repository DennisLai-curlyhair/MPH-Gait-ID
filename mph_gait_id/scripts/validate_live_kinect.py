#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.realtime.detection import (  # noqa: E402
    DEFAULT_YOLO_SEG_MODEL,
    YoloPersonDetector,
    YoloSamPersonDetector,
)
from mph_gait_id.realtime.devices import (  # noqa: E402
    AzureKinectSource,
)
from mph_gait_id.realtime.preprocessing import (  # noqa: E402
    filter_person_pointcloud,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate live Azure Kinect, YOLO, SAM, and point filtering."
    )
    parser.add_argument("--capture-frames", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def _prepared(frame, detection):
    return filter_person_pointcloud(
        frame,
        detection,
        num_points=1024,
        near_mm=400.0,
        far_mm=8000.0,
        depth_margin_mm=550.0,
        min_person_points=96,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = AzureKinectSource()
    frames = []
    capture_started = time.perf_counter()
    try:
        status = source.open()
        for _ in range(max(1, int(args.capture_frames))):
            frame = source.read()
            if frame is not None:
                frames.append(frame)
    finally:
        source.close()
    capture_elapsed = time.perf_counter() - capture_started
    if not frames:
        raise RuntimeError("Kinect opened but returned no synchronized frames")
    frame = frames[-1]
    cloud = frame.point_cloud_mm
    finite = np.isfinite(cloud).all(axis=2)
    valid_depth = finite & (cloud[..., 2] > 0)
    output = {
        "status": "captured",
        "device_status": status.message,
        "device_serial": status.serial,
        "frames": len(frames),
        "capture_fps": round(len(frames) / max(capture_elapsed, 1e-6), 2),
        "color_shape": list(frame.color_bgr.shape),
        "cloud_shape": list(cloud.shape),
        "depth_shape": (
            list(frame.depth_image.shape) if frame.depth_image is not None else None
        ),
        "finite_xyz": int(finite.sum()),
        "valid_depth": int(valid_depth.sum()),
        "alignment": "sdk_color_aligned" if frame.point_cloud_color_aligned else None,
    }

    yolo = YoloPersonDetector(device=args.device)
    started = time.perf_counter()
    yolo_detection = yolo.detect(frame)
    output["yolo_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if yolo_detection is None:
        output["status"] = "captured_no_person"
        output["message"] = "Kinect works, but YOLO found no person in the final frame"
        print(json.dumps(output, indent=2))
        return 3
    yolo_prepared = _prepared(frame, yolo_detection)
    output["yolo"] = {
        "bbox": list(yolo_detection.bbox_xyxy),
        "confidence": round(float(yolo_detection.confidence), 4),
        "foreground_points": int(len(yolo_prepared.foreground_points_mm)),
    }
    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    yolo_seg = YoloPersonDetector(
        weights=DEFAULT_YOLO_SEG_MODEL,
        device=args.device,
        require_mask=True,
    )
    started = time.perf_counter()
    yolo_seg_detection = yolo_seg.detect(frame)
    output["yolo_seg_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if yolo_seg_detection is None or yolo_seg_detection.mask is None:
        output["status"] = "captured_yolo_seg_no_mask"
        output["message"] = "YOLO segmentation returned no usable person mask"
        print(json.dumps(output, indent=2))
        return 4
    yolo_seg_prepared = _prepared(frame, yolo_seg_detection)
    yolo_seg_points = int(len(yolo_seg_prepared.foreground_points_mm))
    yolo_points = int(len(yolo_prepared.foreground_points_mm))
    output["yolo_seg"] = {
        "bbox": list(yolo_seg_detection.bbox_xyxy),
        "confidence": round(float(yolo_seg_detection.confidence), 4),
        "mask_pixels": int(yolo_seg_detection.mask.sum()),
        "foreground_points": yolo_seg_points,
        "retained_vs_bbox": round(yolo_seg_points / max(yolo_points, 1), 4),
        "sampled_shape": list(yolo_seg_prepared.sampled_points_mm.shape),
    }
    del yolo_seg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sam = YoloSamPersonDetector(device=args.device, refresh_interval=1)
    started = time.perf_counter()
    sam_detection = sam.detect(frame)
    output["yolo_sam_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if sam_detection is None or sam_detection.mask is None:
        output["status"] = "captured_sam_no_mask"
        output["message"] = "YOLO found a person, but SAM returned no usable mask"
        print(json.dumps(output, indent=2))
        return 5
    sam_prepared = _prepared(frame, sam_detection)
    sam_points = int(len(sam_prepared.foreground_points_mm))
    yolo_points = int(len(yolo_prepared.foreground_points_mm))
    output["yolo_sam"] = {
        "bbox": list(sam_detection.bbox_xyxy),
        "confidence": round(float(sam_detection.confidence), 4),
        "mask_pixels": int(sam_detection.mask.sum()),
        "foreground_points": sam_points,
        "retained_vs_bbox": round(sam_points / max(yolo_points, 1), 4),
        "sampled_shape": list(sam_prepared.sampled_points_mm.shape),
    }
    output["status"] = "ok"
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
