#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = SYSTEM_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from mph_gait_id.realtime.detection import (  # noqa: E402
    DEFAULT_SAM_CHECKPOINT,
    YoloSamPersonDetector,
)
from mph_gait_id.realtime.preprocessing import (  # noqa: E402
    filter_person_pointcloud,
)
from mph_gait_id.realtime.types import (  # noqa: E402
    Detection,
    SensorFrame,
)


class _FixedPersonBox:
    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox

    def detect(self, frame: SensorFrame) -> Detection:
        return Detection(
            bbox_xyxy=self.bbox,
            confidence=0.99,
            detector="synthetic fixed person box",
        )


def _synthetic_frame() -> tuple[SensorFrame, np.ndarray, tuple[int, int, int, int]]:
    height, width = 360, 640
    image = np.full((height, width, 3), (95, 110, 120), dtype=np.uint8)
    # Add a nearby wall/furniture-like background within the YOLO box.
    cv2.rectangle(image, (215, 40), (435, 335), (75, 125, 145), -1)
    person = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(person, (320, 75), 34, 255, -1)
    cv2.rectangle(person, (275, 105), (365, 245), 255, -1)
    cv2.line(person, (290, 235), (260, 330), 255, 30)
    cv2.line(person, (350, 235), (385, 330), 255, 30)
    cv2.line(person, (280, 130), (235, 225), 255, 24)
    cv2.line(person, (360, 130), (405, 225), 255, 24)
    image[person.astype(bool)] = (45, 65, 205)

    columns = np.arange(width, dtype=np.float32)[None, :]
    rows = np.arange(height, dtype=np.float32)[:, None]
    cloud = np.empty((height, width, 3), dtype=np.float32)
    cloud[..., 0] = (columns - width / 2.0) * 3.0
    cloud[..., 1] = (rows - height / 2.0) * 3.0
    cloud[..., 2] = 2300.0
    cloud[..., 2][person.astype(bool)] = 1800.0
    bbox = (210, 35, 435, 340)
    return (
        SensorFrame(
            index=0,
            timestamp=time.time(),
            color_bgr=image,
            point_cloud_mm=cloud,
            point_cloud_color_aligned=True,
            source_name="synthetic_sam_validation",
        ),
        person.astype(bool),
        bbox,
    )


def main() -> int:
    frame, expected_person, bbox = _synthetic_frame()
    started = time.perf_counter()
    detector = YoloSamPersonDetector(
        sam_checkpoint=DEFAULT_SAM_CHECKPOINT,
        device="auto",
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    detector.yolo = _FixedPersonBox(bbox)
    detection_times = []
    detection = None
    for _ in range(detector.refresh_interval):
        started = time.perf_counter()
        detection = detector.detect(frame)
        detection_times.append((time.perf_counter() - started) * 1000.0)
    if detection is None or detection.mask is None:
        raise RuntimeError("SAM returned no usable mask for the validation frame")
    mask = detection.mask
    x1, y1, x2, y2 = bbox
    outside_box = mask.copy()
    outside_box[y1:y2, x1:x2] = False
    if outside_box.any():
        raise RuntimeError("SAM mask escaped the YOLO bbox constraint")
    prepared = filter_person_pointcloud(
        frame,
        detection,
        num_points=1024,
        depth_margin_mm=700.0,
        min_person_points=64,
    )
    intersection = int((mask & expected_person).sum())
    union = int((mask | expected_person).sum())
    print(
        json.dumps(
            {
                "status": "ok",
                "sam_model_type": detector.model_type,
                "device": detector.device,
                "checkpoint": detector.checkpoint,
                "load_ms": round(load_ms, 1),
                "sam_refresh_ms": round(detection_times[0], 1),
                "tracked_mask_mean_ms": round(
                    float(np.mean(detection_times[1:])), 1
                ),
                "amortized_detection_ms": round(
                    float(np.mean(detection_times)), 1
                ),
                "refresh_interval": detector.refresh_interval,
                "mask_pixels": int(mask.sum()),
                "synthetic_mask_iou": round(intersection / max(union, 1), 4),
                "foreground_points": int(len(prepared.foreground_points_mm)),
                "sampled_shape": list(prepared.sampled_points_mm.shape),
                "alignment": prepared.diagnostics["alignment"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
