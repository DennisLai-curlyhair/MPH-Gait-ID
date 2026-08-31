from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from .types import Detection, SensorFrame


# The deployed pytorch environment uses Ultralytics 8.0.x. YOLOv8n is the
# newest small COCO detector that this installed runtime loads reliably.
DEFAULT_YOLO_MODEL = "model_weights/yolov8n.pt"
DEFAULT_YOLO_SEG_MODEL = "model_weights/yolov8n-seg.pt"
DEFAULT_SAM_MODEL_TYPE = "vit_b"
DEFAULT_SAM_CHECKPOINT_SHA256 = (
    "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
)
DEFAULT_SAM_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "model_weights"
    / "sam_vit_b_01ec64.pth"
)


def resolve_yolo_weights_reference(weights: str | Path | None) -> str:
    """Return a local checkpoint path or an Ultralytics pretrained model ID."""

    raw = str(weights or "").strip() or DEFAULT_YOLO_MODEL
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    if not candidate.is_absolute():
        package_candidate = Path(__file__).resolve().parents[1] / candidate
        if package_candidate.is_file():
            return str(package_candidate.resolve())
    if candidate.name == raw and candidate.suffix.lower() == ".pt":
        # A simple name such as yolov8n.pt is resolved/downloaded by Ultralytics.
        return raw
    raise FileNotFoundError(
        f"YOLO weights are missing: {candidate}. Select an existing local "
        f"checkpoint or use a pretrained model ID such as {DEFAULT_YOLO_MODEL}."
    )


@lru_cache(maxsize=4)
def _checkpoint_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sam_checkpoint(checkpoint: str | Path | None) -> str:
    """Resolve the local Meta SAM checkpoint shipped with or selected in the app."""

    raw = str(checkpoint or "").strip()
    candidate = Path(raw).expanduser() if raw else DEFAULT_SAM_CHECKPOINT
    if not candidate.is_absolute():
        package_candidate = Path(__file__).resolve().parents[1] / candidate
        if package_candidate.is_file():
            candidate = package_candidate
    if not candidate.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint is missing: {candidate}. Download the official Meta "
            "sam_vit_b_01ec64.pth checkpoint or select an existing local file."
        )
    candidate = candidate.resolve()
    stat = candidate.stat()
    actual_hash = _checkpoint_sha256(
        str(candidate),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    if actual_hash != DEFAULT_SAM_CHECKPOINT_SHA256:
        raise ValueError(
            "SAM checkpoint SHA256 does not match Meta's official ViT-B weight: "
            f"{candidate}"
        )
    return str(candidate)


class PersonDetector(Protocol):
    name: str

    def detect(self, frame: SensorFrame) -> Detection | None: ...


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float = 0.0,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * float(padding)
    pad_y = (y2 - y1) * float(padding)
    return (
        max(0, min(width - 1, int(round(x1 - pad_x)))),
        max(0, min(height - 1, int(round(y1 - pad_y)))),
        max(1, min(width, int(round(x2 + pad_x)))),
        max(1, min(height, int(round(y2 + pad_y)))),
    )


class YoloPersonDetector:
    """Ultralytics YOLO person detector with optional instance masks."""

    name = "YOLO person"

    def __init__(
        self,
        weights: str | Path = DEFAULT_YOLO_MODEL,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "auto",
        bbox_padding: float = 0.03,
        require_mask: bool = False,
    ) -> None:
        reference = resolve_yolo_weights_reference(weights)
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "YOLO mode requires the optional 'ultralytics' package. "
                "Install realtime requirements before using Azure Kinect mode."
            ) from exc
        try:
            self.model = YOLO(reference)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load YOLO model '{reference}'. The default pretrained "
                "model is downloaded on first use, so an internet connection is "
                "required once; alternatively select an existing local checkpoint."
            ) from exc
        self.weights_reference = reference
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = None if device == "auto" else device
        self.bbox_padding = float(bbox_padding)
        self.require_mask = bool(require_mask)
        self.name = (
            "YOLO person segmentation" if self.require_mask else "YOLO person"
        )

    def detect(self, frame: SensorFrame) -> Detection | None:
        kwargs = {
            "source": frame.color_bgr,
            "classes": [0],
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results:
            return None
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None
        confidence = boxes.conf.detach().float().cpu().numpy()
        coordinates = boxes.xyxy.detach().float().cpu().numpy()
        areas = (coordinates[:, 2] - coordinates[:, 0]) * (
            coordinates[:, 3] - coordinates[:, 1]
        )
        selected = int(np.argmax(confidence * np.sqrt(np.maximum(areas, 1.0))))
        height, width = frame.color_bgr.shape[:2]
        bbox = _clamp_bbox(
            tuple(float(value) for value in coordinates[selected]),
            width,
            height,
            padding=self.bbox_padding,
        )
        mask = None
        if getattr(result, "masks", None) is not None:
            masks = result.masks.data.detach().float().cpu().numpy()
            if selected < len(masks):
                mask = cv2.resize(
                    masks[selected],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
        if self.require_mask and mask is None:
            raise RuntimeError(
                "The selected YOLO segmentation mode did not return an instance "
                "mask. Select an official segmentation checkpoint such as "
                "model_weights/yolov8n-seg.pt."
            )
        return Detection(
            bbox_xyxy=bbox,
            confidence=float(confidence[selected]),
            mask=mask,
            detector=self.name,
            person_count=int(len(boxes)),
        )


class YoloSamPersonDetector:
    """YOLO person detection refined by Meta SAM using the person box prompt."""

    name = "YOLO + SAM person mask"

    def __init__(
        self,
        yolo_weights: str | Path = DEFAULT_YOLO_MODEL,
        sam_checkpoint: str | Path = DEFAULT_SAM_CHECKPOINT,
        sam_model_type: str = DEFAULT_SAM_MODEL_TYPE,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "auto",
        refresh_interval: int = 10,
    ) -> None:
        checkpoint = resolve_sam_checkpoint(sam_checkpoint)
        self.yolo = YoloPersonDetector(
            weights=yolo_weights,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except Exception as exc:
            raise RuntimeError(
                "YOLO + SAM mode requires Meta's official segment-anything package. "
                "Install requirements-realtime.txt before using this mode."
            ) from exc
        model_type = str(sam_model_type).strip().lower()
        if model_type not in sam_model_registry:
            raise ValueError(
                f"Unsupported SAM model type: {model_type}. "
                f"Available types: {', '.join(sorted(sam_model_registry))}"
            )
        if device == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM was configured for CUDA, but CUDA is unavailable")
        else:
            device_name = str(device)
        try:
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=torch.device(device_name))
            sam.eval()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "Not enough GPU memory to load SAM. Close other GPU applications, "
                    "select CPU, or use YOLO bbox mode."
                ) from exc
            raise
        self.predictor = SamPredictor(sam)
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device_name
        self.refresh_interval = max(1, int(refresh_interval))
        self._frame_counter = 0
        self._previous_mask: np.ndarray | None = None
        self._previous_bbox: tuple[int, int, int, int] | None = None

    @staticmethod
    def _move_previous_mask(
        mask: np.ndarray,
        previous_bbox: tuple[int, int, int, int],
        current_bbox: tuple[int, int, int, int],
        output_shape: tuple[int, int],
    ) -> np.ndarray | None:
        px1, py1, px2, py2 = previous_bbox
        x1, y1, x2, y2 = current_bbox
        crop = mask[py1:py2, px1:px2]
        if crop.size == 0 or x2 <= x1 or y2 <= y1:
            return None
        moved = cv2.resize(
            crop.astype(np.uint8),
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        output = np.zeros(output_shape, dtype=bool)
        output[y1:y2, x1:x2] = moved
        return output

    def detect(self, frame: SensorFrame) -> Detection | None:
        detection = self.yolo.detect(frame)
        if detection is None:
            self._previous_mask = None
            self._previous_bbox = None
            return None
        self._frame_counter += 1
        refresh_sam = (
            self._previous_mask is None
            or self._previous_bbox is None
            or (self._frame_counter - 1) % self.refresh_interval == 0
        )
        if not refresh_sam:
            mask = self._move_previous_mask(
                self._previous_mask,
                self._previous_bbox,
                detection.bbox_xyxy,
                frame.color_bgr.shape[:2],
            )
            if mask is not None and int(mask.sum()) >= 64:
                self._previous_mask = mask
                self._previous_bbox = detection.bbox_xyxy
                return Detection(
                    bbox_xyxy=detection.bbox_xyxy,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    mask=mask,
                    detector=f"{self.name} (tracked)",
                    person_count=detection.person_count,
                )
        rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB)
        box = np.asarray(detection.bbox_xyxy, dtype=np.float32)
        center_point = np.asarray(
            [[(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]],
            dtype=np.float32,
        )
        try:
            with torch.inference_mode():
                self.predictor.set_image(rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=center_point,
                    point_labels=np.ones(1, dtype=np.int32),
                    box=box,
                    multimask_output=True,
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "SAM ran out of GPU memory while segmenting a frame. "
                    "Close other GPU applications, select CPU, or use YOLO bbox mode."
                ) from exc
            raise RuntimeError(f"SAM person-mask inference failed: {exc}") from exc
        if len(masks) == 0:
            return None
        selected = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        mask = np.asarray(masks[selected], dtype=bool)
        if mask.shape != frame.color_bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.color_bgr.shape[1], frame.color_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # A box-prompt mask should stay within the detector's person box. This
        # prevents an adjacent wall/furniture region from leaking into XYZ.
        x1, y1, x2, y2 = detection.bbox_xyxy
        inside_box = np.zeros_like(mask)
        inside_box[y1:y2, x1:x2] = True
        mask &= inside_box
        if int(mask.sum()) < 64:
            self._previous_mask = None
            self._previous_bbox = None
            return None
        self._previous_mask = mask
        self._previous_bbox = detection.bbox_xyxy
        return Detection(
            bbox_xyxy=detection.bbox_xyxy,
            confidence=detection.confidence,
            class_name=detection.class_name,
            mask=mask,
            detector=self.name,
            person_count=detection.person_count,
        )


class ReplayDepthPersonDetector:
    """Local HOG detector with depth-only fallback for server replay tests."""

    name = "Replay HOG/depth diagnostic"

    def __init__(
        self,
        near_mm: float = 500.0,
        far_mm: float = 4500.0,
        min_area_ratio: float = 0.005,
        bbox_padding: float = 0.05,
    ) -> None:
        self.near_mm = float(near_mm)
        self.far_mm = float(far_mm)
        self.min_area_ratio = float(min_area_ratio)
        self.bbox_padding = float(bbox_padding)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._previous_bbox: tuple[int, int, int, int] | None = None

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
        second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
        return intersection / max(1.0, first_area + second_area - intersection)

    def _detect_hog(self, frame: SensorFrame) -> Detection | None:
        color = frame.color_bgr
        height, width = color.shape[:2]
        scale = min(1.0, 960.0 / max(1, width))
        working = (
            cv2.resize(
                color,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else color
        )
        rectangles, weights = self._hog.detectMultiScale(
            working,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.03,
        )
        if not len(rectangles):
            return None
        candidates: list[tuple[tuple[int, int, int, int], float, float]] = []
        diagonal = max(float(np.hypot(width, height)), 1.0)
        for rectangle, raw_weight in zip(rectangles, weights):
            x, y, box_width, box_height = [float(value) / scale for value in rectangle]
            bbox = _clamp_bbox(
                (x, y, x + box_width, y + box_height),
                width,
                height,
                padding=self.bbox_padding,
            )
            weight = float(raw_weight)
            continuity = 0.0
            if self._previous_bbox is not None:
                previous = self._previous_bbox
                current_center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
                previous_center = (
                    (previous[0] + previous[2]) * 0.5,
                    (previous[1] + previous[3]) * 0.5,
                )
                distance = float(np.hypot(
                    current_center[0] - previous_center[0],
                    current_center[1] - previous_center[1],
                )) / diagonal
                continuity = 2.5 * self._iou(bbox, previous) - 2.0 * distance
            candidates.append((bbox, weight, weight + continuity))
        bbox, weight, _ = max(candidates, key=lambda item: item[2])
        if weight < 0.8:
            return None
        if self._previous_bbox is not None:
            previous = self._previous_bbox
            previous_center = (
                (previous[0] + previous[2]) * 0.5,
                (previous[1] + previous[3]) * 0.5,
            )
            current_center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
            center_jump = float(np.hypot(
                current_center[0] - previous_center[0],
                current_center[1] - previous_center[1],
            )) / diagonal
            if center_jump > 0.22 and self._iou(bbox, previous) == 0.0:
                return None
        self._previous_bbox = bbox
        confidence = float(1.0 / (1.0 + np.exp(-weight)))
        return Detection(
            bbox_xyxy=bbox,
            confidence=confidence,
            mask=None,
            detector=self.name,
        )

    def detect(self, frame: SensorFrame) -> Detection | None:
        hog_detection = self._detect_hog(frame)
        if hog_detection is not None:
            return hog_detection
        if self._previous_bbox is not None:
            return None

        # Depth-only fallback is retained for replay sequences where HOG cannot
        # initialize. It is intentionally unavailable to the live source.
        cloud = frame.point_cloud_mm
        z = cloud[..., 2]
        valid = np.isfinite(z) & (z >= self.near_mm) & (z <= self.far_mm)
        if not valid.any():
            return None
        # Keep the nearest robust depth band, then the largest connected component.
        values = z[valid]
        low = float(np.percentile(values, 3))
        near_band = valid & (z <= low + 1200.0)
        mask = (near_band.astype(np.uint8) * 255)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((11, 7), dtype=np.uint8),
            iterations=2,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((5, 3), dtype=np.uint8),
            iterations=1,
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None
        candidates = [
            index
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA]
            >= self.min_area_ratio * mask.shape[0] * mask.shape[1]
        ]
        if not candidates:
            return None
        center_x = mask.shape[1] * 0.5

        def score(index: int) -> float:
            area = float(stats[index, cv2.CC_STAT_AREA])
            component_x = float(stats[index, cv2.CC_STAT_LEFT]) + 0.5 * float(
                stats[index, cv2.CC_STAT_WIDTH]
            )
            center_penalty = 1.0 + abs(component_x - center_x) / max(center_x, 1.0)
            return area / center_penalty

        selected = max(candidates, key=score)
        component = labels == selected
        x = int(stats[selected, cv2.CC_STAT_LEFT])
        y = int(stats[selected, cv2.CC_STAT_TOP])
        width = int(stats[selected, cv2.CC_STAT_WIDTH])
        height = int(stats[selected, cv2.CC_STAT_HEIGHT])
        color_height, color_width = frame.color_bgr.shape[:2]
        scale_x = color_width / component.shape[1]
        scale_y = color_height / component.shape[0]
        bbox = _clamp_bbox(
            (x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y),
            color_width,
            color_height,
            padding=self.bbox_padding,
        )
        color_mask = cv2.resize(
            component.astype(np.uint8),
            (color_width, color_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        confidence = min(
            0.99,
            float(stats[selected, cv2.CC_STAT_AREA])
            / max(1.0, 0.12 * component.size),
        )
        return Detection(
            bbox_xyxy=bbox,
            confidence=confidence,
            mask=color_mask,
            detector=self.name,
        )
