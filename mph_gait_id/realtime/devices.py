from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .types import DeviceStatus, SensorFrame


FRAME_NUMBER = re.compile(r"(\d+)$")


def _azure_config(pyk4a, fps: int = 30):
    fps_value = {
        5: pyk4a.FPS.FPS_5,
        15: pyk4a.FPS.FPS_15,
        30: pyk4a.FPS.FPS_30,
    }.get(int(fps), pyk4a.FPS.FPS_30)
    return pyk4a.Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        color_format=pyk4a.ImageFormat.COLOR_BGRA32,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        camera_fps=fps_value,
        synchronized_images_only=True,
    )


def _azure_access_error(exc: Exception) -> str:
    detail = str(exc).strip()
    detail_text = f" ({type(exc).__name__}: {detail})" if detail else f" ({type(exc).__name__})"
    return (
        "Azure Kinect was detected, but its cameras could not be opened"
        f"{detail_text}. Close Azure Kinect Viewer, Camera, Teams/Webex/Zoom/OBS, and "
        "other running instances of this application. Then enable Windows "
        "Settings > Privacy & security > Camera > Camera access and "
        "'Let desktop apps access your camera', reconnect the Kinect, and retry."
    )


def _start_azure_device(
    pyk4a,
    *,
    device_id: int = 0,
    fps: int = 30,
    retry_delays_s: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
):
    """Open Kinect with backoff for Media Foundation's delayed device release."""

    last_error: Exception | None = None
    for delay in retry_delays_s:
        if delay > 0:
            time.sleep(float(delay))
        device = pyk4a.PyK4A(
            _azure_config(pyk4a, fps),
            device_id=int(device_id),
        )
        try:
            # PyK4A.start() also starts IMU. This application never consumes
            # IMU samples, and on Sensor SDK 1.4.1 that unused IMU session can
            # leave the color camera inaccessible to the next process. Start
            # only the cameras and close them symmetrically below.
            device.open()
            device._start_cameras()
            device._gait_cameras_started = True
            return device
        except Exception as exc:
            last_error = exc
            try:
                if getattr(device, "_gait_cameras_started", False):
                    device._stop_cameras()
                if getattr(device, "opened", False):
                    device.close()
            except Exception:
                pass
    assert last_error is not None
    raise last_error


def _stop_azure_device(device) -> None:
    if device is None:
        return
    try:
        if getattr(device, "_gait_cameras_started", False):
            device._stop_cameras()
            device._gait_cameras_started = False
    finally:
        if getattr(device, "opened", False):
            device.close()


class FrameSource(Protocol):
    def open(self) -> DeviceStatus: ...

    def read(self) -> SensorFrame | None: ...

    def close(self) -> None: ...


def _number(path: Path) -> int:
    match = FRAME_NUMBER.search(path.stem)
    return int(match.group(1)) if match else -1


def probe_azure_kinect(check_access: bool = False) -> DeviceStatus:
    try:
        import pyk4a
    except Exception as exc:
        return DeviceStatus(
            backend="pyk4a",
            connected=False,
            message=(
                "pyk4a / Azure Kinect Sensor SDK is not ready: "
                f"{type(exc).__name__}: {exc}"
            ),
            dependency_ready=False,
        )

    try:
        count = int(pyk4a.connected_device_count())
    except Exception as exc:
        return DeviceStatus(
            backend="pyk4a",
            connected=False,
            message=f"Azure Kinect probe failed: {type(exc).__name__}: {exc}",
            dependency_ready=True,
        )
    if count > 0 and check_access:
        device = None
        started = False
        try:
            device = _start_azure_device(pyk4a)
            started = True
        except Exception as exc:
            return DeviceStatus(
                backend="pyk4a",
                connected=False,
                device_count=count,
                message=_azure_access_error(exc),
                dependency_ready=True,
            )
        finally:
            if started and device is not None:
                try:
                    _stop_azure_device(device)
                except Exception:
                    pass
    return DeviceStatus(
        backend="pyk4a",
        connected=count > 0,
        device_count=count,
        message=(
            f"Detected {count} Azure Kinect DK device(s)"
            if count
            else "No Azure Kinect DK was detected"
        ),
        dependency_ready=True,
    )


class AzureKinectSource:
    """Azure Kinect DK source using SDK-calibrated color-aligned point clouds."""

    def __init__(self, device_id: int = 0, fps: int = 30) -> None:
        self.device_id = int(device_id)
        self.fps = int(fps)
        self._device = None
        self._index = 0
        self._status: DeviceStatus | None = None

    def open(self) -> DeviceStatus:
        status = probe_azure_kinect()
        if not status.connected:
            raise RuntimeError(status.message)
        import pyk4a

        try:
            self._device = _start_azure_device(
                pyk4a,
                device_id=self.device_id,
                fps=self.fps,
            )
        except Exception as exc:
            self._device = None
            raise RuntimeError(_azure_access_error(exc)) from exc
        serial = None
        try:
            serial = str(self._device.serial)
        except Exception:
            serial = None
        self._status = DeviceStatus(
            backend="pyk4a",
            connected=True,
            device_count=status.device_count,
            serial=serial,
            message=f"Azure Kinect DK ready{f' ({serial})' if serial else ''}",
            dependency_ready=True,
        )
        return self._status

    def read(self) -> SensorFrame | None:
        if self._device is None:
            raise RuntimeError("Azure Kinect source has not been opened")
        capture = self._device.get_capture(timeout=1000)
        color = capture.color
        point_cloud = capture.transformed_depth_point_cloud
        if color is None or point_cloud is None:
            return None
        # Detach every returned array from PyK4A's native capture buffers. The
        # UI keeps recent SensorFrames alive; zero-copy views would therefore
        # keep the Media Foundation camera session open after source.close().
        color_bgr = np.array(color[..., :3], copy=True)
        cloud = np.array(point_cloud, dtype=np.float32, copy=True)
        if cloud.ndim != 3 or cloud.shape[:2] != color_bgr.shape[:2]:
            raise RuntimeError(
                "Azure SDK did not return a color-aligned point cloud; "
                f"color={color_bgr.shape}, cloud={cloud.shape}"
            )
        frame = SensorFrame(
            index=self._index,
            timestamp=time.monotonic(),
            color_bgr=color_bgr,
            point_cloud_mm=cloud,
            point_cloud_color_aligned=True,
            depth_image=np.array(capture.transformed_depth, copy=True)
            if capture.transformed_depth is not None
            else None,
            source_name="azure_kinect_dk",
            metadata={"device_serial": self._status.serial if self._status else None},
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._device is not None:
            try:
                _stop_azure_device(self._device)
            finally:
                self._device = None


class ReplaySource:
    """Replay the supplied synchronized RGB / raw organized point-cloud sample."""

    def __init__(
        self,
        root: str | Path,
        fps: float = 10.0,
        loop: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.fps = max(0.1, float(fps))
        self.loop = bool(loop)
        self._records: list[tuple[int, Path, Path, Path | None]] = []
        self._cursor = 0
        self._next_time = 0.0

    def open(self) -> DeviceStatus:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Replay directory does not exist: {self.root}")
        colors = {_number(path): path for path in self.root.glob("Color_image_*.png")}
        clouds = {_number(path): path for path in self.root.glob("Raw_data_*.npy")}
        depths = {_number(path): path for path in self.root.glob("Depth_image_*.png")}
        shared = sorted(set(colors) & set(clouds))
        if not shared:
            raise FileNotFoundError(
                "Replay needs synchronized Color_image_*.png and Raw_data_*.npy"
            )
        self._records = [
            (number, colors[number], clouds[number], depths.get(number))
            for number in shared
        ]
        self._cursor = 0
        self._next_time = time.perf_counter()
        return DeviceStatus(
            backend="replay",
            connected=True,
            device_count=0,
            message=f"Replay ready: {len(self._records)} synchronized frames",
            dependency_ready=True,
        )

    def read(self) -> SensorFrame | None:
        if not self._records:
            raise RuntimeError("Replay source has not been opened")
        now = time.perf_counter()
        if now < self._next_time:
            time.sleep(self._next_time - now)
        self._next_time = max(self._next_time + 1.0 / self.fps, time.perf_counter())
        if self._cursor >= len(self._records):
            if not self.loop:
                return None
            self._cursor = 0
        number, color_path, cloud_path, depth_path = self._records[self._cursor]
        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color is None:
            raise RuntimeError(f"Unable to read replay color frame: {color_path}")
        flat_cloud = np.asarray(np.load(cloud_path), dtype=np.float32)
        if flat_cloud.ndim != 2 or flat_cloud.shape[1] < 3:
            raise RuntimeError(f"Invalid replay point cloud: {cloud_path}")
        point_count = flat_cloud.shape[0]
        if point_count == 640 * 576:
            cloud = flat_cloud[:, :3].reshape(576, 640, 3)
        else:
            side = int(round(point_count ** 0.5))
            if side * side != point_count:
                raise RuntimeError(
                    "Replay raw_data.npy is not an organized point cloud; "
                    f"point_count={point_count}"
                )
            cloud = flat_cloud[:, :3].reshape(side, side, 3)
        depth = (
            cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_path is not None
            else None
        )
        frame = SensorFrame(
            index=number,
            timestamp=time.monotonic(),
            color_bgr=color,
            point_cloud_mm=cloud,
            point_cloud_color_aligned=False,
            depth_image=depth,
            source_name=self.root.name,
            metadata={
                "color_path": str(color_path),
                "point_cloud_path": str(cloud_path),
                "alignment": "reference_pinhole_projection",
                "pinhole_projection": {
                    "reference_size": [1280, 720],
                    "fx": 913.0,
                    "fy": 913.0,
                    "cx": 640.0,
                    "cy": 360.0,
                    "x_offset": -15.0,
                    "y_offset": 58.0,
                },
            },
        )
        self._cursor += 1
        return frame

    def close(self) -> None:
        self._records = []
