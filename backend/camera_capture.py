"""Camera backends for robot face tracking (USB V4L2 / optional Picamera2)."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np

from robot_config import CameraConfig


class CameraCapture(ABC):
    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def capture_rgb(self) -> np.ndarray:
        """Full-resolution frame in RGB order (H, W, 3)."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class UsbCamera(CameraCapture):
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._main_size = (int(cfg.main_res[0]), int(cfg.main_res[1]))

    def start(self) -> None:
        index = int(self._cfg.device_index)
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"USB webcam not available (index {index}). "
                "Check /dev/video0 and lsusb."
            )
        w, h = self._main_size
        # MJPEG often gives a sharper 640x480 stream than raw YUYV on Pi USB webcams.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("USB webcam opened but read() failed")
        actual = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"USB webcam ready: index {index}, requested {w}x{h}, actual {actual[0]}x{actual[1]}")

    def capture_rgb(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("USB camera not started")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("USB webcam read failed")
        # OpenCV returns BGR; YuNet/stream paths expect RGB like Picamera2 RGB888.
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class PicameraCapture(CameraCapture):
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._picam2 = None
        self._main_size = (int(cfg.main_res[0]), int(cfg.main_res[1]))

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 not installed. Use camera.backend: usb or "
                "sudo apt install python3-picamera2"
            ) from e
        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"format": "RGB888", "size": self._main_size},
            buffer_count=1,
        )
        self._picam2.configure(config)
        self._picam2.start()
        print(f"Picamera2 started: main {self._main_size[0]}x{self._main_size[1]}")

    def capture_rgb(self) -> np.ndarray:
        if self._picam2 is None:
            raise RuntimeError("Picamera2 not started")
        return self._picam2.capture_array()

    def close(self) -> None:
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self._picam2 = None


def create_camera(cfg: CameraConfig) -> CameraCapture:
    backend = (cfg.backend or "usb").strip().lower()
    if backend == "usb":
        return UsbCamera(cfg)
    if backend in ("picamera", "picamera2", "csi"):
        return PicameraCapture(cfg)
    raise ValueError(f"Unknown camera.backend: {cfg.backend!r} (use usb or picamera)")


def open_camera(cfg: CameraConfig) -> CameraCapture:
    cam = create_camera(cfg)
    cam.start()
    return cam
