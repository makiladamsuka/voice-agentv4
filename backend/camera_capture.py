"""Picamera2 (CSI) capture helper for tests and tools — no USB webcam support."""

from __future__ import annotations

import numpy as np

from robot_config import CameraConfig


class PicameraCapture:
    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._picam2 = None
        self._main_size = (int(cfg.main_res[0]), int(cfg.main_res[1]))

    def start(self) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "picamera2 not installed — run: sudo apt install python3-picamera2"
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


def open_camera(cfg: CameraConfig) -> PicameraCapture:
    cam = PicameraCapture(cfg)
    cam.start()
    return cam
