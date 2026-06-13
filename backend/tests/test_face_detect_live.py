#!/usr/bin/env python3
"""Live YuNet face detection test — run from backend/ while robot is stopped."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_color import (
    PICAMERA_MAIN_FORMAT,
    probe_yunet_bgr_mode,
    to_detection_bgr,
)
from robot_config import load_config

CFG = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
DETECT_RES = tuple(CFG.camera.detect_res)
MODEL = Path(__file__).resolve().parent.parent / CFG.camera.face_model_path


def main() -> int:
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not installed")
        return 1

    if not MODEL.is_file():
        print(f"Missing model: {MODEL}")
        return 1

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"format": PICAMERA_MAIN_FORMAT, "size": tuple(CFG.camera.main_res)},
            buffer_count=1,
        )
    )
    picam2.start()
    detector = cv2.FaceDetectorYN.create(
        model=str(MODEL),
        config="",
        input_size=DETECT_RES,
        score_threshold=CFG.camera.confidence_threshold,
        nms_threshold=CFG.camera.nms_threshold,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )
    try:
        for i in range(5):
            capture = cv2.resize(picam2.capture_array(), DETECT_RES)
            if i == 0:
                mode, n = probe_yunet_bgr_mode(
                    detector, capture, input_size=DETECT_RES, rotate_180=CFG.camera.rotate_180
                )
                print(f"Boot probe: mode={mode} faces={n}")
            bgr = to_detection_bgr(capture, rotate_180=CFG.camera.rotate_180)
            detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            faces = detector.detect(bgr)
            count = 0 if faces[1] is None else len(faces[1])
            print(f"Frame {i + 1}: {count} face(s)")
            if count:
                f = max(faces[1], key=lambda x: x[2] * x[3])
                print(f"  largest box score area={f[2] * f[3]:.0f} conf~{f[-1] if len(f) > 8 else '?'}")
        return 0
    finally:
        picam2.stop()
        picam2.close()


if __name__ == "__main__":
    raise SystemExit(main())
