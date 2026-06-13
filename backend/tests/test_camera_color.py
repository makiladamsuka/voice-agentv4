#!/usr/bin/env python3
"""Verify Picamera2 RGB888 → BGR pipeline (run while robot is stopped)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera_color import (
    PICAMERA_MAIN_FORMAT,
    log_color_pipeline_verification,
    rgb_to_bgr,
    verify_color_pipeline,
)


def main() -> int:
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not installed")
        return 1

    out_dir = Path(__file__).resolve().parent.parent / "static"
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"format": PICAMERA_MAIN_FORMAT, "size": (640, 480)},
        buffer_count=1,
    )
    picam2.configure(config)
    picam2.start()
    try:
        rgb = picam2.capture_array()
        stats = verify_color_pipeline(rgb, save_dir=out_dir)
        log_color_pipeline_verification(stats)
        bgr = rgb_to_bgr(rgb)
        import cv2

        cv2.imwrite(str(out_dir / "verify_bgr.jpg"), bgr)
        print(f"Saved {out_dir / 'verify_stream.jpg'} and verify_bgr.jpg")
        return 0 if stats["roundtrip_ok"] and stats["rb_consistent"] else 2
    finally:
        picam2.stop()
        picam2.close()


if __name__ == "__main__":
    raise SystemExit(main())
