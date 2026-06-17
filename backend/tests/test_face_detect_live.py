#!/usr/bin/env python3
"""
Live YuNet face detection + ESP32 head servo tracking test.

Stop start_robot.py / robot_eyes.py first — only one process can own camera/serial.

  cd backend && python tests/test_face_detect_live.py --port /dev/ttyUSB0

Ctrl+C homes the head and exits.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2

import _bootstrap  # noqa: F401

from arduino_servo import ArduinoServoLink
from camera_color import (
    PICAMERA_MAIN_FORMAT,
    probe_yunet_bgr_mode,
    to_detection_bgr,
)
from elastic_head_motion import HeadMotionParams, clamp, tick_spring, tick_toward
from robot_config import load_config

BACKEND_ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(BACKEND_ROOT / "config.yaml")
DETECT_RES = tuple(CFG.camera.detect_res)
MODEL = BACKEND_ROOT / CFG.camera.face_model_path

LOOP_HZ = 20.0
STATUS_HZ = 4.0
NO_FACE_SPRING_K = 6.0
NO_FACE_SPRING_DAMP = 5.0

_stop = False


def _on_signal(_signum: int, _frame: object) -> None:
    global _stop
    _stop = True


def apply_deadzone_norm(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value >= 0.0 else -1.0
    return sign * (abs(value) - deadzone) / max(1e-6, 1.0 - deadzone)


def head_motion_from_servo(sv) -> tuple[HeadMotionParams, HeadMotionParams]:
    pan = HeadMotionParams(
        max_vel_pos=float(sv.pan_max_vel),
        max_vel_neg=float(sv.pan_max_vel),
        accel=float(sv.pan_accel),
        decel=float(sv.pan_decel),
        vel_blend=float(sv.head_vel_blend),
        goal_deadband_deg=float(sv.goal_deadband_deg),
        track_gain=float(getattr(sv, "pan_track_gain", 0.0)),
    )
    tilt = HeadMotionParams(
        max_vel_pos=float(sv.tilt_max_vel_up),
        max_vel_neg=float(sv.tilt_max_vel_down),
        accel=float(sv.tilt_accel),
        decel=float(sv.tilt_decel),
        vel_blend=float(getattr(sv, "tilt_head_vel_blend", sv.head_vel_blend)),
        decel_boost_dir=-1.0,
        decel_boost_mult=float(sv.tilt_decel_down_mult),
        goal_deadband_deg=float(sv.goal_deadband_deg),
        track_gain=float(getattr(sv, "tilt_track_gain", 2.2)),
    )
    return pan, tilt


def face_to_servo_targets(
    aim_cx: float,
    aim_cy: float,
    *,
    frame_w: int,
    frame_h: int,
    pan_min: float,
    pan_max: float,
    tilt_min: float,
    tilt_max: float,
    pan_track_range: float,
    tilt_track_range: float,
    deadzone_x: float,
    deadzone_y: float,
) -> tuple[float, float]:
    """Map detect-frame pixel aim point to pan/tilt degrees (robot_eyes logic)."""
    norm_x = -((aim_cx / frame_w - 0.5) * 2.0)
    norm_y = -((aim_cy / frame_h - 0.5) * 2.0)
    norm_x = apply_deadzone_norm(norm_x, deadzone_x)
    norm_y = apply_deadzone_norm(norm_y, deadzone_y)
    pan_center = (pan_min + pan_max) * 0.5
    tilt_center = (tilt_min + tilt_max) * 0.5
    pan = clamp(pan_center + norm_x * pan_track_range, pan_min, pan_max)
    tilt = clamp(tilt_center + norm_y * tilt_track_range, tilt_min, tilt_max)
    return pan, tilt


def largest_face_center(faces) -> tuple[float, float] | None:
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda x: x[2] * x[3])
    return f[0] + f[2] * 0.5, f[1] + f[3] * 0.5


def run_tracking(
    link: ArduinoServoLink | None,
    *,
    duration_sec: float | None,
) -> None:
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not installed")
        raise SystemExit(1)

    if not MODEL.is_file():
        print(f"Missing model: {MODEL}")
        raise SystemExit(1)

    sv = CFG.servo
    ft = CFG.face_tracking
    pan_min, pan_max = float(sv.pan_min), float(sv.pan_max)
    tilt_min, tilt_max = float(sv.tilt_min), float(sv.tilt_max)
    pan_center = (pan_min + pan_max) * 0.5
    tilt_center = (tilt_min + tilt_max) * 0.5
    pan_motion, tilt_motion = head_motion_from_servo(sv)
    servo_alpha = float(ft.face_track_servo_alpha)

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

    pan_current = pan_center
    tilt_current = tilt_center
    pan_vel = 0.0
    tilt_vel = 0.0
    target_pan = pan_center
    target_tilt = tilt_center
    boot_probed = False

    if link is not None:
        link.write_angles(pan_current, tilt_current, force=True)

    dt = 1.0 / LOOP_HZ
    status_interval = 1.0 / STATUS_HZ
    next_status = time.monotonic()
    deadline = None if duration_sec is None else time.monotonic() + duration_sec

    print(
        f"Face tracking — center P{pan_center:.1f} T{tilt_center:.1f}  "
        f"servo={'ESP32' if link else 'off'}  Ctrl+C to quit"
    )

    try:
        while not _stop:
            loop_start = time.monotonic()
            if deadline is not None and loop_start >= deadline:
                break

            capture = cv2.resize(picam2.capture_array(), DETECT_RES)
            if not boot_probed:
                mode, n = probe_yunet_bgr_mode(
                    detector,
                    capture,
                    input_size=DETECT_RES,
                    rotate_180=CFG.camera.rotate_180,
                )
                print(f"Boot probe: mode={mode} faces={n}")
                boot_probed = True

            bgr = to_detection_bgr(capture, rotate_180=CFG.camera.rotate_180)
            detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            result = detector.detect(bgr)
            faces = result[1]
            center = largest_face_center(faces)
            face_count = 0 if faces is None else len(faces)

            if center is not None:
                mapped_pan, mapped_tilt = face_to_servo_targets(
                    center[0],
                    center[1],
                    frame_w=DETECT_RES[0],
                    frame_h=DETECT_RES[1],
                    pan_min=pan_min,
                    pan_max=pan_max,
                    tilt_min=tilt_min,
                    tilt_max=tilt_max,
                    pan_track_range=float(sv.pan_track_range),
                    tilt_track_range=float(sv.tilt_track_range),
                    deadzone_x=float(ft.face_track_deadzone_x),
                    deadzone_y=float(ft.face_track_deadzone_y),
                )
                target_pan += (mapped_pan - target_pan) * servo_alpha
                target_tilt += (mapped_tilt - target_tilt) * servo_alpha
            else:
                target_pan, pan_vel = tick_spring(
                    target_pan, pan_vel, pan_center, dt, k=NO_FACE_SPRING_K, damp=NO_FACE_SPRING_DAMP
                )
                target_tilt, tilt_vel = tick_spring(
                    target_tilt,
                    tilt_vel,
                    tilt_center,
                    dt,
                    k=NO_FACE_SPRING_K,
                    damp=NO_FACE_SPRING_DAMP,
                )

            pan_current, pan_vel = tick_toward(
                pan_current,
                pan_vel,
                target_pan,
                dt,
                lo=pan_min,
                hi=pan_max,
                params=pan_motion,
            )
            tilt_current, tilt_vel = tick_toward(
                tilt_current,
                tilt_vel,
                target_tilt,
                dt,
                lo=tilt_min,
                hi=tilt_max,
                params=tilt_motion,
            )

            if link is not None:
                link.write_angles(pan_current, tilt_current)

            now = time.monotonic()
            if now >= next_status:
                if center is not None:
                    print(
                        f"faces={face_count}  "
                        f"aim=({center[0]:.0f},{center[1]:.0f})  "
                        f"head P{pan_current:.1f} T{tilt_current:.1f}"
                    )
                else:
                    print(f"faces=0  head P{pan_current:.1f} T{tilt_current:.1f} (centering)")
                next_status = now + status_interval

            elapsed = time.monotonic() - loop_start
            sleep_for = dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        picam2.stop()
        picam2.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live YuNet face detection with ESP32 head servo tracking"
    )
    parser.add_argument("--port", default="", help="ESP32 serial port (default auto)")
    parser.add_argument("--baud", type=int, default=CFG.servo.arduino_baud)
    parser.add_argument(
        "--no-servo",
        action="store_true",
        help="Run face detection only (no serial / head motion)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default run until Ctrl+C)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    sv = CFG.servo
    pan_center = (sv.pan_min + sv.pan_max) * 0.5
    tilt_center = (sv.tilt_min + sv.tilt_max) * 0.5

    link: ArduinoServoLink | None = None
    if not args.no_servo:
        link = ArduinoServoLink(port=args.port or sv.arduino_port, baud=args.baud)
        if not link.connect():
            print("Failed to connect to ESP32. Use --no-servo to test camera only.")
            return 1

    try:
        run_tracking(link, duration_sec=args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        if link is not None:
            print(f"Homing head to P{pan_center:.1f} T{tilt_center:.1f}...")
            link.write_angles(pan_center, tilt_center, force=True)
            time.sleep(0.2)
            link.close(home_pan=pan_center, home_tilt=tilt_center)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
