"""Picamera2 color paths — detection and dashboard are independent.

Picamera2 ``RGB888`` capture (numpy H×W×3).
  • YuNet / YOLO: OpenCV **BGR** (auto-probed at boot: rgb_to_bgr vs native_bgr)
  • Dashboard: **RGB** PIL JPEG (``stream_swap_rb`` only affects display)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PICAMERA_MAIN_FORMAT = "RGB888"

# Set at boot by probe_yunet_bgr_mode(); used every vision frame.
DETECTION_BGR_MODE = "rgb_to_bgr"


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 3 and bgr.shape[2] == 3:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr


def channel_means_rgb(rgb: np.ndarray) -> tuple[float, float, float]:
    r, g, b = cv2.split(rgb.astype(np.float32))
    return float(r.mean()), float(g.mean()), float(b.mean())


def to_detection_bgr(capture: np.ndarray, *, rotate_180: bool = False, mode: str | None = None) -> np.ndarray:
    """BGR frame for YuNet/YOLO on the rotated detect view."""
    use_mode = mode or DETECTION_BGR_MODE
    oriented = cv2.rotate(capture, cv2.ROTATE_180) if rotate_180 else capture
    if use_mode == "native_bgr":
        return oriented
    return rgb_to_bgr(oriented)


def to_stream_rgb(
    capture: np.ndarray,
    *,
    rotate_180: bool = False,
    display_swap_rb: bool = False,
) -> np.ndarray:
    """RGB for dashboard overlays. Preview uses unrotated resize (coords mapped separately)."""
    rgb = capture
    if display_swap_rb:
        rgb = rgb[:, :, ::-1]
    return rgb


def gray_world_white_balance_rgb(rgb: np.ndarray, strength: float) -> np.ndarray:
    strength = max(0.0, min(1.0, strength))
    if strength <= 0.0:
        return rgb
    img = rgb.astype(np.float32)
    r, g, b = cv2.split(img)
    mr, mg, mb = r.mean(), g.mean(), b.mean()
    mgray = (mr + mg + mb) / 3.0
    corrected = []
    for ch, mean in zip((r, g, b), (mr, mg, mb)):
        adj = np.clip(ch * (mgray / max(mean, 1e-6)), 0, 255)
        corrected.append(ch * (1.0 - strength) + adj * strength)
    return cv2.merge(corrected).astype(np.uint8)


def encode_stream_jpeg_rgb(rgb: np.ndarray, quality: int, *, swap_rb: bool = False) -> bytes | None:
    if swap_rb:
        rgb = rgb[:, :, ::-1]
    try:
        img = Image.fromarray(np.ascontiguousarray(rgb))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
        return buf.getvalue()
    except Exception:
        return None


def _enhance_face_bgr(bgr: np.ndarray) -> np.ndarray:
    """Boost local contrast — helps YuNet in dim or flat lighting."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def detect_faces_yunet(
    detector: Any,
    capture: np.ndarray,
    *,
    input_size: tuple[int, int],
    rotate_180: bool = False,
) -> tuple[np.ndarray | None, str]:
    """Run YuNet on both BGR layouts (+ CLAHE). Return best face set."""
    global DETECTION_BGR_MODE
    w, h = input_size
    best_faces = None
    best_mode = DETECTION_BGR_MODE
    best_area = 0.0

    for mode in ("rgb_to_bgr", "native_bgr"):
        bgr = to_detection_bgr(capture, rotate_180=rotate_180, mode=mode)
        for _, frame in ((mode, bgr), (mode, _enhance_face_bgr(bgr))):
            detector.setInputSize((w, h))
            result = detector.detect(frame)
            if result[1] is None:
                continue
            faces = result[1]
            largest = max(faces, key=lambda f: float(f[2]) * float(f[3]))
            area = float(largest[2]) * float(largest[3])
            if area > best_area:
                best_area = area
                best_faces = faces
                best_mode = mode

    if best_faces is not None:
        DETECTION_BGR_MODE = best_mode
    return best_faces, best_mode


def probe_yunet_bgr_mode(
    detector: Any,
    capture: np.ndarray,
    *,
    input_size: tuple[int, int],
    rotate_180: bool = False,
) -> tuple[str, int]:
    """Pick BGR layout that yields the most YuNet faces on a live frame."""
    faces, mode = detect_faces_yunet(
        detector, capture, input_size=input_size, rotate_180=rotate_180
    )
    count = 0 if faces is None else len(faces)
    return mode, count


def verify_color_pipeline(rgb_frame: np.ndarray, *, save_dir: Path | None = None) -> dict:
    bgr = rgb_to_bgr(rgb_frame)
    rgb_means = channel_means_rgb(rgb_frame)
    bgr_b, bgr_g, bgr_r = cv2.split(bgr.astype(np.float32))
    bgr_means = (float(bgr_b.mean()), float(bgr_g.mean()), float(bgr_r.mean()))

    jpg = encode_stream_jpeg_rgb(rgb_frame, quality=85)
    decode_ok = jpg is not None
    if decode_ok and save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            cv2.imwrite(str(save_dir / "verify_stream.jpg"), decoded)

    rb_consistent = (
        abs(rgb_means[0] - bgr_r.mean()) < 0.01
        and abs(rgb_means[2] - bgr_b.mean()) < 0.01
    )
    return {
        "picamera_format": PICAMERA_MAIN_FORMAT,
        "rb_consistent": rb_consistent,
        "rgb_means": rgb_means,
        "bgr_means": bgr_means,
        "jpeg_decode_ok": decode_ok,
    }


def log_color_pipeline_verification(stats: dict, *, detection_mode: str, face_probe_count: int) -> None:
    rgb_r, rgb_g, rgb_b = stats["rgb_means"]
    bgr_b, bgr_g, bgr_r = stats["bgr_means"]
    print(
        f"Color: {PICAMERA_MAIN_FORMAT} | YuNet BGR mode={detection_mode} "
        f"(probe faces={face_probe_count}) | dashboard PIL RGB"
    )
    print(
        f"  Capture RGB R={rgb_r:.1f} G={rgb_g:.1f} B={rgb_b:.1f}  "
        f"BGR B={bgr_b:.1f} G={bgr_g:.1f} R={bgr_r:.1f}"
    )
    if face_probe_count == 0:
        print("  Face probe: no face in startup frame — detection mode may auto-fix when you appear")
