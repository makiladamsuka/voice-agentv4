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
PICAMERA_PREVIEW_FORMAT = "XBGR8888"  # rpicam-hello / create_preview_configuration

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
    rgb = frame_to_rgb(capture, legacy_swap_rb=display_swap_rb)
    return rgb


def frame_to_rgb(frame: np.ndarray, *, legacy_swap_rb: bool = False) -> np.ndarray:
    """Convert Picamera2 capture to display RGB (PIL/JPEG).

    Picamera2 format → numpy layout (see picamera2 request._get_pil_mode):
      XBGR8888 preview: [R, G, B, X]  — take :3, already RGB
      RGB888 video:     [B, G, R]    — swap channels (or set legacy_swap_rb)
    """
    if frame.ndim == 3 and frame.shape[2] == 4:
        return np.ascontiguousarray(frame[:, :, :3])
    rgb = np.ascontiguousarray(frame)
    if legacy_swap_rb:
        rgb = rgb[:, :, ::-1]
    return rgb


def configure_picamera(
    picam2,
    main_res: tuple[int, int],
    *,
    use_preview_pipeline: bool = True,
    buffer_count: int = 2,
) -> object:
    """Configure Picamera2. Preview pipeline matches rpicam-hello (sRGB ISP)."""
    size = (int(main_res[0]), int(main_res[1]))
    if use_preview_pipeline:
        config = picam2.create_preview_configuration(
            main={"size": size},
            buffer_count=buffer_count,
        )
    else:
        config = picam2.create_video_configuration(
            main={"format": PICAMERA_MAIN_FORMAT, "size": size},
            buffer_count=buffer_count,
        )
    picam2.configure(config)
    return config


def configure_wide_fov_camera(
    picam2,
    main_res: tuple[int, int],
    *,
    raw_sensor_res: tuple[int, int] = (3280, 2464),
    buffer_count: int = 1,
) -> object:
    """Full-sensor crop for widest FOV (trackingeyes2 style). Needs cma=256 on Pi."""
    main_size = (int(main_res[0]), int(main_res[1]))
    raw_size = (int(raw_sensor_res[0]), int(raw_sensor_res[1]))
    config = picam2.create_video_configuration(
        main={"format": PICAMERA_MAIN_FORMAT, "size": main_size},
        raw={"size": raw_size},
        buffer_count=buffer_count,
    )
    picam2.configure(config)
    picam2.set_controls({"ScalerCrop": (0, 0, raw_size[0], raw_size[1])})
    return config


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


def detection_size_for_main(
    main_res: tuple[int, int],
    *,
    max_width: int = 1280,
) -> tuple[int, int]:
    """Detect resolution with the same aspect ratio as the sensor (no vertical squash)."""
    w, h = main_res
    dw = int(max_width)
    dh = int(round(dw * h / w))
    if dh % 2:
        dh += 1
    return dw, dh


def assert_detection_aspect_matches(
    main_res: tuple[int, int],
    detect_res: tuple[int, int],
    *,
    stream_res: tuple[int, int] | None = None,
) -> None:
    """Warn when detect/stream aspect differs from the sensor — YuNet boxes drift."""
    main_aspect = main_res[0] / main_res[1]
    detect_aspect = detect_res[0] / detect_res[1]
    if abs(main_aspect - detect_aspect) > 0.02:
        print(
            f"Warning: detect_res {detect_res} aspect {detect_aspect:.3f} "
            f"!= sensor {main_res} aspect {main_aspect:.3f}. "
            "Face boxes and eye landmarks will be misaligned."
        )
    if stream_res is not None:
        stream_aspect = stream_res[0] / stream_res[1]
        if abs(main_aspect - stream_aspect) > 0.02:
            print(
                f"Warning: stream_res {stream_res} aspect {stream_aspect:.3f} "
                f"!= sensor aspect {main_aspect:.3f}."
            )


def _yunet_best_face_score(faces: np.ndarray) -> float:
    """Largest confidence-weighted face box (YuNet col 14 = score)."""
    best = 0.0
    for face in faces:
        area = float(face[2]) * float(face[3])
        conf = float(face[14]) if len(face) > 14 else 1.0
        best = max(best, area * conf)
    return best


def detect_faces_yunet_fast(
    detector: Any,
    oriented_frame: np.ndarray,
    *,
    input_size: tuple[int, int],
    mode: str | None = None,
) -> np.ndarray | None:
    """Single-mode YuNet detect on an already-oriented frame (hot path)."""
    use_mode = mode or DETECTION_BGR_MODE
    if use_mode == "native_bgr":
        detect_frame = oriented_frame
    else:
        detect_frame = rgb_to_bgr(oriented_frame)
    w, h = input_size
    detector.setInputSize((w, h))
    result = detector.detect(detect_frame)
    if result[1] is None:
        return None
    return result[1]


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
            score = _yunet_best_face_score(faces)
            if score > best_area:
                best_area = score
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


def apply_camera_controls(
    picam2,
    *,
    awb_mode: str = "auto",
    colour_gains: list[float] | None = None,
    sharpness: float | None = 1.0,
    noise_reduction: str = "high",
) -> None:
    """Apply libcamera ISP controls; skips unsupported options on older libcamera builds."""
    try:
        from libcamera import controls as lc
    except ImportError:
        return

    awb_modes = {
        "auto": lc.AwbModeEnum.Auto,
        "daylight": lc.AwbModeEnum.Daylight,
        "cloudy": lc.AwbModeEnum.Cloudy,
        "tungsten": lc.AwbModeEnum.Tungsten,
        "fluorescent": lc.AwbModeEnum.Fluorescent,
        "incandescent": lc.AwbModeEnum.Incandescent,
        "indoor": lc.AwbModeEnum.Indoor,
    }
    ctrl: dict = {}
    if colour_gains and len(colour_gains) >= 2:
        ctrl["AwbEnable"] = False
        ctrl["ColourGains"] = (float(colour_gains[0]), float(colour_gains[1]))
    else:
        ctrl["AwbMode"] = awb_modes.get(str(awb_mode).lower(), lc.AwbModeEnum.Auto)
    if sharpness is not None and float(sharpness) > 0:
        ctrl["Sharpness"] = float(sharpness)

    nr_enum = getattr(lc, "NoiseReductionModeEnum", None)
    if nr_enum is None:
        draft = getattr(lc, "draft", None)
        nr_enum = getattr(draft, "NoiseReductionModeEnum", None) if draft else None
    if nr_enum is not None and noise_reduction:
        nr_modes = {
            "off": nr_enum.Off,
            "minimal": nr_enum.Minimal,
            "fast": nr_enum.Fast,
            "high": nr_enum.HighQuality,
            "highquality": nr_enum.HighQuality,
        }
        nr = nr_modes.get(str(noise_reduction).lower())
        if nr is not None:
            ctrl["NoiseReductionMode"] = nr

    try:
        picam2.set_controls(ctrl)
        print(f"Camera controls: {ctrl}")
    except Exception as exc:
        print(f"Warning: camera controls not applied: {exc}")
