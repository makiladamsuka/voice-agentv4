"""Load robot tuning from config.yaml into typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH: Path | None = None
_CACHED: RobotConfig | None = None

# Fields safe to change at runtime (no camera/display restart).
RESTART_REQUIRED_PREFIXES = ("display.", "camera.main_res", "camera.detect_res", "stream.port", "stream.host")

TUNING_FIELDS: list[dict[str, Any]] = [
    {"path": "face_tracking.face_track_smooth_alpha", "label": "Face smooth alpha", "min": 0.05, "max": 0.35, "step": 0.01},
    {"path": "face_tracking.face_track_deadzone_x", "label": "Face deadzone X", "min": 0.0, "max": 0.15, "step": 0.01},
    {"path": "face_tracking.face_track_deadzone_y", "label": "Face deadzone Y", "min": 0.0, "max": 0.15, "step": 0.01},
    {"path": "face_tracking.face_track_intensity", "label": "Face track intensity", "min": 0.3, "max": 1.0, "step": 0.05},
    {"path": "face_tracking.face_track_servo_alpha", "label": "Servo follow alpha", "min": 0.1, "max": 0.8, "step": 0.05},
    {"path": "camera.confidence_threshold", "label": "Face confidence", "min": 0.3, "max": 0.9, "step": 0.05},
    {"path": "emotion.min_hold_sec", "label": "Emotion min hold (s)", "min": 0.1, "max": 2.0, "step": 0.05},
    {"path": "emotion.switch_cooldown_sec", "label": "Emotion cooldown (s)", "min": 0.1, "max": 1.0, "step": 0.05},
    {"path": "eyes.max_x_offset", "label": "Max eye X offset", "min": 10, "max": 40, "step": 1},
    {"path": "eyes.max_y_offset", "label": "Max eye Y offset", "min": 8, "max": 30, "step": 1},
    {"path": "eyes.blink_speed_min", "label": "Blink speed min", "min": 1.0, "max": 6.0, "step": 0.1},
    {"path": "eyes.blink_speed_max", "label": "Blink speed max", "min": 2.0, "max": 8.0, "step": 0.1},
    {"path": "servo.smoothing", "label": "Servo smoothing", "min": 0.02, "max": 0.4, "step": 0.02},
    {"path": "stream.fps", "label": "Stream FPS", "min": 2, "max": 15, "step": 1},
    {"path": "stream.jpeg_quality", "label": "JPEG quality", "min": 30, "max": 95, "step": 5},
]


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> None:
    if not data:
        return
    for key, value in data.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)

@dataclass
class DisplayConfig:
    screen_width: int = 128
    screen_height: int = 160
    eye_color: list[int] = field(default_factory=lambda: [255, 255, 255])
    bg_color: list[int] = field(default_factory=lambda: [0, 0, 0])
    eye_size: int = 126
    floor_y_offset: int = 5


@dataclass
class CameraConfig:
    face_model_path: str = "face_detection_yunet_2023mar.onnx"
    main_res: list[int] = field(default_factory=lambda: [1280, 720])
    detect_res: list[int] = field(default_factory=lambda: [640, 360])
    stream_res: list[int] = field(default_factory=lambda: [320, 180])
    confidence_threshold: float = 0.6
    nms_threshold: float = 0.3
    rotate_180: bool = True
    stream_swap_rb: bool = True


@dataclass
class StreamConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    fps: int = 8
    jpeg_quality: int = 70
    render_fps: int = 24
    vision_fps: int = 10


@dataclass
class EyesConfig:
    max_x_offset: float = 30
    max_y_offset: float = 22
    face_roll_mult: float = 0.0
    face_roll_max_deg: float = 10.0
    eye_bound_margin: int = 8
    min_eye_scale: float = 0.85
    max_eye_scale: float = 1.28
    max_top_lid: float = 0.90
    max_bottom_lid: float = 0.82
    eye_move_footprint_x: float = 0.34
    eye_move_footprint_y: float = 0.36
    eye_render_pad_x: float = 4.0
    eye_render_pad_y: float = 6.0
    eye_motion_clamp_scale: float = 0.82
    blink_speed_min: float = 3.2
    blink_speed_max: float = 4.2
    look_side_offset: float = 16.0
    eye_head_ratio: float = 0.88
    eye_head_ratio_face: float = 0.38
    eye_head_smooth_alpha: float = 0.14
    head_eye_pan_sign: float = 1.0
    head_eye_tilt_sign: float = -1.0
    sleep_tilt_deg: float = 8.0
    jerk_amplitude: float = 9.0
    jerk_duration: float = 0.30


@dataclass
class DebugConfig:
    emotions: bool = False
    emotion_reason: bool = False
    amplitude: bool = False


@dataclass
class FaceTrackingConfig:
    close_face_enter_ratio: float = 0.05
    close_face_exit_ratio: float = 0.042
    far_face_area_ratio: float = 0.018
    far_squint_chance: float = 0.08
    far_squint_min_sec: float = 0.22
    far_squint_max_sec: float = 0.55
    face_track_emotions: list[str] = field(default_factory=lambda: [
        "attentive", "engaged", "warm", "content", "curious_intense",
        "amused", "thinking", "cheerful", "happy", "idle",
    ])
    face_track_default: str = "attentive"
    face_track_intensity: float = 0.85
    face_track_smooth_alpha: float = 0.12
    face_track_smooth_alpha_idle: float = 0.15
    face_track_deadzone_x: float = 0.06
    face_track_deadzone_y: float = 0.07
    face_track_servo_alpha: float = 0.38
    face_present_hold_sec: float = 0.50
    face_absent_before_scan_sec: float = 1.0
    face_stable_before_track_sec: float = 0.6
    face_acquire_snap_alpha: float = 0.55
    face_acquire_snap_duration_sec: float = 0.40
    face_scan_cooldown_after_lock_sec: float = 6.0
    no_face_sleepy_sec: float = 120.0
    no_face_bored_sec: float = 180.0
    no_face_idle_blend_min_sec: float = 2.0
    no_face_idle_blend_max_sec: float = 3.4
    no_face_idle_blend_stages: int = 2
    no_face_wander_sec: float = 150.0
    wander_peek_min_sec: float = 5.0
    wander_peek_max_sec: float = 8.0
    wander_peek_chance: float = 0.85
    wander_search_pan_amp_deg: float = 18.0
    wander_search_tilt_amp_deg: float = 12.0
    wander_search_period_sec: float = 14.0
    wander_search_tilt_phase_k: float = 1.3
    sad_return_sec: float = 10.0
    sad_nod_tilt_deg: float = 10.0
    sad_nod_count: float = 2.0
    no_face_sad_recenter_alpha: float = 0.12
    wander_emotions: list[str] = field(default_factory=lambda: [
        "curious_intense", "uncertain", "thinking", "attentive", "curious",
    ])
    settled_sleepy_variety_min_sec: float = 20.0
    settled_sleepy_variety_max_sec: float = 40.0
    no_face_idle_pan_deg: float = 14.0
    no_face_idle_tilt_deg: float = 10.0
    no_face_idle_eye_x: float = 8.0
    no_face_idle_eye_y: float = 10.0
    chat_ready_recenter_alpha: float = 0.18
    chat_ready_min_sec: float = 120.0
    wake_tilt_jerk_deg: float = 12.0
    wake_tilt_jerk_sec: float = 0.35
    wake_surprise_sec: float = 0.35
    awake_conv_prev: list[str] = field(default_factory=lambda: ["waiting", "awkward"])
    awake_conv_active: list[str] = field(default_factory=lambda: ["listening", "speaking", "thinking"])
    no_face_recenter_sec: float = 1.5
    no_face_recenter_alpha: float = 0.06


@dataclass
class EmotionConfig:
    min_hold_sec: float = 0.70
    speak_hold_sec: float = 0.45
    switch_cooldown_sec: float = 0.35
    excited_burst_sec: float = 0.65
    router_stable_sec: float = 0.12
    side_look_enter_offset: float = 7.0
    side_look_exit_offset: float = 4.5
    side_look_switch_cooldown_sec: float = 0.22
    multi_face_debounce_sec: float = 0.16
    jerk_cooldown_sec: float = 0.60
    social_mode_min_sec: float = 0.70
    social_mode_max_sec: float = 2.00
    happy_min_gap_sec: float = 2.80
    speak_emotions: list[str] = field(default_factory=lambda: [
        "engaged", "excited", "cheerful", "amused", "warm", "happy", "content",
    ])
    speak_social_min_sec: float = 0.80
    speak_social_max_sec: float = 1.20
    connected_solo_emotions: list[str] = field(default_factory=lambda: [
        "cheerful", "happy", "content", "warm", "engaged", "excited",
    ])
    lazy_emotions: list[str] = field(default_factory=lambda: ["sleepy", "bored", "idle", "awkward"])


@dataclass
class GazeConfig:
    lock_after_face_sec: float = 3.0
    min_gap_min_sec: float = 4.0
    min_gap_max_sec: float = 6.0
    ambient_scan_min_sec: float = 8.0
    ambient_scan_max_sec: float = 15.0
    no_face_search_min_scans: int = 3
    no_face_scan_trigger_chance: float = 0.55
    no_face_scan_retry_min_sec: float = 3.0
    no_face_scan_retry_max_sec: float = 7.0
    no_face_scan_servo_pan_deg: float = 12.0
    no_face_scan_servo_tilt_deg: float = 10.0
    no_face_scan_tilt_phase: float = 0.35
    solo_upbeat_min_sec: float = 25.0
    social_release_min_sec: float = 20.0
    social_release_max_sec: float = 30.0
    brief_x: float = 12.0
    brief_y: float = 7.0
    think_x: float = 18.0
    think_y: float = 12.0
    scan_x: float = 24.0
    scan_y: float = 11.0
    release_x: float = 18.0
    release_y: float = 6.0
    servo_pan_per_px: float = 0.14
    servo_tilt_per_px: float = 0.12


@dataclass
class ServoConfig:
    enabled: bool = True
    pan_ch: int = 0
    tilt_ch: int = 1
    pan_min: float = 40.0
    pan_max: float = 130.0
    tilt_min: float = 80.0
    tilt_max: float = 130.0
    pulse_min: int = 450
    pulse_max: int = 2600
    smoothing: float = 0.10
    loop_delay: float = 0.01
    max_step_deg: float = 1.4
    deadzone_deg: float = 0.22
    pan_track_range: float = 26.0
    tilt_track_range: float = 24.0
    target_filter_alpha: float = 0.30
    track_damp_alpha_scale: float = 0.40
    track_damp_slow_thresh: float = 0.02
    track_damp_fast_thresh: float = 0.008
    conv_nod_deg: float = 6.0
    conv_nod_hz: float = 5.5
    conv_think_bob_deg: float = 3.0
    conv_think_bob_hz: float = 2.2
    talk_nod_tilt_mult: float = 10.0
    talk_sway_pan_mult: float = 7.5
    talk_gesture_pan_mult: float = 0.0
    talk_gesture_tilt_mult_face: float = 10.0
    talk_gesture_tilt_mult_no_face: float = 10.0
    face_talk_punch_scale: float = 0.45
    face_talk_af_thresh: float = 0.04


@dataclass
class RobotConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    eyes: EyesConfig = field(default_factory=EyesConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    face_tracking: FaceTrackingConfig = field(default_factory=FaceTrackingConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_config_path() -> Path | None:
    return _CONFIG_PATH


def get_by_path(cfg: RobotConfig, path: str) -> Any:
    obj: Any = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_by_path(cfg: RobotConfig, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def is_restart_required(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") or path.startswith(p)
               for p in RESTART_REQUIRED_PREFIXES)


def get_tuning_schema(cfg: RobotConfig) -> list[dict[str, Any]]:
    out = []
    for field in TUNING_FIELDS:
        entry = dict(field)
        entry["value"] = get_by_path(cfg, field["path"])
        out.append(entry)
    return out


def patch_config(cfg: RobotConfig, patches: list[dict[str, Any]]) -> list[str]:
    """Apply patches; returns list of errors (empty on success)."""
    errors: list[str] = []
    for patch in patches:
        path = patch.get("path")
        if not path:
            errors.append("missing path")
            continue
        if is_restart_required(path):
            errors.append(f"{path} requires restart")
            continue
        if not any(path == f["path"] for f in TUNING_FIELDS):
            # Allow debug booleans and any known dataclass field
            try:
                get_by_path(cfg, path)
            except AttributeError:
                errors.append(f"unknown path: {path}")
                continue
        try:
            set_by_path(cfg, path, patch["value"])
        except (AttributeError, TypeError) as e:
            errors.append(f"{path}: {e}")
    return errors


def save_config(cfg: RobotConfig, path: Path | None = None) -> Path:
    if path is None:
        path = _CONFIG_PATH or Path(__file__).parent / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)
    return path


def load_config(path: Path | None = None) -> RobotConfig:
    global _CONFIG_PATH, _CACHED
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    _CONFIG_PATH = path

    cfg = RobotConfig()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _merge_dataclass(cfg, raw)

    _CACHED = cfg
    return cfg


def reload() -> RobotConfig:
    """Phase 3 hook — re-read config.yaml from disk."""
    if _CONFIG_PATH is None:
        return load_config()
    return load_config(_CONFIG_PATH)


if __name__ == "__main__":
    cfg = load_config()
    print(yaml.dump(cfg.to_dict(), default_flow_style=False, sort_keys=False))
