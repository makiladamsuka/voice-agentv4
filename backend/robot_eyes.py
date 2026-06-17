#!/usr/bin/env python3
"""
Face Tracking Eyes for Dual SPI Displays (Picamera2)
Combines face tracking (YuNet) with dual SPI display output (ST7735).

Optional: ESP32 pan/tilt face following via USB serial (PCA9685).
"""

import signal
import subprocess
import time
import math
import random
import sys
import io
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import cv2
import socket
import json
from pathlib import Path

from robot_config import (
    get_config_path,
    get_tuning_schema,
    load_config,
    patch_config,
    save_config,
)
from servo_driver import create_servo_driver
from elastic_head_motion import (
    HeadMotionParams,
    OrganicWanderSearch,
    clamp,
    scale_head_motion,
    tick_toward,
    wander_search_tilt_from_eye_level,
)
from head_servo_axes import check_servo_channel_config
from tof_presence import TofPresenceTracker, TofSnapshot, TofPresence, sanitize_tof_snapshot
from tof_approach import TofApproachController
from animation_player import AnimationPlayer
from botango_loader import (
    DEFAULT_ARM_NEUTRALS,
    _parse_setup,
    format_servo_stop_pose,
    load_botango_commands_file,
    neutral_arm_degrees,
    servo_stop_pose,
)
from person_detector import PersonDetector
from camera_color import (
    apply_camera_controls,
    assert_detection_aspect_matches,
    configure_picamera,
    configure_wide_fov_camera,
    detect_faces_yunet,
    detect_faces_yunet_fast,
    frame_to_rgb,
    log_color_pipeline_verification,
    probe_yunet_bgr_mode,
    to_detection_bgr,
    verify_color_pipeline,
)
from surroundings_emotion import (
    SurroundingsEmotionConfig as SurroundingsEmotionRuntimeConfig,
    SurroundingsEmotionController,
)

# Shared amplitude state written by the UDP thread, read by the render loop
udp_emotion_override = None
udp_emotion_until = 0.0
amplitude_fast = 0.0   # alpha=0.6, syllable-level micro-reactions
amplitude_slow = 0.0   # alpha=0.05, emotional momentum
amplitude_prev_fast = 0.0  # previous frame fast value, used for derivative
udp_speak_pulse = 0.0  # 1.0 when agent is speaking, 0.0 otherwise

# Layer 2: Conversation state (listening | thinking | speaking | waiting)
udp_conv_state = "waiting"        # current conversation phase
udp_conv_emotion = "attentive"    # emotion the conv-state wants to display

# Hardware / Display Imports
import board
import busio
import digitalio
from PIL import Image, ImageDraw
try:
    from adafruit_rgb_display import st7735
except ImportError:
    print("Error: adafruit-circuitpython-rgb-display not found.")
    print("pip3 install adafruit-circuitpython-rgb-display")
    sys.exit(1)

# Camera Import
try:
    from picamera2 import Picamera2
except ImportError:
    print("Error: picamera2 not found. Please install with: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    from libcamera import controls as libcamera_controls
except ImportError:
    libcamera_controls = None

# --- Configuration (loaded from config.yaml) ---
_cfg_path = Path(__file__).parent / "config.yaml"
cfg = load_config(_cfg_path)
_STATIC_DIR = Path(__file__).parent / "static"

_d = cfg.display
_c = cfg.camera
_s = cfg.stream
_e = cfg.eyes
_db = cfg.debug
_ft = cfg.face_tracking
_em = cfg.emotion
_gz = cfg.gaze
_sv = cfg.servo
_base = cfg.base
_tof = cfg.tof
_se = cfg.surroundings_emotion

SCREEN_WIDTH = _d.screen_width
SCREEN_HEIGHT = _d.screen_height
EYE_COLOR = tuple(_d.eye_color)
BG_COLOR = tuple(_d.bg_color)
EYE_SIZE = _d.eye_size
FLOOR_Y = SCREEN_HEIGHT - _d.floor_y_offset

FACE_MODEL_PATH = _c.face_model_path
BODY_MODEL_PATH = _c.body_model_path
BODY_ENABLED = _c.body_enabled
BODY_CONFIDENCE_THRESHOLD = _c.body_confidence_threshold
BODY_NMS_THRESHOLD = _c.body_nms_threshold
BODY_INPUT_SIZE = _c.body_input_size
BODY_DETECT_STRIDE = _c.body_detect_stride
BODY_TRACK_SERVO_ALPHA = _c.body_track_servo_alpha
BODY_AIM_Y_RATIO = _c.body_aim_y_ratio
CAMERA_MAIN_RES = tuple(_c.main_res)
CAMERA_RES = tuple(_c.detect_res)
STREAM_RES = tuple(_c.stream_res)
CAMERA_WIDE_FOV = _c.wide_fov
CAMERA_RAW_SENSOR_RES = tuple(_c.raw_sensor_res)
CAMERA_USE_PREVIEW = _c.use_preview_pipeline and not CAMERA_WIDE_FOV
CONFIDENCE_THRESHOLD = _c.confidence_threshold
NMS_THRESHOLD = _c.nms_threshold
CAMERA_ROTATE_180 = _c.rotate_180
STREAM_SWAP_RB = _c.stream_swap_rb
CAMERA_AWB_MODE = _c.awb_mode
CAMERA_COLOUR_GAINS = _c.colour_gains
CAMERA_SHARPNESS = _c.sharpness
CAMERA_NOISE_REDUCTION = _c.noise_reduction
STREAM_WHITE_BALANCE = _c.stream_white_balance
STREAM_WB_STRENGTH = _c.stream_wb_strength

MAX_X_OFFSET = _e.max_x_offset
MAX_Y_OFFSET = _e.max_y_offset
FACE_ROLL_MULT = _e.face_roll_mult
FACE_ROLL_MAX_DEG = _e.face_roll_max_deg
EYE_BOUND_MARGIN = _e.eye_bound_margin
MIN_EYE_SCALE = _e.min_eye_scale
MAX_EYE_SCALE = _e.max_eye_scale
MAX_TOP_LID = _e.max_top_lid
MAX_BOTTOM_LID = _e.max_bottom_lid
EYE_MOVE_FOOTPRINT_X = _e.eye_move_footprint_x
EYE_MOVE_FOOTPRINT_Y = _e.eye_move_footprint_y
EYE_RENDER_PAD_X = _e.eye_render_pad_x
EYE_RENDER_PAD_Y = _e.eye_render_pad_y
EYE_MOTION_CLAMP_SCALE = _e.eye_motion_clamp_scale
BLINK_SPEED_MIN = _e.blink_speed_min
BLINK_SPEED_MAX = _e.blink_speed_max
LOOK_SIDE_OFFSET = _e.look_side_offset

DEBUG_EMOTIONS = _db.emotions
DEBUG_EMOTION_REASON = _db.emotion_reason
DEBUG_AMPLITUDE = _db.amplitude

CLOSE_FACE_ENTER_RATIO = _ft.close_face_enter_ratio
CLOSE_FACE_EXIT_RATIO = _ft.close_face_exit_ratio
FAR_FACE_AREA_RATIO = _ft.far_face_area_ratio
FAR_SQUINT_CHANCE = _ft.far_squint_chance
FAR_SQUINT_MIN_SEC = _ft.far_squint_min_sec
FAR_SQUINT_MAX_SEC = _ft.far_squint_max_sec
NO_FACE_SLEEPY_SEC = _ft.no_face_sleepy_sec
NO_FACE_BORED_SEC = _ft.no_face_bored_sec
NO_FACE_IDLE_BLEND_MIN_SEC = _ft.no_face_idle_blend_min_sec
NO_FACE_IDLE_BLEND_MAX_SEC = _ft.no_face_idle_blend_max_sec
NO_FACE_IDLE_BLEND_STAGES = _ft.no_face_idle_blend_stages
EMOTION_MIN_HOLD_SEC = _em.min_hold_sec
EMOTION_SPEAK_HOLD_SEC = _em.speak_hold_sec
EMOTION_SWITCH_COOLDOWN_SEC = _em.switch_cooldown_sec
EXCITED_BURST_SEC = _em.excited_burst_sec
ROUTER_EMOTION_STABLE_SEC = _em.router_stable_sec
SIDE_LOOK_ENTER_OFFSET = _em.side_look_enter_offset
SIDE_LOOK_EXIT_OFFSET = _em.side_look_exit_offset
SIDE_LOOK_SWITCH_COOLDOWN_SEC = _em.side_look_switch_cooldown_sec
MULTI_FACE_DEBOUNCE_SEC = _em.multi_face_debounce_sec
JERK_COOLDOWN_SEC = _em.jerk_cooldown_sec
SOCIAL_MODE_MIN_SEC = _em.social_mode_min_sec
SOCIAL_MODE_MAX_SEC = _em.social_mode_max_sec
HAPPY_MIN_GAP_SEC = _em.happy_min_gap_sec

FACE_TRACK_EMOTIONS = tuple(_ft.face_track_emotions)
FACE_TRACK_DEFAULT = _ft.face_track_default
SPEAK_EMOTIONS = tuple(_em.speak_emotions)
SPEAK_SOCIAL_MIN_SEC = _em.speak_social_min_sec
SPEAK_SOCIAL_MAX_SEC = _em.speak_social_max_sec
CONNECTED_SOLO_EMOTIONS = tuple(_em.connected_solo_emotions)
LAZY_EMOTIONS = frozenset(_em.lazy_emotions)
FACE_TRACK_INTENSITY = _ft.face_track_intensity
FACE_TRACK_SMOOTH_ALPHA = _ft.face_track_smooth_alpha
FACE_TRACK_SMOOTH_ALPHA_IDLE = _ft.face_track_smooth_alpha_idle
FACE_TRACK_DEADZONE_X = _ft.face_track_deadzone_x
FACE_TRACK_DEADZONE_Y = _ft.face_track_deadzone_y
FACE_TRACK_SERVO_ALPHA = _ft.face_track_servo_alpha
FACE_TRACK_TILT_SIGN = _ft.face_track_tilt_sign
FACE_PRESENT_HOLD_SEC = _ft.face_present_hold_sec
FACE_ABSENT_BEFORE_SCAN_SEC = _ft.face_absent_before_scan_sec
FACE_STABLE_BEFORE_TRACK_SEC = _ft.face_stable_before_track_sec
FACE_ACQUIRE_SNAP_ALPHA = _ft.face_acquire_snap_alpha
FACE_ACQUIRE_SNAP_DURATION_SEC = _ft.face_acquire_snap_duration_sec
FACE_SCAN_COOLDOWN_AFTER_LOCK_SEC = _ft.face_scan_cooldown_after_lock_sec
NO_FACE_WANDER_SEC = _ft.no_face_wander_sec
WANDER_PEEK_MIN_SEC = _ft.wander_peek_min_sec
WANDER_PEEK_MAX_SEC = _ft.wander_peek_max_sec
WANDER_PEEK_CHANCE = _ft.wander_peek_chance
WANDER_SEARCH_PAN_AMP_DEG = _ft.wander_search_pan_amp_deg
WANDER_SEARCH_PAN_STEP_MIN_DEG = _ft.wander_search_pan_step_min_deg
WANDER_SEARCH_PAN_STEP_MAX_DEG = _ft.wander_search_pan_step_max_deg
WANDER_SEARCH_HOLD_MIN_SEC = _ft.wander_search_hold_min_sec
WANDER_SEARCH_HOLD_MAX_SEC = _ft.wander_search_hold_max_sec
WANDER_SEARCH_THINKING_HOLD_CHANCE = _ft.wander_search_thinking_hold_chance
WANDER_SEARCH_THINKING_HOLD_MIN_SEC = _ft.wander_search_thinking_hold_min_sec
WANDER_SEARCH_THINKING_HOLD_MAX_SEC = _ft.wander_search_thinking_hold_max_sec
WANDER_SEARCH_LONG_STARE_CHANCE = _ft.wander_search_long_stare_chance
WANDER_SEARCH_JUMP_CHANCE = _ft.wander_search_jump_chance
WANDER_SEARCH_ARRIVAL_DEG = _ft.wander_search_arrival_deg
WANDER_SEARCH_TILT_MAX_UP_DEG = _ft.wander_search_tilt_max_up_deg
WANDER_SEARCH_TILT_MAX_DOWN_DEG = _ft.wander_search_tilt_max_down_deg
WANDER_SEARCH_TILT_RECENTER_ALPHA = _ft.wander_search_tilt_recenter_alpha
WANDER_SIDE_LOOK_PAN_DEG = _ft.wander_side_look_pan_deg
WANDER_SEARCH_TILT_AMP_DEG = _ft.wander_search_tilt_amp_deg
WANDER_TILT_TARGET_ALPHA = _ft.wander_tilt_target_alpha
WANDER_PAN_TARGET_ALPHA = _ft.wander_pan_target_alpha
SEARCH_BASE_EDGE_DEG = _ft.search_base_edge_deg
SEARCH_BASE_NUDGE_DEG = _ft.search_base_nudge_deg
SEARCH_BASE_COOLDOWN_SEC = _ft.search_base_cooldown_sec
WANDER_BASE_FOLLOW_CHANCE = _ft.wander_base_follow_chance
WANDER_BASE_FOLLOW_DEG = _ft.wander_base_follow_deg
WANDER_BASE_FOLLOW_MIN_PAN_DEG = _ft.wander_base_follow_min_pan_deg
WANDER_BASE_FOLLOW_MIN_DRIFT_VEL = _ft.wander_base_follow_min_drift_vel
WANDER_BASE_FOLLOW_COOLDOWN_SEC = _ft.wander_base_follow_cooldown_sec
WANDER_BASE_FOLLOW_EVAL_SEC = _ft.wander_base_follow_eval_sec
FACE_BASE_ALIVE_ENABLED = _ft.face_base_alive_enabled
FACE_BASE_ALIVE_DEG = _ft.face_base_alive_deg
FACE_BASE_ALIVE_MAX_DEG = _ft.face_base_alive_max_deg
FACE_BASE_ALIVE_MIN_SEC = _ft.face_base_alive_min_sec
FACE_BASE_ALIVE_MAX_SEC = _ft.face_base_alive_max_sec
FACE_BASE_EDGE_NORM = _ft.face_base_edge_norm
FACE_BASE_EDGE_NUDGE_DEG = _ft.face_base_edge_nudge_deg
FACE_BASE_EDGE_PAN_EDGE_DEG = _ft.face_base_edge_pan_edge_deg
FACE_BASE_COOLDOWN_SEC = _ft.face_base_cooldown_sec
FACE_BASE_HEAD_COMP_ALPHA = _ft.face_base_head_comp_alpha
BASE_HOME_DEG = 0.0
BASE_MAX_DEG_FROM_ZERO = _base.max_deg_from_zero
BASE_MAX_NUDGE_DEG = _base.max_nudge_deg
BASE_ERROR_BACKOFF_SEC = _base.error_backoff_sec
SAD_RETURN_SEC = _ft.sad_return_sec
SAD_NOD_TILT_DEG = _ft.sad_nod_tilt_deg
SAD_NOD_COUNT = _ft.sad_nod_count
NO_FACE_SAD_RECENTER_ALPHA = _ft.no_face_sad_recenter_alpha
WANDER_EMOTIONS = tuple(_ft.wander_emotions)
SETTLED_SLEEPY_VARIETY_MIN_SEC = _ft.settled_sleepy_variety_min_sec
SETTLED_SLEEPY_VARIETY_MAX_SEC = _ft.settled_sleepy_variety_max_sec
NO_FACE_IDLE_PAN_DEG = _ft.no_face_idle_pan_deg
NO_FACE_IDLE_TILT_DEG = _ft.no_face_idle_tilt_deg
NO_FACE_IDLE_EYE_X = _ft.no_face_idle_eye_x
NO_FACE_IDLE_EYE_Y = _ft.no_face_idle_eye_y
CHAT_READY_RECENTER_ALPHA = _ft.chat_ready_recenter_alpha
CHAT_READY_MIN_SEC = _ft.chat_ready_min_sec
WAKE_TILT_JERK_DEG = _ft.wake_tilt_jerk_deg
WAKE_TILT_JERK_SEC = _ft.wake_tilt_jerk_sec
WAKE_SURPRISE_SEC = _ft.wake_surprise_sec
AWAKE_CONV_PREV = tuple(_ft.awake_conv_prev)
AWAKE_CONV_ACTIVE = tuple(_ft.awake_conv_active)

GAZE_LOCK_AFTER_FACE_SEC = _gz.lock_after_face_sec
GAZE_MIN_GAP_MIN_SEC = _gz.min_gap_min_sec
GAZE_MIN_GAP_MAX_SEC = _gz.min_gap_max_sec
GAZE_AMBIENT_SCAN_MIN_SEC = _gz.ambient_scan_min_sec
GAZE_AMBIENT_SCAN_MAX_SEC = _gz.ambient_scan_max_sec
NO_FACE_SEARCH_MIN_SCANS = _gz.no_face_search_min_scans
NO_FACE_SCAN_TRIGGER_CHANCE = _gz.no_face_scan_trigger_chance
NO_FACE_SCAN_RETRY_MIN_SEC = _gz.no_face_scan_retry_min_sec
NO_FACE_SCAN_RETRY_MAX_SEC = _gz.no_face_scan_retry_max_sec
NO_FACE_SCAN_SERVO_PAN_DEG = _gz.no_face_scan_servo_pan_deg
NO_FACE_SCAN_SERVO_TILT_DEG = _gz.no_face_scan_servo_tilt_deg
NO_FACE_SCAN_TILT_PHASE = _gz.no_face_scan_tilt_phase
SOLO_UPBEAT_MIN_SEC = _gz.solo_upbeat_min_sec
GAZE_SOCIAL_RELEASE_MIN_SEC = _gz.social_release_min_sec
GAZE_SOCIAL_RELEASE_MAX_SEC = _gz.social_release_max_sec
GAZE_BRIEF_X = _gz.brief_x
GAZE_BRIEF_Y = _gz.brief_y
GAZE_THINK_X = _gz.think_x
GAZE_THINK_Y = _gz.think_y
GAZE_SCAN_X = _gz.scan_x
GAZE_SCAN_Y = _gz.scan_y
GAZE_RELEASE_X = _gz.release_x
GAZE_RELEASE_Y = _gz.release_y
GAZE_SERVO_PAN_PER_PX = _gz.servo_pan_per_px
GAZE_SERVO_TILT_PER_PX = _gz.servo_tilt_per_px

# --- Emotion Presets ---
EMOTION_PRESETS = {
    "idle": {"scale_w": 1.0, "scale_h": 1.0, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "happy": {"scale_w": 1.10, "scale_h": 0.84, "top_lid": 0.0, "bottom_lid": 0.30, "lid_angle": -6.0, "mirror_angle": True},
    "excited": {"scale_w": 1.14, "scale_h": 0.80, "top_lid": 0.0, "bottom_lid": 0.24, "lid_angle": 0.0, "mirror_angle": True},
    "bored": {"scale_w": 1.03, "scale_h": 0.78, "top_lid": 0.48, "bottom_lid": 0.12, "lid_angle": 0.0, "mirror_angle": True},
    "sad": {"scale_w": 0.98, "scale_h": 1.08, "top_lid": 0.20, "bottom_lid": 0.0, "lid_angle": 10.0, "mirror_angle": True},
    "angry": {"scale_w": 1.02, "scale_h": 0.90, "top_lid": 0.24, "bottom_lid": 0.0, "lid_angle": -14.0, "mirror_angle": True},
    "surprised": {"scale_w": 0.98, "scale_h": 1.12, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "suspicious": {"scale_w": 1.06, "scale_h": 0.74, "top_lid": 0.38, "bottom_lid": 0.35, "lid_angle": 0.0, "mirror_angle": True},
    "sleepy": {"scale_w": 1.04, "scale_h": 0.88, "top_lid": 0.56, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "looking_left_natural": {"scale_w": 1.02, "scale_h": 0.98, "top_lid": 0.0, "bottom_lid": 0.05, "lid_angle": -3.0, "mirror_angle": False},
    "looking_right_natural": {"scale_w": 1.02, "scale_h": 0.98, "top_lid": 0.0, "bottom_lid": 0.05, "lid_angle": 3.0, "mirror_angle": False},
    "looking_left_happy": {"scale_w": 1.10, "scale_h": 0.84, "top_lid": 0.0, "bottom_lid": 0.30, "lid_angle": -6.0, "mirror_angle": False},
    "looking_right_happy": {"scale_w": 1.10, "scale_h": 0.84, "top_lid": 0.0, "bottom_lid": 0.30, "lid_angle": 6.0, "mirror_angle": False},
    "thinking": {"scale_w": 1.00, "scale_h": 0.92, "top_lid": 0.06, "bottom_lid": 0.02, "lid_angle": 0.0, "mirror_angle": True},
    "concentrating": {"scale_w": 0.96, "scale_h": 0.84, "top_lid": 0.16, "bottom_lid": 0.08, "lid_angle": 0.0, "mirror_angle": True},
    "remembering": {"scale_w": 1.04, "scale_h": 1.03, "top_lid": 0.02, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "attentive": {"scale_w": 1.08, "scale_h": 1.06, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "engaged": {"scale_w": 1.02, "scale_h": 1.00, "top_lid": 0.04, "bottom_lid": 0.06, "lid_angle": 5.0, "mirror_angle": True},
    "amused": {"scale_w": 1.00, "scale_h": 0.98, "top_lid": 0.0, "bottom_lid": 0.14, "lid_angle": 3.0, "mirror_angle": True},
    "warm": {"scale_w": 1.06, "scale_h": 1.00, "top_lid": 0.0, "bottom_lid": 0.16, "lid_angle": 2.0, "mirror_angle": True},
    "curious_intense": {"scale_w": 1.04, "scale_h": 1.05, "top_lid": 0.0, "bottom_lid": 0.06, "lid_angle": 8.0, "mirror_angle": False},
    "nodding": {"scale_w": 1.00, "scale_h": 1.00, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": 0.0, "mirror_angle": True},
    "awkward": {"scale_w": 0.96, "scale_h": 0.93, "top_lid": 0.10, "bottom_lid": 0.10, "lid_angle": 0.0, "mirror_angle": True},
    "uncertain": {"scale_w": 0.98, "scale_h": 0.96, "top_lid": 0.08, "bottom_lid": 0.04, "lid_angle": 0.0, "mirror_angle": True},
    "apologetic": {"scale_w": 0.95, "scale_h": 0.92, "top_lid": 0.14, "bottom_lid": 0.04, "lid_angle": 6.0, "mirror_angle": True},
    "proud": {"scale_w": 1.06, "scale_h": 1.02, "top_lid": 0.0, "bottom_lid": 0.0, "lid_angle": -2.0, "mirror_angle": True},
    "playful": {"scale_w": 1.02, "scale_h": 1.00, "top_lid": 0.0, "bottom_lid": 0.06, "lid_angle": 0.0, "mirror_angle": False},
    "cheerful": {"scale_w": 1.08, "scale_h": 0.86, "top_lid": 0.0, "bottom_lid": 0.22, "lid_angle": -4.0, "mirror_angle": True},
    "content": {"scale_w": 1.05, "scale_h": 0.96, "top_lid": 0.0, "bottom_lid": 0.10, "lid_angle": 2.0, "mirror_angle": True},
    "looking_left_cheerful": {"scale_w": 1.08, "scale_h": 0.86, "top_lid": 0.0, "bottom_lid": 0.22, "lid_angle": -4.0, "mirror_angle": False},
    "looking_right_cheerful": {"scale_w": 1.08, "scale_h": 0.86, "top_lid": 0.0, "bottom_lid": 0.22, "lid_angle": 4.0, "mirror_angle": False},
    "squint": {"scale_w": 1.0, "scale_h": 0.62, "top_lid": 0.42, "bottom_lid": 0.35, "lid_angle": 0.0, "mirror_angle": True},
}

# Tracking runs continuously, so keep intensities lower than manual tuner max
# to preserve the softer round-eye style and avoid harsh lid bands.
EMOTION_INTENSITY = {
    "idle": 0.45,
    "looking_left_natural": 0.50,
    "looking_right_natural": 0.50,
    "looking_left_happy": 0.52,
    "looking_right_happy": 0.52,
    "happy": 0.55,
    "excited": 0.62,
    "surprised": 0.70,
    "sad": 0.60,
    "angry": 0.58,
    "suspicious": 0.56,
    "sleepy": 0.62,
    "bored": 0.58,
    "thinking": 0.52,
    "concentrating": 0.58,
    "remembering": 0.50,
    "attentive": 0.56,
    "engaged": 0.54,
    "amused": 0.50,
    "warm": 0.52,
    "curious_intense": 0.56,
    "nodding": 0.45,
    "awkward": 0.48,
    "uncertain": 0.48,
    "apologetic": 0.50,
    "proud": 0.54,
    "playful": 0.50,
    "cheerful": 0.54,
    "content": 0.50,
    "looking_left_cheerful": 0.52,
    "looking_right_cheerful": 0.52,
    "squint": 0.85,
}

SOLO_MOOD_TO_EMOTION = {
    "cheerful": "cheerful",
    "content": "content",
    "playful": "playful",
    "warm": "warm",
    "neutral": "idle",
}

SPECIAL_EMOTIONS = ["happy", "suspicious", "sleepy"]

STREAM_ENABLED = _s.enabled
STREAM_HOST = _s.host
STREAM_PORT = _s.port
STREAM_FPS = _s.fps
STREAM_JPEG_QUALITY = _s.jpeg_quality
RENDER_FPS = _s.render_fps
VISION_FPS = _s.vision_fps

TOF_ENABLED = _tof.enabled
TOF_POLL_HZ = _tof.poll_hz
TOF_PRESENT_MAX_MM = _tof.present_max_mm
TOF_ABSENT_MIN_MM = _tof.absent_min_mm
TOF_MIN_VALID_MM = int(_tof.min_valid_mm)
TOF_DEBOUNCE_PRESENT_SEC = _tof.debounce_present_sec
TOF_DEBOUNCE_ABSENT_SEC = _tof.debounce_absent_sec
_tof_approach = _tof.approach
TOF_APPROACH_ENABLED = _tof_approach.enabled
TOF_APPROACH_HEAD_TURN_DEG = _tof_approach.head_turn_deg
TOF_APPROACH_PAN_STEP_DEG = _tof_approach.pan_step_deg
TOF_APPROACH_BOOT_PAN_STEP_DEG = _tof_approach.boot_pan_step_deg
TOF_APPROACH_ARRIVAL_DEG = _tof_approach.arrival_deg
TOF_APPROACH_USE_BASE = _tof_approach.use_base
TOF_APPROACH_BASE_NUDGE_DEG = _tof_approach.base_nudge_deg
TOF_APPROACH_MAX_BASE_NUDGES = _tof_approach.max_base_nudges_per_event
TOF_APPROACH_CONFIRM_DELAY_SEC = _tof_approach.confirm_delay_sec
TOF_APPROACH_LOCKOUT_SEC = _tof_approach.lockout_sec
TOF_APPROACH_LEFT_RIGHT_ONLY = _tof_approach.left_right_only
TOF_APPROACH_BOOT_ORIENT = _tof_approach.boot_orient
TOF_APPROACH_STARTUP_GRACE_SEC = _tof_approach.startup_grace_sec
TOF_APPROACH_TILT_RECENTER_ALPHA = _tof_approach.tilt_recenter_alpha

ENABLE_SERVO = _sv.enabled
PAN_CH = _sv.pan_ch
TILT_CH = _sv.tilt_ch
PAN_MIN = _sv.pan_min
PAN_MAX = _sv.pan_max
TILT_MIN = _sv.tilt_min
TILT_MAX = _sv.tilt_max
PULSE_MIN = _sv.pulse_min
PULSE_MAX = _sv.pulse_max
SMOOTHING = _sv.smoothing
SERVO_LOOP_DELAY = _sv.loop_delay
MAX_SERVO_STEP_DEG = _sv.max_step_deg
SERVO_DEADZONE_DEG = _sv.deadzone_deg
GOAL_DEADBAND_DEG = _sv.goal_deadband_deg
HEAD_SEND_MIN_DELTA_DEG = _sv.head_send_min_delta_deg
PAN_TRACK_RANGE = _sv.pan_track_range
TILT_TRACK_RANGE = _sv.tilt_track_range
TARGET_FILTER_ALPHA = _sv.target_filter_alpha
TRACK_DAMP_ALPHA_SCALE = _sv.track_damp_alpha_scale
TRACK_DAMP_SLOW_THRESH = _sv.track_damp_slow_thresh
TRACK_DAMP_FAST_THRESH = _sv.track_damp_fast_thresh
CONV_NOD_DEG = _sv.conv_nod_deg
CONV_NOD_HZ = _sv.conv_nod_hz
CONV_THINK_BOB_DEG = _sv.conv_think_bob_deg
CONV_THINK_BOB_HZ = _sv.conv_think_bob_hz
TALK_NOD_TILT_MULT = _sv.talk_nod_tilt_mult
TALK_SWAY_PAN_MULT = _sv.talk_sway_pan_mult
TALK_GESTURE_PAN_MULT = _sv.talk_gesture_pan_mult
TALK_GESTURE_TILT_MULT_FACE = _sv.talk_gesture_tilt_mult_face
TALK_GESTURE_TILT_MULT_NO_FACE = _sv.talk_gesture_tilt_mult_no_face
FACE_TALK_PUNCH_SCALE = _sv.face_talk_punch_scale
FACE_TALK_AF_THRESH = _sv.face_talk_af_thresh
BASE_ENABLED = _base.enabled
NO_FACE_RECENTER_SEC = _ft.no_face_recenter_sec
NO_FACE_RECENTER_ALPHA = _ft.no_face_recenter_alpha
EYE_HEAD_RATIO = _e.eye_head_ratio
EYE_HEAD_RATIO_FACE = _e.eye_head_ratio_face
EYE_HEAD_RATIO_WANDER = _e.eye_head_ratio_wander
EYE_HEAD_SMOOTH_ALPHA = _e.eye_head_smooth_alpha
HEAD_PAN_PX_PER_DEG = MAX_X_OFFSET / PAN_TRACK_RANGE
HEAD_TILT_PX_PER_DEG = MAX_Y_OFFSET / TILT_TRACK_RANGE
HEAD_EYE_PAN_SIGN = _e.head_eye_pan_sign
HEAD_EYE_TILT_SIGN = _e.head_eye_tilt_sign
SLEEP_TILT_DEG = _e.sleep_tilt_deg
JERK_AMPLITUDE = _e.jerk_amplitude
JERK_DURATION = _e.jerk_duration

_config_lock = threading.Lock()


def _head_motion_params_from_servo(sv) -> tuple[HeadMotionParams, HeadMotionParams]:
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


PAN_MOTION, TILT_MOTION = _head_motion_params_from_servo(_sv)

VALID_CONV_STATES = frozenset({
    "listening", "thinking", "nodding", "remembering",
    "concentrating", "speaking", "waiting",
})


def sync_config_from_cfg() -> None:
    """Push cfg dataclass values into module-level runtime globals."""
    global MAX_X_OFFSET, MAX_Y_OFFSET, FACE_ROLL_MULT, FACE_ROLL_MAX_DEG
    global EYE_BOUND_MARGIN, MIN_EYE_SCALE, MAX_EYE_SCALE, MAX_TOP_LID, MAX_BOTTOM_LID
    global EYE_MOVE_FOOTPRINT_X, EYE_MOVE_FOOTPRINT_Y, EYE_RENDER_PAD_X, EYE_RENDER_PAD_Y
    global EYE_MOTION_CLAMP_SCALE, BLINK_SPEED_MIN, BLINK_SPEED_MAX, LOOK_SIDE_OFFSET
    global DEBUG_EMOTIONS, DEBUG_EMOTION_REASON, DEBUG_AMPLITUDE
    global CLOSE_FACE_ENTER_RATIO, CLOSE_FACE_EXIT_RATIO, FAR_FACE_AREA_RATIO
    global FAR_SQUINT_CHANCE, FAR_SQUINT_MIN_SEC, FAR_SQUINT_MAX_SEC
    global NO_FACE_SLEEPY_SEC, NO_FACE_BORED_SEC, NO_FACE_IDLE_BLEND_MIN_SEC
    global NO_FACE_IDLE_BLEND_MAX_SEC, NO_FACE_IDLE_BLEND_STAGES
    global EMOTION_MIN_HOLD_SEC, EMOTION_SPEAK_HOLD_SEC, EMOTION_SWITCH_COOLDOWN_SEC
    global EXCITED_BURST_SEC, ROUTER_EMOTION_STABLE_SEC, SIDE_LOOK_ENTER_OFFSET
    global SIDE_LOOK_EXIT_OFFSET, SIDE_LOOK_SWITCH_COOLDOWN_SEC, MULTI_FACE_DEBOUNCE_SEC
    global JERK_COOLDOWN_SEC, SOCIAL_MODE_MIN_SEC, SOCIAL_MODE_MAX_SEC, HAPPY_MIN_GAP_SEC
    global FACE_TRACK_EMOTIONS, FACE_TRACK_DEFAULT, SPEAK_EMOTIONS
    global SPEAK_SOCIAL_MIN_SEC, SPEAK_SOCIAL_MAX_SEC, CONNECTED_SOLO_EMOTIONS, LAZY_EMOTIONS
    global FACE_TRACK_INTENSITY, FACE_TRACK_SMOOTH_ALPHA, FACE_TRACK_SMOOTH_ALPHA_IDLE
    global FACE_TRACK_DEADZONE_X, FACE_TRACK_DEADZONE_Y, FACE_TRACK_SERVO_ALPHA
    global FACE_TRACK_TILT_SIGN
    global FACE_PRESENT_HOLD_SEC, FACE_ABSENT_BEFORE_SCAN_SEC, FACE_STABLE_BEFORE_TRACK_SEC
    global FACE_ACQUIRE_SNAP_ALPHA, FACE_ACQUIRE_SNAP_DURATION_SEC
    global FACE_SCAN_COOLDOWN_AFTER_LOCK_SEC, NO_FACE_WANDER_SEC
    global WANDER_PEEK_MIN_SEC, WANDER_PEEK_MAX_SEC, WANDER_PEEK_CHANCE
    global WANDER_SEARCH_PAN_AMP_DEG, WANDER_SEARCH_PAN_STEP_MIN_DEG, WANDER_SEARCH_PAN_STEP_MAX_DEG
    global WANDER_SEARCH_HOLD_MIN_SEC, WANDER_SEARCH_HOLD_MAX_SEC, WANDER_SEARCH_JUMP_CHANCE
    global WANDER_SEARCH_THINKING_HOLD_CHANCE, WANDER_SEARCH_THINKING_HOLD_MIN_SEC
    global WANDER_SEARCH_THINKING_HOLD_MAX_SEC, WANDER_SEARCH_LONG_STARE_CHANCE
    global WANDER_SEARCH_ARRIVAL_DEG, WANDER_SEARCH_TILT_MAX_UP_DEG, WANDER_SEARCH_TILT_MAX_DOWN_DEG
    global WANDER_SEARCH_TILT_RECENTER_ALPHA, WANDER_SIDE_LOOK_PAN_DEG
    global WANDER_SEARCH_TILT_AMP_DEG
    global WANDER_TILT_TARGET_ALPHA, WANDER_PAN_TARGET_ALPHA
    global SEARCH_BASE_EDGE_DEG, SEARCH_BASE_NUDGE_DEG, SEARCH_BASE_COOLDOWN_SEC
    global WANDER_BASE_FOLLOW_CHANCE, WANDER_BASE_FOLLOW_DEG
    global WANDER_BASE_FOLLOW_MIN_PAN_DEG, WANDER_BASE_FOLLOW_MIN_DRIFT_VEL
    global WANDER_BASE_FOLLOW_COOLDOWN_SEC, WANDER_BASE_FOLLOW_EVAL_SEC
    global FACE_BASE_ALIVE_ENABLED, FACE_BASE_ALIVE_DEG, FACE_BASE_ALIVE_MAX_DEG
    global FACE_BASE_ALIVE_MIN_SEC, FACE_BASE_ALIVE_MAX_SEC
    global FACE_BASE_EDGE_NORM, FACE_BASE_EDGE_NUDGE_DEG, FACE_BASE_EDGE_PAN_EDGE_DEG
    global FACE_BASE_COOLDOWN_SEC, FACE_BASE_HEAD_COMP_ALPHA
    global BASE_MAX_DEG_FROM_ZERO
    global BASE_MAX_NUDGE_DEG, BASE_ERROR_BACKOFF_SEC
    global SAD_RETURN_SEC, SAD_NOD_TILT_DEG, SAD_NOD_COUNT
    global NO_FACE_SAD_RECENTER_ALPHA, WANDER_EMOTIONS
    global SETTLED_SLEEPY_VARIETY_MIN_SEC, SETTLED_SLEEPY_VARIETY_MAX_SEC
    global NO_FACE_IDLE_PAN_DEG, NO_FACE_IDLE_TILT_DEG, NO_FACE_IDLE_EYE_X, NO_FACE_IDLE_EYE_Y
    global CHAT_READY_RECENTER_ALPHA, CHAT_READY_MIN_SEC, WAKE_TILT_JERK_DEG
    global WAKE_TILT_JERK_SEC, WAKE_SURPRISE_SEC, AWAKE_CONV_PREV, AWAKE_CONV_ACTIVE
    global GAZE_LOCK_AFTER_FACE_SEC, GAZE_MIN_GAP_MIN_SEC, GAZE_MIN_GAP_MAX_SEC
    global GAZE_AMBIENT_SCAN_MIN_SEC, GAZE_AMBIENT_SCAN_MAX_SEC, NO_FACE_SEARCH_MIN_SCANS
    global NO_FACE_SCAN_TRIGGER_CHANCE, NO_FACE_SCAN_RETRY_MIN_SEC, NO_FACE_SCAN_RETRY_MAX_SEC
    global NO_FACE_SCAN_SERVO_PAN_DEG, NO_FACE_SCAN_SERVO_TILT_DEG, NO_FACE_SCAN_TILT_PHASE
    global SOLO_UPBEAT_MIN_SEC, GAZE_SOCIAL_RELEASE_MIN_SEC, GAZE_SOCIAL_RELEASE_MAX_SEC
    global GAZE_BRIEF_X, GAZE_BRIEF_Y, GAZE_THINK_X, GAZE_THINK_Y, GAZE_SCAN_X, GAZE_SCAN_Y
    global GAZE_RELEASE_X, GAZE_RELEASE_Y, GAZE_SERVO_PAN_PER_PX, GAZE_SERVO_TILT_PER_PX
    global STREAM_FPS, STREAM_JPEG_QUALITY, RENDER_FPS, VISION_FPS, ENABLE_SERVO
    global SMOOTHING, SERVO_LOOP_DELAY, MAX_SERVO_STEP_DEG, SERVO_DEADZONE_DEG
    global GOAL_DEADBAND_DEG, PAN_MOTION, TILT_MOTION, BASE_ENABLED
    global HEAD_SEND_MIN_DELTA_DEG
    global PAN_TRACK_RANGE, TILT_TRACK_RANGE, TARGET_FILTER_ALPHA
    global TRACK_DAMP_ALPHA_SCALE, TRACK_DAMP_SLOW_THRESH, TRACK_DAMP_FAST_THRESH
    global CONV_NOD_DEG, CONV_NOD_HZ, CONV_THINK_BOB_DEG, CONV_THINK_BOB_HZ
    global TALK_NOD_TILT_MULT, TALK_SWAY_PAN_MULT, TALK_GESTURE_PAN_MULT
    global TALK_GESTURE_TILT_MULT_FACE, TALK_GESTURE_TILT_MULT_NO_FACE
    global FACE_TALK_PUNCH_SCALE, FACE_TALK_AF_THRESH
    global NO_FACE_RECENTER_SEC, NO_FACE_RECENTER_ALPHA
    global EYE_HEAD_RATIO, EYE_HEAD_RATIO_FACE, EYE_HEAD_RATIO_WANDER, EYE_HEAD_SMOOTH_ALPHA
    global HEAD_PAN_PX_PER_DEG, HEAD_TILT_PX_PER_DEG
    global HEAD_EYE_PAN_SIGN, HEAD_EYE_TILT_SIGN, SLEEP_TILT_DEG
    global JERK_AMPLITUDE, JERK_DURATION
    global CONFIDENCE_THRESHOLD, NMS_THRESHOLD, CAMERA_ROTATE_180, STREAM_SWAP_RB
    global CAMERA_AWB_MODE, CAMERA_COLOUR_GAINS, STREAM_WHITE_BALANCE, STREAM_WB_STRENGTH
    global BODY_MODEL_PATH, BODY_ENABLED, BODY_CONFIDENCE_THRESHOLD, BODY_NMS_THRESHOLD
    global BODY_INPUT_SIZE, BODY_DETECT_STRIDE, BODY_TRACK_SERVO_ALPHA, BODY_AIM_Y_RATIO

    _e = cfg.eyes
    _db = cfg.debug
    _ft = cfg.face_tracking
    _em = cfg.emotion
    _gz = cfg.gaze
    _sv = cfg.servo
    _s = cfg.stream
    _c = cfg.camera

    MAX_X_OFFSET = _e.max_x_offset
    MAX_Y_OFFSET = _e.max_y_offset
    FACE_ROLL_MULT = _e.face_roll_mult
    FACE_ROLL_MAX_DEG = _e.face_roll_max_deg
    EYE_BOUND_MARGIN = _e.eye_bound_margin
    MIN_EYE_SCALE = _e.min_eye_scale
    MAX_EYE_SCALE = _e.max_eye_scale
    MAX_TOP_LID = _e.max_top_lid
    MAX_BOTTOM_LID = _e.max_bottom_lid
    EYE_MOVE_FOOTPRINT_X = _e.eye_move_footprint_x
    EYE_MOVE_FOOTPRINT_Y = _e.eye_move_footprint_y
    EYE_RENDER_PAD_X = _e.eye_render_pad_x
    EYE_RENDER_PAD_Y = _e.eye_render_pad_y
    EYE_MOTION_CLAMP_SCALE = _e.eye_motion_clamp_scale
    BLINK_SPEED_MIN = _e.blink_speed_min
    BLINK_SPEED_MAX = _e.blink_speed_max
    LOOK_SIDE_OFFSET = _e.look_side_offset
    DEBUG_EMOTIONS = _db.emotions
    DEBUG_EMOTION_REASON = _db.emotion_reason
    DEBUG_AMPLITUDE = _db.amplitude
    CLOSE_FACE_ENTER_RATIO = _ft.close_face_enter_ratio
    CLOSE_FACE_EXIT_RATIO = _ft.close_face_exit_ratio
    FAR_FACE_AREA_RATIO = _ft.far_face_area_ratio
    FAR_SQUINT_CHANCE = _ft.far_squint_chance
    FAR_SQUINT_MIN_SEC = _ft.far_squint_min_sec
    FAR_SQUINT_MAX_SEC = _ft.far_squint_max_sec
    NO_FACE_SLEEPY_SEC = _ft.no_face_sleepy_sec
    NO_FACE_BORED_SEC = _ft.no_face_bored_sec
    NO_FACE_IDLE_BLEND_MIN_SEC = _ft.no_face_idle_blend_min_sec
    NO_FACE_IDLE_BLEND_MAX_SEC = _ft.no_face_idle_blend_max_sec
    NO_FACE_IDLE_BLEND_STAGES = _ft.no_face_idle_blend_stages
    EMOTION_MIN_HOLD_SEC = _em.min_hold_sec
    EMOTION_SPEAK_HOLD_SEC = _em.speak_hold_sec
    EMOTION_SWITCH_COOLDOWN_SEC = _em.switch_cooldown_sec
    EXCITED_BURST_SEC = _em.excited_burst_sec
    ROUTER_EMOTION_STABLE_SEC = _em.router_stable_sec
    SIDE_LOOK_ENTER_OFFSET = _em.side_look_enter_offset
    SIDE_LOOK_EXIT_OFFSET = _em.side_look_exit_offset
    SIDE_LOOK_SWITCH_COOLDOWN_SEC = _em.side_look_switch_cooldown_sec
    MULTI_FACE_DEBOUNCE_SEC = _em.multi_face_debounce_sec
    JERK_COOLDOWN_SEC = _em.jerk_cooldown_sec
    SOCIAL_MODE_MIN_SEC = _em.social_mode_min_sec
    SOCIAL_MODE_MAX_SEC = _em.social_mode_max_sec
    HAPPY_MIN_GAP_SEC = _em.happy_min_gap_sec
    FACE_TRACK_EMOTIONS = tuple(_ft.face_track_emotions)
    FACE_TRACK_DEFAULT = _ft.face_track_default
    SPEAK_EMOTIONS = tuple(_em.speak_emotions)
    SPEAK_SOCIAL_MIN_SEC = _em.speak_social_min_sec
    SPEAK_SOCIAL_MAX_SEC = _em.speak_social_max_sec
    CONNECTED_SOLO_EMOTIONS = tuple(_em.connected_solo_emotions)
    LAZY_EMOTIONS = frozenset(_em.lazy_emotions)
    FACE_TRACK_INTENSITY = _ft.face_track_intensity
    FACE_TRACK_SMOOTH_ALPHA = _ft.face_track_smooth_alpha
    FACE_TRACK_SMOOTH_ALPHA_IDLE = _ft.face_track_smooth_alpha_idle
    FACE_TRACK_DEADZONE_X = _ft.face_track_deadzone_x
    FACE_TRACK_DEADZONE_Y = _ft.face_track_deadzone_y
    FACE_TRACK_SERVO_ALPHA = _ft.face_track_servo_alpha
    FACE_TRACK_TILT_SIGN = _ft.face_track_tilt_sign
    FACE_PRESENT_HOLD_SEC = _ft.face_present_hold_sec
    FACE_ABSENT_BEFORE_SCAN_SEC = _ft.face_absent_before_scan_sec
    FACE_STABLE_BEFORE_TRACK_SEC = _ft.face_stable_before_track_sec
    FACE_ACQUIRE_SNAP_ALPHA = _ft.face_acquire_snap_alpha
    FACE_ACQUIRE_SNAP_DURATION_SEC = _ft.face_acquire_snap_duration_sec
    FACE_SCAN_COOLDOWN_AFTER_LOCK_SEC = _ft.face_scan_cooldown_after_lock_sec
    NO_FACE_WANDER_SEC = _ft.no_face_wander_sec
    WANDER_PEEK_MIN_SEC = _ft.wander_peek_min_sec
    WANDER_PEEK_MAX_SEC = _ft.wander_peek_max_sec
    WANDER_PEEK_CHANCE = _ft.wander_peek_chance
    WANDER_SEARCH_PAN_AMP_DEG = _ft.wander_search_pan_amp_deg
    WANDER_SEARCH_PAN_STEP_MIN_DEG = _ft.wander_search_pan_step_min_deg
    WANDER_SEARCH_PAN_STEP_MAX_DEG = _ft.wander_search_pan_step_max_deg
    WANDER_SEARCH_HOLD_MIN_SEC = _ft.wander_search_hold_min_sec
    WANDER_SEARCH_HOLD_MAX_SEC = _ft.wander_search_hold_max_sec
    WANDER_SEARCH_THINKING_HOLD_CHANCE = _ft.wander_search_thinking_hold_chance
    WANDER_SEARCH_THINKING_HOLD_MIN_SEC = _ft.wander_search_thinking_hold_min_sec
    WANDER_SEARCH_THINKING_HOLD_MAX_SEC = _ft.wander_search_thinking_hold_max_sec
    WANDER_SEARCH_LONG_STARE_CHANCE = _ft.wander_search_long_stare_chance
    WANDER_SEARCH_JUMP_CHANCE = _ft.wander_search_jump_chance
    WANDER_SEARCH_ARRIVAL_DEG = _ft.wander_search_arrival_deg
    WANDER_SEARCH_TILT_MAX_UP_DEG = _ft.wander_search_tilt_max_up_deg
    WANDER_SEARCH_TILT_MAX_DOWN_DEG = _ft.wander_search_tilt_max_down_deg
    WANDER_SEARCH_TILT_RECENTER_ALPHA = _ft.wander_search_tilt_recenter_alpha
    WANDER_SIDE_LOOK_PAN_DEG = _ft.wander_side_look_pan_deg
    WANDER_SEARCH_TILT_AMP_DEG = _ft.wander_search_tilt_amp_deg
    WANDER_TILT_TARGET_ALPHA = _ft.wander_tilt_target_alpha
    WANDER_PAN_TARGET_ALPHA = _ft.wander_pan_target_alpha
    SEARCH_BASE_EDGE_DEG = _ft.search_base_edge_deg
    SEARCH_BASE_NUDGE_DEG = _ft.search_base_nudge_deg
    SEARCH_BASE_COOLDOWN_SEC = _ft.search_base_cooldown_sec
    WANDER_BASE_FOLLOW_CHANCE = _ft.wander_base_follow_chance
    WANDER_BASE_FOLLOW_DEG = _ft.wander_base_follow_deg
    WANDER_BASE_FOLLOW_MIN_PAN_DEG = _ft.wander_base_follow_min_pan_deg
    WANDER_BASE_FOLLOW_MIN_DRIFT_VEL = _ft.wander_base_follow_min_drift_vel
    WANDER_BASE_FOLLOW_COOLDOWN_SEC = _ft.wander_base_follow_cooldown_sec
    WANDER_BASE_FOLLOW_EVAL_SEC = _ft.wander_base_follow_eval_sec
    FACE_BASE_ALIVE_ENABLED = _ft.face_base_alive_enabled
    FACE_BASE_ALIVE_DEG = _ft.face_base_alive_deg
    FACE_BASE_ALIVE_MAX_DEG = _ft.face_base_alive_max_deg
    FACE_BASE_ALIVE_MIN_SEC = _ft.face_base_alive_min_sec
    FACE_BASE_ALIVE_MAX_SEC = _ft.face_base_alive_max_sec
    FACE_BASE_EDGE_NORM = _ft.face_base_edge_norm
    FACE_BASE_EDGE_NUDGE_DEG = _ft.face_base_edge_nudge_deg
    FACE_BASE_EDGE_PAN_EDGE_DEG = _ft.face_base_edge_pan_edge_deg
    FACE_BASE_COOLDOWN_SEC = _ft.face_base_cooldown_sec
    FACE_BASE_HEAD_COMP_ALPHA = _ft.face_base_head_comp_alpha
    SAD_RETURN_SEC = _ft.sad_return_sec
    SAD_NOD_TILT_DEG = _ft.sad_nod_tilt_deg
    SAD_NOD_COUNT = _ft.sad_nod_count
    NO_FACE_SAD_RECENTER_ALPHA = _ft.no_face_sad_recenter_alpha
    WANDER_EMOTIONS = tuple(_ft.wander_emotions)
    SETTLED_SLEEPY_VARIETY_MIN_SEC = _ft.settled_sleepy_variety_min_sec
    SETTLED_SLEEPY_VARIETY_MAX_SEC = _ft.settled_sleepy_variety_max_sec
    NO_FACE_IDLE_PAN_DEG = _ft.no_face_idle_pan_deg
    NO_FACE_IDLE_TILT_DEG = _ft.no_face_idle_tilt_deg
    NO_FACE_IDLE_EYE_X = _ft.no_face_idle_eye_x
    NO_FACE_IDLE_EYE_Y = _ft.no_face_idle_eye_y
    CHAT_READY_RECENTER_ALPHA = _ft.chat_ready_recenter_alpha
    CHAT_READY_MIN_SEC = _ft.chat_ready_min_sec
    WAKE_TILT_JERK_DEG = _ft.wake_tilt_jerk_deg
    WAKE_TILT_JERK_SEC = _ft.wake_tilt_jerk_sec
    WAKE_SURPRISE_SEC = _ft.wake_surprise_sec
    AWAKE_CONV_PREV = tuple(_ft.awake_conv_prev)
    AWAKE_CONV_ACTIVE = tuple(_ft.awake_conv_active)
    GAZE_LOCK_AFTER_FACE_SEC = _gz.lock_after_face_sec
    GAZE_MIN_GAP_MIN_SEC = _gz.min_gap_min_sec
    GAZE_MIN_GAP_MAX_SEC = _gz.min_gap_max_sec
    GAZE_AMBIENT_SCAN_MIN_SEC = _gz.ambient_scan_min_sec
    GAZE_AMBIENT_SCAN_MAX_SEC = _gz.ambient_scan_max_sec
    NO_FACE_SEARCH_MIN_SCANS = _gz.no_face_search_min_scans
    NO_FACE_SCAN_TRIGGER_CHANCE = _gz.no_face_scan_trigger_chance
    NO_FACE_SCAN_RETRY_MIN_SEC = _gz.no_face_scan_retry_min_sec
    NO_FACE_SCAN_RETRY_MAX_SEC = _gz.no_face_scan_retry_max_sec
    NO_FACE_SCAN_SERVO_PAN_DEG = _gz.no_face_scan_servo_pan_deg
    NO_FACE_SCAN_SERVO_TILT_DEG = _gz.no_face_scan_servo_tilt_deg
    NO_FACE_SCAN_TILT_PHASE = _gz.no_face_scan_tilt_phase
    SOLO_UPBEAT_MIN_SEC = _gz.solo_upbeat_min_sec
    GAZE_SOCIAL_RELEASE_MIN_SEC = _gz.social_release_min_sec
    GAZE_SOCIAL_RELEASE_MAX_SEC = _gz.social_release_max_sec
    GAZE_BRIEF_X = _gz.brief_x
    GAZE_BRIEF_Y = _gz.brief_y
    GAZE_THINK_X = _gz.think_x
    GAZE_THINK_Y = _gz.think_y
    GAZE_SCAN_X = _gz.scan_x
    GAZE_SCAN_Y = _gz.scan_y
    GAZE_RELEASE_X = _gz.release_x
    GAZE_RELEASE_Y = _gz.release_y
    GAZE_SERVO_PAN_PER_PX = _gz.servo_pan_per_px
    GAZE_SERVO_TILT_PER_PX = _gz.servo_tilt_per_px
    STREAM_FPS = _s.fps
    STREAM_JPEG_QUALITY = _s.jpeg_quality
    RENDER_FPS = _s.render_fps
    VISION_FPS = _s.vision_fps
    ENABLE_SERVO = _sv.enabled
    SMOOTHING = _sv.smoothing
    SERVO_LOOP_DELAY = _sv.loop_delay
    MAX_SERVO_STEP_DEG = _sv.max_step_deg
    SERVO_DEADZONE_DEG = _sv.deadzone_deg
    GOAL_DEADBAND_DEG = _sv.goal_deadband_deg
    BASE_ENABLED = cfg.base.enabled
    BASE_MAX_DEG_FROM_ZERO = cfg.base.max_deg_from_zero
    BASE_MAX_NUDGE_DEG = cfg.base.max_nudge_deg
    BASE_ERROR_BACKOFF_SEC = cfg.base.error_backoff_sec
    PAN_MOTION, TILT_MOTION = _head_motion_params_from_servo(_sv)
    HEAD_SEND_MIN_DELTA_DEG = _sv.head_send_min_delta_deg
    PAN_TRACK_RANGE = _sv.pan_track_range
    TILT_TRACK_RANGE = _sv.tilt_track_range
    TARGET_FILTER_ALPHA = _sv.target_filter_alpha
    TRACK_DAMP_ALPHA_SCALE = _sv.track_damp_alpha_scale
    TRACK_DAMP_SLOW_THRESH = _sv.track_damp_slow_thresh
    TRACK_DAMP_FAST_THRESH = _sv.track_damp_fast_thresh
    CONV_NOD_DEG = _sv.conv_nod_deg
    CONV_NOD_HZ = _sv.conv_nod_hz
    CONV_THINK_BOB_DEG = _sv.conv_think_bob_deg
    CONV_THINK_BOB_HZ = _sv.conv_think_bob_hz
    TALK_NOD_TILT_MULT = _sv.talk_nod_tilt_mult
    TALK_SWAY_PAN_MULT = _sv.talk_sway_pan_mult
    TALK_GESTURE_PAN_MULT = _sv.talk_gesture_pan_mult
    TALK_GESTURE_TILT_MULT_FACE = _sv.talk_gesture_tilt_mult_face
    TALK_GESTURE_TILT_MULT_NO_FACE = _sv.talk_gesture_tilt_mult_no_face
    FACE_TALK_PUNCH_SCALE = _sv.face_talk_punch_scale
    FACE_TALK_AF_THRESH = _sv.face_talk_af_thresh
    NO_FACE_RECENTER_SEC = _ft.no_face_recenter_sec
    NO_FACE_RECENTER_ALPHA = _ft.no_face_recenter_alpha
    EYE_HEAD_RATIO = _e.eye_head_ratio
    EYE_HEAD_RATIO_FACE = _e.eye_head_ratio_face
    EYE_HEAD_RATIO_WANDER = _e.eye_head_ratio_wander
    EYE_HEAD_SMOOTH_ALPHA = _e.eye_head_smooth_alpha
    HEAD_PAN_PX_PER_DEG = MAX_X_OFFSET / PAN_TRACK_RANGE
    HEAD_TILT_PX_PER_DEG = MAX_Y_OFFSET / TILT_TRACK_RANGE
    HEAD_EYE_PAN_SIGN = _e.head_eye_pan_sign
    HEAD_EYE_TILT_SIGN = _e.head_eye_tilt_sign
    SLEEP_TILT_DEG = _e.sleep_tilt_deg
    JERK_AMPLITUDE = _e.jerk_amplitude
    JERK_DURATION = _e.jerk_duration
    CONFIDENCE_THRESHOLD = _c.confidence_threshold
    NMS_THRESHOLD = _c.nms_threshold
    CAMERA_ROTATE_180 = _c.rotate_180
    STREAM_SWAP_RB = _c.stream_swap_rb
    CAMERA_AWB_MODE = _c.awb_mode
    CAMERA_COLOUR_GAINS = _c.colour_gains
    CAMERA_SHARPNESS = _c.sharpness
    CAMERA_NOISE_REDUCTION = _c.noise_reduction
    STREAM_WHITE_BALANCE = _c.stream_white_balance
    STREAM_WB_STRENGTH = _c.stream_wb_strength
    BODY_MODEL_PATH = _c.body_model_path
    BODY_ENABLED = _c.body_enabled
    BODY_CONFIDENCE_THRESHOLD = _c.body_confidence_threshold
    BODY_NMS_THRESHOLD = _c.body_nms_threshold
    BODY_INPUT_SIZE = _c.body_input_size
    BODY_DETECT_STRIDE = _c.body_detect_stride
    BODY_TRACK_SERVO_ALPHA = _c.body_track_servo_alpha
    BODY_AIM_Y_RATIO = _c.body_aim_y_ratio

    if "detector" in globals() and detector is not None:
        try:
            detector.setScoreThreshold(CONFIDENCE_THRESHOLD)
            detector.setNMSThreshold(NMS_THRESHOLD)
        except Exception:
            pass


def apply_config_patches(patches: list[dict], save: bool = False) -> dict:
    with _config_lock:
        errors = patch_config(cfg, patches)
        if errors:
            return {"ok": False, "errors": errors}
        sync_config_from_cfg()
        saved_path = None
        if save:
            saved_path = str(save_config(cfg, get_config_path()))
    return {"ok": True, "config": cfg.to_dict(), "saved": saved_path}


def handle_api_trigger(data: dict) -> dict:
    global udp_emotion_override, udp_emotion_until
    global udp_conv_state, udp_conv_emotion
    global wake_request_ts, amplitude_fast, amplitude_slow, udp_speak_pulse
    global animation_arm_targets

    action = data.get("action")
    if action == "emotion":
        emotion = data.get("emotion", "happy")
        if emotion not in EMOTION_PRESETS:
            return {"ok": False, "error": f"unknown emotion: {emotion}"}
        hold = float(data.get("hold_sec", 8.0))
        udp_emotion_override = emotion
        udp_emotion_until = time.time() + hold
        return {"ok": True, "emotion": emotion, "hold_sec": hold}
    if action == "conv_state":
        state = data.get("state", "listening")
        if state not in VALID_CONV_STATES:
            return {"ok": False, "error": f"unknown state: {state}"}
        emotion = data.get("emotion", "attentive")
        udp_conv_state = state
        udp_conv_emotion = emotion
        return {"ok": True, "state": state, "emotion": emotion}
    if action == "wake":
        wake_request_ts = time.time()
        return {"ok": True, "action": "wake"}
    if action == "speaking":
        amplitude_fast = float(data.get("amplitude_fast", 0.4))
        amplitude_slow = float(data.get("amplitude_slow", 0.3))
        udp_speak_pulse = 1.0
        udp_conv_state = "speaking"
        udp_conv_emotion = "engaged"
        return {"ok": True, "action": "speaking"}
    if action == "animation":
        clip_id = str(data.get("clip_id", "")).strip()
        if not clip_id:
            return {"ok": False, "error": "missing clip_id"}
        loop = bool(data.get("loop", False))
        with animation_lock:
            ok = animation_player.play(clip_id, loop=loop)
        if not ok:
            return {"ok": False, "error": f"unknown clip_id: {clip_id}"}
        return {"ok": True, "action": "animation", "clip_id": clip_id, "loop": loop}
    if action == "animation_stop":
        with animation_lock:
            animation_player.stop()
            animation_arm_targets = dict(neutral_arm_targets)
        return {
            "ok": True,
            "action": "animation_stop",
            "stop_pose": servo_stop_pose(neutral_arm_targets),
        }
    return {"ok": False, "error": f"unknown action: {action}"}


def _resolve_emotion_name(emotion_name: str) -> str | None:
    """Map router aliases to registered presets."""
    if emotion_name == "curious":
        return "curious_intense"
    if emotion_name in EMOTION_PRESETS:
        return emotion_name
    return None


# --- BlockyEye Class (PIL Version with emotion controls) ---
class RoundEye:
    def __init__(self, x, y, scale=1.0, is_left=True):
        self.base_x, self.base_y = x, y
        self.current_pos = [float(x), float(y)]
        self.target_pos = [float(x), float(y)]

        self.vel_x = 0.0
        self.vel_y = 0.0

        self.base_w = EYE_SIZE * scale
        self.base_h = EYE_SIZE * scale

        self.current_w = self.base_w
        self.current_h = self.base_h
        self.target_w = self.base_w
        self.target_h = self.base_h

        self.vel_w = 0.0
        self.vel_h = 0.0

        self.w = self.base_w
        self.h = self.base_h

        self.current_rotation = 0.0
        self.target_rotation = 0.0
        self.rot_sensitivity = random.uniform(0.3, 0.5)
        self.rot_speed = random.uniform(0.15, 0.25)

        self.is_left = is_left
        self.blink_state = "IDLE"
        self.vy = 0
        self.blink_speed_mult = 1.0

        self.target_scale_w = 1.0
        self.target_scale_h = 1.0
        self.scale_w = 1.0
        self.scale_h = 1.0
        self.scale_w_vel = 0.0
        self.scale_h_vel = 0.0
        self.top_lid = 0.0
        self.bottom_lid = 0.0
        self.lid_angle = 0.0
        self.top_lid_vel = 0.0
        self.bottom_lid_vel = 0.0
        self.lid_angle_vel = 0.0
        self.target_top_lid = 0.0
        self.target_bottom_lid = 0.0
        self.target_lid_angle = 0.0
        self.current_emotion = "idle"
        self.happy_phase = random.uniform(0.0, math.pi * 2)
        self.happy_burst_until = 0.0
        self.surprise_shock_until = 0.0
        self.look_entry_until = 0.0
        self.release_bounce_active = False
        self.release_bounce_start = 0.0
        self.release_bounce_duration = 0.24
        self.release_bounce_frequency = 3.2
        self.release_bounce_decay = 14.0
        self.release_bounce_strength = 0.028

        # Emotion transition blending state.
        self.transition_active = False
        self.transition_start = 0.0
        self.transition_duration = 0.20
        self.transition_from_scale_w = 1.0
        self.transition_from_scale_h = 1.0
        self.transition_from_top_lid = 0.0
        self.transition_from_bottom_lid = 0.0
        self.transition_from_lid_angle = 0.0
        self.transition_to_scale_w = 1.0
        self.transition_to_scale_h = 1.0
        self.transition_to_top_lid = 0.0
        self.transition_to_bottom_lid = 0.0
        self.transition_to_lid_angle = 0.0

        self.noise_t = random.uniform(0, 100)

    def _visible_half_extents(self, w: float, h: float, rotation_deg: float = None):
        """Compute conservative half extents for the visible eye footprint.

        Includes rotation and eyelid overdraw so clamping keeps all pixels on-screen.
        """
        if rotation_deg is None:
            rotation_deg = self.current_rotation

        w = max(6.0, float(w))
        h = max(6.0, float(h))

        theta = math.radians(rotation_deg)
        cos_t = abs(math.cos(theta))
        sin_t = abs(math.sin(theta))

        # Axis-aligned bounding box of the rotated ellipse draw area.
        rot_half_w = (w * cos_t + h * sin_t) * 0.5
        rot_half_h = (w * sin_t + h * cos_t) * 0.5

        # Eyelids can extend outside the ellipse during expressive states.
        lid_extra_top = max(0.0, h * self.top_lid + 32.0)
        lid_extra_bottom = max(0.0, h * self.bottom_lid + 13.0)

        vis_half_w = max(8.0, rot_half_w + EYE_RENDER_PAD_X)
        vis_half_h = max(8.0, rot_half_h + max(lid_extra_top, lid_extra_bottom) + EYE_RENDER_PAD_Y)
        return vis_half_w, vis_half_h

    def _clamp_positions_in_bounds(self):
        # Enforce both target and current center to remain inside the panel using
        # a conservative visible footprint (size + rotation + eyelid overdraw).
        vis_half_w, vis_half_h = self._visible_half_extents(self.current_w, self.current_h)
        min_x = vis_half_w + EYE_BOUND_MARGIN
        max_x = SCREEN_WIDTH - vis_half_w - EYE_BOUND_MARGIN
        min_y = vis_half_h + EYE_BOUND_MARGIN
        max_y = SCREEN_HEIGHT - vis_half_h - EYE_BOUND_MARGIN

        if min_x > max_x:
            min_x = max_x = SCREEN_WIDTH * 0.5
        if min_y > max_y:
            min_y = max_y = SCREEN_HEIGHT * 0.5

        self.target_pos[0] = clamp(self.target_pos[0], min_x, max_x)
        self.target_pos[1] = clamp(self.target_pos[1], min_y, max_y)
        self.current_pos[0] = clamp(self.current_pos[0], min_x, max_x)
        self.current_pos[1] = clamp(self.current_pos[1], min_y, max_y)

    def _motion_clamp_bounds(self):
        """Return a slightly looser clamp used only for motion targets."""
        vis_half_w, vis_half_h = self._visible_half_extents(self.current_w, self.current_h)
        motion_half_w = max(8.0, vis_half_w * EYE_MOTION_CLAMP_SCALE)
        motion_half_h = max(8.0, vis_half_h * EYE_MOTION_CLAMP_SCALE)

        min_x = motion_half_w + EYE_BOUND_MARGIN
        max_x = SCREEN_WIDTH - motion_half_w - EYE_BOUND_MARGIN
        min_y = motion_half_h + EYE_BOUND_MARGIN
        max_y = SCREEN_HEIGHT - motion_half_h - EYE_BOUND_MARGIN

        if min_x > max_x:
            min_x = max_x = SCREEN_WIDTH * 0.5
        if min_y > max_y:
            min_y = max_y = SCREEN_HEIGHT * 0.5

        return min_x, max_x, min_y, max_y

    def start_blink(self, speed_mult=None):
        if self.blink_state == "IDLE":
            self.blink_state = "DROPPING"
            if speed_mult is not None:
                self.blink_speed_mult = speed_mult
            else:
                self.blink_speed_mult = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            self.vy = 48 * self.blink_speed_mult

    def _ease_in_out(self, alpha: float) -> float:
        alpha = max(0.0, min(1.0, alpha))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def _transition_duration_for(self, previous_emotion: str, next_emotion: str) -> float:
        no_face_blends = {"uncertain", "curious", "warm", "attentive", "idle", "cheerful", "content"}
        if previous_emotion == next_emotion:
            return 0.12
        if next_emotion in ("excited", "surprised"):
            return 0.13
        if previous_emotion.startswith("looking_") and next_emotion.startswith("looking_"):
            return 0.14
        if previous_emotion in no_face_blends or next_emotion in no_face_blends:
            return 0.42
        if next_emotion in ("sleepy", "bored"):
            return 0.30
        return 0.22

    def set_emotion(self, emotion_name: str, intensity: float = 1.0):
        resolved = _resolve_emotion_name(emotion_name)
        if resolved is None:
            return
        emotion_name = resolved

        now = time.time()
        previous_emotion = self.current_emotion
        changing_emotion = emotion_name != self.current_emotion

        if emotion_name == "happy" and self.current_emotion != "happy":
            self.happy_burst_until = now + 0.35
        if emotion_name == "surprised" and self.current_emotion != "surprised":
            self.surprise_shock_until = now + 0.18
        if previous_emotion == "sleepy" and emotion_name == "surprised":
            self.release_bounce_active = True
            self.release_bounce_start = now
        else:
            self.release_bounce_active = False
        if emotion_name.startswith("looking_") and self.current_emotion != emotion_name:
            self.look_entry_until = now + 0.16
            global jerk_until, jerk_direction, face_tracking_active
            if not face_tracking_active:
                jerk_direction = -1.0 if "left" in emotion_name else 1.0
                jerk_until = now + JERK_DURATION

        self.current_emotion = emotion_name
        preset = EMOTION_PRESETS[emotion_name]
        idle = EMOTION_PRESETS["idle"]

        intensity = max(0.0, min(1.0, intensity))
        scale_w = idle["scale_w"] + (preset["scale_w"] - idle["scale_w"]) * intensity
        scale_h = idle["scale_h"] + (preset["scale_h"] - idle["scale_h"]) * intensity
        top_lid = idle["top_lid"] + (preset["top_lid"] - idle["top_lid"]) * intensity
        bottom_lid = idle["bottom_lid"] + (preset["bottom_lid"] - idle["bottom_lid"]) * intensity
        lid_angle = idle["lid_angle"] + (preset["lid_angle"] - idle["lid_angle"]) * intensity

        if preset.get("mirror_angle", True) and not self.is_left and abs(lid_angle) > 0:
            lid_angle = -lid_angle

        # Blend target shape over a short transition window to avoid hard snaps.
        self.transition_from_scale_w = self.target_scale_w
        self.transition_from_scale_h = self.target_scale_h
        self.transition_from_top_lid = self.target_top_lid
        self.transition_from_bottom_lid = self.target_bottom_lid
        self.transition_from_lid_angle = self.target_lid_angle
        self.transition_to_scale_w = scale_w
        self.transition_to_scale_h = scale_h
        self.transition_to_top_lid = top_lid
        self.transition_to_bottom_lid = bottom_lid
        self.transition_to_lid_angle = lid_angle
        self.transition_start = now
        self.transition_duration = self._transition_duration_for(previous_emotion, emotion_name)
        self.transition_active = True

        if changing_emotion:
            # Reduce spring carry-over so new emotions don't produce one-frame artifacts.
            self.scale_w_vel *= 0.35
            self.scale_h_vel *= 0.35
            self.top_lid_vel *= 0.35
            self.bottom_lid_vel *= 0.35
            self.lid_angle_vel *= 0.35
            self.vel_w *= 0.35
            self.vel_h *= 0.35

    def update(self):
        now = time.time()

        if self.transition_active:
            if self.transition_duration <= 0.0:
                blend = 1.0
            else:
                blend = (now - self.transition_start) / self.transition_duration
            eased = self._ease_in_out(blend)

            self.target_scale_w = self.transition_from_scale_w + (self.transition_to_scale_w - self.transition_from_scale_w) * eased
            self.target_scale_h = self.transition_from_scale_h + (self.transition_to_scale_h - self.transition_from_scale_h) * eased
            self.target_top_lid = self.transition_from_top_lid + (self.transition_to_top_lid - self.transition_from_top_lid) * eased
            self.target_bottom_lid = self.transition_from_bottom_lid + (self.transition_to_bottom_lid - self.transition_from_bottom_lid) * eased
            self.target_lid_angle = self.transition_from_lid_angle + (self.transition_to_lid_angle - self.transition_from_lid_angle) * eased

            if blend >= 1.0:
                self.transition_active = False

        if self.blink_state == "IDLE":
            t = now + self.noise_t
            noise_x = (math.sin(t * 1.3) * 0.2 + math.sin(t * 0.7) * 0.1)
            noise_y = (math.cos(t * 1.1) * 0.2 + math.cos(t * 0.9) * 0.1)

            target_x_phys = self.target_pos[0] + noise_x
            target_y_phys = self.target_pos[1] + noise_y

            burst_active = now < self.happy_burst_until
            if burst_active:
                target_y_phys -= 8.0

            if self.current_emotion == "happy":
                ht = now * 6.0 + self.happy_phase
                target_y_phys -= 2.2 + math.sin(ht) * 1.8
                target_x_phys += math.sin(ht * 1.7) * 1.2
            elif (
                not face_tracking_active
                and self.current_emotion.startswith("looking_")
                and "left" in self.current_emotion
            ):
                target_x_phys -= LOOK_SIDE_OFFSET
            elif (
                not face_tracking_active
                and self.current_emotion.startswith("looking_")
                and "right" in self.current_emotion
            ):
                target_x_phys += LOOK_SIDE_OFFSET

            look_entry_active = (
                not face_tracking_active
                and self.current_emotion.startswith("looking_")
                and now < self.look_entry_until
            )
            if look_entry_active:
                side_sign = -1.0 if "left" in self.current_emotion else 1.0
                target_x_phys = self.base_x + side_sign * (LOOK_SIDE_OFFSET * 0.9)
                target_y_phys = self.base_y

            dx = target_x_phys - self.current_pos[0]
            dy = target_y_phys - self.current_pos[1]

            speed_x = 0.20
            speed_y = 0.22
            if dy < -1.0:
                speed_y = 0.14
            elif dy > 1.0:
                speed_y = 0.38
            if look_entry_active:
                speed_x = 0.42
                speed_y = 0.18

            self.current_pos[0] += dx * speed_x
            self.current_pos[1] += dy * speed_y

            self.vel_x = dx * speed_x
            self.vel_y = dy * speed_y

            rel_x = self.current_pos[0] - self.base_x
            rel_y = self.current_pos[1] - self.base_y
            look_rot = (rel_x * 0.5 + rel_y * 0.8) * self.rot_sensitivity
            if self.current_emotion == "happy":
                look_rot += math.sin(now * 8.0 + self.happy_phase) * 1.2
            final_target_rot = look_rot + self.target_rotation
            self.current_rotation += (final_target_rot - self.current_rotation) * self.rot_speed

            t = now
            breath_w = (math.sin(t * 1.5 + self.base_x) * 1.5 + math.sin(t * 0.5) * 1.0)
            breath_h = (math.cos(t * 1.8 + self.base_y) * 1.5 + math.cos(t * 0.6) * 1.0)

            move_stretch_x = (dx * speed_x) * 2.5
            move_stretch_y = (dy * speed_y) * 2.5
            if self.current_emotion == "surprised":
                move_stretch_x = 0.0
                move_stretch_y = 0.0
            elif self.current_emotion.startswith("looking_"):
                if look_entry_active:
                    move_stretch_x = 0.0
                    move_stretch_y = 0.0
                else:
                    move_stretch_x *= 0.45
                    move_stretch_y *= 0.45

            k = 0.22
            d = 0.84
            if self.current_emotion == "surprised":
                if now < self.surprise_shock_until:
                    k = 0.46
                    d = 0.44
                else:
                    k = 0.20
                    d = 0.72
            self.scale_w_vel = (self.scale_w_vel + (self.target_scale_w - self.scale_w) * k) * d
            self.scale_h_vel = (self.scale_h_vel + (self.target_scale_h - self.scale_h) * k) * d
            self.scale_w += self.scale_w_vel
            self.scale_h += self.scale_h_vel
            self.scale_w = max(MIN_EYE_SCALE, min(MAX_EYE_SCALE, self.scale_w))
            self.scale_h = max(MIN_EYE_SCALE, min(MAX_EYE_SCALE, self.scale_h))

            self.top_lid_vel = (self.top_lid_vel + (self.target_top_lid - self.top_lid) * k) * d
            self.bottom_lid_vel = (self.bottom_lid_vel + (self.target_bottom_lid - self.bottom_lid) * k) * d
            self.lid_angle_vel = (self.lid_angle_vel + (self.target_lid_angle - self.lid_angle) * k) * d

            self.top_lid += self.top_lid_vel
            self.bottom_lid += self.bottom_lid_vel
            self.lid_angle += self.lid_angle_vel

            if self.release_bounce_active:
                elapsed = now - self.release_bounce_start
                if elapsed <= self.release_bounce_duration:
                    bounce = math.exp(-self.release_bounce_decay * elapsed) * math.sin(math.tau * self.release_bounce_frequency * elapsed + math.pi / 2)
                    self.top_lid = max(0.0, min(MAX_TOP_LID, self.top_lid - bounce * self.release_bounce_strength))
                    self.current_pos[1] -= bounce * 0.35
                else:
                    self.release_bounce_active = False
            self.top_lid = max(0.0, min(MAX_TOP_LID, self.top_lid))
            self.bottom_lid = max(0.0, min(MAX_BOTTOM_LID, self.bottom_lid))
            self.lid_angle = max(-22.0, min(22.0, self.lid_angle))

            self.target_w = (self.base_w * self.scale_w) + breath_w + (move_stretch_x * 0.5)
            self.target_h = (self.base_h * self.scale_h) + breath_h - (move_stretch_y * 0.2)

        elif self.blink_state == "DROPPING":
            self.vy += 12 * self.blink_speed_mult
            self.current_pos[1] += self.vy
            self.current_w = self.base_w - 12
            self.current_h = self.base_h + 18
            self.target_w = self.current_w
            self.target_h = self.current_h

            if self.current_pos[1] + self.current_h // 2 >= FLOOR_Y:
                self.current_pos[1] = FLOOR_Y - self.current_h // 2
                self.blink_state = "SQUASHING"
                self.velocity = [0.0, 0.0]

        elif self.blink_state == "SQUASHING":
            squeeze_speed = 58 * self.blink_speed_mult
            spread_speed = 38 * self.blink_speed_mult
            self.current_h -= squeeze_speed
            self.current_w += spread_speed
            self.current_pos[1] = FLOOR_Y - self.current_h // 2

            if self.current_h <= 25:
                self.current_h = 25
                self.blink_state = "JUMPING"

        elif self.blink_state == "JUMPING":
            recovery_speed = max(0.15, min(0.95, 0.82 * self.blink_speed_mult))
            self.current_h += (self.base_h - self.current_h) * recovery_speed
            self.current_w += (self.base_w - self.current_w) * recovery_speed

            self.vel_x = (self.vel_x + (self.target_pos[0] - self.current_pos[0]) * 0.12) * 0.82
            self.current_pos[0] += self.vel_x

            target_y = self.target_pos[1]
            self.current_pos[1] += (target_y - self.current_pos[1]) * 0.88

            if abs(self.current_h - self.base_h) < 5 and abs(self.current_pos[1] - target_y) < 5:
                self.current_h = self.base_h
                self.current_w = self.base_w
                self.blink_state = "IDLE"
                self.vy = 0
                self.vel_x = 0
                self.vel_y = 0

        if self.blink_state == "IDLE":
            k = 0.08
            d = 0.90
            force_w = (self.target_w - self.current_w) * k
            self.vel_w = (self.vel_w + force_w) * d
            self.current_w += self.vel_w

            force_h = (self.target_h - self.current_h) * k
            self.vel_h = (self.vel_h + force_h) * d
            self.current_h += self.vel_h
        else:
            self.vel_w = 0
            self.vel_h = 0

        self.current_w = max(6.0, min(float(SCREEN_WIDTH - 6), self.current_w))
        self.current_h = max(6.0, min(float(SCREEN_HEIGHT - 6), self.current_h))
        self._clamp_positions_in_bounds()
        self.w = self.current_w
        self.h = self.current_h

    def draw_solid_eye(self, draw, x, y, w, h, color, pupil_offset=(0, 0)):
        draw.ellipse([x, y, x + w, y + h], fill=color)

    def draw_eyelids(self, eye_img, rect):
        x0, y0, x1, y1 = rect
        w = int(x1 - x0)
        h = int(y1 - y0)
        lid_color = BG_COLOR

        if self.top_lid > 0.01:
            lid_h = int(h * self.top_lid)
            lid_src = Image.new("RGBA", (int(w * 2.1), int(lid_h + 64)), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y0 - 32)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

        if self.bottom_lid > 0.01:
            lid_h = int(h * self.bottom_lid)
            lid_src = Image.new("RGBA", (int(w * 2.1), int(lid_h + 28)), (*lid_color, 255))
            if abs(self.lid_angle) > 0.1:
                lid_src = lid_src.rotate(self.lid_angle, resample=Image.BICUBIC, expand=True)
            lid_x = int(x0 + w / 2 - lid_src.width / 2)
            lid_y = int(y1 + 13 - lid_src.height)
            eye_img.alpha_composite(lid_src, (lid_x, lid_y))

    def draw(self, bg_image):
        draw_w = max(6, min(int(self.w), SCREEN_WIDTH - 4))
        draw_h = max(6, min(int(self.h), SCREEN_HEIGHT - 4))

        # Render-time safety clamp: keep the visible eye footprint inside the panel.
        # Uses strict geometric extents so no rotated/lidded pixels leave the display.
        vis_half_w, vis_half_h = self._visible_half_extents(draw_w, draw_h, self.current_rotation)
        min_cx = vis_half_w + EYE_BOUND_MARGIN
        max_cx = SCREEN_WIDTH - vis_half_w - EYE_BOUND_MARGIN
        min_cy = vis_half_h + EYE_BOUND_MARGIN
        max_cy = SCREEN_HEIGHT - vis_half_h - EYE_BOUND_MARGIN
        if min_cx > max_cx:
            min_cx = max_cx = SCREEN_WIDTH * 0.5
        if min_cy > max_cy:
            min_cy = max_cy = SCREEN_HEIGHT * 0.5
        render_cx = clamp(self.current_pos[0], min_cx, max_cx)
        render_cy = clamp(self.current_pos[1], min_cy, max_cy)

        eye_img_size = int(max(self.base_w, self.base_h) * 2.6)
        eye_img = Image.new("RGBA", (eye_img_size, eye_img_size), (0, 0, 0, 0))
        eye_draw = ImageDraw.Draw(eye_img)

        off_x = max(-1, min(1, (self.current_pos[0] - self.base_x) / 30.0))
        off_y = max(-1, min(1, (self.current_pos[1] - self.base_y) / 20.0))

        cx, cy = eye_img_size / 2, eye_img_size / 2
        x0 = cx - draw_w / 2
        y0 = cy - draw_h / 2
        x1 = cx + draw_w / 2
        y1 = cy + draw_h / 2

        self.draw_solid_eye(eye_draw, x0, y0, draw_w, draw_h, EYE_COLOR, (off_x, off_y))
        self.draw_eyelids(eye_img, (x0, y0, x1, y1))

        rotated = eye_img.rotate(self.current_rotation, resample=Image.BICUBIC, expand=True)

        # Center using rotated size (not source size) so expand=True doesn't shift the eye.
        paste_x = int(render_cx - rotated.width / 2)
        paste_y = int(render_cy - rotated.height / 2)
        bg_image.alpha_composite(rotated, (paste_x, paste_y))


# --- MJPEG Streaming Server ---
latest_frame = None
frame_lock = threading.Lock()
stream_server = None


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MJPEGHandler(BaseHTTPRequestHandler):
    def _json_safe(self, value):
        """Coerce numpy scalars and other non-JSON types for API responses."""
        if isinstance(value, dict):
            return {k: self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        type_name = type(value).__name__
        if type_name in ("bool_", "bool8"):
            return bool(value)
        if type_name.startswith("int"):
            return int(value)
        if type_name.startswith("float"):
            return float(value)
        return value

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(self._json_safe(payload)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, filename: str, content_type: str):
        path = _STATIC_DIR / filename
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _stream_mjpeg(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                with frame_lock:
                    frame = None if latest_frame is None else latest_frame.copy()

                if frame is None:
                    time.sleep(0.05)
                    continue

                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(
                    ".jpg",
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(STREAM_JPEG_QUALITY)],
                )
                if not ok:
                    time.sleep(0.05)
                    continue
                jpg = buf.tobytes()

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / max(1, STREAM_FPS))
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/", "/debug"):
            self._serve_static("debug.html", "text/html; charset=utf-8")
            return

        if path == "/api/state":
            self._send_json(get_runtime_state())
            return

        if path == "/api/config":
            self._send_json(cfg.to_dict())
            return

        if path == "/api/tuning":
            self._send_json({"fields": get_tuning_schema(cfg)})
            return

        if path == "/api/tof":
            self._send_json(get_tof_api_payload())
            return

        if path in ("/tof", "/tof_viz.html"):
            self._serve_static("tof_viz.html", "text/html; charset=utf-8")
            return

        if path == "/stream":
            self._stream_mjpeg()
            return

        self.send_error(404)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            data = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"ok": False, "error": "invalid json"}, 400)
            return

        try:
            if path == "/api/config":
                patches = data.get("patches", [])
                save = bool(data.get("save", False))
                result = apply_config_patches(patches, save=save)
                status = 200 if result.get("ok") else 400
                self._send_json(result, status)
                return

            if path == "/api/trigger":
                result = handle_api_trigger(data)
                status = 200 if result.get("ok") else 400
                self._send_json(result, status)
                return

            self.send_error(404)
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, 500)

    def log_message(self, format, *args):
        return


def start_stream_server():
    global stream_server
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), MJPEGHandler)
    thread = threading.Thread(target=stream_server.serve_forever, daemon=True)
    thread.start()
    print(f"Debug dashboard: http://{STREAM_HOST}:{STREAM_PORT}/debug")
    print(f"MJPEG stream:     http://{STREAM_HOST}:{STREAM_PORT}/stream")


# --- Display Setup ---
print("Initializing Displays (Dual SPI)...")
disp_l = None
disp_r = None

# SPI 0 (Left Screen)
try:
    spi0 = board.SPI()
    disp_l = st7735.ST7735R(
        spi0, 
        rotation=0, 
        baudrate=24000000, 
        bgr=True,
        cs=digitalio.DigitalInOut(board.CE1),   
        dc=digitalio.DigitalInOut(board.D24),   
        rst=digitalio.DigitalInOut(board.D25)
    )
except Exception as e:
    print(f"Error init Left Display (SPI0): {e}")

# SPI 1 (Right Screen)
try:
    spi1 = busio.SPI(clock=board.D21, MOSI=board.D20, MISO=board.D19)
    disp_r = st7735.ST7735R(
        spi1, 
        rotation=0, 
        baudrate=24000000, 
        bgr=True,
        cs=digitalio.DigitalInOut(board.D18),   
        dc=digitalio.DigitalInOut(board.D23),   
        rst=digitalio.DigitalInOut(board.D27)
    )
except Exception as e:
    print(f"Error init Right Display (SPI1): {e}")


# Connect ESP32 before Picamera2 — shared USB bus on Pi can block boot/handshake.
_boot_servo_driver = None
if ENABLE_SERVO:
    print("Connecting ESP32 (before camera USB)...")
    check_servo_channel_config(cfg.servo.pan_ch, cfg.servo.tilt_ch, backend=cfg.servo.backend)
    _boot_servo_driver = create_servo_driver(cfg, max_attempts=2, retry_delay_sec=1.5)


def _apply_camera_colour_controls(picam2: Picamera2) -> None:
    """Set AWB / manual colour gains to reduce orange indoor cast."""
    apply_camera_controls(
        picam2,
        awb_mode=CAMERA_AWB_MODE,
        colour_gains=CAMERA_COLOUR_GAINS,
        sharpness=CAMERA_SHARPNESS,
        noise_reduction=CAMERA_NOISE_REDUCTION,
    )


# --- Camera & Face Detector Setup ---
print("Initializing Picamera2...")
picam2 = None
try:
    picam2 = Picamera2()

    if CAMERA_WIDE_FOV:
        configure_wide_fov_camera(
            picam2,
            CAMERA_MAIN_RES,
            raw_sensor_res=CAMERA_RAW_SENSOR_RES,
        )
        pipeline = f"wide-FOV video/RGB888 raw {CAMERA_RAW_SENSOR_RES[0]}x{CAMERA_RAW_SENSOR_RES[1]}"
    else:
        configure_picamera(picam2, CAMERA_MAIN_RES, use_preview_pipeline=CAMERA_USE_PREVIEW)
        pipeline = "preview/sRGB" if CAMERA_USE_PREVIEW else "video/RGB888"
    picam2.start()
    _apply_camera_colour_controls(picam2)
    if CAMERA_WIDE_FOV:
        print(
            f"Camera started ({pipeline}): "
            f"main {CAMERA_MAIN_RES[0]}x{CAMERA_MAIN_RES[1]}, "
            f"detect {CAMERA_RES[0]}x{CAMERA_RES[1]}"
        )
        print("  Tip: full-sensor mode needs cma=256 in /boot/firmware/config.txt")
    else:
        print(
            f"Camera started ({pipeline}): "
            f"main {CAMERA_MAIN_RES[0]}x{CAMERA_MAIN_RES[1]}, "
            f"detect {CAMERA_RES[0]}x{CAMERA_RES[1]}"
        )
    assert_detection_aspect_matches(CAMERA_MAIN_RES, CAMERA_RES, stream_res=STREAM_RES)
except Exception as e:
    print(f"Error starting Picamera2: {e}")
    if CAMERA_WIDE_FOV:
        print("  If capture failed, add cma=256 to /boot/firmware/config.txt and reboot.")
    sys.exit(1)

print("Initializing YuNet Face Detector...")
try:
    if not Path(FACE_MODEL_PATH).exists():
        print(f"Error: Face model not found at {FACE_MODEL_PATH}")
        sys.exit(1)
        
    detector = cv2.FaceDetectorYN.create(
        model=FACE_MODEL_PATH,
        config="",
        input_size=CAMERA_RES,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )
    print("YuNet initialized.")
except Exception as e:
    print(f"Error initializing detector: {e}")
    sys.exit(1)

person_detector = None


def _boot_color_probe():
    """Non-blocking YuNet BGR layout check (runs after face tracking starts)."""
    global _detection_bgr_mode
    try:
        _boot_capture = picam2.capture_array()
        if CAMERA_WIDE_FOV:
            _boot_detect = cv2.resize(_boot_capture, CAMERA_RES, interpolation=cv2.INTER_AREA)
            if CAMERA_ROTATE_180:
                _boot_detect = cv2.rotate(_boot_detect, cv2.ROTATE_180)
            _boot_rgb = (
                cv2.cvtColor(_boot_detect, cv2.COLOR_BGR2RGB)
                if STREAM_SWAP_RB
                else _boot_detect
            )
        else:
            _boot_rgb = frame_to_rgb(
                _boot_capture,
                legacy_swap_rb=STREAM_SWAP_RB and not CAMERA_USE_PREVIEW,
            )
            _boot_detect = cv2.resize(_boot_rgb, CAMERA_RES, interpolation=cv2.INTER_AREA)
        mode, face_count = probe_yunet_bgr_mode(
            detector,
            _boot_detect,
            input_size=CAMERA_RES,
            rotate_180=False if CAMERA_WIDE_FOV else CAMERA_ROTATE_180,
        )
        _detection_bgr_mode = mode
        _color_stats = verify_color_pipeline(_boot_rgb)
        log_color_pipeline_verification(
            _color_stats,
            detection_mode=mode,
            face_probe_count=face_count,
        )
    except Exception as e:
        print(f"Warning: face color probe skipped: {e}")


def _init_person_detector_background():
    """YOLO body fallback — optional, loaded after face tracking is live."""
    global person_detector
    if not BODY_ENABLED:
        return
    print("Initializing YOLOv8 person detector (background)...")
    try:
        if not Path(BODY_MODEL_PATH).exists():
            print(f"Warning: Body model not found at {BODY_MODEL_PATH}")
            print("  Run: python tools/download_models.py yolov8n.onnx")
            return
        person_detector = PersonDetector(
            BODY_MODEL_PATH,
            confidence_threshold=BODY_CONFIDENCE_THRESHOLD,
            nms_threshold=BODY_NMS_THRESHOLD,
            input_size=BODY_INPUT_SIZE,
        )
        print("YOLOv8 person detector initialized.")
    except Exception as e:
        print(f"Warning: Person detector disabled: {e}")
        person_detector = None


def _load_animations_background():
    loaded = _load_default_animation_clips()
    if loaded:
        print(f"Animation clips ready ({loaded} loaded)")


# --- Eye Objects ---
center_x = SCREEN_WIDTH / 2
center_y = SCREEN_HEIGHT / 2

left_eye = RoundEye(center_x, center_y, scale=1.0, is_left=True)
right_eye = RoundEye(center_x, center_y, scale=1.0, is_left=False)
# Keep both eyes using identical dynamics to avoid drift during blink phases.
right_eye.noise_t = left_eye.noise_t
right_eye.rot_sensitivity = left_eye.rot_sensitivity
right_eye.rot_speed = left_eye.rot_speed
right_eye.happy_phase = left_eye.happy_phase
left_eye.set_emotion("idle", EMOTION_INTENSITY["idle"])
right_eye.set_emotion("idle", EMOTION_INTENSITY["idle"])

surroundings_controller = SurroundingsEmotionController(
    cfg=SurroundingsEmotionRuntimeConfig(
        no_face_grace_sec=_se.no_face_grace_sec,
        no_person_hold_min_sec=_se.no_person_hold_min_sec,
        no_person_hold_max_sec=_se.no_person_hold_max_sec,
        person_hold_min_sec=_se.person_hold_min_sec,
        person_hold_max_sec=_se.person_hold_max_sec,
        direction_trigger_norm_x=_se.direction_trigger_norm_x,
        direction_hold_min_sec=_se.direction_hold_min_sec,
        direction_hold_max_sec=_se.direction_hold_max_sec,
        direction_cooldown_sec=_se.direction_cooldown_sec,
        close_face_enter_ratio=_se.close_face_enter_ratio,
        far_face_area_ratio=_se.far_face_area_ratio,
        near_exit_ratio=_se.near_exit_ratio,
        far_exit_ratio=_se.far_exit_ratio,
        emotion_history_len=_se.emotion_history_len,
    )
)
prev_surroundings_x = 0.0
prev_surroundings_y = 0.0
prev_surroundings_rot = 0.0

# Animation Loop Vars
running = True
servo_running = False
next_blink_time = time.time() + random.uniform(3, 6)
last_blink_time = time.time()
smoothed_x_off = 0.0
smoothed_y_off = 0.0
smoothed_rotation = 0.0
smoothed_head_eye_x = 0.0
smoothed_head_eye_y = 0.0
solo_mood = "neutral"
solo_mood_until = 0.0
current_emotion = "idle"  # Track current emotion to avoid redundant updates
emotion_last_switch_ts = time.time()
emotion_last_normal_switch_ts = time.time()
emotion_force_until = 0.0
router_face_present_prev = False
router_face_close = False
router_multi_face_prev = False
router_candidate_emotion = current_emotion
router_candidate_since = time.time()
side_dir_state = 0
side_dir_last_switch_ts = time.time()
multi_face_candidate = False
multi_face_candidate_since = time.time()
multi_face_stable = False
social_mode = "neutral"
social_mode_until = time.time()
last_happy_ts = 0.0
no_face_since_ts = time.time()
no_face_scan_checks = 0
no_face_blend_emotion = "idle"
no_face_blend_until = 0.0
no_face_blend_queue = []

target_lock = threading.Lock()
target_x_off = 0.0
target_y_off = 0.0
target_rotation = 0.0
target_squint = 0.0
target_face_present = False
target_body_present = False
target_face_area_ratio = 0.0
target_face_count = 0
squint_until = 0.0

# Servo shared state
servo_state_lock = threading.Lock()
servo_thread = None
servo_driver = None
servo_target_pan = (PAN_MIN + PAN_MAX) * 0.5
servo_target_tilt = (TILT_MIN + TILT_MAX) * 0.5
servo_current_pan = servo_target_pan
servo_current_tilt = servo_target_tilt
servo_pan_vel = 0.0
servo_tilt_vel = 0.0
last_base_fov_nudge_ts = 0.0
last_face_base_nudge_ts = 0.0
base_error_until = 0.0
next_face_base_alive_ts = time.time() + random.uniform(8.0, 16.0)
last_wander_base_follow_ts = 0.0
last_wander_base_follow_eval_ts = 0.0
last_face_seen_ts = time.time()
_pan_center_init = (PAN_MIN + PAN_MAX) * 0.5
_tilt_center_init = (TILT_MIN + TILT_MAX) * 0.5
last_face_pan = _pan_center_init
last_face_tilt = _tilt_center_init
last_face_norm_x = 0.0
last_face_norm_y = 0.0
# No-face priority FSM: tracking > wandering > sad_return > settled > chat_ready
no_face_mode = "tracking"
wander_until = 0.0
sad_return_until = 0.0
sad_return_start = 0.0
next_wander_peek_ts = 0.0
chat_ready_until = 0.0
wake_tilt_jerk_until = 0.0
wake_request_ts = 0.0
session_active = False
PRESENCE_ARRIVAL_COOLDOWN_SEC = 45.0
presence_arrival_last_ts = 0.0
prev_tof_center_present = False
_voice_notify_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_VOICE_NOTIFY_ADDR = ("127.0.0.1", 9001)
wander_search_phase = 0.0  # legacy
wander_search_last_ts = time.time()
organic_wander_search = OrganicWanderSearch()
search_tilt_from_eye_level_deg = 0.0
wander_pan_speed_scale = 1.0
ever_had_face = False
pending_face_stable_since = None
router_face_stable_prev = False
_last_body_det = None
_last_body_det_ts = 0.0
vision_frame_idx = 0
settled_solo_emotion = "sleepy"
settled_emotion_until = 0.0
speak_social_mode = "engaged"
speak_social_until = 0.0
emotion_trace = []
_last_emotion_debug_ts = 0.0
_last_amplitude_debug_ts = 0.0

# Jerk animation state
jerk_until = 0.0      # Timestamp when jerk ends
jerk_direction = 0.0  # -1 for left, +1 for right, 0 for no jerk
jerk_cooldown_until = 0.0
face_tracking_active = False

# Gaze aversion manager state
gaze_state = "ENGAGED"
gaze_event_active = False
gaze_event_start = 0.0
gaze_event_to_sec = 0.0
gaze_event_hold_sec = 0.0
gaze_event_back_sec = 0.0
gaze_event_pause_sec = 0.0
gaze_event_target_x = 0.0
gaze_event_target_y = 0.0
gaze_override_x = 0.0
gaze_override_y = 0.0
scan_emotion_override = None
no_face_scan_completed_pulse = False
gaze_reengage_until = 0.0
next_talk_saccade_ts = 0.0
gaze_next_allowed_ts = time.time()
gaze_next_scan_ts = time.time() + random.uniform(GAZE_AMBIENT_SCAN_MIN_SEC, GAZE_AMBIENT_SCAN_MAX_SEC)
gaze_next_release_ts = time.time() + random.uniform(GAZE_SOCIAL_RELEASE_MIN_SEC, GAZE_SOCIAL_RELEASE_MAX_SEC)
face_present_since_ts = None

# Servo aversion offsets (added on top of face tracking servo targets)
servo_aversion_pan_offset = 0.0
servo_aversion_tilt_offset = 0.0

animation_lock = threading.Lock()
animation_player = AnimationPlayer()
neutral_arm_targets: dict[str, float] = dict(DEFAULT_ARM_NEUTRALS)
animation_arm_targets: dict[str, float] = dict(neutral_arm_targets)
arm_current_smoothed: dict[str, float] = dict(neutral_arm_targets)
_last_servo_frame_ts = 0.0
_last_sent_pan: float | None = None
_last_sent_tilt: float | None = None
SERVO_FRAME_INTERVAL = 1.0 / 30.0
ARM_ANIM_BLEND = 0.45
BOTANGO_COMMANDS_FILE = "AnimationCommands.json"

tof_lock = threading.Lock()
tof_snapshot = TofSnapshot.empty()
tof_poll_ok = False
tof_presence = TofPresence(False, False, False, False, 0)
tof_tracker = TofPresenceTracker(
    present_max_mm=TOF_PRESENT_MAX_MM,
    absent_min_mm=TOF_ABSENT_MIN_MM,
    debounce_present_sec=TOF_DEBOUNCE_PRESENT_SEC,
    debounce_absent_sec=TOF_DEBOUNCE_ABSENT_SEC,
)


def _make_tof_approach_controller() -> TofApproachController:
    return TofApproachController(
        enabled=TOF_APPROACH_ENABLED and TOF_ENABLED,
        head_turn_deg=TOF_APPROACH_HEAD_TURN_DEG,
        present_max_mm=TOF_PRESENT_MAX_MM,
        pan_step_deg=TOF_APPROACH_PAN_STEP_DEG,
        boot_pan_step_deg=TOF_APPROACH_BOOT_PAN_STEP_DEG,
        arrival_deg=TOF_APPROACH_ARRIVAL_DEG,
        use_base=TOF_APPROACH_USE_BASE,
        base_nudge_deg=TOF_APPROACH_BASE_NUDGE_DEG,
        max_base_nudges_per_event=TOF_APPROACH_MAX_BASE_NUDGES,
        confirm_delay_sec=TOF_APPROACH_CONFIRM_DELAY_SEC,
        lockout_sec=TOF_APPROACH_LOCKOUT_SEC,
        left_right_only=TOF_APPROACH_LEFT_RIGHT_ONLY,
        boot_orient=TOF_APPROACH_BOOT_ORIENT,
        startup_grace_sec=TOF_APPROACH_STARTUP_GRACE_SEC,
        pan_min=PAN_MIN,
        pan_max=PAN_MAX,
    )


tof_approach_controller = _make_tof_approach_controller()


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def apply_deadzone_norm(value: float, deadzone: float) -> float:
    """Zero small normalized offsets so tracking holds steady near center."""
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value >= 0.0 else -1.0
    return sign * (abs(value) - deadzone) / max(1e-6, 1.0 - deadzone)


def weighted_pick(weighted_items):
    total = 0.0
    for _, w in weighted_items:
        total += max(0.0, float(w))
    if total <= 0.0:
        return weighted_items[0][0]
    r = random.uniform(0.0, total)
    acc = 0.0
    for value, w in weighted_items:
        acc += max(0.0, float(w))
        if r <= acc:
            return value
    return weighted_items[-1][0]


def _smoothstep01(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def is_upbeat_session() -> bool:
    return (
        session_active
        or no_face_mode == "chat_ready"
        or udp_conv_state in AWAKE_CONV_ACTIVE
    )


def voice_emotion_active(now: float) -> bool:
    """Voice agent UDP layers take priority over surroundings-driven emotions."""
    if session_active:
        return True
    if udp_emotion_override and now < udp_emotion_until:
        return True
    if udp_conv_state in (
        "listening",
        "speaking",
        "thinking",
        "nodding",
        "remembering",
        "concentrating",
    ):
        return True
    if udp_conv_state == "waiting" and udp_conv_emotion == "awkward":
        return True
    return False


def trace_emotion(step: str, emotion: str):
    global emotion_trace
    emotion_trace.append(f"{step}->{emotion}")


def pick_upbeat_solo_emotion() -> str:
    return weighted_pick([
        ("cheerful", 0.28),
        ("happy", 0.22),
        ("content", 0.18),
        ("warm", 0.14),
        ("engaged", 0.12),
        ("excited", 0.06),
    ])


def pick_speak_emotion() -> str:
    return weighted_pick([
        ("engaged", 0.28),
        ("cheerful", 0.22),
        ("excited", 0.16),
        ("warm", 0.12),
        ("amused", 0.10),
        ("happy", 0.07),
        ("content", 0.05),
    ])


def trigger_settled_wake(now: float):
    """Perk up from settled idle when a live chat session starts (no face in frame)."""
    global no_face_mode, chat_ready_until, wake_tilt_jerk_until, emotion_force_until
    global jerk_cooldown_until, current_emotion

    no_face_mode = "chat_ready"
    chat_ready_until = now + CHAT_READY_MIN_SEC
    wake_tilt_jerk_until = now + WAKE_TILT_JERK_SEC
    emotion_force_until = now + WAKE_SURPRISE_SEC
    jerk_cooldown_until = now + JERK_COOLDOWN_SEC
    clear_gaze_aversion()
    surprise_intensity = EMOTION_INTENSITY.get("surprised", 0.72)
    left_eye.set_emotion("surprised", surprise_intensity)
    right_eye.set_emotion("surprised", surprise_intensity)
    left_eye.surprise_shock_until = now + WAKE_SURPRISE_SEC
    right_eye.surprise_shock_until = now + WAKE_SURPRISE_SEC
    current_emotion = "surprised"


def start_wander_peek():
    """Look-around peek while searching for someone (no-face wandering)."""
    global scan_emotion_override

    if random.random() < 0.35:
        sx = random.choice([-1.0, 1.0])
    else:
        sx = 1.0 if last_face_norm_x >= 0.0 else -1.0
    tilt_sign = -1.0 if random.random() < 0.75 else 1.0
    start_gaze_event(
        "AVERT_SCAN",
        sx * random.uniform(GAZE_SCAN_X * 0.55, GAZE_SCAN_X * 0.85),
        tilt_sign * random.uniform(GAZE_SCAN_Y * 0.05, GAZE_SCAN_Y * 0.20),
        to_sec=0.45,
        hold_sec=random.uniform(0.6, 1.0),
        back_sec=0.45,
    )
    scan_emotion_override = random.choice(WANDER_EMOTIONS)


def clear_gaze_aversion():
    """Cancel ambient look-away so face re-acquire starts on the face."""
    global gaze_state, gaze_event_active, gaze_override_x, gaze_override_y
    global scan_emotion_override, no_face_scan_completed_pulse
    global servo_aversion_pan_offset, servo_aversion_tilt_offset

    gaze_event_active = False
    gaze_state = "ENGAGED"
    gaze_override_x = 0.0
    gaze_override_y = 0.0
    scan_emotion_override = None
    no_face_scan_completed_pulse = False
    with servo_state_lock:
        servo_aversion_pan_offset = 0.0
        servo_aversion_tilt_offset = 0.0


def start_gaze_event(kind: str, x: float, y: float, to_sec: float, hold_sec: float, back_sec: float):
    global gaze_state, gaze_event_active, gaze_event_start, gaze_event_to_sec
    global gaze_event_hold_sec, gaze_event_back_sec, gaze_event_target_x, gaze_event_target_y
    global scan_emotion_override, solo_mood

    gaze_state = kind
    gaze_event_active = True
    gaze_event_start = time.time()
    gaze_event_to_sec = max(0.01, to_sec)
    gaze_event_hold_sec = max(0.0, hold_sec)
    gaze_event_back_sec = max(0.01, back_sec)
    gaze_event_target_x = float(x)
    gaze_event_target_y = float(y)
    if kind == "AVERT_SCAN":
        if no_face_mode == "wandering":
            scan_emotion_override = weighted_pick([
                ("curious_intense", 0.40),
                ("uncertain", 0.30),
                ("thinking", 0.20),
                ("attentive", 0.10),
            ])
        elif solo_mood in ("cheerful", "content"):
            scan_emotion_override = (
                "looking_right_cheerful" if gaze_event_target_x >= 0 else "looking_left_cheerful"
            )
        else:
            scan_emotion_override = (
                "looking_right_natural" if gaze_event_target_x >= 0 else "looking_left_natural"
            )
    else:
        scan_emotion_override = None


def update_gaze_manager(now: float):
    global gaze_state, gaze_event_active, gaze_override_x, gaze_override_y, gaze_reengage_until
    global servo_aversion_pan_offset, servo_aversion_tilt_offset, scan_emotion_override
    global no_face_scan_completed_pulse, no_face_mode

    if not gaze_event_active:
        gaze_override_x = 0.0
        gaze_override_y = 0.0
        scan_emotion_override = None
        no_face_scan_completed_pulse = False
        with servo_state_lock:
            servo_aversion_pan_offset = 0.0
            servo_aversion_tilt_offset = 0.0
        return

    elapsed = now - gaze_event_start
    t1 = gaze_event_to_sec
    t2 = t1 + gaze_event_hold_sec
    t3 = t2 + gaze_event_back_sec

    if elapsed <= t1:
        a = _smoothstep01(elapsed / max(0.001, t1))
        gaze_override_x = gaze_event_target_x * a
        gaze_override_y = gaze_event_target_y * a
    elif elapsed <= t2:
        gaze_override_x = gaze_event_target_x
        gaze_override_y = gaze_event_target_y
    elif elapsed <= t3:
        a = _smoothstep01((elapsed - t2) / max(0.001, gaze_event_back_sec))
        gaze_override_x = gaze_event_target_x * (1.0 - a)
        gaze_override_y = gaze_event_target_y * (1.0 - a)
    else:
        finished_kind = gaze_state
        gaze_event_active = False
        gaze_state = "ENGAGED"
        gaze_override_x = 0.0
        gaze_override_y = 0.0
        scan_emotion_override = None
        no_face_scan_completed_pulse = finished_kind == "AVERT_SCAN"
        gaze_reengage_until = now + 0.28

    with servo_state_lock:
        if gaze_state == "AVERT_SCAN":
            denom = max(1e-3, abs(gaze_event_target_x))
            scan_progress = gaze_override_x / denom
            scan_progress = clamp(scan_progress, -1.0, 1.0)
            scan_dir = 1.0 if gaze_event_target_x >= 0.0 else -1.0
            servo_aversion_pan_offset = scan_dir * scan_progress * NO_FACE_SCAN_SERVO_PAN_DEG
            tilt_arc = (
                math.sin(scan_progress * math.pi + NO_FACE_SCAN_TILT_PHASE)
                * NO_FACE_SCAN_SERVO_TILT_DEG
            )
            servo_aversion_tilt_offset = tilt_arc + gaze_override_y * GAZE_SERVO_TILT_PER_PX
        else:
            servo_aversion_pan_offset = gaze_override_x * GAZE_SERVO_PAN_PER_PX
            servo_aversion_tilt_offset = gaze_override_y * GAZE_SERVO_TILT_PER_PX


def head_center_angles() -> tuple[float, float]:
    return (PAN_MIN + PAN_MAX) * 0.5, (TILT_MIN + TILT_MAX) * 0.5


def _tilt_down_from_center(offset_deg: float) -> float:
    """Lower servo tilt = head down (toward TILT_MIN)."""
    _, tilt_center = head_center_angles()
    return clamp(tilt_center - abs(offset_deg), TILT_MIN, TILT_MAX)


def _apply_detection_aim_point(
    aim_cx: float,
    aim_cy: float,
    *,
    servo_alpha: float,
) -> tuple[float, float, float, float, float | None, float | None]:
    """Map detect-frame pixel aim point to eye offsets and optional servo targets."""
    global servo_target_pan, servo_target_tilt
    global last_face_pan, last_face_tilt, last_face_norm_x, last_face_norm_y

    norm_x = -((aim_cx / CAMERA_RES[0] - 0.5) * 2.0)
    # Match main branch: face lower in frame -> look down.
    norm_y = -((aim_cy / CAMERA_RES[1] - 0.5) * 2.0)
    norm_x = apply_deadzone_norm(norm_x, FACE_TRACK_DEADZONE_X)
    norm_y = apply_deadzone_norm(norm_y, FACE_TRACK_DEADZONE_Y)

    local_x = max(-MAX_X_OFFSET, min(MAX_X_OFFSET, norm_x * MAX_X_OFFSET))
    local_y = max(-MAX_Y_OFFSET, min(MAX_Y_OFFSET, norm_y * MAX_Y_OFFSET))

    mapped_pan = None
    mapped_tilt = None
    if ENABLE_SERVO and servo_driver is not None:
        pan_center = (PAN_MIN + PAN_MAX) * 0.5
        tilt_center = (TILT_MIN + TILT_MAX) * 0.5
        mapped_pan = clamp(pan_center + (norm_x * PAN_TRACK_RANGE), PAN_MIN, PAN_MAX)
        mapped_tilt = clamp(
            tilt_center + (norm_y * TILT_TRACK_RANGE), TILT_MIN, TILT_MAX
        )
        with servo_state_lock:
            servo_target_pan = servo_target_pan + (mapped_pan - servo_target_pan) * servo_alpha
            servo_target_tilt = servo_target_tilt + (mapped_tilt - servo_target_tilt) * servo_alpha
            last_face_pan = mapped_pan
            last_face_tilt = mapped_tilt
            last_face_norm_x = norm_x
            last_face_norm_y = norm_y

    return local_x, local_y, norm_x, norm_y, mapped_pan, mapped_tilt


def _base_motion_allowed(now: float) -> bool:
    if not BASE_ENABLED or servo_driver is None:
        return False
    if not getattr(servo_driver, "base_motion_allowed", True):
        return False
    if now < base_error_until:
        return False
    return hasattr(servo_driver, "write_base_relative_clamped")


def _record_base_error(now: float, reason: str) -> None:
    global base_error_until
    base_error_until = now + BASE_ERROR_BACKOFF_SEC
    print(f"Base motion paused {BASE_ERROR_BACKOFF_SEC:.0f}s ({reason})")
    if servo_driver is not None and hasattr(servo_driver, "write_base_stop"):
        try:
            servo_driver.write_base_stop()
        except Exception:
            pass


def _maybe_search_base_fov_nudge(
    pan_current: float,
    *,
    face_locked: bool,
    mode: str,
) -> None:
    """Small base rotation when pan nears limit during no-face search."""
    now = time.time()
    if (
        not _base_motion_allowed(now)
        or face_locked
        or mode != "wandering"
        or tof_approach_controller.suppresses_wander_base()
    ):
        return

    if now - last_base_fov_nudge_ts < SEARCH_BASE_COOLDOWN_SEC:
        return

    dist_min = pan_current - PAN_MIN
    dist_max = PAN_MAX - pan_current
    nudge_deg = 0.0
    if dist_max <= SEARCH_BASE_EDGE_DEG:
        nudge_deg = -SEARCH_BASE_NUDGE_DEG
    elif dist_min <= SEARCH_BASE_EDGE_DEG:
        nudge_deg = SEARCH_BASE_NUDGE_DEG
    else:
        return

    _apply_wander_base_nudge(nudge_deg, now)


def _apply_wander_base_nudge(
    nudge_deg: float,
    now: float,
    *,
    face_tracking: bool = False,
) -> bool:
    """Base nudge with head pan counter-rotation to hold gaze."""
    global last_base_fov_nudge_ts, servo_target_pan, servo_current_pan, servo_pan_vel
    global _last_sent_pan

    if abs(nudge_deg) < 0.2:
        return False
    if not _base_motion_allowed(now):
        return False
    try:
        st = servo_driver.query_base_status()
        if st is not None and st.busy:
            return False
        ok = servo_driver.write_base_relative_clamped(
            nudge_deg,
            max_from_zero=BASE_MAX_DEG_FROM_ZERO,
            max_nudge_deg=BASE_MAX_NUDGE_DEG,
            wait=True,
        )
        if not ok:
            _record_base_error(now, "ERR B or timeout")
            return False
        last_base_fov_nudge_ts = now
        pan_center = (PAN_MIN + PAN_MAX) * 0.5
        comp_alpha = (
            clamp(float(FACE_BASE_HEAD_COMP_ALPHA), 0.2, 1.0)
            if face_tracking
            else WANDER_PAN_TARGET_ALPHA
        )
        with servo_state_lock:
            compensated = clamp(
                servo_target_pan - nudge_deg,
                PAN_MIN,
                PAN_MAX,
            )
            if (
                not face_tracking
                and abs(compensated - pan_center) < SEARCH_BASE_NUDGE_DEG
            ):
                compensated = pan_center
            servo_target_pan += (compensated - servo_target_pan) * comp_alpha
            if face_tracking:
                servo_current_pan = clamp(
                    servo_current_pan - nudge_deg,
                    PAN_MIN,
                    PAN_MAX,
                )
                servo_pan_vel = 0.0
                _last_sent_pan = None
        return True
    except Exception as e:
        print(f"Base wander nudge failed: {e}")
        _record_base_error(now, str(e))
        return False


def _maybe_wander_base_follow_nudge(
    pan_current: float,
    *,
    face_locked: bool,
    mode: str,
) -> None:
    """Occasionally rotate base slightly in the direction the head is drifting."""
    global last_wander_base_follow_ts, last_wander_base_follow_eval_ts

    now = time.time()
    if (
        not _base_motion_allowed(now)
        or face_locked
        or mode != "wandering"
        or tof_approach_controller.suppresses_wander_base()
    ):
        return

    if now - last_wander_base_follow_eval_ts < WANDER_BASE_FOLLOW_EVAL_SEC:
        return
    last_wander_base_follow_eval_ts = now

    if now - last_wander_base_follow_ts < WANDER_BASE_FOLLOW_COOLDOWN_SEC:
        return
    if now - last_base_fov_nudge_ts < 2.5:
        return

    drift_vel = organic_wander_search.drift_vel
    if abs(drift_vel) < WANDER_BASE_FOLLOW_MIN_DRIFT_VEL:
        return

    pan_center = (PAN_MIN + PAN_MAX) * 0.5
    head_pan = pan_current - pan_center
    if abs(head_pan) < WANDER_BASE_FOLLOW_MIN_PAN_DEG:
        return
    if random.random() > WANDER_BASE_FOLLOW_CHANCE:
        return

    # Same sign convention as FOV edge nudge: drift right → negative base cmd.
    nudge_deg = -WANDER_BASE_FOLLOW_DEG if drift_vel > 0 else WANDER_BASE_FOLLOW_DEG
    if _apply_wander_base_nudge(nudge_deg, now):
        last_wander_base_follow_ts = now


def _maybe_face_track_base_nudges(
    pan_current: float,
    *,
    face_locked: bool,
    now: float,
) -> None:
    """Small base rotations while face-tracking — alive micro-moves and edge assist."""
    global last_face_base_nudge_ts, next_face_base_alive_ts

    if (
        not _base_motion_allowed(now)
        or not face_locked
        or tof_approach_controller.suppresses_wander_base()
    ):
        return

    if now - last_face_base_nudge_ts < FACE_BASE_COOLDOWN_SEC:
        return
    if now - last_base_fov_nudge_ts < 2.0:
        return

    with servo_state_lock:
        norm_x = last_face_norm_x

    dist_min = pan_current - PAN_MIN
    dist_max = PAN_MAX - pan_current
    head_near_limit = (
        dist_min <= FACE_BASE_EDGE_PAN_EDGE_DEG
        or dist_max <= FACE_BASE_EDGE_PAN_EDGE_DEG
    )
    face_off_center = abs(norm_x) >= FACE_BASE_EDGE_NORM

    nudge_deg = 0.0
    cap = max(0.5, float(FACE_BASE_ALIVE_MAX_DEG))

    if face_off_center or head_near_limit:
        if face_off_center:
            off = (abs(norm_x) - FACE_BASE_EDGE_NORM) / max(
                1e-6, 1.0 - FACE_BASE_EDGE_NORM
            )
            mag = min(FACE_BASE_EDGE_NUDGE_DEG, cap) * clamp(off, 0.0, 1.0)
            nudge_deg = -math.copysign(mag, norm_x) if norm_x != 0.0 else 0.0
        if head_near_limit:
            limit_nudge = min(FACE_BASE_EDGE_NUDGE_DEG, cap)
            if dist_max <= FACE_BASE_EDGE_PAN_EDGE_DEG:
                nudge_deg = -limit_nudge
            elif dist_min <= FACE_BASE_EDGE_PAN_EDGE_DEG:
                nudge_deg = limit_nudge
    elif FACE_BASE_ALIVE_ENABLED and now >= next_face_base_alive_ts:
        next_face_base_alive_ts = now + random.uniform(
            FACE_BASE_ALIVE_MIN_SEC,
            FACE_BASE_ALIVE_MAX_SEC,
        )
        alive_mag = min(FACE_BASE_ALIVE_DEG, cap)
        if abs(norm_x) > 0.12:
            nudge_deg = -math.copysign(alive_mag * 0.55, norm_x)
        else:
            nudge_deg = random.choice([-1.0, 1.0]) * alive_mag * random.uniform(0.4, 0.9)

    if abs(nudge_deg) < 0.35:
        return
    nudge_deg = clamp(nudge_deg, -cap, cap)
    if _apply_wander_base_nudge(nudge_deg, now, face_tracking=True):
        last_face_base_nudge_ts = now


def _servo_home_frame() -> dict[str, float]:
    pan_home, tilt_home = head_center_angles()
    pose = servo_stop_pose(neutral_arm_targets)
    return {
        "P": pan_home,
        "T": tilt_home,
        "A0=": pose["arm_0"],
        "A1=": pose["arm_1"],
        "A2=": pose["arm_2"],
        "A3=": pose["arm_3"],
    }


def _home_servos_on_shutdown() -> None:
    """Move head + arms to neutral, then release the serial link."""
    global animation_arm_targets, arm_current_smoothed
    if servo_driver is None:
        return
    with animation_lock:
        animation_player.stop()
        animation_arm_targets = dict(neutral_arm_targets)
        arm_current_smoothed = dict(neutral_arm_targets)
    pan_home, tilt_home = head_center_angles()
    arms = dict(neutral_arm_targets)
    homed = False
    try:
        if BASE_ENABLED and hasattr(servo_driver, "write_base_stop"):
            servo_driver.write_base_stop()
        if BASE_ENABLED and hasattr(servo_driver, "write_base_absolute"):
            try:
                servo_driver.write_base_absolute(BASE_HOME_DEG, wait=True)
                print(f"Base homed to {BASE_HOME_DEG:.1f}°")
            except Exception as e:
                print(f"Base home failed: {e}")
        if hasattr(servo_driver, "write_home_pose"):
            homed = servo_driver.write_home_pose(
                pan_home, tilt_home, arms, wait_ack=True
            )
        elif hasattr(servo_driver, "write_servo_frame"):
            homed = servo_driver.write_servo_frame(_servo_home_frame(), wait_ack=True)
        else:
            try:
                homed = servo_driver.write_angles(pan_home, tilt_home, force=True)
            except TypeError:
                homed = servo_driver.write_angles(pan_home, tilt_home)
        if homed:
            time.sleep(0.35)
            print(
                f"Servos homed: P={pan_home:.1f} T={tilt_home:.1f} "
                f"arms {arms['arm_0']:.0f}/{arms['arm_1']:.0f}/"
                f"{arms['arm_2']:.0f}/{arms['arm_3']:.0f} deg"
            )
    except Exception as e:
        print(f"Servo home failed: {e}")
    try:
        servo_driver.close(
            home_pan=pan_home,
            home_tilt=tilt_home,
            arm_neutrals=arms,
            skip_home=homed,
        )
        print("Serial closed.")
    except TypeError:
        servo_driver.close(home_pan=pan_home, home_tilt=tilt_home)
        print("Serial closed.")
    except Exception as e:
        print(e)


def _blend_track(base: float, sample_value: float, mode: str, weight: float) -> float:
    w = max(0.0, min(1.0, float(weight)))
    if str(mode).lower() == "override":
        return base + (sample_value - base) * w
    return base + (sample_value * w)


def _load_default_animation_clips() -> int:
    clips_dir = Path(__file__).parent / "animations"
    if not clips_dir.exists():
        return 0
    loaded = 0
    global animation_arm_targets, arm_current_smoothed, neutral_arm_targets

    botango_path = clips_dir / BOTANGO_COMMANDS_FILE
    if botango_path.exists():
        try:
            clips = load_botango_commands_file(botango_path)
            for clip in clips:
                animation_player.register_clip(clip)
                loaded += 1
            with botango_path.open(encoding="utf-8") as f:
                raw = json.load(f)
            controllers = raw if isinstance(raw, list) else [raw]
            for controller in controllers:
                setup_text = controller.get("Setup", {}).get("Controller Setup Commands", "")
                effectors = _parse_setup(setup_text)
                neutrals = neutral_arm_degrees(effectors)
                if neutrals:
                    neutral_arm_targets.update(neutrals)
                    animation_arm_targets.update(neutrals)
                    arm_current_smoothed.update(neutrals)
            print(
                f"Loaded {len(clips)} Botango animation(s) from {BOTANGO_COMMANDS_FILE}: "
                + ", ".join(c.clip_id for c in clips)
            )
            print(format_servo_stop_pose(servo_stop_pose(neutral_arm_targets)))
        except Exception as e:
            print(f"Botango animation load failed ({BOTANGO_COMMANDS_FILE}): {e}")

    for fp in sorted(clips_dir.glob("*.json")):
        if fp.name == BOTANGO_COMMANDS_FILE:
            continue
        try:
            animation_player.load_clip_file(fp)
            loaded += 1
        except Exception as e:
            print(f"Animation clip load failed ({fp.name}): {e}")
    return loaded


def get_tof_api_payload() -> dict:
    """Flat ToF snapshot for radar viz (/api/tof)."""
    with tof_lock:
        snap = tof_snapshot.as_dict()
        pres = tof_presence.as_dict()
        bearing = tof_approach_controller.last_bearing_deg
        approach_active = bool(tof_approach_controller.active)
        phase = tof_approach_controller.phase_name()
        latched = tof_approach_controller.latched_sector
    return {
        **snap,
        "presence": pres,
        "bearing_deg": bearing,
        "target_pan_offset_deg": bearing,
        "approach_active": approach_active,
        "approach_phase": phase,
        "latched_sector": latched,
        "enabled": bool(TOF_ENABLED),
        "use_base": bool(TOF_APPROACH_USE_BASE),
    }


def _tof_firmware_ok() -> bool:
    if tof_poll_ok:
        return True
    if servo_driver is None:
        return False
    link = getattr(servo_driver, "_link", None)
    return bool(getattr(link, "tof_capable", False))


def get_tof_state() -> dict:
    with tof_lock:
        snap = tof_snapshot.as_dict()
        pres = tof_presence.as_dict()
        bearing = tof_approach_controller.last_bearing_deg
        approach_active = bool(tof_approach_controller.active)
        phase = tof_approach_controller.phase_name()
        latched = tof_approach_controller.latched_sector
    snap_ts = float(snap.get("timestamp") or 0.0)
    age_sec = round(time.time() - snap_ts, 2) if snap_ts > 0 else None
    any_valid = any(
        snap.get(k) for k in ("left_valid", "center_valid", "right_valid")
    )
    firmware_ok = _tof_firmware_ok()
    if not TOF_ENABLED:
        status = "disabled"
    elif not firmware_ok:
        status = "no_firmware"
    elif snap_ts <= 0:
        status = "waiting"
    elif age_sec is not None and age_sec > 3.0:
        status = "stale"
    elif any_valid:
        status = "live"
    else:
        status = "clear"
    return {
        "enabled": bool(TOF_ENABLED),
        "firmware_ok": firmware_ok,
        "status": status,
        "age_sec": age_sec,
        "poll_hz": float(TOF_POLL_HZ),
        "snapshot": snap,
        "presence": pres,
        "bearing_deg": bearing,
        "target_pan_offset_deg": bearing,
        "approach_active": approach_active,
        "approach_phase": phase,
        "latched_sector": latched,
        "use_base": bool(TOF_APPROACH_USE_BASE),
    }


def tof_worker():
    global tof_snapshot, tof_presence, prev_tof_center_present, tof_poll_ok
    if not TOF_ENABLED or servo_driver is None:
        return
    interval = 1.0 / max(0.5, float(TOF_POLL_HZ))
    last_error_log = 0.0
    while running:
        try:
            with tof_lock:
                need_init = tof_snapshot.timestamp <= 0
            poll_timeout = 8.0 if need_init else 0.35
            snap = servo_driver.poll_tof(timeout=poll_timeout)
            if snap is not None:
                tof_poll_ok = True
                snap = sanitize_tof_snapshot(
                    snap,
                    min_valid_mm=TOF_MIN_VALID_MM,
                )
                presence = tof_tracker.update(snap)
                with tof_lock:
                    tof_snapshot = snap
                    tof_presence = presence
                    center_rising = presence.center and not prev_tof_center_present
                    prev_tof_center_present = presence.center
                tof_approach_controller.sync_presence(presence, time.time())
                if center_rising:
                    _notify_voice_presence_arrival(time.time())
        except Exception as e:
            now = time.time()
            if now - last_error_log > 5.0:
                print(f"ToF poll error: {e}")
                last_error_log = now
        time.sleep(interval)


def _tof_approach_skip_motion(now: float) -> bool:
    return (
        no_face_mode == "sad_return"
        or now < jerk_until
        or gaze_event_active
    )


def _apply_tof_approach_action(action, now: float) -> None:
    """Update head pan/tilt toward sector look angle (eye level, head servos)."""
    global servo_target_pan, servo_target_tilt
    _, tilt_center = head_center_angles()
    with servo_state_lock:
        if abs(action.pan_delta_deg) >= 0.05:
            servo_target_pan = clamp(
                servo_target_pan + action.pan_delta_deg,
                PAN_MIN,
                PAN_MAX,
            )
        if tof_approach_controller.drives_motion() or abs(action.pan_delta_deg) >= 0.05:
            alpha = max(0.05, TOF_APPROACH_TILT_RECENTER_ALPHA)
            servo_target_tilt += (tilt_center - servo_target_tilt) * alpha
    if abs(action.base_nudge_deg) >= 0.2 and BASE_ENABLED:
        ok = _apply_wander_base_nudge(action.base_nudge_deg, now)
        if not ok:
            print(
                f"ToF base nudge failed ({action.base_nudge_deg:+.1f}°) "
                f"sector={tof_approach_controller.latched_sector}"
            )
        else:
            print(
                f"ToF base nudge {action.base_nudge_deg:+.1f}° "
                f"sector={tof_approach_controller.latched_sector}"
            )


def _maybe_tof_approach_turn(
    pan_current: float,
    *,
    pan_target: float | None = None,
    face_locked: bool,
    now: float,
) -> None:
    if not TOF_APPROACH_ENABLED or not TOF_ENABLED:
        return
    with tof_lock:
        snap = tof_snapshot
        pres = tof_presence
        action = tof_approach_controller.tick(
            snap,
            pres,
            face_locked=face_locked,
            pan_current=pan_current,
            pan_target=pan_target,
            skip_motion=_tof_approach_skip_motion(now),
            now=now,
        )
    if action is not None:
        _apply_tof_approach_action(action, now)


def servo_worker():
    global servo_current_pan, servo_current_tilt, servo_pan_vel, servo_tilt_vel
    global servo_running, jerk_until, jerk_direction
    global amplitude_fast, amplitude_slow, udp_speak_pulse, udp_conv_state
    global current_emotion, gaze_event_active, no_face_mode, sad_return_start, wake_tilt_jerk_until
    global animation_arm_targets, arm_current_smoothed, _last_servo_frame_ts, wander_pan_speed_scale
    if servo_driver is None:
        return

    while servo_running:
        now = time.time()

        with target_lock:
            face_locked = target_face_present

        with servo_state_lock:
            pan_current = servo_current_pan
            pan_target = servo_target_pan
        _maybe_tof_approach_turn(
            pan_current,
            pan_target=pan_target,
            face_locked=face_locked,
            now=now,
        )

        with servo_state_lock:
            pan_target = servo_target_pan
            tilt_target = servo_target_tilt
            pan_current = servo_current_pan
            tilt_current = servo_current_tilt
            pan_avert = servo_aversion_pan_offset
            tilt_avert = servo_aversion_tilt_offset

        # Apply jerk oscillation if active
        jerk_offset = 0.0
        if now < jerk_until and jerk_direction != 0.0:
            # Elapsed time within jerk window (0.0 to JERK_DURATION)
            elapsed = now - (jerk_until - JERK_DURATION)
            # Normalize to 0-1 phase
            phase = elapsed / JERK_DURATION
            # Sine wave oscillation: quick outward jerk, return, small reverse jerk
            jerk_offset = jerk_direction * JERK_AMPLITUDE * math.sin(phase * math.pi * 2.0)

        # Conversation-state nods — not while face locked, sad return, or settled idle.
        conv_tilt = 0.0
        if not face_locked and no_face_mode in ("wandering", "chat_ready"):
            if (
                no_face_mode == "wandering"
                and not organic_wander_search.moving
                and organic_wander_search.pause_kind in ("thinking", "long_stare")
            ):
                bob_scale = 1.15 if organic_wander_search.pause_kind == "thinking" else 0.82
                conv_tilt = (
                    math.sin(now * CONV_THINK_BOB_HZ * math.tau)
                    * CONV_THINK_BOB_DEG
                    * bob_scale
                )
            elif udp_conv_state == "nodding":
                conv_tilt = math.sin(now * CONV_NOD_HZ * math.tau) * CONV_NOD_DEG
            elif udp_conv_state in ("thinking", "concentrating", "remembering"):
                conv_tilt = math.sin(now * CONV_THINK_BOB_HZ * math.tau) * CONV_THINK_BOB_DEG

        # Speech-driven gestures: tilt nods only (no pan sweep while talking).
        subtle_pan = 0.0
        subtle_tilt = 0.0
        speaking = udp_speak_pulse > 0.0 or udp_conv_state == "speaking"
        if speaking:
            subtle_pan = TALK_GESTURE_PAN_MULT
            tilt_mult = (
                TALK_GESTURE_TILT_MULT_FACE
                if face_locked
                else TALK_GESTURE_TILT_MULT_NO_FACE
            )
            subtle_tilt = math.cos(now * 3.2) * (amplitude_fast * tilt_mult)

        sad_nod_tilt = 0.0
        if no_face_mode == "sad_return" and sad_return_start > 0.0:
            elapsed = now - sad_return_start
            phase = elapsed * (SAD_NOD_COUNT * math.tau / max(0.1, SAD_RETURN_SEC))
            sad_nod_tilt = -abs(math.sin(phase)) * SAD_NOD_TILT_DEG

        wake_tilt = 0.0
        if no_face_mode == "chat_ready" and now < wake_tilt_jerk_until:
            elapsed = now - (wake_tilt_jerk_until - WAKE_TILT_JERK_SEC)
            phase = clamp(elapsed / max(0.01, WAKE_TILT_JERK_SEC), 0.0, 1.0)
            wake_tilt = WAKE_TILT_JERK_DEG * math.sin(phase * math.pi)

        sleep_tilt = 0.0
        if current_emotion == "sleepy" and not gaze_event_active:
            sleep_tilt = -abs(SLEEP_TILT_DEG)

        anim_samples = {}
        with animation_lock:
            anim_samples = animation_player.sample(now)

        pan_target_blended = pan_target
        tilt_target_blended = tilt_target
        if "head_pan" in anim_samples:
            s = anim_samples["head_pan"]
            pan_target_blended = _blend_track(pan_target_blended, s.value, s.mode, s.weight)
        if "head_tilt" in anim_samples:
            s = anim_samples["head_tilt"]
            tilt_target_blended = _blend_track(tilt_target_blended, s.value, s.mode, s.weight)

        if not anim_samples:
            arm_targets = dict(neutral_arm_targets)
        else:
            arm_targets = dict(animation_arm_targets)
            for arm_track in ("arm_0", "arm_1", "arm_2", "arm_3"):
                if arm_track in anim_samples:
                    s = anim_samples[arm_track]
                    base = neutral_arm_targets.get(arm_track, 90.0)
                    arm_targets[arm_track] = _blend_track(base, s.value, s.mode, s.weight)
        animation_arm_targets = arm_targets

        animating = bool(anim_samples)
        arm_alpha = ARM_ANIM_BLEND if animating else 0.18
        for arm_track in ("arm_0", "arm_1", "arm_2", "arm_3"):
            tgt = arm_targets.get(arm_track, neutral_arm_targets.get(arm_track, 90.0))
            cur = arm_current_smoothed.get(arm_track, tgt)
            if animating or abs(tgt - cur) > 0.4:
                arm_current_smoothed[arm_track] = cur + (tgt - cur) * arm_alpha
            else:
                arm_current_smoothed[arm_track] = tgt

        pan_goal = clamp(
            pan_target_blended + jerk_offset + pan_avert + subtle_pan,
            PAN_MIN,
            PAN_MAX,
        )
        tilt_goal = clamp(
            tilt_target_blended
            + tilt_avert
            + conv_tilt
            + subtle_tilt
            + sad_nod_tilt
            + wake_tilt
            + sleep_tilt,
            TILT_MIN,
            TILT_MAX,
        )

        pan_motion = PAN_MOTION
        if no_face_mode == "wandering" and not face_locked:
            pan_motion = scale_head_motion(PAN_MOTION, wander_pan_speed_scale)

        pan_current, servo_pan_vel = tick_toward(
            pan_current,
            servo_pan_vel,
            pan_goal,
            SERVO_LOOP_DELAY,
            lo=PAN_MIN,
            hi=PAN_MAX,
            params=pan_motion,
        )
        tilt_current, servo_tilt_vel = tick_toward(
            tilt_current,
            servo_tilt_vel,
            tilt_goal,
            SERVO_LOOP_DELAY,
            lo=TILT_MIN,
            hi=TILT_MAX,
            params=TILT_MOTION,
        )

        _maybe_search_base_fov_nudge(
            pan_current,
            face_locked=face_locked,
            mode=no_face_mode,
        )
        if not tof_approach_controller.active:
            _maybe_wander_base_follow_nudge(
                pan_current,
                face_locked=face_locked,
                mode=no_face_mode,
            )
        _maybe_face_track_base_nudges(
            pan_current,
            face_locked=face_locked,
            now=now,
        )

        global _last_sent_pan, _last_sent_tilt
        head_moved = (
            _last_sent_pan is None
            or _last_sent_tilt is None
            or abs(pan_current - _last_sent_pan) >= HEAD_SEND_MIN_DELTA_DEG
            or abs(tilt_current - _last_sent_tilt) >= HEAD_SEND_MIN_DELTA_DEG
        )
        send_frame = (
            animating
            or head_moved
            or (now - _last_servo_frame_ts) >= SERVO_FRAME_INTERVAL
        )
        if send_frame:
            try:
                frame_tokens: dict[str, float] = {"P": pan_current, "T": tilt_current}
                frame_tokens["A0="] = clamp(
                    arm_current_smoothed.get("arm_0", neutral_arm_targets.get("arm_0", 0.0)), 0.0, 180.0
                )
                frame_tokens["A1="] = clamp(
                    arm_current_smoothed.get("arm_1", neutral_arm_targets.get("arm_1", 180.0)), 0.0, 180.0
                )
                frame_tokens["A2="] = clamp(
                    arm_current_smoothed.get("arm_2", neutral_arm_targets.get("arm_2", 90.0)), 0.0, 180.0
                )
                frame_tokens["A3="] = clamp(
                    arm_current_smoothed.get("arm_3", neutral_arm_targets.get("arm_3", 90.0)), 0.0, 180.0
                )
                if hasattr(servo_driver, "write_servo_frame"):
                    servo_driver.write_servo_frame(frame_tokens)
                else:
                    servo_driver.write_angles(pan_current, tilt_current)
                _last_servo_frame_ts = now
                _last_sent_pan = pan_current
                _last_sent_tilt = tilt_current
            except Exception as e:
                print(f"Servo write error: {e}")

        with servo_state_lock:
            servo_current_pan = pan_current
            servo_current_tilt = tilt_current

        time.sleep(SERVO_LOOP_DELAY)

def clamp_eye_target(eye):
    # Keep the motion target inside a slightly looser region so subtle drift still
    # exists, while the final draw-time clamp prevents any pixel overflow.
    min_x, max_x, min_y, max_y = eye._motion_clamp_bounds()
    eye.target_pos[0] = max(min_x, min(max_x, eye.target_pos[0]))
    eye.target_pos[1] = max(min_y, min(max_y, eye.target_pos[1]))


def trigger_synced_blink(speed_mult):
    # Align blink start conditions so both displays animate the same phase.
    avg_y = (left_eye.current_pos[1] + right_eye.current_pos[1]) * 0.5
    avg_w = (left_eye.current_w + right_eye.current_w) * 0.5
    avg_h = (left_eye.current_h + right_eye.current_h) * 0.5
    for eye in (left_eye, right_eye):
        eye.blink_state = "IDLE"
        eye.vy = 0
        eye.current_pos[1] = avg_y
        eye.current_w = avg_w
        eye.current_h = avg_h
        eye.w = avg_w
        eye.h = avg_h
    left_eye.start_blink(speed_mult)
    right_eye.start_blink(speed_mult)


def mirror_blink_state(master, slave):
    # Force exact blink phase matching once a blink is active.
    slave.blink_state = master.blink_state
    slave.vy = master.vy
    slave.current_pos[1] = master.current_pos[1]
    slave.current_w = master.current_w
    slave.current_h = master.current_h
    slave.target_w = master.target_w
    slave.target_h = master.target_h
    slave.w = master.w
    slave.h = master.h


def mirror_full_state(master, slave):
    # Keep both eyes identical by driving one master state.
    slave.blink_state = master.blink_state
    slave.vy = master.vy
    slave.current_pos[0] = master.current_pos[0]
    slave.current_pos[1] = master.current_pos[1]
    slave.target_pos[0] = master.target_pos[0]
    slave.target_pos[1] = master.target_pos[1]
    slave.current_w = master.current_w
    slave.current_h = master.current_h
    slave.target_w = master.target_w
    slave.target_h = master.target_h
    slave.current_rotation = master.current_rotation
    slave.target_rotation = master.target_rotation
    slave.scale_w = master.scale_w
    slave.scale_h = master.scale_h
    slave.target_scale_w = master.target_scale_w
    slave.target_scale_h = master.target_scale_h
    slave.top_lid = master.top_lid
    slave.bottom_lid = master.bottom_lid
    slave.lid_angle = master.lid_angle
    slave.target_top_lid = master.target_top_lid
    slave.target_bottom_lid = master.target_bottom_lid
    slave.target_lid_angle = master.target_lid_angle
    slave.w = master.w
    slave.h = master.h


def _read_cpu_temp_c() -> float | None:
    """SoC temperature in °C (Raspberry Pi thermal zone or vcgencmd)."""
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text(encoding="utf-8").strip()
        return round(int(raw) / 1000.0, 1)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            text=True,
            timeout=1.0,
        )
        # temp=84.2'C
        return round(float(out.split("=")[1].split("'")[0]), 1)
    except Exception:
        return None


def get_runtime_state() -> dict:
    """Thread-safe snapshot of live robot state for the debug dashboard."""
    with target_lock:
        face = {
            "present": target_face_present,
            "body_present": target_body_present,
            "count": target_face_count,
            "area_ratio": round(target_face_area_ratio, 4),
            "x_offset": round(smoothed_x_off, 2),
            "y_offset": round(smoothed_y_off, 2),
            "rotation": round(smoothed_rotation, 2),
            "tracking_active": face_tracking_active,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }

    with servo_state_lock:
        servo = {
            "enabled": ENABLE_SERVO,
            "pan": round(servo_current_pan, 2),
            "tilt": round(servo_current_tilt, 2),
            "target_pan": round(servo_target_pan, 2),
            "target_tilt": round(servo_target_tilt, 2),
            "tilt_from_eye_level_deg": round(search_tilt_from_eye_level_deg, 2),
        }
    with animation_lock:
        active_clip = animation_player.active_clip_id()
    servo["animation_clip"] = active_clip
    if servo_driver is not None:
        link = getattr(servo_driver, "_link", None)
        servo["serial_connected"] = bool(
            getattr(servo_driver, "serial_connected", False)
            or (link is not None and link.connected)
        )
    else:
        servo["serial_connected"] = False

    cpu_temp_c = _read_cpu_temp_c()

    return {
        "timestamp": time.time(),
        "system": {
            "cpu_temp_c": cpu_temp_c,
        },
        "session_active": session_active,
        "emotion": {
            "current": current_emotion,
            "candidate": router_candidate_emotion,
            "trace": list(emotion_trace[-8:]),
            "no_face_mode": no_face_mode,
            "gaze_state": gaze_state,
            "social_mode": social_mode,
            "speak_social_mode": speak_social_mode,
        },
        "udp": {
            "conv_state": udp_conv_state,
            "conv_emotion": udp_conv_emotion,
            "emotion_override": udp_emotion_override,
            "emotion_override_remaining": round(max(0.0, udp_emotion_until - time.time()), 2),
            "amplitude_fast": round(amplitude_fast, 4),
            "amplitude_slow": round(amplitude_slow, 4),
            "speak_pulse": round(udp_speak_pulse, 4),
        },
        "face": face,
        "servo": servo,
        "tof": get_tof_state(),
    }


def _notify_voice_presence_arrival(now: float) -> None:
    """Tell voice_agent someone approached during an active LiveKit session."""
    global presence_arrival_last_ts
    if not session_active:
        return
    if udp_conv_state == "speaking":
        return
    if now - presence_arrival_last_ts < PRESENCE_ARRIVAL_COOLDOWN_SEC:
        return
    presence_arrival_last_ts = now
    try:
        _voice_notify_sock.sendto(
            json.dumps({"command": "presence_arrival"}).encode("utf-8"),
            _VOICE_NOTIFY_ADDR,
        )
    except OSError:
        pass


def udp_worker():
    global udp_emotion_override, udp_emotion_until, udp_speak_pulse
    global amplitude_fast, amplitude_slow
    global udp_conv_state, udp_conv_emotion, wake_request_ts, session_active
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 9000))
    sock.settimeout(0.5)
    print("UDP Listener active on port 9000")
    while running:
        try:
            data, _ = sock.recvfrom(1024)
            msg = json.loads(data.decode("utf-8"))

            # Layer 3: Real amplitude signals from AmplitudeTTS (40ms cadence)
            if "amplitude_fast" in msg:
                amplitude_fast = float(msg["amplitude_fast"])
                amplitude_slow = float(msg["amplitude_slow"])
                # Lower threshold (0.005) to capture whispers better
                udp_speak_pulse = 1.0 if amplitude_fast > 0.005 else 0.0

            # Layer 2: Conversation state override
            if msg.get("command") == "conv_state":
                new_state = msg.get("state", "waiting")
                udp_conv_state = new_state
                udp_conv_emotion = msg.get("emotion", "attentive")
                # Clear stale speak pulse when not talking (old 0.5 hack blocked tilt tracking).
                if new_state != "speaking" and amplitude_fast <= 0.005:
                    udp_speak_pulse = 0.0

            # Layer 1: VADER emotion backdrop
            if msg.get("command") == "emotion":
                udp_emotion_override = msg.get("emotion")
                udp_emotion_until = time.time() + 8.0   # VADER holds for 8s

            # Legacy fallback binary pulse
            if "speak_pulse" in msg:
                udp_speak_pulse = float(msg["speak_pulse"])

            if msg.get("command") == "wake":
                wake_request_ts = time.time()

            if msg.get("command") == "session_active":
                session_active = bool(msg.get("active", False))

        except socket.timeout:
            pass
        except Exception:
            pass

# MJPEG debug overlay colors (cv2 BGR tuples on RGB stream buffer → see comment)
# Face: green box + yellow eye dots | Body: magenta box + magenta aim dot
_FACE_BOX_BGR = (0, 255, 0)
_FACE_EYE_BGR = (0, 255, 255)
_BODY_BOX_BGR = (255, 0, 255)
_BODY_AIM_BGR = (255, 0, 255)
_DEBUG_OVERLAY_THICKNESS = 3


def map_coords_to_stream_preview(fx, fy, fw, fh, re_x, re_y, le_x, le_y):
    """Map detection coords to MJPEG stream (same oriented frame when wide FOV)."""
    w, h = CAMERA_RES[0], CAMERA_RES[1]
    if not CAMERA_WIDE_FOV and CAMERA_ROTATE_180:
        fx = w - fx - fw
        fy = h - fy - fh
        re_x, re_y = w - re_x, h - re_y
        le_x, le_y = w - le_x, h - le_y
    scale_x = STREAM_RES[0] / w
    scale_y = STREAM_RES[1] / h
    return (
        int(fx * scale_x), int(fy * scale_y),
        int(fw * scale_x), int(fh * scale_y),
        int(re_x * scale_x), int(re_y * scale_y),
        int(le_x * scale_x), int(le_y * scale_y),
    )


def _prepare_vision_frames(large_frame):
    """Return (detect_frame, stream_frame) for the active camera pipeline."""
    if CAMERA_WIDE_FOV:
        frame = cv2.resize(large_frame, CAMERA_RES, interpolation=cv2.INTER_AREA)
        if CAMERA_ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        stream_frame = None
        if STREAM_ENABLED:
            stream_frame = cv2.resize(frame, STREAM_RES, interpolation=cv2.INTER_AREA)
            if STREAM_SWAP_RB:
                stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)
        return frame, stream_frame

    rgb_frame = frame_to_rgb(
        large_frame,
        legacy_swap_rb=STREAM_SWAP_RB and not CAMERA_USE_PREVIEW,
    )
    frame_raw = cv2.resize(rgb_frame, CAMERA_RES, interpolation=cv2.INTER_AREA)
    stream_frame = None
    if STREAM_ENABLED:
        if STREAM_RES == CAMERA_MAIN_RES:
            stream_frame = rgb_frame.copy()
        else:
            stream_frame = cv2.resize(rgb_frame, STREAM_RES, interpolation=cv2.INTER_AREA)
    return frame_raw, stream_frame


def _detect_faces_on_frame(frame_raw):
    if CAMERA_WIDE_FOV:
        return detect_faces_yunet_fast(
            detector,
            frame_raw,
            input_size=CAMERA_RES,
        )
    detected_faces, _ = detect_faces_yunet(
        detector,
        frame_raw,
        input_size=CAMERA_RES,
        rotate_180=CAMERA_ROTATE_180,
    )
    return detected_faces


def vision_worker():
    global ever_had_face, organic_wander_search, no_face_mode
    global running, target_x_off, target_y_off, target_rotation, target_squint
    global target_face_present, target_body_present, target_face_area_ratio, target_face_count
    global squint_until, latest_frame, servo_target_pan, servo_target_tilt, last_face_seen_ts, next_talk_saccade_ts
    global last_face_pan, last_face_tilt, last_face_norm_x, last_face_norm_y, no_face_mode
    global wander_search_phase
    global amplitude_fast, amplitude_slow, amplitude_prev_fast
    global udp_conv_state, udp_conv_emotion, udp_speak_pulse
    global _last_body_det, _last_body_det_ts, vision_frame_idx

    interval = 1.0 / max(1.0, float(VISION_FPS))
    next_tick = time.perf_counter()

    while running:
        try:
            vision_frame_idx += 1
            large_frame = picam2.capture_array()
            frame_raw, stream_frame = _prepare_vision_frames(large_frame)

            local_x = 0.0
            local_y = 0.0
            local_rot = 0.0
            local_squint = 0.0
            has_face = False
            has_body = False
            face_area_ratio = 0.0
            face_count = 0

            if frame_raw is not None and frame_raw.size > 0:
                detected_faces = _detect_faces_on_frame(frame_raw)

                if detected_faces is not None:
                    has_face = True
                    face_count = len(detected_faces)
                    largest_face = max(detected_faces, key=lambda f: f[2] * f[3])

                    fx, fy, fw, fh = largest_face[0:4]
                    re_x, re_y = largest_face[4], largest_face[5]
                    le_x, le_y = largest_face[6], largest_face[7]

                    if STREAM_ENABLED and stream_frame is not None:
                        fx_s, fy_s, fw_s, fh_s, re_x_s, re_y_s, le_x_s, le_y_s = map_coords_to_stream_preview(
                            fx, fy, fw, fh, re_x, re_y, le_x, le_y,
                        )
                        cv2.rectangle(
                            stream_frame,
                            (fx_s, fy_s),
                            (fx_s + fw_s, fy_s + fh_s),
                            _FACE_BOX_BGR,
                            _DEBUG_OVERLAY_THICKNESS,
                        )
                        cv2.circle(stream_frame, (re_x_s, re_y_s), 6, _FACE_EYE_BGR, -1)
                        cv2.circle(stream_frame, (le_x_s, le_y_s), 6, _FACE_EYE_BGR, -1)
                        cv2.putText(
                            stream_frame,
                            "FACE",
                            (fx_s, max(18, fy_s - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            _FACE_BOX_BGR,
                            2,
                            cv2.LINE_AA,
                        )

                    face_cx = (fx + fw / 2) / CAMERA_RES[0]
                    face_cy = (fy + fh / 2) / CAMERA_RES[1]

                    local_x, local_y, norm_x, norm_y, _, _ = _apply_detection_aim_point(
                        fx + fw / 2,
                        fy + fh / 2,
                        servo_alpha=FACE_TRACK_SERVO_ALPHA,
                    )

                    # Distance-based emotion: squint when far, excited when close
                    face_area_ratio = (fw * fh) / float(CAMERA_RES[0] * CAMERA_RES[1])
                    now = time.time()
                    
                    # Check for far-distance squinting
                    if face_area_ratio < FAR_FACE_AREA_RATIO:
                        if now > squint_until and random.random() < FAR_SQUINT_CHANCE:
                            squint_until = now + random.uniform(FAR_SQUINT_MIN_SEC, FAR_SQUINT_MAX_SEC)
                        if now < squint_until:
                            local_squint = 1.0
                    else:
                        squint_until = 0.0
                    
                    dx = re_x - le_x
                    dy = re_y - le_y
                    if dx != 0:
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        local_rot = max(-FACE_ROLL_MAX_DEG, min(FACE_ROLL_MAX_DEG, -angle_deg * FACE_ROLL_MULT))
                else:
                    squint_until = 0.0
                    if (
                        BODY_ENABLED
                        and person_detector is not None
                        and vision_frame_idx % max(1, BODY_DETECT_STRIDE) == 0
                    ):
                        body_bgr = to_detection_bgr(
                            frame_raw,
                            rotate_180=False if CAMERA_WIDE_FOV else CAMERA_ROTATE_180,
                        )
                        _last_body_det = person_detector.detect_largest(body_bgr)
                        _last_body_det_ts = time.time()

                    body_det = (
                        _last_body_det
                        if _last_body_det is not None
                        and (time.time() - _last_body_det_ts) < 0.75
                        else None
                    )
                    if body_det is not None:
                        has_body = True
                        bx, by, bw, bh = (
                            body_det.x,
                            body_det.y,
                            body_det.w,
                            body_det.h,
                        )
                        aim_cx = body_det.cx
                        aim_cy = body_det.aim_y(BODY_AIM_Y_RATIO)

                        if STREAM_ENABLED and stream_frame is not None:
                            bx_s, by_s, bw_s, bh_s, _, _, _, _ = map_coords_to_stream_preview(
                                int(bx),
                                int(by),
                                int(bw),
                                int(bh),
                                0,
                                0,
                                0,
                                0,
                            )
                            cv2.rectangle(
                                stream_frame,
                                (bx_s, by_s),
                                (bx_s + bw_s, by_s + bh_s),
                                _BODY_BOX_BGR,
                                _DEBUG_OVERLAY_THICKNESS,
                            )
                            ax_s, ay_s, _, _, _, _, _, _ = map_coords_to_stream_preview(
                                int(aim_cx),
                                int(aim_cy),
                                0,
                                0,
                                int(aim_cx),
                                int(aim_cy),
                                0,
                                0,
                            )
                            cv2.circle(stream_frame, (ax_s, ay_s), 6, _BODY_AIM_BGR, -1)
                            cv2.putText(
                                stream_frame,
                                "BODY",
                                (bx_s, max(18, by_s - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                _BODY_BOX_BGR,
                                2,
                                cv2.LINE_AA,
                            )

                        local_x, local_y, _, _, _, _ = _apply_detection_aim_point(
                            aim_cx,
                            aim_cy,
                            servo_alpha=BODY_TRACK_SERVO_ALPHA,
                        )
                        face_area_ratio = (bw * bh) / float(
                            CAMERA_RES[0] * CAMERA_RES[1]
                        )

                now_vis = time.time()
                if has_face or has_body:
                    last_face_seen_ts = now_vis
                    if has_face:
                        global ever_had_face
                        ever_had_face = True
                presence_locked = has_face or has_body or (
                    now_vis - last_face_seen_ts
                ) < FACE_PRESENT_HOLD_SEC
                face_locked = presence_locked

                if not face_locked:
                    if no_face_mode == "wandering":
                        # Eyes ride head sweep; stale last-face offsets fought pan/tilt motion.
                        local_x = 0.0
                        local_y = 0.0
                    elif no_face_mode == "sad_return":
                        local_x = NO_FACE_IDLE_EYE_X
                        local_y = NO_FACE_IDLE_EYE_Y
                    elif no_face_mode == "chat_ready":
                        local_x = 0.0
                        local_y = 0.0
                    else:
                        local_x = NO_FACE_IDLE_EYE_X
                        local_y = NO_FACE_IDLE_EYE_Y

                if ENABLE_SERVO and servo_driver is not None and not face_locked:
                    pan_center = (PAN_MIN + PAN_MAX) * 0.5
                    tilt_center = (TILT_MIN + TILT_MAX) * 0.5
                    pan_idle = clamp(
                        pan_center + NO_FACE_IDLE_PAN_DEG, PAN_MIN, PAN_MAX
                    )
                    tilt_idle = _tilt_down_from_center(NO_FACE_IDLE_TILT_DEG)
                    with tof_lock:
                        tof_drives_motion = (
                            TOF_APPROACH_ENABLED
                            and TOF_ENABLED
                            and tof_approach_controller.drives_motion()
                        )
                    with servo_state_lock:
                        if no_face_mode == "wandering" and not tof_drives_motion:
                            global organic_wander_search, search_tilt_from_eye_level_deg
                            global wander_pan_speed_scale
                            pan_current = servo_current_pan
                            wpan, wtilt = organic_wander_search.tick(
                                now_vis,
                                pan_center=pan_center,
                                tilt_center=tilt_center,
                                pan_current=pan_current,
                                pan_min=PAN_MIN,
                                pan_max=PAN_MAX,
                                tilt_min=TILT_MIN,
                                tilt_max=TILT_MAX,
                                amp_deg=WANDER_SEARCH_PAN_AMP_DEG,
                                step_min_deg=WANDER_SEARCH_PAN_STEP_MIN_DEG,
                                step_max_deg=WANDER_SEARCH_PAN_STEP_MAX_DEG,
                                hold_min_sec=WANDER_SEARCH_HOLD_MIN_SEC,
                                hold_max_sec=WANDER_SEARCH_HOLD_MAX_SEC,
                                jump_chance=WANDER_SEARCH_JUMP_CHANCE,
                                arrival_deg=WANDER_SEARCH_ARRIVAL_DEG,
                                tilt_max_up_deg=WANDER_SEARCH_TILT_MAX_UP_DEG,
                                tilt_max_down_deg=WANDER_SEARCH_TILT_MAX_DOWN_DEG,
                                thinking_hold_chance=WANDER_SEARCH_THINKING_HOLD_CHANCE,
                                thinking_hold_min_sec=WANDER_SEARCH_THINKING_HOLD_MIN_SEC,
                                thinking_hold_max_sec=WANDER_SEARCH_THINKING_HOLD_MAX_SEC,
                                long_stare_chance=WANDER_SEARCH_LONG_STARE_CHANCE,
                            )
                            search_tilt_from_eye_level_deg = wander_search_tilt_from_eye_level(
                                servo_target_tilt, tilt_center
                            )
                            wander_pan_speed_scale = organic_wander_search.move_speed_scale
                            speed = wander_pan_speed_scale
                            hold_scale = 0.45
                            if not organic_wander_search.moving:
                                if organic_wander_search.pause_kind == "thinking":
                                    hold_scale = 0.28
                                elif organic_wander_search.pause_kind == "long_stare":
                                    hold_scale = 0.32
                                elif organic_wander_search.pause_kind == "glance":
                                    hold_scale = 0.52
                            pan_alpha = WANDER_PAN_TARGET_ALPHA * speed * (
                                1.35 if organic_wander_search.moving else hold_scale
                            )
                            servo_target_pan += (wpan - servo_target_pan) * pan_alpha
                            servo_target_tilt += (
                                wtilt - servo_target_tilt
                            ) * WANDER_TILT_TARGET_ALPHA
                        elif no_face_mode == "sad_return":
                            servo_target_pan = servo_target_pan + (
                                pan_idle - servo_target_pan
                            ) * NO_FACE_SAD_RECENTER_ALPHA
                            servo_target_tilt = servo_target_tilt + (
                                tilt_idle - servo_target_tilt
                            ) * NO_FACE_SAD_RECENTER_ALPHA
                        elif no_face_mode == "chat_ready":
                            servo_target_pan = servo_target_pan + (
                                pan_center - servo_target_pan
                            ) * CHAT_READY_RECENTER_ALPHA
                            servo_target_tilt = servo_target_tilt + (
                                tilt_center - servo_target_tilt
                            ) * CHAT_READY_RECENTER_ALPHA
                        elif no_face_mode == "settled":
                            if now_vis - last_face_seen_ts > NO_FACE_RECENTER_SEC:
                                servo_target_pan = servo_target_pan + (
                                    pan_idle - servo_target_pan
                                ) * NO_FACE_RECENTER_ALPHA
                                servo_target_tilt = servo_target_tilt + (
                                    tilt_idle - servo_target_tilt
                                ) * NO_FACE_RECENTER_ALPHA

                with target_lock:
                    target_x_off = local_x
                    target_y_off = local_y
                    target_rotation = local_rot
                    target_squint = local_squint
                    target_face_present = presence_locked
                    target_body_present = has_body and not has_face
                    target_face_area_ratio = face_area_ratio
                    target_face_count = face_count

                if STREAM_ENABLED and stream_frame is not None:
                    with frame_lock:
                        latest_frame = stream_frame

        except Exception as e:
            print(f"Capture/Detect Error: {e}")

        next_tick += interval
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()


def _start_servo_hardware(driver) -> None:
    """Center head, start servo worker (+ ToF when enabled)."""
    global servo_driver, servo_running, servo_thread

    if driver is None or servo_running:
        return
    servo_driver = driver
    pan_c, tilt_c = head_center_angles()
    try:
        servo_driver.write_angles(pan_c, tilt_c, force=True)
    except TypeError:
        servo_driver.write_angles(pan_c, tilt_c)
    print(
        f"Head servos centered (ESP32): P={pan_c:.1f} T={tilt_c:.1f} "
        f"— face tracking enabled"
    )
    servo_running = True
    servo_thread = threading.Thread(target=servo_worker, daemon=True)
    servo_thread.start()
    if TOF_ENABLED:
        threading.Thread(target=tof_worker, daemon=True).start()
        print(f"ToF presence polling at {TOF_POLL_HZ:.1f} Hz (ESP32 F command)")
        if TOF_APPROACH_ENABLED:
            print(
                "ToF approach awareness enabled "
                f"(grace {TOF_APPROACH_STARTUP_GRACE_SEC:.1f}s, one-shot orient)"
            )


def _servo_connect_worker() -> None:
    """Background ESP32 connect when early boot or camera load delays READY."""
    global running
    delay_sec = 5.0
    while running and not servo_running:
        driver = create_servo_driver(cfg, max_attempts=2, retry_delay_sec=2.0)
        if driver is not None:
            _start_servo_hardware(driver)
            return
        time.sleep(delay_sec)


print("Starting face tracking...")
if ENABLE_SERVO:
    if _boot_servo_driver is not None:
        _start_servo_hardware(_boot_servo_driver)
    elif not servo_running:
        print(
            "Head servo driver unavailable at startup; "
            "retrying ESP32 connect in background..."
        )
        threading.Thread(target=_servo_connect_worker, daemon=True).start()

if STREAM_ENABLED:
    try:
        start_stream_server()
    except Exception as e:
        print(f"Error starting debug HTTP server: {e}")

vision_thread = threading.Thread(target=vision_worker, daemon=True)
vision_thread.start()
udp_thread = threading.Thread(target=udp_worker, daemon=True)
udp_thread.start()
threading.Thread(target=_boot_color_probe, daemon=True).start()
threading.Thread(target=_init_person_detector_background, daemon=True).start()
threading.Thread(target=_load_animations_background, daemon=True).start()
print("Face tracking active — eyes and servos follow camera")


def _handle_shutdown_signal(signum, frame):
    global running, servo_running
    running = False
    servo_running = False
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _handle_shutdown_signal)
signal.signal(signal.SIGTERM, _handle_shutdown_signal)

try:
    # ── Tracking & Emotion History (Pre-Initialization to prevent crashes) ──
    prev_face_area_ratio = 0.0
    face_count_history = []
    router_face_count_prev = 0
    router_face_present_prev = False
    router_multi_face_prev = False
    router_face_close = False
    face_present_since_ts = 0.0
    no_face_since_ts = time.time()
    no_face_scan_checks = 0
    
    multi_face_candidate = False
    multi_face_stable = False
    multi_face_candidate_since = 0.0
    
    no_face_blend_until = 0.0
    no_face_blend_queue = []
    no_face_blend_emotion = "idle"
    settled_solo_emotion = "sleepy"
    settled_emotion_until = 0.0
    
    side_dir_state = 0
    side_dir_last_switch_ts = 0.0
    side_look_active = False
    
    router_candidate_emotion = "idle"
    router_candidate_since = 0.0
    social_mode = "neutral"
    social_mode_until = 0.0
    
    smoothed_x_off = 0.0
    smoothed_y_off = 0.0
    smoothed_rotation = 0.0
    smoothed_head_eye_x = 0.0
    smoothed_head_eye_y = 0.0
    solo_mood = "neutral"
    solo_mood_until = 0.0
    face_acquire_until = 0.0
    prev_udp_conv_state = "waiting"
    speak_social_mode = "engaged"
    speak_social_until = 0.0

    print("🎨 Render Loop initialized. Starting displays...")
    
    while running:
        loop_start = time.perf_counter()
        now = time.time()
        
        # 1. Update Tracking Context
        with target_lock:
            local_target_x = target_x_off
            local_target_y = target_y_off
            local_target_rot = target_rotation
            local_target_squint = target_squint
            local_face_present = target_face_present
            local_face_area_ratio = target_face_area_ratio
            local_face_count = target_face_count
        
        area_ratio_delta = abs(local_face_area_ratio - prev_face_area_ratio)
        
        # Track face count changes over last 3 seconds
        if local_face_count != router_face_count_prev:
            face_count_history.append(now)
        face_count_history = [t for t in face_count_history if now - t < 3.0]
        face_count_changes = len(face_count_history)
        router_face_count_prev = local_face_count
        
        # Snapshot current for next frame
        prev_face_area_ratio = local_face_area_ratio

        # Debounced face lock (vision_worker hold) — not raw YuNet edges.
        face_entered = local_face_present and not router_face_present_prev
        face_lost = (not local_face_present) and router_face_present_prev

        if local_face_present:
            if pending_face_stable_since is None:
                pending_face_stable_since = now
        else:
            pending_face_stable_since = None
        face_stable_acquire = (
            local_face_present
            and pending_face_stable_since is not None
            and (now - pending_face_stable_since) >= FACE_STABLE_BEFORE_TRACK_SEC
        )
        face_stable_entered = face_stable_acquire and not router_face_stable_prev

        if face_stable_entered:
            face_present_since_ts = now
            _notify_voice_presence_arrival(now)
            face_acquire_until = now + FACE_ACQUIRE_SNAP_DURATION_SEC
            gaze_next_release_ts = now + random.uniform(GAZE_SOCIAL_RELEASE_MIN_SEC, GAZE_SOCIAL_RELEASE_MAX_SEC)
            gaze_next_scan_ts = now + FACE_SCAN_COOLDOWN_AFTER_LOCK_SEC
            gaze_next_allowed_ts = now
            no_face_blend_until = 0.0
            no_face_blend_queue = []
            smoothed_x_off = local_target_x
            smoothed_y_off = local_target_y
            smoothed_head_eye_x = 0.0
            smoothed_head_eye_y = 0.0
            no_face_mode = "tracking"
            wander_until = 0.0
            sad_return_until = 0.0
            sad_return_start = 0.0
            next_wander_peek_ts = 0.0
            wander_search_phase = 0.0
            wander_search_last_ts = now
            organic_wander_search.reset(
                (PAN_MIN + PAN_MAX) * 0.5, (TILT_MIN + TILT_MAX) * 0.5, now
            )
            search_tilt_from_eye_level_deg = 0.0
            chat_ready_until = 0.0
            wake_tilt_jerk_until = 0.0
            clear_gaze_aversion()
        elif not local_face_present:
            face_present_since_ts = None

        # Smooth tracking — fast catch-up right after face lock.
        if now < face_acquire_until:
            smooth_alpha = FACE_ACQUIRE_SNAP_ALPHA
        elif local_face_present:
            smooth_alpha = FACE_TRACK_SMOOTH_ALPHA
        else:
            smooth_alpha = FACE_TRACK_SMOOTH_ALPHA_IDLE
        smoothed_x_off = smoothed_x_off + (local_target_x - smoothed_x_off) * smooth_alpha
        smoothed_y_off = smoothed_y_off + (local_target_y - smoothed_y_off) * smooth_alpha
        smoothed_rotation = smoothed_rotation + (local_target_rot - smoothed_rotation) * smooth_alpha

        # 2. Update Eye Targets (gaze overrides are layered later)
        left_eye.target_pos[0] = left_eye.base_x + smoothed_x_off
        left_eye.target_pos[1] = left_eye.base_y + smoothed_y_off
        clamp_eye_target(left_eye)

        right_eye.target_pos[0] = left_eye.target_pos[0]
        right_eye.target_pos[1] = left_eye.target_pos[1]
        left_eye.target_rotation = 0.0
        right_eye.target_rotation = 0.0

        # Natural emotion routing with timing gates and hysteresis.

        if face_lost:
            clear_gaze_aversion()
            no_face_mode = "wandering"
            wander_until = now + NO_FACE_WANDER_SEC
            sad_return_until = 0.0
            sad_return_start = 0.0
            wander_search_phase = random.uniform(0.0, math.tau)
            wander_search_last_ts = now
            organic_wander_search.reset(
                (PAN_MIN + PAN_MAX) * 0.5, (TILT_MIN + TILT_MAX) * 0.5, now
            )
            search_tilt_from_eye_level_deg = 0.0
            next_wander_peek_ts = now + random.uniform(
                WANDER_PEEK_MIN_SEC, WANDER_PEEK_MAX_SEC
            )
            first_blend = weighted_pick([
                ("uncertain", 0.35),
                ("curious_intense", 0.25),
                ("warm", 0.20),
                ("attentive", 0.20),
            ])
            second_options = [
                e for e in ("uncertain", "curious_intense", "warm", "attentive") if e != first_blend
            ]
            no_face_blend_queue = [first_blend]
            if NO_FACE_IDLE_BLEND_STAGES >= 2 and second_options:
                no_face_blend_queue.append(random.choice(second_options))
            no_face_blend_emotion = no_face_blend_queue[0]
            no_face_blend_until = now + random.uniform(NO_FACE_IDLE_BLEND_MIN_SEC, NO_FACE_IDLE_BLEND_MAX_SEC)

        if (not local_face_present) and no_face_blend_queue and now >= no_face_blend_until:
            no_face_blend_queue.pop(0)
            if no_face_blend_queue:
                no_face_blend_emotion = no_face_blend_queue[0]
                no_face_blend_until = now + random.uniform(NO_FACE_IDLE_BLEND_MIN_SEC, NO_FACE_IDLE_BLEND_MAX_SEC)
            else:
                no_face_blend_until = 0.0

        # Debounce multi-face state to avoid flicker from detector instability.
        multi_face_raw = local_face_count >= 2
        if multi_face_raw != multi_face_candidate:
            multi_face_candidate = multi_face_raw
            multi_face_candidate_since = now
        if (
            multi_face_stable != multi_face_candidate
            and (now - multi_face_candidate_since) >= MULTI_FACE_DEBOUNCE_SEC
        ):
            multi_face_stable = multi_face_candidate
        multi_face_entered = multi_face_stable and not router_multi_face_prev

        # Side-look hysteresis and direction cooldown reduce left/right chatter.
        pan_center = (PAN_MIN + PAN_MAX) * 0.5
        with servo_state_lock:
            render_pan_cur = servo_current_pan
        head_pan_deg_render = render_pan_cur - pan_center
        head_eye_preview_x = (
            head_pan_deg_render
            * HEAD_PAN_PX_PER_DEG
            * EYE_HEAD_RATIO_WANDER
            * HEAD_EYE_PAN_SIGN
        )
        if no_face_mode == "wandering" and not gaze_event_active:
            side_look_source_x = head_eye_preview_x
            next_side_dir = side_dir_state
            if side_dir_state == 0:
                if head_pan_deg_render >= WANDER_SIDE_LOOK_PAN_DEG:
                    next_side_dir = 1
                elif head_pan_deg_render <= -WANDER_SIDE_LOOK_PAN_DEG:
                    next_side_dir = -1
            else:
                if abs(head_pan_deg_render) <= WANDER_SIDE_LOOK_PAN_DEG * 0.45:
                    next_side_dir = 0
                elif head_pan_deg_render >= WANDER_SIDE_LOOK_PAN_DEG:
                    next_side_dir = 1
                elif head_pan_deg_render <= -WANDER_SIDE_LOOK_PAN_DEG:
                    next_side_dir = -1
        else:
            side_look_source_x = smoothed_x_off
            abs_x = abs(side_look_source_x)
            next_side_dir = side_dir_state
            if side_dir_state == 0:
                if abs_x >= SIDE_LOOK_ENTER_OFFSET:
                    next_side_dir = 1 if side_look_source_x >= 0 else -1
            else:
                if abs_x <= SIDE_LOOK_EXIT_OFFSET:
                    next_side_dir = 0
                else:
                    candidate_dir = 1 if side_look_source_x >= 0 else -1
                    if (
                        candidate_dir != side_dir_state
                        and abs_x >= SIDE_LOOK_ENTER_OFFSET
                        and (now - side_dir_last_switch_ts) >= SIDE_LOOK_SWITCH_COOLDOWN_SEC
                    ):
                        next_side_dir = candidate_dir
        if next_side_dir != side_dir_state:
            side_dir_state = next_side_dir
            side_dir_last_switch_ts = now

        face_tracking_active = local_face_present

        if local_face_present:
            no_face_since_ts = now
            no_face_scan_checks = 0
            if (not router_face_close) and (local_face_area_ratio >= CLOSE_FACE_ENTER_RATIO):
                router_face_close = True
            elif router_face_close and (local_face_area_ratio < CLOSE_FACE_EXIT_RATIO):
                router_face_close = False
        else:
            router_face_close = False

        should_squint = local_target_squint > 0.5
        side_look_active = side_dir_state != 0
        side_right = side_dir_state >= 0
        target_emotion_raw = "idle"
        emotion_trace = []
        upbeat = is_upbeat_session()
        speak_da = amplitude_fast - amplitude_prev_fast
        use_surroundings_emotions = not voice_emotion_active(now)

        if use_surroundings_emotions:
            dx_activity = abs(local_target_x - prev_surroundings_x) / max(1.0, float(MAX_X_OFFSET))
            dy_activity = abs(local_target_y - prev_surroundings_y) / max(1.0, float(MAX_Y_OFFSET))
            dr_activity = abs(local_target_rot - prev_surroundings_rot) / max(1.0, float(FACE_ROLL_MAX_DEG))
            surroundings_activity = min(1.0, (dx_activity + dy_activity + dr_activity) / 3.0)
            prev_surroundings_x = local_target_x
            prev_surroundings_y = local_target_y
            prev_surroundings_rot = local_target_rot

            if no_face_mode == "sad_return" and not local_face_present:
                target_emotion_raw = "sad"
            elif no_face_mode == "wandering" and gaze_event_active and scan_emotion_override:
                target_emotion_raw = scan_emotion_override
            else:
                surroundings_pick = surroundings_controller.tick(
                    now=now,
                    face_detected=local_face_present,
                    face_area_ratio=local_face_area_ratio,
                    face_norm_x=last_face_norm_x,
                    squint_hint=1.0 if should_squint else 0.0,
                    activity=surroundings_activity,
                    wander_mode=no_face_mode == "wandering",
                )
                if surroundings_pick:
                    target_emotion_raw = surroundings_pick
                elif local_face_present:
                    target_emotion_raw = surroundings_controller.current_emotion or FACE_TRACK_DEFAULT
                elif no_face_mode == "settled":
                    target_emotion_raw = settled_solo_emotion
                else:
                    target_emotion_raw = surroundings_controller.current_emotion or "idle"
            trace_emotion("surroundings", target_emotion_raw)

        # Pick a short-lived social mode to keep expressions varied and lifelike.
        if not use_surroundings_emotions and now >= social_mode_until:
            if local_face_present:
                if router_face_close:
                    social_mode = weighted_pick([
                        ("engaged", 0.30),
                        ("warm", 0.25),
                        ("amused", 0.20),
                        ("cheerful", 0.15),
                        ("content", 0.10),
                    ])
                elif multi_face_stable:
                    social_mode = weighted_pick([
                        ("engaged", 0.35),
                        ("thinking", 0.30),
                        ("attentive", 0.25),
                        ("curious_intense", 0.10),
                    ])
                elif local_face_area_ratio < FAR_FACE_AREA_RATIO:
                    social_mode = weighted_pick([
                        ("curious_intense", 0.40),
                        ("attentive", 0.35),
                        ("engaged", 0.15),
                        ("squint", 0.10),
                    ])
                else:
                    social_mode = weighted_pick([
                        ("attentive", 0.35),
                        ("engaged", 0.25),
                        ("warm", 0.20),
                        ("content", 0.20),
                    ])
            elif no_face_mode == "wandering":
                if upbeat:
                    social_mode = weighted_pick([
                        ("curious_intense", 0.30),
                        ("cheerful", 0.25),
                        ("attentive", 0.20),
                        ("warm", 0.15),
                        ("engaged", 0.10),
                    ])
                else:
                    social_mode = weighted_pick([
                        ("thinking", 0.28),
                        ("concentrating", 0.22),
                        ("uncertain", 0.22),
                        ("curious_intense", 0.15),
                        ("attentive", 0.13),
                    ])
            elif upbeat:
                social_mode = pick_upbeat_solo_emotion()
            else:
                social_mode = weighted_pick([
                    ("neutral", 0.70),
                    ("curious", 0.30),
                ])
            social_mode_until = now + random.uniform(SOCIAL_MODE_MIN_SEC, SOCIAL_MODE_MAX_SEC)

        if not use_surroundings_emotions and not local_face_present:
            no_face_elapsed_solo = now - no_face_since_ts
            block_solo_upbeat = (
                no_face_mode in ("wandering", "sad_return", "settled")
                and not session_active
            )
            if (
                not block_solo_upbeat
                and no_face_elapsed_solo >= SOLO_UPBEAT_MIN_SEC
                and now >= solo_mood_until
            ):
                solo_mood = weighted_pick([
                    ("cheerful", 0.40),
                    ("content", 0.30),
                    ("playful", 0.20),
                    ("warm", 0.10),
                ])
                solo_mood_until = now + random.uniform(SOCIAL_MODE_MIN_SEC, SOCIAL_MODE_MAX_SEC)

        if not use_surroundings_emotions and local_face_present:
            if should_squint:
                target_emotion_raw = "squint"
            elif router_face_close:
                target_emotion_raw = social_mode if social_mode in FACE_TRACK_EMOTIONS else "engaged"
            elif multi_face_stable:
                target_emotion_raw = social_mode if social_mode in FACE_TRACK_EMOTIONS else "engaged"
            elif local_face_area_ratio < FAR_FACE_AREA_RATIO:
                target_emotion_raw = (
                    social_mode if social_mode in FACE_TRACK_EMOTIONS else "curious_intense"
                )
            elif social_mode in FACE_TRACK_EMOTIONS:
                target_emotion_raw = social_mode
            else:
                target_emotion_raw = FACE_TRACK_DEFAULT
        elif not use_surroundings_emotions:
            no_face_elapsed = now - no_face_since_ts
            if no_face_mode == "sad_return":
                target_emotion_raw = "sad"
            elif no_face_mode == "wandering":
                if gaze_event_active and gaze_state == "AVERT_SCAN" and scan_emotion_override:
                    target_emotion_raw = scan_emotion_override
                elif abs(head_pan_deg_render) >= WANDER_SIDE_LOOK_PAN_DEG:
                    target_emotion_raw = (
                        "looking_right_natural"
                        if head_pan_deg_render > 0
                        else "looking_left_natural"
                    )
                elif (
                    not organic_wander_search.moving
                    and organic_wander_search.hold_emotion_hint in WANDER_EMOTIONS
                ):
                    target_emotion_raw = organic_wander_search.hold_emotion_hint
                elif social_mode in WANDER_EMOTIONS:
                    target_emotion_raw = social_mode
                else:
                    target_emotion_raw = "curious_intense"
            elif no_face_mode == "settled" and not session_active:
                if now >= settled_emotion_until:
                    settled_solo_emotion = (
                        "idle" if random.random() < 0.25 else "sleepy"
                    )
                    settled_emotion_until = now + random.uniform(
                        SETTLED_SLEEPY_VARIETY_MIN_SEC,
                        SETTLED_SLEEPY_VARIETY_MAX_SEC,
                    )
                target_emotion_raw = settled_solo_emotion
            elif gaze_event_active and gaze_state == "AVERT_SCAN" and scan_emotion_override:
                target_emotion_raw = scan_emotion_override
            elif no_face_blend_queue and now < no_face_blend_until:
                target_emotion_raw = no_face_blend_emotion
            elif upbeat and no_face_mode not in ("wandering", "sad_return", "settled"):
                if no_face_elapsed >= NO_FACE_BORED_SEC:
                    target_emotion_raw = pick_upbeat_solo_emotion()
                elif no_face_elapsed >= SOLO_UPBEAT_MIN_SEC and solo_mood in SOLO_MOOD_TO_EMOTION:
                    target_emotion_raw = SOLO_MOOD_TO_EMOTION[solo_mood]
                else:
                    target_emotion_raw = pick_upbeat_solo_emotion()
            elif no_face_elapsed >= NO_FACE_BORED_SEC:
                target_emotion_raw = "warm"
            elif (
                no_face_elapsed >= NO_FACE_SLEEPY_SEC
                and no_face_scan_checks >= NO_FACE_SEARCH_MIN_SCANS
            ):
                target_emotion_raw = "sleepy"
            elif (
                no_face_mode not in ("wandering", "sad_return", "settled")
                and no_face_elapsed >= SOLO_UPBEAT_MIN_SEC
                and solo_mood in SOLO_MOOD_TO_EMOTION
            ):
                target_emotion_raw = SOLO_MOOD_TO_EMOTION[solo_mood]
            else:
                target_emotion_raw = "idle"
        if not use_surroundings_emotions:
            trace_emotion("router_raw", target_emotion_raw)

        # Avoid repetitive smiling streaks by enforcing a happy cooldown.
        if (
            target_emotion_raw in ("happy", "looking_left_happy", "looking_right_happy")
            and (now - last_happy_ts) < HAPPY_MIN_GAP_SEC
        ):
            if side_look_active:
                target_emotion_raw = "looking_right_natural" if side_right else "looking_left_natural"
            else:
                target_emotion_raw = "warm"

        # ── LAYER 4: VISION-BASED TRIGGERS ────────────────────────────────────
        # Surprised: large area jump only (not on first face frame — avoids smile-like burst)
        area_delta = abs(local_face_area_ratio - prev_face_area_ratio)
        if (not face_entered) and area_delta > 0.15:
            target_emotion_raw = "surprised"
            emotion_force_until = now + 1.5
            trace_emotion("vision_surprised", target_emotion_raw)

        # Suspicious: unstable face count when not actively tracking one person
        if (not local_face_present) and face_count_changes > 3:
            target_emotion_raw = "suspicious"
            emotion_force_until = now + 2.0
            trace_emotion("vision_suspicious", target_emotion_raw)

        # Playful: Talking but no face is here (skip lazy playful when upbeat)
        if not local_face_present and udp_speak_pulse > 0.1 and not upbeat:
            target_emotion_raw = "playful"
            trace_emotion("vision_playful", target_emotion_raw)
        # ─────────────────────────────────────────────────────────────────────

        # Debounce route output
        if target_emotion_raw != router_candidate_emotion:
            router_candidate_emotion = target_emotion_raw
            router_candidate_since = now
        if (now - router_candidate_since) >= ROUTER_EMOTION_STABLE_SEC:
            target_emotion = router_candidate_emotion
        else:
            target_emotion = current_emotion
        trace_emotion("debounced", target_emotion)

        # ── 3-LAYER EMOTION PRIORITY ──────────────────────────────────────────
        # Layer 1 (VADER) overrides the face-tracker baseline
        if udp_emotion_override and now < udp_emotion_until:
            target_emotion = udp_emotion_override
            trace_emotion("layer1_vader", target_emotion)

        # Layer 2 (Conversation state) overrides everything when robot is active
        # High priority "handshake" states
        if udp_conv_state in ("listening", "thinking", "nodding", "remembering", "concentrating"):
            conv_emo = udp_conv_emotion
            if upbeat and conv_emo in LAZY_EMOTIONS:
                conv_emo = "attentive"
            target_emotion = conv_emo
            trace_emotion("layer2_conv", target_emotion)
        
        # Speaking: varied expressions driven by amplitude + rotation
        elif udp_conv_state == "speaking":
            if udp_emotion_override and now < udp_emotion_until:
                target_emotion = udp_emotion_override
                trace_emotion("layer1_vader_speak", target_emotion)
            elif amplitude_slow > 0.5:
                target_emotion = weighted_pick([("excited", 0.55), ("happy", 0.45)])
                trace_emotion("layer2_speak_slow", target_emotion)
            elif speak_da > 0.08:
                target_emotion = weighted_pick([("amused", 0.55), ("cheerful", 0.45)])
                trace_emotion("layer2_speak_spike", target_emotion)
            elif local_face_present and target_emotion in FACE_TRACK_EMOTIONS:
                trace_emotion("layer2_speak_face_track", target_emotion)
            else:
                if now >= speak_social_until:
                    speak_social_mode = pick_speak_emotion()
                    speak_social_until = now + random.uniform(
                        SPEAK_SOCIAL_MIN_SEC, SPEAK_SOCIAL_MAX_SEC
                    )
                target_emotion = speak_social_mode
                trace_emotion("layer2_speak_rotate", target_emotion)
        
        # Waiting/Idle: VADER mood holds, but conv-state can trigger awkward
        elif udp_conv_state == "waiting":
            if udp_conv_emotion == "awkward" and upbeat:
                target_emotion = "cheerful"
                trace_emotion("layer2_waiting_upbeat", target_emotion)
            elif udp_conv_emotion == "awkward":
                target_emotion = "awkward"
                trace_emotion("layer2_waiting_awkward", target_emotion)
            elif udp_emotion_override and now < udp_emotion_until:
                target_emotion = udp_emotion_override
                trace_emotion("layer1_vader_wait", target_emotion)
            elif upbeat:
                target_emotion = pick_upbeat_solo_emotion()
                trace_emotion("layer2_waiting_upbeat_default", target_emotion)
        # ─────────────────────────────────────────────────────────────────────

        if no_face_mode == "sad_return" and not local_face_present:
            target_emotion = "sad"
            trace_emotion("override_sad", target_emotion)

        if now < emotion_force_until and not local_face_present and no_face_mode == "chat_ready":
            target_emotion = "surprised"
            trace_emotion("override_wake_surprise", target_emotion)

        alone_no_face_idle = (
            not local_face_present
            and no_face_mode in ("wandering", "sad_return", "settled")
            and not is_upbeat_session()
        )
        if upbeat and target_emotion in LAZY_EMOTIONS and not alone_no_face_idle:
            target_emotion = pick_upbeat_solo_emotion()
            trace_emotion("upbeat_block_lazy", target_emotion)

        trace_emotion("final", target_emotion)

        if (
            multi_face_entered
            and now >= jerk_cooldown_until
            and not local_face_present
        ):
            jerk_direction = -1.0 if smoothed_x_off < 0 else 1.0
            jerk_until = now + JERK_DURATION
            jerk_cooldown_until = now + JERK_COOLDOWN_SEC

        # Debug: Show emotion state at each frame
        if DEBUG_EMOTIONS:
            state_info = (
                f"[{target_emotion:20}] face_present={local_face_present} faces={local_face_count} "
                f"area_ratio={local_face_area_ratio:.3f} close={router_face_close} side={side_dir_state} mode={social_mode}"
            )
            if local_face_present:
                burst_remaining = max(0, emotion_force_until - now)
                state_info += f" burst_rem={burst_remaining:.2f}s"
            else:
                no_face_elapsed = now - no_face_since_ts
                scan_eta = max(0.0, gaze_next_scan_ts - now)
                state_info += (
                    f" no_face_elapsed={no_face_elapsed:.2f}s"
                    f" scan_checks={no_face_scan_checks}/{NO_FACE_SEARCH_MIN_SCANS}"
                    f" scan_eta={scan_eta:.2f}s"
                )
            print(state_info)

        if DEBUG_EMOTION_REASON and (now - _last_emotion_debug_ts) >= 5.0:
            _last_emotion_debug_ts = now
            why = emotion_trace[-1] if emotion_trace else "n/a"
            print(
                f"  EMOTION_HEARTBEAT: {current_emotion:12} | last_step={why} "
                f"| mode={no_face_mode} session={int(session_active)} "
                f"conv={udp_conv_state} face={int(local_face_present)}"
            )

        if target_emotion != current_emotion:
            immediate_excited = face_entered and target_emotion == "excited"
            min_hold = (
                EMOTION_SPEAK_HOLD_SEC
                if udp_conv_state == "speaking"
                else EMOTION_MIN_HOLD_SEC
            )
            hold_ok = (now - emotion_last_switch_ts) >= min_hold
            cooldown_ok = (now - emotion_last_normal_switch_ts) >= EMOTION_SWITCH_COOLDOWN_SEC
            prev_emotion = current_emotion
            if immediate_excited or (hold_ok and cooldown_ok):
                target_intensity = EMOTION_INTENSITY.get(target_emotion, 0.55)
                if local_face_present and target_emotion in FACE_TRACK_EMOTIONS:
                    target_intensity = min(target_intensity, FACE_TRACK_INTENSITY)
                left_eye.set_emotion(target_emotion, target_intensity)
                current_emotion = target_emotion
                if current_emotion in ("happy", "looking_left_happy", "looking_right_happy"):
                    last_happy_ts = now
                emotion_last_switch_ts = now
                if not immediate_excited:
                    emotion_last_normal_switch_ts = now

                if DEBUG_EMOTION_REASON or DEBUG_EMOTIONS:
                    why = " | ".join(emotion_trace) if emotion_trace else "unknown"
                    print(
                        f"  EMOTION: {prev_emotion:12} -> {current_emotion:12} | why: {why} "
                        f"| mode={no_face_mode} session={int(session_active)} "
                        f"conv={udp_conv_state} face={int(local_face_present)} "
                        f"af={amplitude_fast:.3f}"
                    )
                    if DEBUG_EMOTIONS and current_emotion.startswith("looking_left"):
                        print(
                            f"  ↺ LOOK_DIR: LEFT  state={gaze_state} x_off={effective_x_off:.2f} y_off={effective_y_off:.2f}"
                        )
                    elif DEBUG_EMOTIONS and current_emotion.startswith("looking_right"):
                        print(
                            f"  ↻ LOOK_DIR: RIGHT state={gaze_state} x_off={effective_x_off:.2f} y_off={effective_y_off:.2f}"
                        )

        router_face_present_prev = local_face_present
        router_face_stable_prev = face_stable_acquire
        router_multi_face_prev = multi_face_stable

        # No-face FSM: search -> sad nod -> sleepy settled.
        if not local_face_present:
            if (
                no_face_mode == "tracking"
                and (now - no_face_since_ts) >= FACE_ABSENT_BEFORE_SCAN_SEC
            ):
                no_face_mode = "wandering"
                wander_until = now + NO_FACE_WANDER_SEC
                sad_return_until = 0.0
                sad_return_start = 0.0
                next_wander_peek_ts = now + random.uniform(
                    WANDER_PEEK_MIN_SEC, WANDER_PEEK_MAX_SEC
                )
                wander_search_phase = random.uniform(0.0, math.pi * 2.0)
                wander_search_last_ts = now
                organic_wander_search.reset(
                    (PAN_MIN + PAN_MAX) * 0.5, (TILT_MIN + TILT_MAX) * 0.5, now
                )
                search_tilt_from_eye_level_deg = 0.0
            if no_face_mode == "wandering":
                if now >= wander_until:
                    no_face_mode = "sad_return"
                    sad_return_start = now
                    sad_return_until = now + SAD_RETURN_SEC
                    clear_gaze_aversion()
                elif (
                    now >= next_wander_peek_ts
                    and not gaze_event_active
                    and not (udp_speak_pulse > 0.0 or udp_conv_state == "speaking")
                ):
                    if random.random() < WANDER_PEEK_CHANCE:
                        start_wander_peek()
                    next_wander_peek_ts = now + random.uniform(
                        WANDER_PEEK_MIN_SEC, WANDER_PEEK_MAX_SEC
                    )
            elif no_face_mode == "sad_return" and now >= sad_return_until:
                no_face_mode = "settled"
                settled_solo_emotion = "sleepy"
                settled_emotion_until = now + random.uniform(
                    SETTLED_SLEEPY_VARIETY_MIN_SEC,
                    SETTLED_SLEEPY_VARIETY_MAX_SEC,
                )
            elif (
                no_face_mode == "chat_ready"
                and now >= chat_ready_until
                and udp_conv_state in ("waiting", "awkward")
            ):
                no_face_mode = "settled"

        conv_wake_edge = (
            no_face_mode == "settled"
            and not local_face_present
            and now >= jerk_cooldown_until
            and udp_conv_state in AWAKE_CONV_ACTIVE
            and prev_udp_conv_state in AWAKE_CONV_PREV
        )
        wake_udp_edge = (
            no_face_mode == "settled"
            and not local_face_present
            and now >= jerk_cooldown_until
            and wake_request_ts > 0.0
            and (now - wake_request_ts) < 0.5
        )
        if conv_wake_edge or wake_udp_edge:
            trigger_settled_wake(now)
            wake_request_ts = 0.0

        # Gaze aversion: cancel search before tick so face lock never gets scan offsets.
        if local_face_present:
            clear_gaze_aversion()
        update_gaze_manager(now)
        if no_face_scan_completed_pulse:
            no_face_scan_completed_pulse = False

        prev_udp_conv_state = udp_conv_state

        pan_center = (PAN_MIN + PAN_MAX) * 0.5
        tilt_center = (TILT_MIN + TILT_MAX) * 0.5
        with servo_state_lock:
            pan_cur = servo_current_pan
            tilt_cur = servo_current_tilt
        head_pan_deg = pan_cur - pan_center
        head_tilt_deg = tilt_cur - tilt_center
        if no_face_mode == "wandering" and not gaze_event_active:
            head_ratio = EYE_HEAD_RATIO_WANDER
            head_smooth_alpha = min(0.38, EYE_HEAD_SMOOTH_ALPHA * 2.4)
        elif local_face_present:
            head_ratio = EYE_HEAD_RATIO_FACE
            head_smooth_alpha = EYE_HEAD_SMOOTH_ALPHA
        else:
            head_ratio = EYE_HEAD_RATIO
            head_smooth_alpha = EYE_HEAD_SMOOTH_ALPHA
        raw_head_eye_x = (
            head_pan_deg * HEAD_PAN_PX_PER_DEG * head_ratio * HEAD_EYE_PAN_SIGN
        )
        raw_head_eye_y = (
            head_tilt_deg * HEAD_TILT_PX_PER_DEG * head_ratio * HEAD_EYE_TILT_SIGN
        )
        smoothed_head_eye_x += (raw_head_eye_x - smoothed_head_eye_x) * head_smooth_alpha
        smoothed_head_eye_y += (raw_head_eye_y - smoothed_head_eye_y) * head_smooth_alpha

        # Gaze peeks already layer eye + servo offsets; skip head coupling to avoid doubling.
        if gaze_event_active:
            head_eye_x = 0.0
            head_eye_y = 0.0
        else:
            head_eye_x = smoothed_head_eye_x
            head_eye_y = smoothed_head_eye_y
        effective_x_off = smoothed_x_off + gaze_override_x + head_eye_x
        effective_y_off = smoothed_y_off + gaze_override_y + head_eye_y
        left_eye.target_pos[0] = left_eye.base_x + effective_x_off
        left_eye.target_pos[1] = left_eye.base_y + effective_y_off
        clamp_eye_target(left_eye)
        right_eye.target_pos[0] = left_eye.target_pos[0]
        right_eye.target_pos[1] = left_eye.target_pos[1]
        
        # 3. Blink Logic
        # If talking, force earlier blinks but don't overdo it!
        if udp_speak_pulse > 0.0 and (next_blink_time - time.time()) > 4.5:
             next_blink_time = time.time() + random.uniform(1.0, 2.5)
        if time.time() > next_blink_time:
            blink_speed = random.uniform(BLINK_SPEED_MIN, BLINK_SPEED_MAX)
            trigger_synced_blink(blink_speed)
            last_blink_time = time.time()
            next_blink_time = time.time() + random.uniform(2.5 if udp_speak_pulse > 0.0 else 3.5, 5.0 if udp_speak_pulse > 0.0 else 7.0)

        # 4. SACCADE FREQUENCY + AMPLITUDE SNAPSHOT
        #    Compute af/sl/da first — used by saccade AND physics blocks below
        af = amplitude_fast
        sl = amplitude_slow
        da = af - amplitude_prev_fast
        amplitude_prev_fast = af

        # AVERT_TALK removed: pan/eye saccades during speech caused unrealistic sweeping.
        # Amplitude-driven eye scale and tilt-only servo nods handle talk sync.

        # Keep idle motion deterministic to avoid perceived micro-jitter.
        
        # 5. Physics Update
        # Drive one master eye and mirror full state every frame for strict sync.
        reengage_bump = 0.0
        if now < gaze_reengage_until:
            phase = (gaze_reengage_until - now) / 0.28
            reengage_bump = max(0.0, min(1.0, phase)) * 0.035
            left_eye.target_scale_w += reengage_bump
            left_eye.target_scale_h += reengage_bump * 0.70
            
        # ── AMPLITUDE-DRIVEN BEHAVIOURS ─────────────────────────────────────
        # af/sl/da already computed above in section 4

        # 1. VERTICAL FLOAT & HORIZONTAL DRIFT — only when not face-tracking
        float_y = 0.0
        drift_x = 0.0
        jitter_x = 0.0
        punch = 0.0
        droop = 0.0
        head_motion_idle = no_face_mode in ("wandering", "sad_return", "chat_ready")
        if not local_face_present and not head_motion_idle:
            float_y = -sl * 35.0
            drift_x = math.sin(now * 4.5) * (sl * 15.0)
            left_eye.target_pos[0] += drift_x
            left_eye.target_pos[1] -= float_y
            jitter_x = da * 14.0
            left_eye.target_pos[0] += jitter_x

        # 2. SYLLABLE PUNCH — no-face full strength; face-present scaled down
        talk_active = udp_speak_pulse > 0.0 or udp_conv_state == "speaking"
        if af > 0.05:
            if da > 0.08:
                punch = da * 0.85
            else:
                punch = af * 0.40
            if local_face_present and talk_active:
                punch *= FACE_TALK_PUNCH_SCALE
            if (not local_face_present) or (local_face_present and talk_active and af > FACE_TALK_AF_THRESH):
                left_eye.target_scale_w += punch * 0.10
                left_eye.target_scale_h += punch

        # 2b. Face-present talk: tiny vertical bounce (no horizontal drift)
        talk_bounce_y = 0.0
        if local_face_present and talk_active and af > FACE_TALK_AF_THRESH:
            talk_bounce_y = af * 2.5
            left_eye.target_pos[1] -= talk_bounce_y

        # 3. LID MICRO-DROOP — only when not face-tracking
        if (
            not local_face_present
            and udp_speak_pulse > 0.0
            and af < 0.025
            and sl > 0.015
        ):
            droop = 0.06
            left_eye.target_scale_h -= droop

        left_eye.update()

        # Clean up temporary target mutations so they don't accumulate
        if af > 0.05:
            applied_punch = (not local_face_present) or (
                local_face_present and talk_active and af > FACE_TALK_AF_THRESH
            )
            if applied_punch:
                left_eye.target_scale_w -= punch * 0.10
                left_eye.target_scale_h -= punch
        if local_face_present and talk_active and af > FACE_TALK_AF_THRESH:
            left_eye.target_pos[1] += talk_bounce_y
        if (
            not local_face_present
            and udp_speak_pulse > 0.0
            and af < 0.025
            and sl > 0.015
        ):
            left_eye.target_scale_h += droop
        left_eye.target_pos[1] += float_y   # undo float before mirror

        if DEBUG_AMPLITUDE and talk_active and (now - _last_amplitude_debug_ts) >= 1.0:
            _last_amplitude_debug_ts = now
            print(
                f"  AMP face={local_face_present} af={af:.3f} sl={sl:.3f} "
                f"pulse={udp_speak_pulse:.1f} conv={udp_conv_state}"
            )
        # ────────────────────────────────────────────────────────────────────
        if reengage_bump > 0.0:
            left_eye.target_scale_w -= reengage_bump
            left_eye.target_scale_h -= reengage_bump * 0.70
        mirror_full_state(left_eye, right_eye)
        
        # 6. Draw
        shared_rgb = None
        if disp_l or disp_r:
            # Render once and present the exact same frame on both displays.
            img = Image.new("RGBA", (SCREEN_WIDTH, SCREEN_HEIGHT), BG_COLOR)
            left_eye.draw(img)
            shared_rgb = img.convert("RGB")

        try:
            if disp_l and shared_rgb is not None:
                disp_l.image(shared_rgb)
            if disp_r and shared_rgb is not None:
                disp_r.image(shared_rgb)
        except Exception as e:
            print(f"Display update error: {e}")

        frame_budget = (1.0 / max(1.0, float(RENDER_FPS))) - (time.perf_counter() - loop_start)
        if frame_budget > 0:
            time.sleep(frame_budget)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    running = False
    servo_running = False
    if vision_thread.is_alive():
        vision_thread.join(timeout=1.0)

    if servo_thread and servo_thread.is_alive():
        servo_thread.join(timeout=0.5)

    _home_servos_on_shutdown()

    # Cleanup attributes
    try:
        if picam2:
            picam2.stop()
            picam2.close()
            print("Camera closed.")
    except Exception as e:
        print(e)

    if stream_server:
        try:
            stream_server.shutdown()
            stream_server.server_close()
            print("MJPEG stream stopped.")
        except Exception as e:
            print(e)
        
    # Clear screens
    black = Image.new("RGB", (SCREEN_WIDTH, SCREEN_HEIGHT), (0, 0, 0))
    if disp_l: disp_l.image(black)
    if disp_r: disp_r.image(black)
    print("Displays cleared.")