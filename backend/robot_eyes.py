#!/usr/bin/env python3
"""
Face Tracking Eyes for Dual SPI Displays (Picamera2)
Combines face tracking (YuNet) with dual SPI display output (ST7735).

Optional: ServoKit pan/tilt face following (non-PID smoothing).
"""

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

udp_emotion_override = None
udp_emotion_until = 0.0
udp_speak_pulse = 0.0

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
    from adafruit_servokit import ServoKit
except ImportError:
    ServoKit = None


# --- Configuration ---
SCREEN_WIDTH = 128
SCREEN_HEIGHT = 160
EYE_COLOR = (255, 255, 255) # White
BG_COLOR = (0, 0, 0)      # Black
EYE_SIZE = 126
FLOOR_Y = SCREEN_HEIGHT - 5

# Camera / Face Tracking Config
FACE_MODEL_PATH = "face_detection_yunet_2023mar.onnx"
# Use a larger 16:9 main stream for wider/detail-rich source frames (wider field of view)
CAMERA_MAIN_RES = (1920, 1080)
# Balanced 16:9 processing for detail + CPU headroom
CAMERA_RES = (640, 360)
STREAM_RES = (320, 180)   # Downscaled for web preview (maintain 16:9, no lag)
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
# Camera adjustments
CAMERA_ROTATE_180 = True
# If stream colors look wrong, swap R/B for MJPEG output
STREAM_SWAP_RB = True

# Eye Interaction Config
MAX_X_OFFSET = 30
MAX_Y_OFFSET = 22
# Disable roll-driven eye tilt so the face doesn't appear to rotate.
FACE_ROLL_MULT = 0.0
FACE_ROLL_MAX_DEG = 10.0
EYE_BOUND_MARGIN = 8
MIN_EYE_SCALE = 0.72
MAX_EYE_SCALE = 1.28
MAX_TOP_LID = 0.90
MAX_BOTTOM_LID = 0.82
EYE_MOVE_FOOTPRINT_X = 0.34
EYE_MOVE_FOOTPRINT_Y = 0.36
EYE_RENDER_PAD_X = 4.0
EYE_RENDER_PAD_Y = 6.0
# Allow slightly more motion than the final visible render clamp so the eyes
# still drift naturally without ever drawing outside the panel.
EYE_MOTION_CLAMP_SCALE = 0.82

# Blink Speed (Higher = Faster)
BLINK_SPEED_MIN = 3.2
BLINK_SPEED_MAX = 4.2
LOOK_SIDE_OFFSET = 16.0

# Debug Mode - set to True to see emotion transitions
DEBUG_EMOTIONS = False

# Distance-based behavior
CLOSE_FACE_ENTER_RATIO = 0.05
CLOSE_FACE_EXIT_RATIO = 0.042
FAR_FACE_AREA_RATIO = 0.018   # Trigger squint when user is far (<1.8% of frame)
FAR_SQUINT_CHANCE = 0.08
FAR_SQUINT_MIN_SEC = 0.22
FAR_SQUINT_MAX_SEC = 0.55
# If face detection drops temporarily, stay expressive and avoid early sleep.
NO_FACE_SLEEPY_SEC = 120.0
NO_FACE_BORED_SEC = 180.0
NO_FACE_IDLE_BLEND_MIN_SEC = 2.0
NO_FACE_IDLE_BLEND_MAX_SEC = 3.4
NO_FACE_IDLE_BLEND_STAGES = 2
EMOTION_MIN_HOLD_SEC = 0.70
EMOTION_SWITCH_COOLDOWN_SEC = 0.35
EXCITED_BURST_SEC = 0.65
ROUTER_EMOTION_STABLE_SEC = 0.12
SIDE_LOOK_ENTER_OFFSET = 7.0
SIDE_LOOK_EXIT_OFFSET = 4.5
SIDE_LOOK_SWITCH_COOLDOWN_SEC = 0.22
MULTI_FACE_DEBOUNCE_SEC = 0.16
JERK_COOLDOWN_SEC = 0.60
# Social expression variability (prevents always-happy robotic feel)
SOCIAL_MODE_MIN_SEC = 0.70
SOCIAL_MODE_MAX_SEC = 2.00
HAPPY_MIN_GAP_SEC = 2.80

# Natural gaze aversion timing and amplitudes.
GAZE_LOCK_AFTER_FACE_SEC = 3.0
GAZE_MIN_GAP_MIN_SEC = 4.0
GAZE_MIN_GAP_MAX_SEC = 6.0
GAZE_AMBIENT_SCAN_MIN_SEC = 8.0
GAZE_AMBIENT_SCAN_MAX_SEC = 15.0
NO_FACE_SEARCH_MIN_SCANS = 3
NO_FACE_SCAN_TRIGGER_CHANCE = 0.55
NO_FACE_SCAN_RETRY_MIN_SEC = 3.0
NO_FACE_SCAN_RETRY_MAX_SEC = 7.0
NO_FACE_SCAN_SERVO_PAN_DEG = 30.0
GAZE_SOCIAL_RELEASE_MIN_SEC = 20.0
GAZE_SOCIAL_RELEASE_MAX_SEC = 30.0

GAZE_BRIEF_X = 12.0
GAZE_BRIEF_Y = 7.0
GAZE_THINK_X = 18.0
GAZE_THINK_Y = 12.0
GAZE_SCAN_X = 24.0
GAZE_SCAN_Y = 8.0
GAZE_RELEASE_X = 18.0
GAZE_RELEASE_Y = 6.0

GAZE_SERVO_PAN_PER_PX = 0.14
GAZE_SERVO_TILT_PER_PX = 0.12

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
    "squint": 0.85,
}

SPECIAL_EMOTIONS = ["happy", "suspicious", "sleepy"]

# MJPEG Stream Config (for headless SSH viewing)
STREAM_ENABLED = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080
STREAM_FPS = 8
STREAM_JPEG_QUALITY = 70
RENDER_FPS = 24
VISION_FPS = 10

# Optional non-PID Servo Tracking
# Enable head servo tracking (pan/tilt) while keeping eye tilt disabled.
ENABLE_SERVO = True
PAN_CH = 0
TILT_CH = 1
PAN_MIN = 40.0
PAN_MAX = 130.0
TILT_MIN = 80.0
TILT_MAX = 130.0
PULSE_MIN = 450
PULSE_MAX = 2600

SMOOTHING = 0.10
SERVO_LOOP_DELAY = 0.01
MAX_SERVO_STEP_DEG = 1.4
SERVO_DEADZONE_DEG = 0.22

PAN_TRACK_RANGE = 26.0
TILT_TRACK_RANGE = 18.0
TARGET_FILTER_ALPHA = 0.30
NO_FACE_RECENTER_SEC = 1.5
NO_FACE_RECENTER_ALPHA = 0.06

# Head Jerk Animation (for looking_left/looking_right emotions)
JERK_AMPLITUDE = 9.0  # Degrees to jerk left/right
JERK_DURATION = 0.30  # Seconds for full jerk animation


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
        no_face_blends = {"uncertain", "curious", "warm", "attentive", "idle"}
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
        if emotion_name not in EMOTION_PRESETS:
            return

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
            # Trigger head jerk in the direction of the gaze
            global jerk_until, jerk_direction
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
            elif self.current_emotion.startswith("looking_") and "left" in self.current_emotion:
                target_x_phys -= LOOK_SIDE_OFFSET
            elif self.current_emotion.startswith("looking_") and "right" in self.current_emotion:
                target_x_phys += LOOK_SIDE_OFFSET

            look_entry_active = self.current_emotion.startswith("looking_") and now < self.look_entry_until
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

            k = 0.12
            d = 0.70
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
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_error(404)
            return

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

                # Encode to JPEG (frame is RGB)
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=STREAM_JPEG_QUALITY)
                jpg = buf.getvalue()

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / max(1, STREAM_FPS))
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format, *args):
        return


def start_stream_server():
    global stream_server
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), MJPEGHandler)
    thread = threading.Thread(target=stream_server.serve_forever, daemon=True)
    thread.start()
    print(f"MJPEG stream started: http://{STREAM_HOST}:{STREAM_PORT}/stream")


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


# --- Camera & Face Detector Setup ---
print("Initializing Picamera2...")
picam2 = None
try:
    picam2 = Picamera2()
    
    # Pi Camera v2: Full sensor resolution (3280x2464) for widest FOV
    # Using full sensor ensures no digital zoom, maximizing field of view
    config = picam2.create_video_configuration(
        main={"format": 'RGB888', "size": CAMERA_MAIN_RES},
        raw={"size": (3280, 2464)}
    )
    picam2.configure(config)
    # Force full sensor area (no cropping)
    picam2.set_controls({"ScalerCrop": (0, 0, 3280, 2464)})
    picam2.start()
    print(f"Camera started: Full sensor (3280x2464) -> Main ({CAMERA_MAIN_RES[0]}x{CAMERA_MAIN_RES[1]}), detect ({CAMERA_RES[0]}x{CAMERA_RES[1]})")
except Exception as e:
    print(f"Error starting Picamera2: {e}")
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


# --- MJPEG Stream ---
if STREAM_ENABLED:
    try:
        start_stream_server()
    except Exception as e:
        print(f"Error starting MJPEG stream: {e}")


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

# Animation Loop Vars
running = True
next_blink_time = time.time() + random.uniform(3, 6)
last_blink_time = time.time()
smoothed_x_off = 0.0
smoothed_y_off = 0.0
smoothed_rotation = 0.0
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
target_face_area_ratio = 0.0
target_face_count = 0
squint_until = 0.0

# Servo shared state
servo_state_lock = threading.Lock()
servo_running = False
servo_thread = None
servo_kit = None
servo_target_pan = (PAN_MIN + PAN_MAX) * 0.5
servo_target_tilt = (TILT_MIN + TILT_MAX) * 0.5
servo_current_pan = servo_target_pan
servo_current_tilt = servo_target_tilt
last_face_seen_ts = time.time()

# Jerk animation state
jerk_until = 0.0      # Timestamp when jerk ends
jerk_direction = 0.0  # -1 for left, +1 for right, 0 for no jerk
jerk_cooldown_until = 0.0

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


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


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


def start_gaze_event(kind: str, x: float, y: float, to_sec: float, hold_sec: float, back_sec: float):
    global gaze_state, gaze_event_active, gaze_event_start, gaze_event_to_sec
    global gaze_event_hold_sec, gaze_event_back_sec, gaze_event_target_x, gaze_event_target_y
    global scan_emotion_override

    gaze_state = kind
    gaze_event_active = True
    gaze_event_start = time.time()
    gaze_event_to_sec = max(0.01, to_sec)
    gaze_event_hold_sec = max(0.0, hold_sec)
    gaze_event_back_sec = max(0.01, back_sec)
    gaze_event_target_x = float(x)
    gaze_event_target_y = float(y)
    if kind == "AVERT_SCAN":
        scan_emotion_override = "looking_right_natural" if gaze_event_target_x >= 0 else "looking_left_natural"
    else:
        scan_emotion_override = None


def update_gaze_manager(now: float):
    global gaze_state, gaze_event_active, gaze_override_x, gaze_override_y, gaze_reengage_until
    global servo_aversion_pan_offset, servo_aversion_tilt_offset, scan_emotion_override
    global no_face_scan_completed_pulse

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
            # Map scan progress to a deliberate +/-30 deg pan sweep for natural searching.
            denom = max(1e-3, abs(gaze_event_target_x))
            scan_progress = gaze_override_x / denom
            scan_progress = clamp(scan_progress, -1.0, 1.0)
            servo_aversion_pan_offset = scan_progress * NO_FACE_SCAN_SERVO_PAN_DEG
            servo_aversion_tilt_offset = gaze_override_y * GAZE_SERVO_TILT_PER_PX
        else:
            servo_aversion_pan_offset = gaze_override_x * GAZE_SERVO_PAN_PER_PX
            servo_aversion_tilt_offset = gaze_override_y * GAZE_SERVO_TILT_PER_PX


def servo_worker():
    global servo_current_pan, servo_current_tilt, servo_running, jerk_until, jerk_direction
    if servo_kit is None:
        return

    while servo_running:
        with servo_state_lock:
            pan_target = servo_target_pan
            tilt_target = servo_target_tilt
            pan_current = servo_current_pan
            tilt_current = servo_current_tilt
            pan_avert = servo_aversion_pan_offset
            tilt_avert = servo_aversion_tilt_offset

        # Apply jerk oscillation if active
        now = time.time()
        jerk_offset = 0.0
        if now < jerk_until and jerk_direction != 0.0:
            # Elapsed time within jerk window (0.0 to JERK_DURATION)
            elapsed = now - (jerk_until - JERK_DURATION)
            # Normalize to 0-1 phase
            phase = elapsed / JERK_DURATION
            # Sine wave oscillation: quick outward jerk, return, small reverse jerk
            jerk_offset = jerk_direction * JERK_AMPLITUDE * math.sin(phase * math.pi * 2.0)

        pan_error = (pan_target + jerk_offset + pan_avert) - pan_current
        tilt_error = (tilt_target + tilt_avert) - tilt_current

        if abs(pan_error) < SERVO_DEADZONE_DEG:
            pan_error = 0.0
        if abs(tilt_error) < SERVO_DEADZONE_DEG:
            tilt_error = 0.0

        pan_step = clamp(pan_error * SMOOTHING, -MAX_SERVO_STEP_DEG, MAX_SERVO_STEP_DEG)
        tilt_step = clamp(tilt_error * SMOOTHING, -MAX_SERVO_STEP_DEG, MAX_SERVO_STEP_DEG)

        pan_current = clamp(pan_current + pan_step, PAN_MIN, PAN_MAX)
        tilt_current = clamp(tilt_current + tilt_step, TILT_MIN, TILT_MAX)

        try:
            servo_kit.servo[PAN_CH].angle = pan_current
            servo_kit.servo[TILT_CH].angle = tilt_current
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


def udp_worker():
    global udp_emotion_override, udp_emotion_until, udp_speak_pulse
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 9000))
    sock.settimeout(0.5)
    print("UDP Listener active on port 9000")
    while running:
        try:
            data, _ = sock.recvfrom(1024)
            msg = json.loads(data.decode("utf-8"))
            if "emotion" in msg:
                udp_emotion_override = msg["emotion"]
                # Lock the emotion for 5 seconds locally or until another comes
                udp_emotion_until = time.time() + 5.0
                print(f"[UDP] Received emotion override: {udp_emotion_override}")
            if "speak_pulse" in msg:
                udp_speak_pulse = float(msg["speak_pulse"])
        except socket.timeout:
            pass
        except Exception as e:
            pass

def vision_worker():
    global running, target_x_off, target_y_off, target_rotation, target_squint
    global target_face_present, target_face_area_ratio, target_face_count
    global squint_until, latest_frame, servo_target_pan, servo_target_tilt, last_face_seen_ts, next_talk_saccade_ts

    interval = 1.0 / max(1.0, float(VISION_FPS))
    next_tick = time.perf_counter()

    while running:
        try:
            # Capture full frame and resize once for detector input
            large_frame = picam2.capture_array()
            frame = cv2.resize(large_frame, CAMERA_RES)

            if CAMERA_ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            local_x = 0.0
            local_y = 0.0
            local_rot = 0.0
            local_squint = 0.0
            has_face = False
            face_area_ratio = 0.0
            face_count = 0

            if frame is not None and frame.size > 0:
                stream_frame = None
                if STREAM_ENABLED:
                    stream_frame = cv2.resize(frame, STREAM_RES)
                    if STREAM_SWAP_RB:
                        stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)
                    scale_x = STREAM_RES[0] / CAMERA_RES[0]
                    scale_y = STREAM_RES[1] / CAMERA_RES[1]

                detector.setInputSize((frame.shape[1], frame.shape[0]))
                faces = detector.detect(frame)

                if faces[1] is not None:
                    has_face = True
                    detected_faces = faces[1]
                    face_count = len(detected_faces)
                    largest_face = max(detected_faces, key=lambda f: f[2] * f[3])

                    fx, fy, fw, fh = largest_face[0:4]
                    re_x, re_y = largest_face[4], largest_face[5]
                    le_x, le_y = largest_face[6], largest_face[7]

                    if STREAM_ENABLED and stream_frame is not None:
                        fx_s, fy_s = int(fx * scale_x), int(fy * scale_y)
                        fw_s, fh_s = int(fw * scale_x), int(fh * scale_y)
                        re_x_s, re_y_s = int(re_x * scale_x), int(re_y * scale_y)
                        le_x_s, le_y_s = int(le_x * scale_x), int(le_y * scale_y)
                        cv2.rectangle(stream_frame, (fx_s, fy_s), (fx_s + fw_s, fy_s + fh_s), (0, 255, 0), 2)
                        cv2.circle(stream_frame, (re_x_s, re_y_s), 5, (255, 0, 0), -1)
                        cv2.circle(stream_frame, (le_x_s, le_y_s), 5, (255, 0, 0), -1)

                    face_cx = (fx + fw / 2) / CAMERA_RES[0]
                    face_cy = (fy + fh / 2) / CAMERA_RES[1]
                    norm_x = -((face_cx - 0.5) * 2.0)
                    norm_y = (face_cy - 0.5) * 2.0

                    local_x = max(-MAX_X_OFFSET, min(MAX_X_OFFSET, norm_x * MAX_X_OFFSET))
                    local_y = max(-MAX_Y_OFFSET, min(MAX_Y_OFFSET, norm_y * MAX_Y_OFFSET))

                    if ENABLE_SERVO and servo_kit is not None:
                        pan_center = (PAN_MIN + PAN_MAX) * 0.5
                        tilt_center = (TILT_MIN + TILT_MAX) * 0.5
                        mapped_pan = clamp(pan_center + (norm_x * PAN_TRACK_RANGE), PAN_MIN, PAN_MAX)
                        mapped_tilt = clamp(tilt_center + (norm_y * TILT_TRACK_RANGE), TILT_MIN, TILT_MAX)
                        with servo_state_lock:
                            servo_target_pan = servo_target_pan + (mapped_pan - servo_target_pan) * TARGET_FILTER_ALPHA
                            servo_target_tilt = servo_target_tilt + (mapped_tilt - servo_target_tilt) * TARGET_FILTER_ALPHA
                            last_face_seen_ts = time.time()

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

                    if ENABLE_SERVO and servo_kit is not None:
                        if time.time() - last_face_seen_ts > NO_FACE_RECENTER_SEC:
                            pan_center = (PAN_MIN + PAN_MAX) * 0.5
                            tilt_center = (TILT_MIN + TILT_MAX) * 0.5
                            with servo_state_lock:
                                # Smoothly glide target back to neutral when face is lost.
                                servo_target_pan = servo_target_pan + (pan_center - servo_target_pan) * NO_FACE_RECENTER_ALPHA
                                servo_target_tilt = servo_target_tilt + (tilt_center - servo_target_tilt) * NO_FACE_RECENTER_ALPHA

                with target_lock:
                    target_x_off = local_x
                    target_y_off = local_y
                    target_rotation = local_rot
                    target_squint = local_squint
                    target_face_present = has_face
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

print("Starting Tracking Loop...")
time.sleep(1.0) # Warmup

if ENABLE_SERVO:
    if ServoKit is None:
        print("ServoKit not installed; running eyes-only mode.")
    else:
        try:
            print("Initializing ServoKit...")
            servo_kit = ServoKit(channels=16)
            servo_kit.servo[PAN_CH].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
            servo_kit.servo[TILT_CH].set_pulse_width_range(PULSE_MIN, PULSE_MAX)
            servo_kit.servo[PAN_CH].angle = servo_current_pan
            servo_kit.servo[TILT_CH].angle = servo_current_tilt
            servo_running = True
            servo_thread = threading.Thread(target=servo_worker, daemon=True)
            servo_thread.start()
            print("Servo tracking enabled.")
        except Exception as e:
            print(f"Servo init failed, continuing eyes-only: {e}")
            servo_kit = None
            servo_running = False

vision_thread = threading.Thread(target=vision_worker, daemon=True)
vision_thread.start()

udp_thread = threading.Thread(target=udp_worker, daemon=True)
udp_thread.start()

try:
    while running:
        loop_start = time.perf_counter()
        now = time.time()

        with target_lock:
            local_target_x = target_x_off
            local_target_y = target_y_off
            local_target_rot = target_rotation
            local_target_squint = target_squint
            local_face_present = target_face_present
            local_face_area_ratio = target_face_area_ratio
            local_face_count = target_face_count

        # Smooth tracking to reduce jitter
        smooth_alpha = 0.15
        smoothed_x_off = smoothed_x_off + (local_target_x - smoothed_x_off) * smooth_alpha
        smoothed_y_off = smoothed_y_off + (local_target_y - smoothed_y_off) * smooth_alpha
        smoothed_rotation = smoothed_rotation + (local_target_rot - smoothed_rotation) * smooth_alpha
        
        # 2. Update Eye Targets (gaze overrides are layered later)
        left_eye.target_pos[0] = left_eye.base_x + smoothed_x_off
        left_eye.target_pos[1] = left_eye.base_y + smoothed_y_off
        clamp_eye_target(left_eye)

        right_eye.target_pos[0] = left_eye.target_pos[0]
        right_eye.target_pos[1] = left_eye.target_pos[1]
        # Keep face position tracking but disable eye tilt rotation.
        left_eye.target_rotation = 0.0
        right_eye.target_rotation = 0.0

        # Natural emotion routing with timing gates and hysteresis.
        face_entered = local_face_present and not router_face_present_prev
        face_lost = (not local_face_present) and router_face_present_prev

        if face_entered:
            face_present_since_ts = now
            gaze_next_release_ts = now + random.uniform(GAZE_SOCIAL_RELEASE_MIN_SEC, GAZE_SOCIAL_RELEASE_MAX_SEC)
            no_face_blend_until = 0.0
            no_face_blend_queue = []
        elif not local_face_present:
            face_present_since_ts = None

        if face_lost:
            first_blend = weighted_pick([
                ("uncertain", 0.35),
                ("curious", 0.25),
                ("warm", 0.20),
                ("attentive", 0.20),
            ])
            second_options = [e for e in ("uncertain", "curious", "warm", "attentive") if e != first_blend]
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
        abs_x = abs(smoothed_x_off)
        next_side_dir = side_dir_state
        if side_dir_state == 0:
            if abs_x >= SIDE_LOOK_ENTER_OFFSET:
                next_side_dir = 1 if smoothed_x_off >= 0 else -1
        else:
            if abs_x <= SIDE_LOOK_EXIT_OFFSET:
                next_side_dir = 0
            else:
                candidate_dir = 1 if smoothed_x_off >= 0 else -1
                if (
                    candidate_dir != side_dir_state
                    and abs_x >= SIDE_LOOK_ENTER_OFFSET
                    and (now - side_dir_last_switch_ts) >= SIDE_LOOK_SWITCH_COOLDOWN_SEC
                ):
                    next_side_dir = candidate_dir
        if next_side_dir != side_dir_state:
            side_dir_state = next_side_dir
            side_dir_last_switch_ts = now

        if face_entered:
            emotion_force_until = now + EXCITED_BURST_SEC

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

        # Pick a short-lived social mode to keep expressions varied and lifelike.
        if now >= social_mode_until:
            if local_face_present:
                if multi_face_stable:
                    social_mode = weighted_pick([
                        ("warm", 0.45),
                        ("curious", 0.40),
                        ("neutral", 0.15),
                    ])
                elif router_face_close:
                    social_mode = weighted_pick([
                        ("warm", 0.35),
                        ("curious", 0.45),
                        ("neutral", 0.20),
                    ])
                else:
                    social_mode = weighted_pick([
                        ("curious", 0.60),
                        ("neutral", 0.30),
                        ("warm", 0.10),
                    ])
            else:
                social_mode = weighted_pick([
                    ("neutral", 0.70),
                    ("curious", 0.30),
                ])
            social_mode_until = now + random.uniform(SOCIAL_MODE_MIN_SEC, SOCIAL_MODE_MAX_SEC)

        if local_face_present:
            if now < emotion_force_until:
                target_emotion_raw = "excited"
            elif multi_face_stable:
                if social_mode == "warm":
                    target_emotion_raw = "looking_right_happy" if side_right else "looking_left_happy"
                elif social_mode == "curious":
                    target_emotion_raw = "looking_right_natural" if side_right else "looking_left_natural"
                else:
                    target_emotion_raw = "engaged"
            elif router_face_close:
                if social_mode == "warm":
                    if side_look_active:
                        target_emotion_raw = "looking_right_happy" if side_right else "looking_left_happy"
                    else:
                        target_emotion_raw = "happy"
                elif social_mode == "curious":
                    if side_look_active:
                        target_emotion_raw = "looking_right_natural" if side_right else "looking_left_natural"
                    else:
                        target_emotion_raw = "curious_intense"
                else:
                    target_emotion_raw = "idle"
            elif should_squint:
                target_emotion_raw = "squint"
            else:
                target_emotion_raw = "looking_right_natural" if side_right else "looking_left_natural"
        else:
            no_face_elapsed = now - no_face_since_ts
            if gaze_event_active and gaze_state == "AVERT_SCAN" and scan_emotion_override:
                target_emotion_raw = scan_emotion_override
            elif no_face_blend_queue and now < no_face_blend_until:
                target_emotion_raw = no_face_blend_emotion
            elif no_face_elapsed >= NO_FACE_BORED_SEC:
                target_emotion_raw = "warm"
            elif no_face_elapsed >= NO_FACE_SLEEPY_SEC and no_face_scan_checks >= NO_FACE_SEARCH_MIN_SCANS:
                target_emotion_raw = "sleepy"
            else:
                target_emotion_raw = "idle"

        # Avoid repetitive smiling streaks by enforcing a happy cooldown.
        if (
            target_emotion_raw in ("happy", "looking_left_happy", "looking_right_happy")
            and (now - last_happy_ts) < HAPPY_MIN_GAP_SEC
        ):
            if side_look_active:
                target_emotion_raw = "looking_right_natural" if side_right else "looking_left_natural"
            else:
                target_emotion_raw = "warm"

        # Debounce route output so brief detector spikes do not force emotion flips.
        if target_emotion_raw != router_candidate_emotion:
            router_candidate_emotion = target_emotion_raw
            router_candidate_since = now
        if (now - router_candidate_since) >= ROUTER_EMOTION_STABLE_SEC:
            target_emotion = router_candidate_emotion
        else:
            target_emotion = current_emotion

        # --- IMPORTANT UDP AI OVERRIDE ---
        if udp_emotion_override and now < udp_emotion_until:
            target_emotion = udp_emotion_override

        if multi_face_entered and now >= jerk_cooldown_until:
            # When a new face appears (2+ total), jerk toward current look direction.
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

        if target_emotion != current_emotion:
            immediate_excited = face_entered and target_emotion == "excited"
            hold_ok = (now - emotion_last_switch_ts) >= EMOTION_MIN_HOLD_SEC
            cooldown_ok = (now - emotion_last_normal_switch_ts) >= EMOTION_SWITCH_COOLDOWN_SEC
            if immediate_excited or (hold_ok and cooldown_ok):
                target_intensity = EMOTION_INTENSITY.get(target_emotion, 0.55)
                left_eye.set_emotion(target_emotion, target_intensity)
                current_emotion = target_emotion
                if current_emotion in ("happy", "looking_left_happy", "looking_right_happy"):
                    last_happy_ts = now
                emotion_last_switch_ts = now
                if not immediate_excited:
                    emotion_last_normal_switch_ts = now
                
                # Debug: Show transition reason
                if DEBUG_EMOTIONS:
                    if immediate_excited:
                        reason = "FACE_ENTERED (immediate burst)"
                    elif local_face_present:
                        if now < emotion_force_until:
                            reason = "BURST_WINDOW"
                        elif multi_face_stable:
                            reason = "MULTI_FACE_2PLUS (happy looking + jerk)"
                        elif router_face_close:
                            reason = "FACE_CLOSE (happy)"
                        elif should_squint:
                            reason = "FAR_SQUINT"
                        else:
                            reason = "FACE_PRESENT (looking)"
                    else:
                        no_face_elapsed = now - no_face_since_ts
                        if no_face_elapsed >= NO_FACE_BORED_SEC:
                            reason = f"NO_FACE_{NO_FACE_BORED_SEC}s_ELAPSED (warm)"
                        elif no_face_elapsed >= NO_FACE_SLEEPY_SEC and no_face_scan_checks >= NO_FACE_SEARCH_MIN_SCANS:
                            reason = f"NO_FACE_{NO_FACE_SLEEPY_SEC}s_ELAPSED (sleepy)"
                        elif gaze_event_active and gaze_state == "AVERT_SCAN":
                            reason = "NO_FACE_SEARCHING (active scan)"
                        elif no_face_blend_queue and now < no_face_blend_until:
                            stage_idx = max(1, NO_FACE_IDLE_BLEND_STAGES - len(no_face_blend_queue) + 1)
                            reason = f"NO_FACE_BLEND_STAGE_{stage_idx} ({no_face_blend_emotion})"
                        else:
                            reason = f"NO_FACE_IDLE_WAITING (scans {no_face_scan_checks}/{NO_FACE_SEARCH_MIN_SCANS})"
                    print(f"  ✓ EMOTION_CHANGE: {current_emotion:8} | {reason}")
                    if current_emotion.startswith("looking_left"):
                        print(
                            f"  ↺ LOOK_DIR: LEFT  state={gaze_state} x_off={effective_x_off:.2f} y_off={effective_y_off:.2f}"
                        )
                    elif current_emotion.startswith("looking_right"):
                        print(
                            f"  ↻ LOOK_DIR: RIGHT state={gaze_state} x_off={effective_x_off:.2f} y_off={effective_y_off:.2f}"
                        )

        router_face_present_prev = local_face_present
        router_multi_face_prev = multi_face_stable

        # Gaze aversion manager: additive offsets on top of normal face tracking.
        update_gaze_manager(now)
        if no_face_scan_completed_pulse:
            if not local_face_present:
                no_face_scan_checks = min(NO_FACE_SEARCH_MIN_SCANS, no_face_scan_checks + 1)
            no_face_scan_completed_pulse = False
        can_avert = (not gaze_event_active) and (now >= gaze_next_allowed_ts)
        if can_avert:
            if local_face_present and face_present_since_ts is not None:
                face_age = now - face_present_since_ts
                if face_age >= GAZE_LOCK_AFTER_FACE_SEC:
                    if current_emotion in ("thinking", "concentrating", "remembering"):
                        sx = random.choice([-1.0, 1.0])
                        start_gaze_event(
                            "AVERT_THINK",
                            sx * random.uniform(GAZE_THINK_X * 0.8, GAZE_THINK_X),
                            -random.uniform(GAZE_THINK_Y * 0.7, GAZE_THINK_Y),
                            to_sec=0.22,
                            hold_sec=random.uniform(0.8, 2.0),
                            back_sec=0.20,
                        )
                        gaze_next_allowed_ts = now + random.uniform(GAZE_MIN_GAP_MIN_SEC, GAZE_MIN_GAP_MAX_SEC)
                    elif now >= gaze_next_release_ts:
                        sx = random.choice([-1.0, 1.0])
                        start_gaze_event(
                            "AVERT_RELEASE",
                            sx * random.uniform(GAZE_RELEASE_X * 0.85, GAZE_RELEASE_X),
                            random.uniform(-GAZE_RELEASE_Y * 0.5, GAZE_RELEASE_Y),
                            to_sec=0.34,
                            hold_sec=random.uniform(1.5, 3.0),
                            back_sec=0.34,
                        )
                        gaze_next_allowed_ts = now + random.uniform(GAZE_MIN_GAP_MIN_SEC, GAZE_MIN_GAP_MAX_SEC)
                        gaze_next_release_ts = now + random.uniform(GAZE_SOCIAL_RELEASE_MIN_SEC, GAZE_SOCIAL_RELEASE_MAX_SEC)
                    elif random.random() < 0.015:
                        sx = random.choice([-1.0, 1.0])
                        start_gaze_event(
                            "AVERT_BRIEF",
                            sx * random.uniform(GAZE_BRIEF_X * 0.8, GAZE_BRIEF_X),
                            random.uniform(-GAZE_BRIEF_Y, GAZE_BRIEF_Y),
                            to_sec=0.14,
                            hold_sec=random.uniform(0.3, 0.8),
                            back_sec=0.16,
                        )
                        gaze_next_allowed_ts = now + random.uniform(GAZE_MIN_GAP_MIN_SEC, GAZE_MIN_GAP_MAX_SEC)
            elif (not local_face_present) and now >= gaze_next_scan_ts:
                if random.random() < NO_FACE_SCAN_TRIGGER_CHANCE:
                    sx = random.choice([-1.0, 1.0])
                    start_gaze_event(
                        "AVERT_SCAN",
                        sx * random.uniform(GAZE_SCAN_X * 0.8, GAZE_SCAN_X),
                        random.uniform(-GAZE_SCAN_Y, GAZE_SCAN_Y),
                        to_sec=1.20,
                        hold_sec=random.uniform(1.8, 3.6),
                        back_sec=1.20,
                    )
                    gaze_next_allowed_ts = now + random.uniform(GAZE_MIN_GAP_MIN_SEC, GAZE_MIN_GAP_MAX_SEC)
                    gaze_next_scan_ts = now + random.uniform(GAZE_AMBIENT_SCAN_MIN_SEC, GAZE_AMBIENT_SCAN_MAX_SEC)
                else:
                    # Keep scanning occasional by skipping some opportunities.
                    gaze_next_scan_ts = now + random.uniform(NO_FACE_SCAN_RETRY_MIN_SEC, NO_FACE_SCAN_RETRY_MAX_SEC)

        effective_x_off = smoothed_x_off + gaze_override_x
        effective_y_off = smoothed_y_off + gaze_override_y
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

        # 4. Conversational Micro-Expressions (darting eyes while speaking)
        if udp_speak_pulse > 0.0 and now >= next_talk_saccade_ts and not gaze_event_active:
            # Dart randomly to emphasize speech
            sx = random.choice([-1.0, 1.0])
            start_gaze_event(
                "AVERT_TALK",
                sx * random.uniform(15.0, 45.0),
                random.uniform(-30.0, 25.0),
                to_sec=random.uniform(0.08, 0.20),
                hold_sec=random.uniform(0.2, 0.6),
                back_sec=random.uniform(0.10, 0.25)
            )
            next_talk_saccade_ts = now + random.uniform(0.4, 1.4)

        # Keep idle motion deterministic to avoid perceived micro-jitter.
        
        # 5. Physics Update
        # Drive one master eye and mirror full state every frame for strict sync.
        reengage_bump = 0.0
        if now < gaze_reengage_until:
            phase = (gaze_reengage_until - now) / 0.28
            reengage_bump = max(0.0, min(1.0, phase)) * 0.035
            left_eye.target_scale_w += reengage_bump
            left_eye.target_scale_h += reengage_bump * 0.70
            
        left_eye.update()
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
    if vision_thread.is_alive():
        vision_thread.join(timeout=1.0)

    if servo_thread and servo_thread.is_alive():
        servo_running = False
        servo_thread.join(timeout=1.0)

    if servo_kit is not None:
        try:
            servo_kit.servo[PAN_CH].angle = None
            servo_kit.servo[TILT_CH].angle = None
            print("Servos relaxed.")
        except Exception as e:
            print(e)

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