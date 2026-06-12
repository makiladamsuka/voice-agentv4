"""
Pan/tilt axis contract — must stay aligned across wiring, firmware, and Python.

| Axis | PCA9685 ch | USB serial | robottest | Face tracking        |
|------|------------|------------|-----------|----------------------|
| Pan  | 4          | P (degrees)| A / D     | norm_x → P command   |
| Tilt | 5          | T (degrees)| W / S     | norm_y → T command   |

Firmware: firmware/head_servo/head_servo.ino (PAN_CH, TILT_CH)
Config:   config.yaml servo.pan_ch / servo.tilt_ch (ServoKit + docs)
"""

from __future__ import annotations

FIRMWARE_PAN_PCA_CH = 4
FIRMWARE_TILT_PCA_CH = 5


def check_servo_channel_config(pan_ch: int, tilt_ch: int, *, backend: str) -> None:
    if backend != "arduino":
        return
    if pan_ch == FIRMWARE_PAN_PCA_CH and tilt_ch == FIRMWARE_TILT_PCA_CH:
        return
    print(
        "WARNING: config servo.pan_ch/tilt_ch "
        f"({pan_ch}/{tilt_ch}) differ from ESP32 firmware "
        f"(PCA {FIRMWARE_PAN_PCA_CH}/{FIRMWARE_TILT_PCA_CH}). "
        "Arduino backend uses firmware channels; update ino or config."
    )


def format_head_servo_map() -> str:
    return (
        f"pan=PCA{FIRMWARE_PAN_PCA_CH} (serial P, keys A/D)  "
        f"tilt=PCA{FIRMWARE_TILT_PCA_CH} (serial T, keys W/S)"
    )
