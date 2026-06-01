"""Head servo backends: Arduino Nano (USB) or Adafruit ServoKit (I2C)."""

from __future__ import annotations

from typing import Optional, Protocol

from arduino_servo import ArduinoServoLink
from robot_config import RobotConfig


class ServoDriver(Protocol):
    def write_angles(self, pan: float, tilt: float) -> bool: ...
    def close(self) -> None: ...


class ArduinoServoDriver:
    def __init__(self, link: ArduinoServoLink):
        self._link = link

    def write_angles(self, pan: float, tilt: float) -> bool:
        return self._link.write_angles(pan, tilt)

    def close(self) -> None:
        self._link.close()


class ServoKitDriver:
    """Existing PCA9685 path — unchanged behavior when backend=servokit."""

    def __init__(self, kit, pan_ch: int, tilt_ch: int):
        self._kit = kit
        self._pan_ch = pan_ch
        self._tilt_ch = tilt_ch

    def write_angles(self, pan: float, tilt: float) -> bool:
        try:
            self._kit.servo[self._pan_ch].angle = pan
            self._kit.servo[self._tilt_ch].angle = tilt
            return True
        except Exception as e:
            print(f"ServoKit write error: {e}")
            return False

    def close(self) -> None:
        try:
            self._kit.servo[self._pan_ch].angle = None
            self._kit.servo[self._tilt_ch].angle = None
            print("Servos relaxed.")
        except Exception as e:
            print(e)


def create_servo_driver(cfg: RobotConfig) -> Optional[ServoDriver]:
    if not cfg.servo.enabled:
        return None

    sv = cfg.servo
    backend = (sv.backend or "arduino").strip().lower()

    if backend == "arduino":
        link = ArduinoServoLink(port=sv.arduino_port, baud=sv.arduino_baud)
        if not link.connect():
            return None
        return ArduinoServoDriver(link)

    if backend == "servokit":
        try:
            from adafruit_servokit import ServoKit
        except ImportError:
            print("ServoKit not installed; running eyes-only mode.")
            return None
        try:
            print("Initializing ServoKit...")
            kit = ServoKit(channels=16)
            kit.servo[sv.pan_ch].set_pulse_width_range(sv.pulse_min, sv.pulse_max)
            kit.servo[sv.tilt_ch].set_pulse_width_range(sv.pulse_min, sv.pulse_max)
            pan_center = (sv.pan_min + sv.pan_max) * 0.5
            tilt_center = (sv.tilt_min + sv.tilt_max) * 0.5
            kit.servo[sv.pan_ch].angle = pan_center
            kit.servo[sv.tilt_ch].angle = tilt_center
            print("Servo tracking enabled (ServoKit).")
            return ServoKitDriver(kit, sv.pan_ch, sv.tilt_ch)
        except Exception as e:
            print(f"ServoKit init failed, continuing eyes-only: {e}")
            return None

    print(f"Unknown servo.backend '{backend}'; use arduino or servokit")
    return None
