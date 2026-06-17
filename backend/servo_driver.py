"""Head servo backends: ESP32 over USB (arduino) or Pi-local ServoKit (legacy)."""

from __future__ import annotations

import time
from typing import Optional, Protocol

from arduino_servo import ArduinoServoLink
from esp32_serial import configure_esp32_link, prepare_esp32_for_live_control
from head_servo_axes import check_servo_channel_config, format_head_servo_map
from robot_config import RobotConfig
from tof_presence import TofSnapshot


class ServoDriver(Protocol):
    def write_angles(self, pan: float, tilt: float) -> bool: ...
    def write_servo_frame(self, values_by_token: dict[str, float]) -> bool: ...
    def close(self) -> None: ...


class ArduinoServoDriver:
    def __init__(self, link: ArduinoServoLink):
        self._link = link
        self.base_motion_allowed = True

    @property
    def serial_connected(self) -> bool:
        return self._link.connected

    def write_angles(self, pan: float, tilt: float, *, force: bool = False) -> bool:
        return self._link.write_angles(pan, tilt, force=force)

    def write_servo_frame(self, values_by_token: dict[str, float], *, wait_ack: bool = False) -> bool:
        return self._link.write_servo_frame(values_by_token, wait_ack=wait_ack)

    def poll_tof(self, timeout: float = 1.0) -> TofSnapshot | None:
        return self._link.poll_tof(timeout)

    def set_tof_stream(self, enabled: bool, hz: float = 5.0) -> bool:
        return self._link.set_tof_stream(enabled, hz)

    def write_home_pose(
        self,
        pan: float,
        tilt: float,
        arm_neutrals: dict[str, float] | None = None,
        *,
        wait_ack: bool = True,
    ) -> bool:
        return self._link.write_home_pose(
            pan, tilt, arm_neutrals, wait_ack=wait_ack
        )

    def write_base_relative(self, deg: float, *, wait: bool = True) -> bool:
        if not self.base_motion_allowed:
            return False
        return self._link.write_base_relative(deg, wait=wait)

    def write_base_relative_clamped(
        self,
        deg: float,
        *,
        max_from_zero: float,
        max_nudge_deg: float | None = None,
        wait: bool = True,
    ) -> bool:
        if not self.base_motion_allowed:
            return False
        if max_nudge_deg is not None and max_nudge_deg > 0:
            cap = float(max_nudge_deg)
            deg = max(-cap, min(cap, deg))
        st = self.query_base_status()
        base_now = st.degrees if st is not None else 0.0
        if st is not None and st.busy:
            return False
        proposed = base_now + deg
        if proposed > max_from_zero:
            deg = max_from_zero - base_now
        elif proposed < -max_from_zero:
            deg = -max_from_zero - base_now
        if abs(deg) < 0.25:
            return False
        return self.write_base_relative(deg, wait=wait)

    def write_base_absolute(self, deg: float, *, wait: bool = True) -> bool:
        return self._link.write_base_absolute(deg, wait=wait)

    def write_base_stop(self) -> bool:
        return self._link.write_base_stop()

    def query_base_status(self):
        return self._link.query_status()

    def close(
        self,
        *,
        home_pan: float | None = None,
        home_tilt: float | None = None,
        arm_neutrals: dict[str, float] | None = None,
        skip_home: bool = False,
    ) -> None:
        self._link.close(
            home_pan=home_pan,
            home_tilt=home_tilt,
            arm_neutrals=arm_neutrals,
            skip_home=skip_home,
        )


class ServoKitDriver:
    """Pi-local PCA9685 via ServoKit — use only if servos are on Pi I2C, not ESP32."""

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

    def write_servo_frame(self, values_by_token: dict[str, float]) -> bool:
        pan = values_by_token.get("P")
        tilt = values_by_token.get("T")
        if pan is None or tilt is None:
            return False
        return self.write_angles(float(pan), float(tilt))

    def close(
        self,
        *,
        home_pan: float | None = None,
        home_tilt: float | None = None,
        arm_neutrals: dict[str, float] | None = None,
        skip_home: bool = False,
    ) -> None:
        del arm_neutrals, skip_home
        try:
            if home_pan is not None and home_tilt is not None:
                self._kit.servo[self._pan_ch].angle = home_pan
                self._kit.servo[self._tilt_ch].angle = home_tilt
                time.sleep(0.15)
            self._kit.servo[self._pan_ch].angle = None
            self._kit.servo[self._tilt_ch].angle = None
            print("Servos relaxed.")
        except Exception as e:
            print(e)


def create_servo_driver(
    cfg: RobotConfig,
    *,
    max_attempts: int = 1,
    retry_delay_sec: float = 2.0,
) -> Optional[ServoDriver]:
    if not cfg.servo.enabled:
        return None

    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        driver = _create_servo_driver_once(cfg)
        if driver is not None:
            return driver
        if attempt + 1 < attempts:
            print(
                f"ESP32 connect retry {attempt + 2}/{attempts} "
                f"in {retry_delay_sec:.0f}s..."
            )
            time.sleep(retry_delay_sec)
    return None


def _create_servo_driver_once(cfg: RobotConfig) -> Optional[ServoDriver]:
    if not cfg.servo.enabled:
        return None

    sv = cfg.servo
    backend = (sv.backend or "arduino").strip().lower()
    check_servo_channel_config(sv.pan_ch, sv.tilt_ch, backend=backend)

    if backend == "arduino":
        link = ArduinoServoLink(port=sv.arduino_port, baud=sv.arduino_baud)
        link.base_move_timeout_sec = float(cfg.base.move_timeout_sec)
        configure_esp32_link(link)
        if not link.connect():
            return None
        try:
            from base_motor_utils import apply_config_cpd_to_nano

            cpd_ok = apply_config_cpd_to_nano(link)
        except Exception as e:
            print(f"Warning: could not apply base CPD to ESP32: {e}")
            cpd_ok = False
        prepare_esp32_for_live_control(link)
        driver = ArduinoServoDriver(link)
        if cfg.base.enabled:
            calibrated = link.is_calibrated()
            driver.base_motion_allowed = bool(cpd_ok and calibrated)
            if not driver.base_motion_allowed:
                print(
                    "Base motor disabled until CPD is calibrated — head servos still active.\n"
                    "  Run: python tests/test_base_motor.py --calibrate-manual --degrees 90 --write-config"
                )
        print(f"Head servos: {format_head_servo_map()}")
        return driver

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
