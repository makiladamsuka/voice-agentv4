#!/usr/bin/env python3
"""
Bench test for 3x VL53L0X on TCA9548A via ESP32 (head_servo firmware).

  cd backend && python tests/test_tof_sensors.py
  python tests/test_tof_sensors.py --port /dev/ttyUSB0
  python tests/test_tof_sensors.py --stream --hz 5
"""

from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401

from arduino_servo import ArduinoServoLink
from esp32_serial import connect_esp32
from robot_config import load_config
from tof_presence import (
    TofPresenceTracker,
    format_tof_channel,
    parse_tof_line,
    sanitize_tof_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll ESP32 ToF presence (L/C/R)")
    parser.add_argument("--port", default="", help="Serial port (default: auto)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--hz", type=float, default=5.0, help="Poll rate when not streaming")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable firmware O stream instead of Pi F polls",
    )
    parser.add_argument(
        "--present-max-mm",
        type=float,
        default=None,
        help="Override config present_max_mm",
    )
    args = parser.parse_args()

    cfg = load_config()
    t = cfg.tof
    present_max = args.present_max_mm if args.present_max_mm is not None else t.present_max_mm
    tracker = TofPresenceTracker(
        present_max_mm=present_max,
        absent_min_mm=t.absent_min_mm,
        debounce_present_sec=t.debounce_present_sec,
        debounce_absent_sec=t.debounce_absent_sec,
    )
    min_valid_mm = int(t.min_valid_mm)

    link = connect_esp32(port=args.port, baud=args.baud, prepare=False)
    if link is None:
        print("Could not connect to ESP32.")
        return 1

    if link._boot_lines:
        print("Boot log:")
        for line in link._boot_lines:
            print(f"  {line}")

    if not link.tof_capable:
        print(
            "\nThis ESP32 image does not answer the ToF (F) command.\n"
            "Stop start_robot.py, then flash:\n"
            "  arduino-cli lib install Adafruit_VL53L0X\n"
            "  arduino-cli compile --fqbn esp32:esp32:esp32 firmware/head_servo\n"
            "  arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 firmware/head_servo\n"
        )
        link.close()
        return 1

    mux_missing = any("WARN TCA9548A" in ln for ln in link._boot_lines)
    if mux_missing:
        print(
            "\nNote: TCA9548A not seen on I2C (0x70-0x77). Check mux 3.3V, SDA=21, SCL=22, "
            "GND. Boot log should list 'I2C scan:' with any devices found.\n"
        )

    if args.stream:
        if not link.set_tof_stream(True, args.hz):
            print("Failed to enable ToF stream (O command).")
            return 1
        print(f"ToF stream enabled at ~{args.hz} Hz — Ctrl+C to stop")
        try:
            while True:
                line = link._ser.readline().decode("utf-8", errors="ignore").strip()  # type: ignore
                if not line:
                    continue
                snap = parse_tof_line(line)
                if snap is None:
                    if line.startswith("TOF") or "WARN TOF" in line:
                        print(line)
                    continue
                snap = sanitize_tof_snapshot(snap, min_valid_mm=min_valid_mm)
                pres = tracker.update(snap)
                print(
                    f"L={format_tof_channel(snap.left_mm, snap.left_valid):>8} "
                    f"C={format_tof_channel(snap.center_mm, snap.center_valid):>8} "
                    f"R={format_tof_channel(snap.right_mm, snap.right_valid):>8}  "
                    f"present L={pres.left} C={pres.center} R={pres.right} "
                    f"count={pres.count_present}"
                )
        except KeyboardInterrupt:
            print("\nStopping stream...")
        finally:
            link.set_tof_stream(False)
            link.close()
        return 0

    interval = 1.0 / max(0.5, args.hz)
    print(f"Polling ToF at {args.hz:.1f} Hz — Ctrl+C to stop")
    print("'clear' = no target. Hold hand ~30–60 cm in front of each sensor.\n")
    try:
        while True:
            snap = link.poll_tof()
            if snap is None:
                print("(no TOF response — check mux/VL53 wiring or I2C on GPIO 21/22)")
            else:
                snap = sanitize_tof_snapshot(snap, min_valid_mm=min_valid_mm)
                pres = tracker.update(snap)
                print(
                    f"L={format_tof_channel(snap.left_mm, snap.left_valid):>8} "
                    f"C={format_tof_channel(snap.center_mm, snap.center_valid):>8} "
                    f"R={format_tof_channel(snap.right_mm, snap.right_valid):>8}  "
                    f"present L={pres.left} C={pres.center} R={pres.right} "
                    f"any={pres.any_present}"
                )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
