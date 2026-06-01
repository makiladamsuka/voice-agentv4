#!/usr/bin/env python3
"""
Standalone pan/tilt servo test for Arduino Nano (head_servo firmware).

Does not start robot_eyes, camera, or voice agent.

  cd backend && python test_head_servos.py
  python test_head_servos.py --verify --hold 5
  python test_head_servos.py --port /dev/ttyUSB0
  python test_head_servos.py --pan 40 --tilt 80
  python test_head_servos.py --sweep
"""

from __future__ import annotations

import argparse
import sys
import time

from arduino_servo import ArduinoServoLink

PAN_MIN = 40.0
PAN_MAX = 130.0
TILT_MIN = 80.0
TILT_MAX = 130.0
PAN_CENTER = 85.0
TILT_CENTER = 105.0
DEFAULT_HOLD_SEC = 5.0

DEMO_STEPS = [
    ("pan left (min)", PAN_MIN, TILT_CENTER),
    ("pan right (max)", PAN_MAX, TILT_CENTER),
    ("center", PAN_CENTER, TILT_CENTER),
    ("center pan", PAN_CENTER, TILT_CENTER),
    ("tilt down (min)", PAN_CENTER, TILT_MIN),
    ("tilt up (max)", PAN_CENTER, TILT_MAX),
    ("return center", PAN_CENTER, TILT_CENTER),
]


def write_and_report(
    link: ArduinoServoLink,
    label: str,
    pan: float,
    tilt: float,
    *,
    verify: bool,
    hold_sec: float,
) -> None:
    print(f"  -> {label}: Pi sent P{pan:.1f} T{tilt:.1f}", end="")
    if not link.write_angles(pan, tilt, force=True, wait_ack=verify):
        print(" — write failed")
        sys.exit(1)
    if verify:
        ack = link._last_ack
        if ack is None:
            print(" — no ACK (re-flash firmware with DEBUG_ACK?)")
        else:
            print(f" → Nano ACK P{ack[0]} T{ack[1]}")
    else:
        print()
    time.sleep(hold_sec)


def run_demo(link: ArduinoServoLink, *, verify: bool, hold_sec: float) -> None:
    print("Running demo sweep (Ctrl+C to stop early)...")
    for label, pan, tilt in DEMO_STEPS:
        write_and_report(link, label, pan, tilt, verify=verify, hold_sec=hold_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test head servos via Arduino Nano USB")
    parser.add_argument("--port", default="", help="Serial port (default: auto USB0/ACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--no-demo", action="store_true", help="Skip demo sweep")
    parser.add_argument("--pan", type=float, default=None, help="Single pan angle (degrees)")
    parser.add_argument("--tilt", type=float, default=None, help="Single tilt angle (degrees)")
    parser.add_argument(
        "--hold",
        type=float,
        default=DEFAULT_HOLD_SEC,
        help=f"Seconds to hold each pose (default {DEFAULT_HOLD_SEC})",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read Nano OK P.. T.. ACK after each command",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run onboard pan bench sweep (S command, ~6s)",
    )
    args = parser.parse_args()

    link = ArduinoServoLink(port=args.port, baud=args.baud)
    if not link.connect():
        print("Failed to connect. Check USB, dialout group, and head_servo firmware (READY).")
        return 1

    print(
        "Watch servos during each hold (not only after Done). "
        "If ACK angles change but servos do not move: check 5V supply, horn screw, pulse on D9."
    )

    try:
        if args.sweep:
            print("Sending bench sweep (S)...")
            if not link.run_bench_sweep():
                print("Sweep command failed.")
                return 1
            print("Sweep finished.")
        elif args.pan is not None and args.tilt is not None:
            pan = max(PAN_MIN, min(PAN_MAX, args.pan))
            tilt = max(TILT_MIN, min(TILT_MAX, args.tilt))
            write_and_report(
                link,
                "manual",
                pan,
                tilt,
                verify=args.verify,
                hold_sec=args.hold,
            )
        elif not args.no_demo:
            run_demo(link, verify=args.verify, hold_sec=args.hold)
        else:
            print("Nothing to do. Use demo, --pan/--tilt, or --sweep.")
            return 1
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted — returning to center.")
    finally:
        link.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
