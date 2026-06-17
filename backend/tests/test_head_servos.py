#!/usr/bin/env python3
"""
Interactive elastic pan/tilt + base test for ESP32 (head_servo firmware).

Velocity ramps up while a key is held and eases down on release — head and base,
no sudden jumps.

  cd backend && python tests/test_head_servos.py --port /dev/ttyUSB0

Keys (inverted tilt, same as testservos2):
  W/S = tilt up/down     A/D = pan left/right
  M/N = base nudge per tap  C = spring to center
  Q or Ctrl+C = home and quit
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
import time

import _bootstrap  # noqa: F401

from arduino_servo import ArduinoServoLink
from esp32_serial import (
    ESP32_SERIAL_SEND_HZ,
    connect_esp32,
    prepare_esp32_for_live_control,
)
from elastic_head_motion import (
    HeadMotionParams,
    clamp,
    tick_axis,
    tick_spring,
)
from robot_config import load_config

LOOP_HZ = 100.0
SERIAL_SEND_HZ = ESP32_SERIAL_SEND_HZ
KEY_TIMEOUT = 0.35
DEBUG_HZ = 8.0
BASE_POLL_HZ = 1.5
DEFAULT_HOLD_SEC = 3.0
CONTROL_KEYS = frozenset("wasd")

# Head elastic — smoother and less aggressive response.
HEAD_VEL_BLEND = 0.30
PAN_MOTION = HeadMotionParams(
    max_vel_pos=40.0,
    max_vel_neg=40.0,
    accel=130.0,
    decel=180.0,
    vel_blend=HEAD_VEL_BLEND,
)
TILT_MOTION = HeadMotionParams(
    max_vel_pos=24.0,
    max_vel_neg=10.0,
    accel=85.0,
    decel=170.0,
    vel_blend=HEAD_VEL_BLEND,
    decel_boost_dir=-1.0,
    decel_boost_mult=2.20,
)

# Base tap (M/N = one small encoder move per press)
BASE_TAP_DEG = 1.5
BASE_TAP_DEBOUNCE_SEC = 0.22

# Spring return to center (home / C key)
SPRING_K = 7.8
SPRING_DAMP = 5.8
HOME_SETTLE_VEL = 0.35
HOME_SETTLE_POS = 0.4
BASE_HOME_DEG = 0.0
BASE_HOME_SETTLE_DEG = 0.5


def _limits() -> tuple[float, float, float, float, float, float]:
    cfg = load_config()
    sv = cfg.servo
    pan_min = float(sv.pan_min)
    pan_max = float(sv.pan_max)
    tilt_min = float(sv.tilt_min)
    tilt_max = float(sv.tilt_max)
    pan_center = (pan_min + pan_max) * 0.5
    tilt_center = (tilt_min + tilt_max) * 0.5
    return pan_min, pan_max, tilt_min, tilt_max, pan_center, tilt_center


def _get_key() -> str | None:
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _drain_keys() -> list[str]:
    keys: list[str] = []
    while True:
        key = _get_key()
        if key is None:
            break
        keys.append(key)
    return keys


def _read_base_deg(link: ArduinoServoLink) -> float | None:
    st = link.query_status()
    return st.degrees if st is not None else None


def _home_base(link: ArduinoServoLink, target_deg: float = BASE_HOME_DEG) -> None:
    _stop_base(link)
    st = link.query_status()
    if (
        st is not None
        and not st.busy
        and abs(st.degrees - target_deg) < BASE_HOME_SETTLE_DEG
    ):
        return
    print(f"Homing base to {target_deg:.1f}°...")
    link.write_base_absolute(target_deg, wait=True)


def _home_robot(
    link: ArduinoServoLink,
    pan: float,
    pan_vel: float,
    tilt: float,
    tilt_vel: float,
    *,
    loop_delay: float,
    pan_center: float,
    tilt_center: float,
    base_home_deg: float = BASE_HOME_DEG,
) -> tuple[float, float]:
    _home_base(link, base_home_deg)
    pan, pan_vel, tilt, tilt_vel = _elastic_home_head(
        link,
        pan,
        pan_vel,
        tilt,
        tilt_vel,
        loop_delay=loop_delay,
        pan_center=pan_center,
        tilt_center=tilt_center,
    )
    return pan, tilt


def _elastic_home_head(
    link: ArduinoServoLink,
    pan: float,
    pan_vel: float,
    tilt: float,
    tilt_vel: float,
    *,
    loop_delay: float,
    pan_center: float,
    tilt_center: float,
) -> tuple[float, float, float, float]:
    print(f"\nHoming head (P{pan_center:.1f} T{tilt_center:.1f})...")
    while True:
        pan, pan_vel = tick_spring(pan, pan_vel, pan_center, loop_delay)
        tilt, tilt_vel = tick_spring(tilt, tilt_vel, tilt_center, loop_delay)
        link.write_angles(pan, tilt)
        if (
            abs(pan - pan_center) < HOME_SETTLE_POS
            and abs(tilt - tilt_center) < HOME_SETTLE_POS
            and abs(pan_vel) < HOME_SETTLE_VEL
            and abs(tilt_vel) < HOME_SETTLE_VEL
        ):
            link.write_angles(pan_center, tilt_center, force=True)
            return pan_center, 0.0, tilt_center, 0.0
        time.sleep(loop_delay)


def _stop_base(link: ArduinoServoLink) -> None:
    link.write_base_stop()


def _queue_base_tap(
    queue: list[float],
    deg: float,
    *,
    last_tap_ts: float,
    now: float,
) -> float:
    if now - last_tap_ts < BASE_TAP_DEBOUNCE_SEC:
        return last_tap_ts
    queue.append(deg)
    return now


def _run_base_tap(link: ArduinoServoLink, queue: list[float]) -> bool:
    if not queue:
        return False
    st = link.query_status()
    if st is not None and st.busy:
        return False
    deg = queue.pop(0)
    ok = link.write_base_relative(deg, wait=True)
    return ok


def _maybe_send_angles(
    link: ArduinoServoLink,
    pan: float,
    tilt: float,
    *,
    last_send_ts: float,
    send_interval: float,
    force: bool = False,
) -> float:
    """Throttle USB serial — only latest pose, avoids ESP32 command backlog."""
    now = time.time()
    if force or (now - last_send_ts) >= send_interval:
        link.write_angles(pan, tilt, force=force)
        return now
    return last_send_ts


def run_interactive(
    link: ArduinoServoLink,
    *,
    loop_delay: float,
    key_timeout: float,
) -> tuple[float, float]:
    pan_min, pan_max, tilt_min, tilt_max, pan_center, tilt_center = _limits()

    pan = pan_center
    tilt = tilt_center
    pan_vel = 0.0
    tilt_vel = 0.0
    base_deg: float | None = _read_base_deg(link)
    base_tap_queue: list[float] = []
    base_tap_busy = False
    last_base_tap_ts = 0.0

    pan_input = 0.0
    tilt_input = 0.0
    spring_center = False

    key_last_seen: dict[str, float] = {}
    last_debug_ts = 0.0
    last_base_poll_ts = 0.0
    last_serial_send_ts = 0.0
    send_interval = 1.0 / max(5.0, SERIAL_SEND_HZ)
    running = True

    link.flush_pending_commands()
    link.write_base_stop()
    link.set_tof_stream(False)
    link.write_angles(pan, tilt, force=True)
    last_serial_send_ts = time.time()

    print("--- Elastic head + base control (ESP32) ---")
    print("  W/S = tilt up/down (inverted)   A/D = pan left/right")
    print(f"  M/N = base nudge ±{BASE_TAP_DEG:.0f}° per tap (M=right, N=left)")
    print("  WASD + M/N together = move head and nudge base")
    print("  C = spring head to center       Q or Ctrl+C = home head + base and quit")
    print(
        f"  Tilt down uses extra braking (gravity). "
        f"Loop {1.0 / loop_delay:.0f} Hz"
    )
    if base_deg is not None:
        print(
            f"  Base zero = {BASE_HOME_DEG:.1f}°  "
            f"current {base_deg:+.1f}° from zero"
        )

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        _drain_keys()
        try:
            while running:
                key_now = time.time()
                for key in _drain_keys():
                    k = key.lower()
                    if k == "q":
                        running = False
                    elif k == "c":
                        spring_center = True
                        pan_input = tilt_input = 0.0
                        for head_key in "wasd":
                            key_last_seen.pop(head_key, None)
                    elif k == "m":
                        last_base_tap_ts = _queue_base_tap(
                            base_tap_queue,
                            -BASE_TAP_DEG,
                            last_tap_ts=last_base_tap_ts,
                            now=key_now,
                        )
                        spring_center = False
                    elif k == "n":
                        last_base_tap_ts = _queue_base_tap(
                            base_tap_queue,
                            BASE_TAP_DEG,
                            last_tap_ts=last_base_tap_ts,
                            now=key_now,
                        )
                        spring_center = False
                    elif k in CONTROL_KEYS:
                        key_last_seen[k] = key_now
                        spring_center = False

                now = key_now
                active_keys = {
                    k
                    for k, ts in key_last_seen.items()
                    if now - ts < key_timeout
                }

                pan_input = 0.0
                tilt_input = 0.0
                if not spring_center:
                    if "a" in active_keys:
                        pan_input -= 1.0
                    if "d" in active_keys:
                        pan_input += 1.0
                    if "w" in active_keys:
                        tilt_input += 1.0
                    if "s" in active_keys:
                        tilt_input -= 1.0

                if spring_center:
                    pan, pan_vel = tick_spring(pan, pan_vel, pan_center, loop_delay)
                    tilt, tilt_vel = tick_spring(tilt, tilt_vel, tilt_center, loop_delay)
                    if (
                        abs(pan - pan_center) < HOME_SETTLE_POS
                        and abs(tilt - tilt_center) < HOME_SETTLE_POS
                        and abs(pan_vel) < HOME_SETTLE_VEL
                        and abs(tilt_vel) < HOME_SETTLE_VEL
                    ):
                        pan, tilt = pan_center, tilt_center
                        pan_vel = tilt_vel = 0.0
                        spring_center = False
                else:
                    pan, pan_vel = tick_axis(
                        pan,
                        pan_vel,
                        pan_input,
                        loop_delay,
                        lo=pan_min,
                        hi=pan_max,
                        params=PAN_MOTION,
                    )
                    tilt, tilt_vel = tick_axis(
                        tilt,
                        tilt_vel,
                        tilt_input,
                        loop_delay,
                        lo=tilt_min,
                        hi=tilt_max,
                        params=TILT_MOTION,
                    )

                last_serial_send_ts = _maybe_send_angles(
                    link,
                    pan,
                    tilt,
                    last_send_ts=last_serial_send_ts,
                    send_interval=send_interval,
                )

                if base_tap_queue and not base_tap_busy:
                    base_tap_busy = True
                    _run_base_tap(link, base_tap_queue)
                    polled = _read_base_deg(link)
                    if polled is not None:
                        base_deg = polled
                    base_tap_busy = False

                if now - last_debug_ts >= 1.0 / DEBUG_HZ:
                    if base_tap_busy:
                        spin_lbl = "move"
                    elif base_tap_queue:
                        spin_lbl = f"q{len(base_tap_queue)}"
                    else:
                        spin_lbl = "stop"
                    driving = bool(active_keys & CONTROL_KEYS) or base_tap_busy
                    if (
                        not driving
                        and now - last_base_poll_ts >= 1.0 / BASE_POLL_HZ
                    ):
                        polled = _read_base_deg(link)
                        if polled is not None:
                            base_deg = polled
                        last_base_poll_ts = now
                    base_pos = f"{base_deg:+.1f}°" if base_deg is not None else "?"
                    sys.stdout.write(
                        f"\r  pan {pan:5.1f}  tilt {tilt:5.1f}  "
                        f"base {base_pos} ({spin_lbl})   "
                    )
                    sys.stdout.flush()
                    last_debug_ts = now

                time.sleep(loop_delay)
        except KeyboardInterrupt:
            print("\nCtrl+C — homing...")
    finally:
        print()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        _stop_base(link)
        base_tap_queue.clear()

    pan, tilt = _home_robot(
        link,
        pan,
        pan_vel,
        tilt,
        tilt_vel,
        loop_delay=loop_delay,
        pan_center=pan_center,
        tilt_center=tilt_center,
        base_home_deg=BASE_HOME_DEG,
    )
    return pan, tilt


def _demo_steps() -> list[tuple[str, float, float]]:
    pan_min, pan_max, tilt_min, tilt_max, pan_center, tilt_center = _limits()
    return [
        ("pan left (min)", pan_min, tilt_center),
        ("pan right (max)", pan_max, tilt_center),
        ("center", pan_center, tilt_center),
        ("tilt down (min)", pan_center, tilt_min),
        ("tilt up (max)", pan_center, tilt_max),
        ("return center", pan_center, tilt_center),
    ]


def run_demo(link: ArduinoServoLink, *, hold_sec: float, loop_delay: float) -> None:
    pan_min, pan_max, tilt_min, tilt_max, pan_center, tilt_center = _limits()
    pan, pan_vel = pan_center, 0.0
    tilt, tilt_vel = tilt_center, 0.0
    print("Elastic demo sweep...")
    for label, goal_pan, goal_tilt in _demo_steps():
        print(f"  -> {label}")
        while True:
            pan, pan_vel = tick_spring(pan, pan_vel, goal_pan, loop_delay, k=6.0)
            tilt, tilt_vel = tick_spring(tilt, tilt_vel, goal_tilt, loop_delay, k=6.0)
            link.write_angles(pan, tilt)
            if (
                abs(pan - goal_pan) < HOME_SETTLE_POS
                and abs(tilt - goal_tilt) < HOME_SETTLE_POS
                and abs(pan_vel) < HOME_SETTLE_VEL
                and abs(tilt_vel) < HOME_SETTLE_VEL
            ):
                break
            time.sleep(loop_delay)
        time.sleep(hold_sec)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Elastic interactive head + base test (ESP32 USB serial)"
    )
    parser.add_argument("--port", default="", help="Serial port (default: auto)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--demo", action="store_true", help="Elastic pose demo sweep")
    parser.add_argument("--pan", type=float, default=None, help="Single pan angle")
    parser.add_argument("--tilt", type=float, default=None, help="Single tilt angle")
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD_SEC)
    parser.add_argument("--sweep", action="store_true", help="Onboard bench sweep (S)")
    parser.add_argument(
        "--loop-delay",
        type=float,
        default=1.0 / LOOP_HZ,
        help=f"Control loop period (default {1.0 / LOOP_HZ:.3f}s)",
    )
    parser.add_argument("--key-timeout", type=float, default=KEY_TIMEOUT)
    args = parser.parse_args()
    loop_delay = max(0.005, args.loop_delay)

    link = connect_esp32(port=args.port, baud=args.baud, prepare=False)
    print("Tip: stop start_robot.py first — only one program can use /dev/ttyUSB0.")
    if link is None:
        print("Failed to connect. Check USB and head_servo firmware (READY).")
        return 1

    try:
        from base_motor_utils import apply_config_cpd_to_nano

        apply_config_cpd_to_nano(link)
    except Exception as e:
        print(f"Note: base CPD not applied ({e})")

    prepare_esp32_for_live_control(link)

    pan_min, pan_max, tilt_min, tilt_max, pan_center, tilt_center = _limits()
    base_start = _read_base_deg(link)
    print(
        f"Head center P{pan_center:.1f} T{tilt_center:.1f}  "
        f"pan [{pan_min:.0f},{pan_max:.0f}]  tilt [{tilt_min:.0f},{tilt_max:.0f}]"
    )
    if base_start is not None:
        print(
            f"Base zero reference {BASE_HOME_DEG:.1f}°  "
            f"(at connect: {base_start:+.1f}°)"
        )

    pan_current, tilt_current = pan_center, tilt_center
    homed = False

    try:
        if args.sweep:
            print("Sending bench sweep (S)...")
            if not link.run_bench_sweep():
                return 1
        elif args.pan is not None and args.tilt is not None:
            pan_g = clamp(args.pan, pan_min, pan_max)
            tilt_g = clamp(args.tilt, tilt_min, tilt_max)
            _elastic_home_head(
                link,
                pan_center,
                0.0,
                tilt_center,
                0.0,
                loop_delay=loop_delay,
                pan_center=pan_g,
                tilt_center=tilt_g,
            )
            time.sleep(args.hold)
            pan_current, tilt_current = pan_g, tilt_g
        elif args.demo:
            run_demo(link, hold_sec=args.hold, loop_delay=loop_delay)
            homed = True
        else:
            pan_current, tilt_current = run_interactive(
                link,
                loop_delay=loop_delay,
                key_timeout=args.key_timeout,
            )
            homed = True
        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted — homing...")
        if not homed:
            pan_current, tilt_current = _home_robot(
                link,
                pan_current,
                0.0,
                tilt_current,
                0.0,
                loop_delay=loop_delay,
                pan_center=pan_center,
                tilt_center=tilt_center,
                base_home_deg=BASE_HOME_DEG,
            )
            homed = True
    finally:
        link.write_base_stop()
        try:
            link.close(
                home_pan=pan_center,
                home_tilt=tilt_center,
                skip_home=homed,
            )
        except TypeError:
            link.close(home_pan=pan_center, home_tilt=tilt_center)

    return 0


if __name__ == "__main__":
    sys.exit(main())
