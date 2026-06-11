#!/usr/bin/env python3
"""
Interactive animation tester — play Botango presets from AnimationCommands.json.

Does not start camera, eyes, or voice agent. Stop start_robot.py first (serial port).

  cd backend && python tests/test_animations.py
  python tests/test_animations.py --port /dev/ttyUSB0
  python tests/test_animations.py --play 3
  python tests/test_animations.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import BACKEND_ROOT

from animation_player import AnimationPlayer
from arduino_servo import ArduinoServoLink
from botango_loader import (
    DEFAULT_ARM_NEUTRALS,
    _parse_setup,
    format_servo_stop_pose,
    load_botango_commands_file,
    neutral_arm_degrees,
    servo_stop_pose,
)

BOTANGO_FILE = BACKEND_ROOT / "animations" / "AnimationCommands.json"

PAN_MIN = 40.0
PAN_MAX = 130.0
TILT_MIN = 80.0
TILT_MAX = 130.0
PAN_CENTER = (PAN_MIN + PAN_MAX) * 0.5
TILT_CENTER = (TILT_MIN + TILT_MAX) * 0.5

LOOP_HZ = 30.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def blend_track(base: float, sample_value: float, mode: str, weight: float) -> float:
    w = max(0.0, min(1.0, float(weight)))
    if str(mode).lower() == "override":
        return base + (sample_value - base) * w
    return base + (sample_value * w)


def load_arm_neutrals() -> dict[str, float]:
    defaults = dict(DEFAULT_ARM_NEUTRALS)
    if not BOTANGO_FILE.exists():
        return defaults
    with BOTANGO_FILE.open(encoding="utf-8") as f:
        raw = json.load(f)
    controllers = raw if isinstance(raw, list) else [raw]
    for controller in controllers:
        setup_text = controller.get("Setup", {}).get("Controller Setup Commands", "")
        effectors = _parse_setup(setup_text)
        neutrals = neutral_arm_degrees(effectors)
        if neutrals:
            defaults.update(neutrals)
    return defaults


def send_frame(
    link: ArduinoServoLink,
    pan: float,
    tilt: float,
    arms: dict[str, float],
) -> None:
    frame = {
        "P": pan,
        "T": tilt,
        "A0=": clamp(arms.get("arm_0", 0.0), 0.0, 180.0),
        "A1=": clamp(arms.get("arm_1", 180.0), 0.0, 180.0),
        "A2=": clamp(arms.get("arm_2", 90.0), 0.0, 180.0),
        "A3=": clamp(arms.get("arm_3", 90.0), 0.0, 180.0),
    }
    link.write_servo_frame(frame)


def play_clip(
    link: ArduinoServoLink,
    player: AnimationPlayer,
    clip_id: str,
    *,
    pan_home: float,
    tilt_home: float,
    arm_home: dict[str, float],
    loop: bool = False,
) -> bool:
    if not player.play(clip_id, loop=loop):
        return False

    pan = pan_home
    tilt = tilt_home
    arms = dict(arm_home)
    interval = 1.0 / LOOP_HZ
    start = time.time()
    print(f"Playing '{clip_id}' ({'loop' if loop else 'oneshot'})...")

    try:
        while True:
            now = time.time()
            samples = player.sample(now)
            if not samples:
                break

            pan_target = pan_home
            tilt_target = tilt_home
            if "head_pan" in samples:
                s = samples["head_pan"]
                pan_target = blend_track(pan_target, s.value, s.mode, s.weight)
            if "head_tilt" in samples:
                s = samples["head_tilt"]
                tilt_target = blend_track(tilt_target, s.value, s.mode, s.weight)

            arm_targets = dict(arms)
            for arm_track in ("arm_0", "arm_1", "arm_2", "arm_3"):
                if arm_track in samples:
                    s = samples[arm_track]
                    base = arm_home[arm_track]
                    arm_targets[arm_track] = blend_track(base, s.value, s.mode, s.weight)

            pan += (pan_target - pan) * 0.35
            tilt += (tilt_target - tilt) * 0.35
            for k in arm_targets:
                arms[k] = arms.get(k, arm_home[k]) + (arm_targets[k] - arms.get(k, arm_home[k])) * 0.45

            pan = clamp(pan, PAN_MIN, PAN_MAX)
            tilt = clamp(tilt, TILT_MIN, TILT_MAX)
            send_frame(link, pan, tilt, arms)
            time.sleep(interval)
    finally:
        player.stop()

    elapsed = time.time() - start
    print(f"Done ({elapsed:.1f}s)")
    return True


def home_pose(link: ArduinoServoLink, arm_home: dict[str, float]) -> None:
    stop_pose = servo_stop_pose(arm_home)
    print(format_servo_stop_pose(stop_pose))
    pan = stop_pose["head_pan"]
    tilt = stop_pose["head_tilt"]
    arms = dict(arm_home)
    for _ in range(40):
        send_frame(link, pan, tilt, arms)
        time.sleep(0.03)
    link.write_angles(pan, tilt, force=True)
    send_frame(link, pan, tilt, arms)
    try:
        link.send_line("V")
        time.sleep(0.05)
    except Exception:
        pass


def print_menu(clips: list) -> None:
    print("\nAnimations (from AnimationCommands.json):")
    print("-" * 48)
    for i, clip in enumerate(clips, start=1):
        tracks = ", ".join(tr.servo for tr in clip.tracks)
        print(f"  {i:2d}. {clip.clip_id:<22} ({clip.duration_ms/1000:.1f}s) [{tracks}]")
    print("-" * 48)
    print("  Enter number to play | r = replay last | h = home | l = list | q = quit")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Botango animation presets by number")
    parser.add_argument("--port", default="", help="Serial port (default: auto USB0/ACM0)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--json", default=str(BOTANGO_FILE), help="Botango export JSON path")
    parser.add_argument("--list", action="store_true", help="List animations and exit")
    parser.add_argument("--play", type=int, default=0, help="Play animation by number (1-based) and exit")
    parser.add_argument("--loop", action="store_true", help="Loop selected animation")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Animation file not found: {json_path}")
        return 1

    clips = load_botango_commands_file(json_path)
    if not clips:
        print("No animations found in JSON.")
        return 1

    player = AnimationPlayer()
    for clip in clips:
        player.register_clip(clip)

    if args.list:
        print_menu(clips)
        return 0

    if args.play:
        if args.play < 1 or args.play > len(clips):
            print(f"Invalid --play {args.play}; choose 1-{len(clips)}")
            return 1

    if not sys.stdin.isatty() and not args.play:
        print("Interactive mode needs a terminal, or use --play N")
        return 1

    link = ArduinoServoLink(port=args.port, baud=args.baud)
    if not link.connect():
        print("Failed to connect. Stop start_robot.py and check USB / firmware READY.")
        return 1

    arm_home = load_arm_neutrals()
    last_idx = 0

    try:
        home_pose(link, arm_home)

        if args.play:
            clip = clips[args.play - 1]
            play_clip(
                link,
                player,
                clip.clip_id,
                pan_home=PAN_CENTER,
                tilt_home=TILT_CENTER,
                arm_home=arm_home,
                loop=args.loop,
            )
            home_pose(link, arm_home)
            return 0

        print_menu(clips)
        while True:
            try:
                raw = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if raw in ("q", "quit", "exit"):
                break
            if raw in ("l", "list"):
                print_menu(clips)
                continue
            if raw in ("h", "home"):
                home_pose(link, arm_home)
                continue
            if raw in ("r", "replay") and last_idx:
                clip = clips[last_idx - 1]
                play_clip(
                    link,
                    player,
                    clip.clip_id,
                    pan_home=PAN_CENTER,
                    tilt_home=TILT_CENTER,
                    arm_home=arm_home,
                )
                home_pose(link, arm_home)
                continue

            try:
                idx = int(raw)
            except ValueError:
                print("Enter a number, h, l, r, or q")
                continue

            if idx < 1 or idx > len(clips):
                print(f"Choose 1-{len(clips)}")
                continue

            last_idx = idx
            clip = clips[idx - 1]
            play_clip(
                link,
                player,
                clip.clip_id,
                pan_home=PAN_CENTER,
                tilt_home=TILT_CENTER,
                arm_home=arm_home,
            )
            home_pose(link, arm_home)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        home_pose(link, arm_home)
        link.close(home_pan=PAN_CENTER, home_tilt=TILT_CENTER)

    return 0


if __name__ == "__main__":
    sys.exit(main())
