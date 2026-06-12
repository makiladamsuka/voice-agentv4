"""Load Bottango AnimationCommands.json exports into runtime animation clips."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from animation_player import AnimationClip, Keyframe, Track, TrackBlend

BOTANGO_POS_SCALE = 8192.0

# Bottango effector id = str(i2c_addr) + str(channel)
BOTANGO_TRACK_MAP: dict[str, str] = {
    "644": "head_pan",   # PCA ch4
    "645": "head_tilt",  # PCA ch5
    "640": "arm_0",      # PCA ch0
    "642": "arm_1",      # PCA ch2
    "648": "arm_2",      # PCA ch8
    "649": "arm_3",      # PCA ch9
}

HEAD_PAN_DEG = (40.0, 130.0)
HEAD_TILT_DEG = (80.0, 130.0)
HEAD_PAN_HOME_DEG = (HEAD_PAN_DEG[0] + HEAD_PAN_DEG[1]) * 0.5
HEAD_TILT_HOME_DEG = (HEAD_TILT_DEG[0] + HEAD_TILT_DEG[1]) * 0.5

# PCA channel, label — all 6 servos
SERVO_STOP_SPECS: tuple[tuple[str, str, str], ...] = (
    ("head_pan", "ch4", "pan"),
    ("head_tilt", "ch5", "tilt"),
    ("arm_0", "ch0", "MG996R_R"),
    ("arm_1", "ch2", "MG996R_L"),
    ("arm_2", "ch8", "SG90_R"),
    ("arm_3", "ch9", "SG90_L"),
)

DEFAULT_ARM_NEUTRALS: dict[str, float] = {
    "arm_0": 0.0,
    "arm_1": 180.0,
    "arm_2": 90.0,
    "arm_3": 90.0,
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass
class BottangoEffector:
    effector_id: str
    pca_channel: int
    min_signal: float
    max_signal: float
    max_change_per_sec: float
    start_signal: float

    def pulse_to_deg(self, pulse: float, *, track: str) -> float:
        lo = self.min_signal
        hi = self.max_signal
        if hi <= lo:
            return pulse
        t = (pulse - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
        if track == "head_pan":
            return _lerp(HEAD_PAN_DEG[0], HEAD_PAN_DEG[1], t)
        if track == "head_tilt":
            return _lerp(HEAD_TILT_DEG[0], HEAD_TILT_DEG[1], t)
        return t * 180.0


@dataclass
class BottangoCurve:
    effector_id: str
    start_ms: float
    duration_ms: float
    start_y: float
    start_cx: float
    start_cy: float
    end_y: float
    end_cx: float
    end_cy: float
    last_u: float = 0.5

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms

    def evaluate(self, t_ms: float) -> float:
        if self.duration_ms <= 0:
            return self.end_y
        u_lower = 0.0
        u_upper = 1.0
        u = self.last_u
        x = t_ms
        for _ in range(48):
            ex, ey = self._evaluate_for_u(u)
            if abs(ex - x) < 1.0:
                self.last_u = u
                return ey
            if ex > x:
                u_upper = u
            else:
                u_lower = u
            u = (u_upper - u_lower) * 0.5 + u_lower
        return self.end_y

    def _evaluate_for_u(self, u: float) -> tuple[float, float]:
        p11x = _lerp(self.start_ms, self.start_ms + self.start_cx, u)
        p11y = _lerp(self.start_y, self.start_y + self.start_cy, u)
        p12x = _lerp(self.start_ms + self.start_cx, self.end_ms + self.end_cx, u)
        p12y = _lerp(self.start_y + self.start_cy, self.end_y + self.end_cy, u)
        p13x = _lerp(self.end_ms + self.end_cx, self.end_ms, u)
        p13y = _lerp(self.end_y + self.end_cy, self.end_y, u)
        p21x = _lerp(p11x, p12x, u)
        p21y = _lerp(p11y, p12y, u)
        p22x = _lerp(p12x, p13x, u)
        p22y = _lerp(p12y, p13y, u)
        return _lerp(p21x, p22x, u), _lerp(p21y, p22y, u)


def _parse_setup(setup_text: str) -> dict[str, BottangoEffector]:
    effectors: dict[str, BottangoEffector] = {}
    for line in setup_text.splitlines():
        line = line.strip()
        if not line.startswith("rSVI2C,"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        addr = int(parts[1])
        ch = int(parts[2])
        effector_id = f"{addr}{ch}"
        effectors[effector_id] = BottangoEffector(
            effector_id=effector_id,
            pca_channel=ch,
            min_signal=float(parts[3]),
            max_signal=float(parts[4]),
            max_change_per_sec=float(parts[5]),
            start_signal=float(parts[6]),
        )
    return effectors


def _parse_sc_line(line: str) -> BottangoCurve | None:
    line = line.strip()
    if not line.startswith("sC,"):
        return None
    parts = line.split(",")
    if len(parts) < 10:
        return None
    return BottangoCurve(
        effector_id=parts[1],
        start_ms=float(parts[2]),
        duration_ms=float(parts[3]),
        start_y=float(parts[4]) / BOTANGO_POS_SCALE,
        start_cx=float(parts[5]),
        start_cy=float(parts[6]) / BOTANGO_POS_SCALE,
        end_y=float(parts[7]) / BOTANGO_POS_SCALE,
        end_cx=float(parts[8]),
        end_cy=float(parts[9]) / BOTANGO_POS_SCALE,
    )


def _movement_to_deg(
    movement: float,
    effector: BottangoEffector,
    track: str,
) -> float:
    pulse = _lerp(effector.min_signal, effector.max_signal, movement)
    return effector.pulse_to_deg(pulse, track=track)


def _sample_animation(
    curves: list[BottangoCurve],
    effectors: dict[str, BottangoEffector],
    *,
    sample_ms: int = 33,
) -> tuple[float, dict[str, list[Keyframe]]]:
    if not curves:
        return 0.0, {}

    duration_ms = max(c.end_ms for c in curves)
    keys_by_track: dict[str, list[Keyframe]] = {}

    t = 0.0
    while t <= duration_ms + 0.5:
        active: dict[str, BottangoCurve] = {}
        for curve in curves:
            track = BOTANGO_TRACK_MAP.get(curve.effector_id)
            if track is None:
                continue
            if curve.start_ms <= t <= curve.end_ms:
                prev = active.get(track)
                if prev is None or curve.start_ms >= prev.start_ms:
                    active[track] = curve

        for track, curve in active.items():
            effector = effectors.get(curve.effector_id)
            if effector is None:
                continue
            movement = curve.evaluate(t)
            deg = _movement_to_deg(movement, effector, track)
            keys_by_track.setdefault(track, []).append(
                Keyframe(t_ms=t, v=deg, ease="linear")
            )
        t += sample_ms

    return duration_ms, keys_by_track


def _default_blend(track: str) -> TrackBlend:
    if track in ("head_pan", "head_tilt"):
        return TrackBlend(mode="override", weight=0.45)
    return TrackBlend(mode="override", weight=1.0)


def clip_from_botango_animation(
    name: str,
    command_text: str,
    effectors: dict[str, BottangoEffector],
    *,
    loop_text: str = "",
    include_loop_segment: bool = False,
) -> AnimationClip | None:
    curves: list[BottangoCurve] = []
    for line in command_text.splitlines():
        curve = _parse_sc_line(line)
        if curve is not None:
            curves.append(curve)

    # Loop segment holds pose for a long time in Bottango; skip for one-shot clips.
    if include_loop_segment and loop_text.strip():
        for line in loop_text.splitlines():
            curve = _parse_sc_line(line)
            if curve is not None:
                curves.append(curve)

    duration_ms, keys_by_track = _sample_animation(curves, effectors)
    if not keys_by_track:
        return None

    tracks: list[Track] = []
    blends: dict[str, TrackBlend] = {}
    for track, keys in sorted(keys_by_track.items()):
        tracks.append(Track(servo=track, units="deg", keys=keys))
        blends[track] = _default_blend(track)

    clip_id = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_") or "botango_clip"
    return AnimationClip(
        clip_id=clip_id,
        duration_ms=duration_ms,
        tracks=tracks,
        blends=blends,
    )


def load_botango_commands_file(path: str | Path) -> list[AnimationClip]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)

    controllers = raw if isinstance(raw, list) else [raw]
    clips: list[AnimationClip] = []
    for controller in controllers:
        setup = controller.get("Setup", {})
        setup_text = setup.get("Controller Setup Commands", "")
        effectors = _parse_setup(setup_text)
        for anim in controller.get("Animations", []):
            name = str(anim.get("Animation Name", "animation"))
            cmd = str(anim.get("Animation Commands", ""))
            loop = str(anim.get("Animation Loop Commands", ""))
            clip = clip_from_botango_animation(name, cmd, effectors, loop_text=loop)
            if clip is not None:
                clips.append(clip)
    return clips


def servo_stop_pose(arm_neutrals: dict[str, float] | None = None) -> dict[str, float]:
    arms = dict(DEFAULT_ARM_NEUTRALS)
    if arm_neutrals:
        arms.update(arm_neutrals)
    return {
        "head_pan": HEAD_PAN_HOME_DEG,
        "head_tilt": HEAD_TILT_HOME_DEG,
        **arms,
    }


def format_servo_stop_pose(pose: dict[str, float]) -> str:
    lines = ["Servo stop positions (deg):"]
    for track, channel, label in SERVO_STOP_SPECS:
        deg = pose.get(track, float("nan"))
        lines.append(f"  {channel} {label}: {deg:.1f}")
    return "\n".join(lines)


def neutral_arm_degrees(effectors: dict[str, BottangoEffector]) -> dict[str, float]:
    out: dict[str, float] = {}
    for effector_id, track in BOTANGO_TRACK_MAP.items():
        if not track.startswith("arm_"):
            continue
        eff = effectors.get(effector_id)
        if eff is None:
            continue
        movement = (eff.start_signal - eff.min_signal) / max(1e-6, eff.max_signal - eff.min_signal)
        movement = max(0.0, min(1.0, movement))
        out[track] = movement * 180.0
    return out
