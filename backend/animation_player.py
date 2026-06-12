"""Playback for Botango-style JSON animation clips."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _smoothstep01(v: float) -> float:
    v = _clamp01(v)
    return v * v * (3.0 - 2.0 * v)


def _ease(alpha: float, easing: str) -> float:
    e = (easing or "linear").strip().lower()
    if e in ("smooth", "smoothstep", "ease"):
        return _smoothstep01(alpha)
    return _clamp01(alpha)


@dataclass
class Keyframe:
    t_ms: float
    v: float
    ease: str = "linear"


@dataclass
class TrackBlend:
    mode: str = "add"  # add | override
    weight: float = 1.0


@dataclass
class Track:
    servo: str
    units: str
    keys: list[Keyframe] = field(default_factory=list)

    def sample(self, t_ms: float) -> float | None:
        if not self.keys:
            return None
        if t_ms <= self.keys[0].t_ms:
            return self.keys[0].v
        if t_ms >= self.keys[-1].t_ms:
            return self.keys[-1].v
        for i in range(len(self.keys) - 1):
            k0 = self.keys[i]
            k1 = self.keys[i + 1]
            if k0.t_ms <= t_ms <= k1.t_ms:
                span = max(1e-6, k1.t_ms - k0.t_ms)
                alpha = (t_ms - k0.t_ms) / span
                a = _ease(alpha, k1.ease)
                return k0.v + (k1.v - k0.v) * a
        return self.keys[-1].v


@dataclass
class AnimationClip:
    clip_id: str
    duration_ms: float
    tracks: list[Track]
    blends: dict[str, TrackBlend]
    source_path: Path | None = None

    @staticmethod
    def from_json(data: dict[str, Any], *, source_path: Path | None = None) -> "AnimationClip":
        clip_id = str(data.get("clip_id", "unnamed"))
        duration_ms = float(data.get("duration_ms", 0))
        tracks: list[Track] = []
        for raw_track in data.get("tracks", []):
            keys: list[Keyframe] = []
            for raw_key in raw_track.get("keys", []):
                keys.append(
                    Keyframe(
                        t_ms=float(raw_key.get("t_ms", 0)),
                        v=float(raw_key.get("v", 0)),
                        ease=str(raw_key.get("ease", "linear")),
                    )
                )
            keys.sort(key=lambda k: k.t_ms)
            if keys and duration_ms <= 0:
                duration_ms = max(duration_ms, keys[-1].t_ms)
            tracks.append(
                Track(
                    servo=str(raw_track.get("servo", "")).strip(),
                    units=str(raw_track.get("units", "deg")).strip().lower(),
                    keys=keys,
                )
            )
        blend_map: dict[str, TrackBlend] = {}
        raw_blend = data.get("blend", {}) or {}
        if isinstance(raw_blend, dict):
            for servo, cfg in raw_blend.items():
                if not isinstance(cfg, dict):
                    continue
                blend_map[str(servo)] = TrackBlend(
                    mode=str(cfg.get("mode", "add")).strip().lower(),
                    weight=float(cfg.get("weight", 1.0)),
                )
        return AnimationClip(
            clip_id=clip_id,
            duration_ms=max(0.0, duration_ms),
            tracks=tracks,
            blends=blend_map,
            source_path=source_path,
        )


@dataclass
class ActiveClip:
    clip: AnimationClip
    started_at: float
    loop: bool = False


@dataclass
class TrackSample:
    value: float
    mode: str
    weight: float


class AnimationPlayer:
    """Simple timeline player for event-triggered clips."""

    def __init__(self) -> None:
        self._active: ActiveClip | None = None
        self._clips_by_id: dict[str, AnimationClip] = {}

    def register_clip(self, clip: AnimationClip) -> None:
        self._clips_by_id[clip.clip_id] = clip

    def load_clip_file(self, path: str | Path) -> AnimationClip:
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            raw = json.load(f)
        clip = AnimationClip.from_json(raw, source_path=p)
        self.register_clip(clip)
        return clip

    def play(self, clip_id: str, *, loop: bool = False, now: float | None = None) -> bool:
        clip = self._clips_by_id.get(clip_id)
        if clip is None:
            return False
        if now is None:
            now = time.time()
        self._active = ActiveClip(clip=clip, started_at=now, loop=loop)
        return True

    def play_clip(self, clip: AnimationClip, *, loop: bool = False, now: float | None = None) -> None:
        self.register_clip(clip)
        if now is None:
            now = time.time()
        self._active = ActiveClip(clip=clip, started_at=now, loop=loop)

    def stop(self) -> None:
        self._active = None

    def active_clip_id(self) -> str | None:
        if self._active is None:
            return None
        return self._active.clip.clip_id

    def sample(self, now: float | None = None) -> dict[str, TrackSample]:
        if self._active is None:
            return {}
        if now is None:
            now = time.time()
        active = self._active
        clip = active.clip
        if clip.duration_ms <= 0.0:
            self._active = None
            return {}
        elapsed_ms = (now - active.started_at) * 1000.0
        if elapsed_ms > clip.duration_ms:
            if active.loop:
                elapsed_ms = elapsed_ms % clip.duration_ms
            else:
                self._active = None
                return {}
        out: dict[str, TrackSample] = {}
        for tr in clip.tracks:
            val = tr.sample(elapsed_ms)
            if val is None:
                continue
            blend = clip.blends.get(tr.servo, TrackBlend(mode="add", weight=1.0))
            out[tr.servo] = TrackSample(value=val, mode=blend.mode, weight=_clamp01(blend.weight))
        return out
