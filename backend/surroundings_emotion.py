"""Surroundings-driven eye emotions ported from tftdisplay/trackingeyes2.py."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

# trackingeyes2 name -> RoundEye preset in robot_eyes.EMOTION_PRESETS
EMOTION_NAME_MAP: dict[str, str] = {
    "looking_left": "looking_left_happy",
    "looking_right": "looking_right_happy",
    "calm": "content",
    "curious": "curious_intense",
    "afraid": "uncertain",
}


def map_surroundings_emotion(name: str) -> str:
    return EMOTION_NAME_MAP.get(name, name)


def weighted_pick(weights: dict[str, float], fallback: str = "idle") -> str:
    total = 0.0
    cleaned: dict[str, float] = {}
    for name, weight in weights.items():
        if weight > 0.0:
            cleaned[name] = float(weight)
            total += float(weight)
    if total <= 0.0:
        return fallback
    r = random.uniform(0.0, total)
    acc = 0.0
    for name, weight in cleaned.items():
        acc += weight
        if r <= acc:
            return name
    return fallback


@dataclass
class SurroundingsEmotionConfig:
    no_face_grace_sec: float = 0.9
    no_person_hold_min_sec: float = 2.8
    no_person_hold_max_sec: float = 5.2
    person_hold_min_sec: float = 1.1
    person_hold_max_sec: float = 2.8
    direction_trigger_norm_x: float = 0.22
    direction_hold_min_sec: float = 0.6
    direction_hold_max_sec: float = 1.2
    direction_cooldown_sec: float = 0.9
    close_face_enter_ratio: float = 0.05
    far_face_area_ratio: float = 0.018
    near_exit_ratio: float = 0.041
    far_exit_ratio: float = 0.0225
    emotion_history_len: int = 3


@dataclass
class SurroundingsEmotionController:
    cfg: SurroundingsEmotionConfig = field(default_factory=SurroundingsEmotionConfig)
    distance_zone: str = "mid"
    no_person_next_emotion: str = "sleepy"
    next_emotion_change_time: float = field(default_factory=lambda: time.time() + random.uniform(1.6, 3.0))
    direction_cooldown_until: float = 0.0
    emotion_history: list[str] = field(default_factory=list)
    last_seen_face_time: float = 0.0
    current_emotion: str = "idle"

    def classify_distance_zone(self, face_area_ratio: float) -> str:
        prev = self.distance_zone
        if prev == "near" and face_area_ratio >= self.cfg.near_exit_ratio:
            return "near"
        if prev == "far" and face_area_ratio <= self.cfg.far_exit_ratio:
            return "far"
        if face_area_ratio >= self.cfg.close_face_enter_ratio:
            return "near"
        if face_area_ratio < self.cfg.far_face_area_ratio:
            return "far"
        return "mid"

    def choose_no_person_emotion(self) -> str:
        base = self.no_person_next_emotion
        self.no_person_next_emotion = "idle" if self.no_person_next_emotion == "sleepy" else "sleepy"
        r = random.random()
        if r < 0.08:
            return "sad"
        if r < 0.14:
            return "calm"
        return base

    def choose_person_emotion(
        self,
        zone: str,
        activity: float,
        squint_hint: float,
    ) -> str:
        if zone == "near":
            weights = {
                "excited": 1.0,
                "happy": 0.55,
                "curious": 0.12,
                "calm": 0.12,
                "surprised": 0.09,
                "afraid": 0.06,
                "angry": 0.04,
            }
            if activity > 0.75:
                weights["surprised"] += 0.20
                weights["afraid"] += 0.08
            if activity < 0.25:
                weights["calm"] += 0.10
        elif zone == "far":
            weights = {
                "curious": 1.0,
                "happy": 0.25,
                "calm": 0.30,
                "squint": 0.22,
                "sad": 0.10,
                "sleepy": 0.08,
                "idle": 0.10,
            }
            if squint_hint > 0.5:
                weights["squint"] += 0.35
        else:
            weights = {
                "happy": 1.0,
                "excited": 0.22,
                "curious": 0.30,
                "calm": 0.28,
                "suspicious": 0.15,
                "surprised": 0.08,
                "angry": 0.04,
                "afraid": 0.03,
            }
            if activity > 0.8:
                weights["surprised"] += 0.20
            if activity < 0.2:
                weights["calm"] += 0.12

        for recent in self.emotion_history[-self.cfg.emotion_history_len :]:
            if recent in weights:
                weights[recent] *= 0.35
        if self.current_emotion in weights:
            weights[self.current_emotion] *= 0.45

        return weighted_pick(weights, fallback="happy")

    def push_history(self, emotion_name: str) -> None:
        self.emotion_history.append(emotion_name)
        if len(self.emotion_history) > self.cfg.emotion_history_len:
            self.emotion_history.pop(0)

    def tick(
        self,
        *,
        now: float,
        face_detected: bool,
        face_area_ratio: float,
        face_norm_x: float,
        squint_hint: float,
        activity: float,
        wander_mode: bool,
    ) -> str | None:
        """Return mapped RoundEye emotion when a surroundings change is due, else None."""
        if face_detected:
            self.last_seen_face_time = now

        person_present = (now - self.last_seen_face_time) <= self.cfg.no_face_grace_sec
        if wander_mode and not person_present:
            return None

        if now < self.next_emotion_change_time:
            return None

        if not person_present:
            raw = self.choose_no_person_emotion()
            hold_sec = random.uniform(
                self.cfg.no_person_hold_min_sec,
                self.cfg.no_person_hold_max_sec,
            )
        else:
            self.distance_zone = self.classify_distance_zone(face_area_ratio)
            can_directional = (
                now >= self.direction_cooldown_until
                and abs(face_norm_x) >= self.cfg.direction_trigger_norm_x
            )
            if can_directional and random.random() < (
                0.5 if self.distance_zone == "mid" else 0.35
            ):
                raw = "looking_right" if face_norm_x > 0 else "looking_left"
                hold_sec = random.uniform(
                    self.cfg.direction_hold_min_sec,
                    self.cfg.direction_hold_max_sec,
                )
                self.direction_cooldown_until = now + hold_sec + self.cfg.direction_cooldown_sec
            else:
                raw = self.choose_person_emotion(
                    self.distance_zone,
                    activity,
                    squint_hint,
                )
                hold_base = random.uniform(
                    self.cfg.person_hold_min_sec,
                    self.cfg.person_hold_max_sec,
                )
                hold_sec = max(
                    self.cfg.person_hold_min_sec,
                    hold_base * (1.12 - 0.42 * activity),
                )

        mapped = map_surroundings_emotion(raw)
        self.current_emotion = mapped
        self.push_history(mapped)
        self.next_emotion_change_time = now + hold_sec
        return mapped
