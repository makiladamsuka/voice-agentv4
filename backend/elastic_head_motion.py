"""Velocity-based elastic pan/tilt motion (shared by test script and robot_eyes)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class HeadMotionParams:
    max_vel_pos: float
    max_vel_neg: float
    accel: float
    decel: float
    vel_blend: float = 0.0
    decel_boost_dir: float = 0.0
    decel_boost_mult: float = 1.0
    goal_deadband_deg: float = 0.08
    track_gain: float = 0.0  # >0 = proportional chase (smoother); 0 = bang-bang


PAN_DEFAULT = HeadMotionParams(
    max_vel_pos=22.0,
    max_vel_neg=22.0,
    accel=55.0,
    decel=85.0,
    vel_blend=0.28,
)

TILT_DEFAULT = HeadMotionParams(
    max_vel_pos=18.0,
    max_vel_neg=10.0,
    accel=45.0,
    decel=70.0,
    vel_blend=0.12,
    decel_boost_dir=-1.0,
    decel_boost_mult=1.85,
    track_gain=2.2,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ramp_velocity(
    vel: float,
    target_vel: float,
    *,
    accel: float,
    decel: float,
    dt: float,
) -> float:
    if abs(target_vel) > 1e-6:
        if vel < target_vel:
            return min(target_vel, vel + accel * dt)
        if vel > target_vel:
            return max(target_vel, vel - accel * dt)
        return vel
    dec = decel * dt
    if vel > 0:
        return max(0.0, vel - dec)
    if vel < 0:
        return min(0.0, vel + dec)
    return 0.0


def tick_axis(
    pos: float,
    vel: float,
    input_dir: float,
    dt: float,
    *,
    lo: float,
    hi: float,
    params: HeadMotionParams,
) -> tuple[float, float]:
    if input_dir > 0.5:
        target_v = params.max_vel_pos
        decel_use = params.decel
    elif input_dir < -0.5:
        target_v = -params.max_vel_neg
        decel_use = params.decel * (
            params.decel_boost_mult if params.decel_boost_dir < 0 else 1.0
        )
    else:
        target_v = 0.0
        decel_use = params.decel * (
            params.decel_boost_mult
            if params.decel_boost_dir != 0.0 and vel * params.decel_boost_dir > 0
            else 1.0
        )

    vel = ramp_velocity(
        vel, target_v, accel=params.accel, decel=decel_use, dt=dt
    )
    if params.vel_blend > 0.0:
        vel += (target_v - vel) * min(1.0, params.vel_blend)
    pos += vel * dt
    if pos <= lo:
        pos = lo
        if vel < 0:
            vel = 0.0
    if pos >= hi:
        pos = hi
        if vel > 0:
            vel = 0.0
    return pos, vel


def tick_toward(
    pos: float,
    vel: float,
    target: float,
    dt: float,
    *,
    lo: float,
    hi: float,
    params: HeadMotionParams,
) -> tuple[float, float]:
    error = target - pos
    if abs(error) <= params.goal_deadband_deg:
        input_dir = 0.0
        target_v = 0.0
    elif params.track_gain > 0.0:
        if error > 0:
            target_v = min(params.max_vel_pos, error * params.track_gain)
        else:
            target_v = -min(params.max_vel_neg, abs(error) * params.track_gain)
        input_dir = 1.0 if target_v > 0.05 else (-1.0 if target_v < -0.05 else 0.0)
        if input_dir == 0.0:
            target_v = 0.0
        else:
            # Proportional mode: ramp directly toward scaled target_v
            vel = ramp_velocity(
                vel, target_v, accel=params.accel, decel=params.decel, dt=dt
            )
            if params.vel_blend > 0.0:
                vel += (target_v - vel) * min(1.0, params.vel_blend)
            pos += vel * dt
            if pos <= lo:
                pos = lo
                if vel < 0:
                    vel = 0.0
            if pos >= hi:
                pos = hi
                if vel > 0:
                    vel = 0.0
            return pos, vel
    elif error > 0:
        input_dir = 1.0
    else:
        input_dir = -1.0
    return tick_axis(
        pos,
        vel,
        input_dir,
        dt,
        lo=lo,
        hi=hi,
        params=params,
    )


def tick_spring(
    pos: float,
    vel: float,
    center: float,
    dt: float,
    *,
    k: float = 9.0,
    damp: float = 6.5,
) -> tuple[float, float]:
    force = -k * (pos - center) - damp * vel
    vel += force * dt
    pos += vel * dt
    return pos, vel


def _eased_sine(value: float, ease_power: float) -> float:
    if ease_power <= 1.0:
        return value
    sign = 1.0 if value >= 0.0 else -1.0
    return sign * (abs(value) ** ease_power)


def wander_search_angles(
    phase: float,
    *,
    pan_anchor: float,
    tilt_anchor: float,
    pan_amp_deg: float,
    tilt_amp_deg: float,
    ease_power: float = 2.0,
    tilt_phase_k: float = 0.35,
    pan_min: float,
    pan_max: float,
    tilt_min: float,
    tilt_max: float,
) -> tuple[float, float]:
    """Slow horizontal-biased no-face search target with eased turnarounds."""
    s = math.sin(phase)
    eased_pan = _eased_sine(s, ease_power)
    pan = clamp(pan_anchor + eased_pan * pan_amp_deg, pan_min, pan_max)

    # Minimal vertical: slow secondary wobble, not full cos swing.
    tilt_wobble = math.sin(phase * tilt_phase_k) * tilt_amp_deg
    tilt = clamp(tilt_anchor + tilt_wobble, tilt_min, tilt_max)
    return pan, tilt


def wander_phase_delta(
    dt: float,
    *,
    period_sec: float,
    phase: float,
    dwell_at_extremes: bool = True,
) -> float:
    """Advance wander phase; slow near pan extremes for a human pause."""
    base_rate = (2.0 * math.pi / max(1.0, period_sec)) * dt
    if not dwell_at_extremes:
        return base_rate
    s = abs(math.sin(phase))
    if s > 0.85:
        return base_rate * 0.62
    return base_rate


@dataclass
class OrganicWanderSearch:
    """Hold-and-step pan search: move to a spot, pause, jump to the next — not a smooth sweep."""

    pan_goal: float = 80.0
    tilt_goal: float = 110.0
    hold_until: float = 0.0
    moving: bool = False
    drift_vel: float = 0.0
    move_speed_scale: float = 1.0
    arrival_deg: float = 2.0
    pause_kind: str = "look"
    hold_emotion_hint: str = "attentive"
    _last_step_deg: float = 0.0
    _next_hold_sec: float = 3.0

    def reset(self, pan_center: float, tilt_center: float, now: float) -> None:
        self.pan_goal = pan_center
        self.tilt_goal = tilt_center
        self.hold_until = now + random.uniform(1.0, 2.8)
        self.moving = False
        self.drift_vel = 0.0
        self.move_speed_scale = 1.0
        self.arrival_deg = 2.0
        self.pause_kind = "look"
        self.hold_emotion_hint = "attentive"
        self._last_step_deg = 0.0
        self._next_hold_sec = 3.0

    @property
    def is_holding(self) -> bool:
        return not self.moving

    def tick(
        self,
        now: float,
        *,
        pan_center: float,
        tilt_center: float,
        pan_current: float,
        pan_min: float,
        pan_max: float,
        tilt_min: float,
        tilt_max: float,
        amp_deg: float,
        step_min_deg: float,
        step_max_deg: float,
        hold_min_sec: float,
        hold_max_sec: float,
        jump_chance: float,
        arrival_deg: float,
        tilt_max_up_deg: float,
        tilt_max_down_deg: float,
        thinking_hold_chance: float = 0.35,
        thinking_hold_min_sec: float = 3.5,
        thinking_hold_max_sec: float = 8.0,
        long_stare_chance: float = 0.12,
    ) -> tuple[float, float]:
        arrived = abs(pan_current - self.pan_goal) <= self.arrival_deg

        if not self.moving:
            if now >= self.hold_until:
                self._pick_new_target(
                    pan_center=pan_center,
                    tilt_center=tilt_center,
                    pan_current=pan_current,
                    pan_min=pan_min,
                    pan_max=pan_max,
                    tilt_min=tilt_min,
                    tilt_max=tilt_max,
                    amp_deg=amp_deg,
                    step_min_deg=step_min_deg,
                    step_max_deg=step_max_deg,
                    jump_chance=jump_chance,
                    arrival_deg=arrival_deg,
                    tilt_max_up_deg=tilt_max_up_deg,
                    tilt_max_down_deg=tilt_max_down_deg,
                )
                self.moving = True
            self.drift_vel = 0.0
        elif arrived:
            self.moving = False
            self.drift_vel = 0.0
            self._assign_pause_kind(
                hold_min_sec=hold_min_sec,
                hold_max_sec=hold_max_sec,
                step_min_deg=step_min_deg,
                step_max_deg=step_max_deg,
                thinking_hold_chance=thinking_hold_chance,
                thinking_hold_min_sec=thinking_hold_min_sec,
                thinking_hold_max_sec=thinking_hold_max_sec,
                long_stare_chance=long_stare_chance,
            )
            self.hold_until = now + self._next_hold_sec
        else:
            self.drift_vel = (
                1.0 if self.pan_goal > pan_current else -1.0
            ) * max(3.0, abs(self.pan_goal - pan_current) * 0.35) * self.move_speed_scale

        return self.pan_goal, self.tilt_goal

    def _pick_new_target(
        self,
        *,
        pan_center: float,
        tilt_center: float,
        pan_current: float,
        pan_min: float,
        pan_max: float,
        tilt_min: float,
        tilt_max: float,
        amp_deg: float,
        step_min_deg: float,
        step_max_deg: float,
        jump_chance: float,
        arrival_deg: float,
        tilt_max_up_deg: float,
        tilt_max_down_deg: float,
    ) -> None:
        margin = 4.0
        step, speed_scale = self._dynamic_step_and_speed(
            step_min_deg, step_max_deg, amp_deg, jump_chance
        )
        self.move_speed_scale = speed_scale
        self.arrival_deg = arrival_deg * random.uniform(0.75, 1.35)

        if random.random() < jump_chance:
            offset = random.uniform(-amp_deg, amp_deg)
            self.pan_goal = clamp(pan_center + offset, pan_min, pan_max)
            self._last_step_deg = abs(self.pan_goal - pan_current)
        else:
            direction = random.choice([-1.0, 1.0])
            self.pan_goal = clamp(pan_current + direction * step, pan_min, pan_max)
            self._last_step_deg = step
            if self.pan_goal >= pan_max - margin:
                self.pan_goal = clamp(
                    pan_current - random.uniform(step * 0.5, step),
                    pan_min,
                    pan_max,
                )
            elif self.pan_goal <= pan_min + margin:
                self.pan_goal = clamp(
                    pan_current + random.uniform(step * 0.5, step),
                    pan_min,
                    pan_max,
                )

        up = max(0.0, tilt_max_up_deg)
        down = max(0.0, tilt_max_down_deg)
        t_lo = clamp(tilt_center - down, tilt_min, tilt_max)
        t_hi = clamp(tilt_center + up, tilt_min, tilt_max)
        if t_lo <= t_hi:
            self.tilt_goal = random.uniform(t_lo, t_hi)
        else:
            self.tilt_goal = tilt_center

    def _assign_pause_kind(
        self,
        *,
        hold_min_sec: float,
        hold_max_sec: float,
        step_min_deg: float,
        step_max_deg: float,
        thinking_hold_chance: float,
        thinking_hold_min_sec: float,
        thinking_hold_max_sec: float,
        long_stare_chance: float,
    ) -> None:
        span = max(0.1, step_max_deg - step_min_deg)
        step_norm = clamp((self._last_step_deg - step_min_deg) / span, 0.0, 1.2)
        base_hold = random.uniform(hold_min_sec, hold_max_sec)
        if step_norm < 0.35:
            base_hold *= random.uniform(0.65, 1.05)
        elif step_norm > 0.85:
            base_hold *= random.uniform(1.15, 1.85)
        if self.move_speed_scale < 0.75:
            base_hold *= random.uniform(1.08, 1.45)
        elif self.move_speed_scale > 1.25:
            base_hold *= random.uniform(0.88, 1.12)

        roll = random.random()
        if roll < thinking_hold_chance:
            self.pause_kind = "thinking"
            self.hold_emotion_hint = random.choice(
                ["thinking", "concentrating", "uncertain", "remembering"]
            )
            self._next_hold_sec = random.uniform(
                thinking_hold_min_sec, thinking_hold_max_sec
            )
        elif roll < thinking_hold_chance + long_stare_chance:
            self.pause_kind = "long_stare"
            self.hold_emotion_hint = random.choice(
                ["attentive", "curious_intense", "thinking"]
            )
            self._next_hold_sec = random.uniform(
                hold_max_sec * 0.95, hold_max_sec * 1.75
            )
        elif step_norm < 0.38:
            self.pause_kind = "glance"
            self.hold_emotion_hint = random.choice(
                ["curious", "uncertain", "curious_intense"]
            )
            self._next_hold_sec = max(1.0, base_hold * random.uniform(0.55, 0.92))
        else:
            self.pause_kind = "look"
            self.hold_emotion_hint = random.choice(
                [
                    "attentive",
                    "uncertain",
                    "thinking",
                    "curious_intense",
                    "curious",
                ]
            )
            self._next_hold_sec = max(1.2, base_hold)

    @staticmethod
    def _dynamic_step_and_speed(
        step_min_deg: float,
        step_max_deg: float,
        amp_deg: float,
        jump_chance: float,
    ) -> tuple[float, float]:
        """Variable glance / turn sizes and per-move speed temperament."""
        lo = min(step_min_deg, step_max_deg)
        hi = max(step_min_deg, step_max_deg)
        mid = (lo + hi) * 0.5
        roll = random.random()

        if roll < 0.14:
            step = random.uniform(lo * 0.12, lo * 0.55)
            speed = random.uniform(0.42, 0.72)
        elif roll < 0.32:
            step = random.uniform(lo * 0.35, lo * 0.95)
            speed = random.uniform(0.55, 0.88)
        elif roll < 0.58:
            step = random.triangular(lo * 0.5, mid, hi * 0.62)
            speed = random.uniform(0.75, 1.12)
        elif roll < 0.78:
            step = random.uniform(mid * 0.7, hi * 1.05)
            speed = random.uniform(0.92, 1.42)
        elif roll < 0.92:
            step = random.uniform(hi * 0.85, min(hi * 1.22, amp_deg))
            speed = random.uniform(1.18, 1.68)
        else:
            step = random.uniform(hi * 1.05, amp_deg * 1.08)
            speed = random.uniform(1.35, 1.85)

        if random.random() < jump_chance * 0.4:
            step = random.uniform(hi * 0.75, amp_deg * random.uniform(0.95, 1.15))
            speed = random.uniform(1.0, 1.62)

        if random.random() < 0.08:
            speed *= random.uniform(0.38, 0.62)
        elif random.random() < 0.10:
            speed *= random.uniform(1.45, 1.9)

        return max(2.0, step), clamp(speed, 0.35, 1.9)


def scale_head_motion(params: HeadMotionParams, scale: float) -> HeadMotionParams:
    s = clamp(scale, 0.45, 1.75)
    return HeadMotionParams(
        max_vel_pos=params.max_vel_pos * s,
        max_vel_neg=params.max_vel_neg * s,
        accel=params.accel * s,
        decel=params.decel * (0.82 + 0.18 * s),
        vel_blend=params.vel_blend,
        decel_boost_dir=params.decel_boost_dir,
        decel_boost_mult=params.decel_boost_mult,
        goal_deadband_deg=params.goal_deadband_deg,
        track_gain=params.track_gain * (0.88 + 0.12 * s),
    )


def wander_search_tilt_from_eye_level(tilt_current: float, tilt_center: float) -> float:
    """Signed offset from eye-level home (+ = slightly up, - = slightly down)."""
    return tilt_current - tilt_center
