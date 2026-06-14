"""One-shot ToF approach: latch sector, orient once, confirm with camera, cooldown."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum

from tof_presence import TofPresence, TofSnapshot


class ApproachPhase(str, Enum):
    IDLE = "idle"
    ORIENTING = "orienting"
    CONFIRM_WAIT = "confirm_wait"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class ApproachAction:
    pan_delta_deg: float
    tilt_target_deg: float | None = None
    base_nudge_deg: float = 0.0


def sector_pan_offset(sector: str, head_turn_deg: float) -> float:
    if sector == "left":
        return -head_turn_deg
    if sector == "right":
        return head_turn_deg
    return 0.0


def pick_latch_sector(
    snapshot: TofSnapshot,
    presence: TofPresence,
    *,
    head_turn_deg: float,
    present_max_mm: float,
    left_right_only: bool,
) -> str | None:
    """Pick closest present sector for a new approach event."""
    sectors = (
        ("left", presence.left, snapshot.left_mm, snapshot.left_valid),
        ("center", presence.center, snapshot.center_mm, snapshot.center_valid),
        ("right", presence.right, snapshot.right_mm, snapshot.right_valid),
    )
    best_name: str | None = None
    best_dist = float("inf")
    for name, present, mm, valid in sectors:
        if left_right_only and name == "center":
            continue
        if not present or not valid or mm < 0 or mm > present_max_mm:
            continue
        if float(mm) < best_dist:
            best_dist = float(mm)
            best_name = name
    return best_name


def detect_rising_edge(
    presence: TofPresence,
    prev: TofPresence,
    *,
    left_right_only: bool,
) -> str | None:
    """Return sector that just became present (debounced rising edge)."""
    for name in ("left", "center", "right"):
        if left_right_only and name == "center":
            continue
        if getattr(presence, name) and not getattr(prev, name):
            return name
    return None


class TofApproachController:
    """
    Latch-on-edge approach controller.

    Avoids spin-chasing static objects by orienting once per event, then waiting
    for camera confirmation or entering cooldown if no face appears.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        head_turn_deg: float = 30.0,
        present_max_mm: float = 1500.0,
        pan_step_deg: float = 4.0,
        boot_pan_step_deg: float = 30.0,
        arrival_deg: float = 3.0,
        use_base: bool = True,
        base_nudge_deg: float = 12.0,
        max_base_nudges_per_event: int = 1,
        confirm_delay_sec: float = 0.6,
        lockout_sec: float = 10.0,
        left_right_only: bool = True,
        boot_orient: bool = True,
        startup_grace_sec: float = 2.5,
        pan_min: float = 40.0,
        pan_max: float = 120.0,
    ):
        self.enabled = enabled
        self.head_turn_deg = head_turn_deg
        self.present_max_mm = present_max_mm
        self.pan_step_deg = pan_step_deg
        self.boot_pan_step_deg = boot_pan_step_deg
        self.arrival_deg = arrival_deg
        self.use_base = use_base
        self.base_nudge_deg = base_nudge_deg
        self.max_base_nudges_per_event = max(0, int(max_base_nudges_per_event))
        self.confirm_delay_sec = confirm_delay_sec
        self.lockout_sec = lockout_sec
        self.left_right_only = left_right_only
        self.boot_orient = boot_orient
        self.startup_grace_sec = startup_grace_sec
        self.pan_min = pan_min
        self.pan_max = pan_max

        self._lock = threading.Lock()
        self.phase = ApproachPhase.IDLE
        self.last_bearing_deg: float | None = None
        self.active = False
        self.latched_sector: str | None = None
        self._latched_offset: float | None = None
        self._prev_presence = TofPresence(False, False, False, False, 0)
        self._base_nudges_used = 0
        self._confirm_deadline = 0.0
        self._cooldown_until = 0.0
        self._base_pending = False
        self._presence_synced = False
        self._started_ts: float | None = None
        self._boot_orient_done = False

    def sync_presence(self, presence: TofPresence, now: float) -> None:
        """Seed previous presence from first live reading (avoids fake rising edges)."""
        with self._lock:
            if not self._presence_synced:
                self._prev_presence = presence
                self._presence_synced = True
                self._started_ts = now

    def _is_armed(self, now: float) -> bool:
        if self._started_ts is None:
            return False
        return now >= self._started_ts + self.startup_grace_sec

    def _reset_event(self) -> None:
        self.latched_sector = None
        self._latched_offset = None
        self._base_nudges_used = 0
        self._confirm_deadline = 0.0
        self._base_pending = False

    def _enter_cooldown(self, now: float) -> None:
        self.phase = ApproachPhase.COOLDOWN
        self._cooldown_until = now + self.lockout_sec
        self._reset_event()
        self.active = False
        self.last_bearing_deg = None

    def _start_event(self, sector: str, now: float) -> None:
        self.latched_sector = sector
        self._latched_offset = sector_pan_offset(sector, self.head_turn_deg)
        self.last_bearing_deg = self._latched_offset
        self._base_nudges_used = 0
        self.phase = ApproachPhase.ORIENTING
        self.active = True
        self._confirm_deadline = 0.0
        self._base_pending = (
            self.use_base and abs(self._latched_offset or 0.0) >= 0.1
        )

    def _pan_toward_target(
        self,
        pan_current: float,
        *,
        pan_target: float | None = None,
        boot: bool,
    ) -> tuple[float, float, bool]:
        """Return (pan_delta, error, head_at_target)."""
        if self._latched_offset is None:
            return 0.0, 0.0, True
        pan_center = (self.pan_min + self.pan_max) * 0.5
        pan_ref = float(pan_target if pan_target is not None else pan_current)
        error = float(self._latched_offset) - (pan_ref - pan_center)
        at_target = abs(error) <= self.arrival_deg
        if at_target:
            return 0.0, error, True
        step_cap = self.boot_pan_step_deg if boot else self.pan_step_deg
        step = min(abs(error), step_cap)
        return math.copysign(step, error), error, False

    def _maybe_base_nudge(self, target_offset: float) -> float:
        if not self.use_base or abs(target_offset) < 0.1:
            return 0.0
        if self._base_nudges_used >= self.max_base_nudges_per_event:
            return 0.0
        self._base_nudges_used += 1
        if target_offset < 0.0:
            return self.base_nudge_deg
        return -self.base_nudge_deg

    def _tick_unlocked(
        self,
        snapshot: TofSnapshot,
        presence: TofPresence,
        *,
        face_locked: bool,
        pan_current: float,
        pan_target: float | None,
        skip_motion: bool,
        now: float,
        boot: bool,
    ) -> ApproachAction | None:
        if not self.enabled or skip_motion:
            self.phase = ApproachPhase.IDLE
            self.active = False
            self.last_bearing_deg = None
            self._reset_event()
            return None

        if face_locked:
            if self.phase != ApproachPhase.IDLE:
                self.phase = ApproachPhase.IDLE
                self._reset_event()
            self.active = False
            self.last_bearing_deg = None
            return None

        armed = self._is_armed(now)

        rising = None
        if armed:
            rising = detect_rising_edge(
                presence,
                self._prev_presence,
                left_right_only=self.left_right_only,
            )
        self._prev_presence = presence

        if (
            armed
            and self.boot_orient
            and not self._boot_orient_done
            and self.phase == ApproachPhase.IDLE
            and now >= self._cooldown_until
        ):
            sector = pick_latch_sector(
                snapshot,
                presence,
                head_turn_deg=self.head_turn_deg,
                present_max_mm=self.present_max_mm,
                left_right_only=self.left_right_only,
            )
            if sector is not None:
                self._start_event(sector, now)
                self._boot_orient_done = True
                boot = True

        if self.phase == ApproachPhase.IDLE:
            self.active = False
            self.last_bearing_deg = None
            if not armed or now < self._cooldown_until:
                return None
            if rising is not None:
                self._start_event(rising, now)

        if self.phase == ApproachPhase.COOLDOWN:
            self.active = False
            self.last_bearing_deg = None
            if now >= self._cooldown_until:
                self.phase = ApproachPhase.IDLE
            return None

        if self.phase not in (ApproachPhase.ORIENTING, ApproachPhase.CONFIRM_WAIT):
            return None

        target_offset = self._latched_offset if self._latched_offset is not None else 0.0
        self.last_bearing_deg = target_offset
        self.active = True

        use_boot_step = boot and self.phase == ApproachPhase.ORIENTING
        pan_delta, _, head_at_target = self._pan_toward_target(
            pan_current,
            pan_target=pan_target,
            boot=use_boot_step,
        )

        base_nudge = 0.0
        if self.phase == ApproachPhase.ORIENTING and self._base_pending:
            self._base_pending = False
            base_nudge = self._maybe_base_nudge(target_offset)

        if self.phase == ApproachPhase.ORIENTING and head_at_target:
            if base_nudge == 0.0 and self._base_nudges_used == 0:
                base_nudge = self._maybe_base_nudge(target_offset)
            self.phase = ApproachPhase.CONFIRM_WAIT
            self._confirm_deadline = now + self.confirm_delay_sec

        if self.phase == ApproachPhase.CONFIRM_WAIT and now >= self._confirm_deadline:
            self._enter_cooldown(now)
            return None

        if abs(pan_delta) < 0.05 and abs(base_nudge) < 0.2:
            if self.phase == ApproachPhase.CONFIRM_WAIT:
                return ApproachAction(pan_delta_deg=0.0)
            return None

        return ApproachAction(
            pan_delta_deg=pan_delta,
            base_nudge_deg=base_nudge,
        )

    def tick(
        self,
        snapshot: TofSnapshot,
        presence: TofPresence,
        *,
        face_locked: bool,
        pan_current: float,
        pan_target: float | None = None,
        skip_motion: bool,
        now: float,
        boot: bool = False,
    ) -> ApproachAction | None:
        with self._lock:
            return self._tick_unlocked(
                snapshot,
                presence,
                face_locked=face_locked,
                pan_current=pan_current,
                pan_target=pan_target,
                skip_motion=skip_motion,
                now=now,
                boot=boot,
            )

    def phase_name(self) -> str:
        return self.phase.value

    def drives_motion(self) -> bool:
        return self.phase in (ApproachPhase.ORIENTING, ApproachPhase.CONFIRM_WAIT)

    def suppresses_wander_base(self) -> bool:
        return self.phase != ApproachPhase.IDLE
