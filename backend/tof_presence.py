"""VL53L0X presence from ESP32 ToF telemetry (TCA9548A mux ch 0/1/2 = L/C/R)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_TOF_RE = re.compile(
    r"^TOF L=(-?\d+) C=(-?\d+) R=(-?\d+) VALID=(\d)(\d)(\d)\s*$"
)

DEFAULT_MIN_VALID_MM = 30


@dataclass(frozen=True)
class TofSnapshot:
    left_mm: int
    center_mm: int
    right_mm: int
    left_valid: bool
    center_valid: bool
    right_valid: bool
    timestamp: float

    @classmethod
    def empty(cls) -> TofSnapshot:
        return cls(-1, -1, -1, False, False, False, 0.0)

    def as_dict(self) -> dict:
        return {
            "left_mm": self.left_mm,
            "center_mm": self.center_mm,
            "right_mm": self.right_mm,
            "left_valid": self.left_valid,
            "center_valid": self.center_valid,
            "right_valid": self.right_valid,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class TofPresence:
    left: bool
    center: bool
    right: bool
    any_present: bool
    count_present: int

    def as_dict(self) -> dict:
        return {
            "left": self.left,
            "center": self.center,
            "right": self.right,
            "any_present": self.any_present,
            "count_present": self.count_present,
        }


def format_tof_channel(mm: int, valid: bool) -> str:
    """Human-readable distance, or 'clear' when no target."""
    if not valid or mm < 0:
        return "clear"
    return f"{mm} mm"


def sanitize_tof_snapshot(
    snap: TofSnapshot,
    *,
    min_valid_mm: int = DEFAULT_MIN_VALID_MM,
) -> TofSnapshot:
    """Drop closer-than-min readings; invalid channels use mm=-1 (display as 'clear')."""

    def one(mm: int, valid: bool) -> tuple[int, bool]:
        if valid and mm >= min_valid_mm:
            return mm, True
        return -1, False

    l_mm, l_v = one(snap.left_mm, snap.left_valid)
    c_mm, c_v = one(snap.center_mm, snap.center_valid)
    r_mm, r_v = one(snap.right_mm, snap.right_valid)
    return TofSnapshot(
        left_mm=l_mm,
        center_mm=c_mm,
        right_mm=r_mm,
        left_valid=l_v,
        center_valid=c_v,
        right_valid=r_v,
        timestamp=snap.timestamp,
    )


def parse_tof_line(line: str) -> TofSnapshot | None:
    match = _TOF_RE.match(line.strip())
    if not match:
        return None
    return TofSnapshot(
        left_mm=int(match.group(1)),
        center_mm=int(match.group(2)),
        right_mm=int(match.group(3)),
        left_valid=match.group(4) == "1",
        center_valid=match.group(5) == "1",
        right_valid=match.group(6) == "1",
        timestamp=time.time(),
    )


class TofPresenceTracker:
    """Debounced per-sector presence from distance thresholds."""

    def __init__(
        self,
        *,
        present_max_mm: float,
        absent_min_mm: float,
        debounce_present_sec: float,
        debounce_absent_sec: float,
    ):
        self.present_max_mm = present_max_mm
        self.absent_min_mm = absent_min_mm
        self.debounce_present_sec = debounce_present_sec
        self.debounce_absent_sec = debounce_absent_sec
        self._stable = [False, False, False]
        self._pending = [False, False, False]
        self._pending_since = [0.0, 0.0, 0.0]

    def _raw_present(self, mm: int, valid: bool) -> bool:
        if not valid or mm < 0:
            return False
        if mm <= self.present_max_mm:
            return True
        if mm >= self.absent_min_mm:
            return False
        return None  # hysteresis band: hold previous

    def update(self, snap: TofSnapshot) -> TofPresence:
        now = time.time()
        readings = (
            (snap.left_mm, snap.left_valid),
            (snap.center_mm, snap.center_valid),
            (snap.right_mm, snap.right_valid),
        )
        for i, (mm, valid) in enumerate(readings):
            raw = self._raw_present(mm, valid)
            if raw is None:
                target = self._stable[i]
            else:
                target = raw
            if target != self._stable[i]:
                if target != self._pending[i]:
                    self._pending[i] = target
                    self._pending_since[i] = now
                debounce = (
                    self.debounce_present_sec
                    if target
                    else self.debounce_absent_sec
                )
                if now - self._pending_since[i] >= debounce:
                    self._stable[i] = target
            else:
                self._pending[i] = target
                self._pending_since[i] = now

        count = sum(self._stable)
        return TofPresence(
            left=self._stable[0],
            center=self._stable[1],
            right=self._stable[2],
            any_present=count > 0,
            count_present=count,
        )
