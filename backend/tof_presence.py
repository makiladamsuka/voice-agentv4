"""VL53L0X presence from ESP32 ToF telemetry (TCA9548A mux ch 0/1/2 = L/C/R).

Reliability fixes (fix/tof-reliability):
  - Rolling median filter (window=5) smooths per-channel readings before thresholding
  - Spike rejection: ignore readings that jump > max_delta_mm from the rolling median
  - Consecutive-agree gating: require N consecutive raw-present/absent before flipping
  - Hold-last-good: keep previous valid reading during transient dropouts
  - Stale timeout: only invalidate after no good reading for stale_timeout_sec
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

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
            "left_mm": int(self.left_mm),
            "center_mm": int(self.center_mm),
            "right_mm": int(self.right_mm),
            "left_valid": bool(self.left_valid),
            "center_valid": bool(self.center_valid),
            "right_valid": bool(self.right_valid),
            "timestamp": float(self.timestamp),
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
            "left": bool(self.left),
            "center": bool(self.center),
            "right": bool(self.right),
            "any_present": bool(self.any_present),
            "count_present": int(self.count_present),
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


# ── Rolling median filter ───────────────────────────────────────────────────


def _median(values: list[int]) -> int:
    """Median of a list of ints."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return -1
    return s[n // 2]


class _ChannelFilter:
    """Per-channel rolling median + spike rejection + hold-last-good."""

    def __init__(
        self,
        *,
        window: int = 5,
        max_spike_mm: int = 600,
        stale_timeout_sec: float = 2.0,
    ):
        self._window = window
        self._max_spike_mm = max_spike_mm
        self._stale_timeout_sec = stale_timeout_sec
        self._history: deque[int] = deque(maxlen=window)
        self._last_good_mm: int = -1
        self._last_good_ts: float = 0.0
        self._filtered_mm: int = -1
        self._filtered_valid: bool = False

    def update(self, mm: int, valid: bool, now: float) -> tuple[int, bool]:
        """Feed a raw reading, return (filtered_mm, filtered_valid)."""
        if not valid or mm < 0:
            # No reading — check if stale
            if now - self._last_good_ts > self._stale_timeout_sec:
                self._filtered_mm = -1
                self._filtered_valid = False
                self._history.clear()
            # else: hold last good value
            return self._filtered_mm, self._filtered_valid

        # Spike rejection: if we have a median, reject wild jumps
        if len(self._history) >= 3:
            current_median = _median(list(self._history))
            if current_median > 0 and abs(mm - current_median) > self._max_spike_mm:
                # Spike detected — don't add to history, don't update
                return self._filtered_mm, self._filtered_valid

        # Add to rolling window
        self._history.append(mm)

        # Need at least 2 samples for a meaningful median
        if len(self._history) < 2:
            return self._filtered_mm, self._filtered_valid

        med = _median(list(self._history))
        self._filtered_mm = med
        self._filtered_valid = med > 0
        self._last_good_mm = med
        self._last_good_ts = now
        return self._filtered_mm, self._filtered_valid


class TofPresenceTracker:
    """Debounced per-sector presence with rolling median filter and spike rejection.

    Improvements over original:
      - Rolling median (window=5) per channel smooths jittery readings
      - Spike rejection prevents single wild readings from triggering presence
      - Consecutive-agree count (consec_agree) requires N matching raw states before flip
      - Hold-last-good during transient dropouts
    """

    def __init__(
        self,
        *,
        present_max_mm: float,
        absent_min_mm: float,
        debounce_present_sec: float,
        debounce_absent_sec: float,
        # New reliability params
        median_window: int = 5,
        max_spike_mm: int = 600,
        stale_timeout_sec: float = 2.0,
        consec_agree: int = 2,
    ):
        self.present_max_mm = present_max_mm
        self.absent_min_mm = absent_min_mm
        self.debounce_present_sec = debounce_present_sec
        self.debounce_absent_sec = debounce_absent_sec
        self.consec_agree = max(1, consec_agree)

        # Per-channel median filters
        self._filters = [
            _ChannelFilter(
                window=median_window,
                max_spike_mm=max_spike_mm,
                stale_timeout_sec=stale_timeout_sec,
            )
            for _ in range(3)
        ]

        self._stable = [False, False, False]
        self._pending = [False, False, False]
        self._pending_since = [0.0, 0.0, 0.0]
        self._consec_count = [0, 0, 0]  # how many consecutive readings agree with pending

    def _raw_present(self, mm: int, valid: bool) -> bool | None:
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
            # Apply per-channel median filter first
            filtered_mm, filtered_valid = self._filters[i].update(mm, valid, now)

            raw = self._raw_present(filtered_mm, filtered_valid)
            if raw is None:
                target = self._stable[i]
            else:
                target = raw

            if target != self._stable[i]:
                if target != self._pending[i]:
                    # New pending direction — reset counters
                    self._pending[i] = target
                    self._pending_since[i] = now
                    self._consec_count[i] = 1
                else:
                    # Same pending direction — increment consecutive count
                    self._consec_count[i] += 1

                debounce = (
                    self.debounce_present_sec
                    if target
                    else self.debounce_absent_sec
                )
                # Require both time debounce AND consecutive-agree count
                if (
                    now - self._pending_since[i] >= debounce
                    and self._consec_count[i] >= self.consec_agree
                ):
                    self._stable[i] = target
            else:
                self._pending[i] = target
                self._pending_since[i] = now
                self._consec_count[i] = 0

        count = sum(self._stable)
        return TofPresence(
            left=self._stable[0],
            center=self._stable[1],
            right=self._stable[2],
            any_present=count > 0,
            count_present=count,
        )
