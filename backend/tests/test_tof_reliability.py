#!/usr/bin/env python3
"""Tests for ToF reliability fixes (fix/tof-reliability).

Tests the Python-side filtering in tof_presence.py and tof_approach.py:
  - Rolling median filter rejects outlier spikes
  - Consecutive-agree gating prevents single-sample state flips
  - Hold-last-good keeps previous value during transient dropouts
  - Stale timeout eventually invalidates dead sensors
  - ConfirmedRisingEdge requires sustained presence before triggering approach

Run: python tests/test_tof_reliability.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tof_presence import TofPresenceTracker, TofSnapshot, TofPresence, _ChannelFilter


def make_snap(left=1000, center=1000, right=1000,
              lv=True, cv=True, rv=True, ts=None):
    """Helper to build TofSnapshot."""
    return TofSnapshot(
        left_mm=left, center_mm=center, right_mm=right,
        left_valid=lv, center_valid=cv, right_valid=rv,
        timestamp=ts or time.time(),
    )


# ── Rolling median filter tests ─────────────────────────────────────────────

def test_median_rejects_spike():
    """A single spike reading amid stable readings should be rejected."""
    filt = _ChannelFilter(window=5, max_spike_mm=600, stale_timeout_sec=2.0)
    now = time.time()

    # Feed 5 stable readings to build median
    for i in range(5):
        mm, valid = filt.update(800, True, now + i * 0.2)

    # Median should be 800
    assert valid, "Should be valid after 5 readings"
    assert abs(mm - 800) < 50, f"Expected ~800, got {mm}"

    # Now feed a spike at 3000mm — should be rejected
    mm_after, valid_after = filt.update(3000, True, now + 1.2)
    assert abs(mm_after - 800) < 50, f"Spike should be rejected, got {mm_after}"
    print("  ✓ Median filter rejects outlier spike")


def test_median_accepts_gradual_change():
    """Gradual distance changes should be tracked correctly."""
    filt = _ChannelFilter(window=5, max_spike_mm=600, stale_timeout_sec=2.0)
    now = time.time()

    # Start at 1000mm
    for i in range(5):
        filt.update(1000, True, now + i * 0.2)

    # Gradually move to 600mm in steps
    for i, dist in enumerate([900, 850, 750, 700, 650, 600]):
        mm, valid = filt.update(dist, True, now + 1.0 + i * 0.2)

    assert valid
    assert mm < 800, f"Should have tracked downward, got {mm}"
    print("  ✓ Median filter tracks gradual changes")


def test_hold_last_good_during_dropout():
    """Transient invalid readings should keep previous value."""
    filt = _ChannelFilter(window=5, max_spike_mm=600, stale_timeout_sec=2.0)
    now = time.time()

    # Build stable reading
    for i in range(5):
        filt.update(1000, True, now + i * 0.2)

    # One invalid reading — should hold last good
    mm, valid = filt.update(-1, False, now + 1.2)
    assert valid, "Should hold last good during transient dropout"
    assert abs(mm - 1000) < 50, f"Should hold ~1000, got {mm}"
    print("  ✓ Hold-last-good works during transient dropout")


def test_stale_timeout():
    """After stale_timeout_sec of no valid reads, should invalidate."""
    filt = _ChannelFilter(window=5, max_spike_mm=600, stale_timeout_sec=1.0)
    now = time.time()

    # Build stable reading
    for i in range(5):
        filt.update(1000, True, now + i * 0.2)

    # Feed invalid for >1s
    for i in range(6):
        mm, valid = filt.update(-1, False, now + 1.5 + i * 0.3)

    assert not valid, "Should invalidate after stale timeout"
    assert mm == -1, f"Should be -1 after stale timeout, got {mm}"
    print("  ✓ Stale timeout invalidates dead sensor")


# ── Presence tracker tests ──────────────────────────────────────────────────

def test_presence_no_false_trigger_on_spike():
    """A single spike reading should NOT trigger presence."""
    tracker = TofPresenceTracker(
        present_max_mm=1500,
        absent_min_mm=2000,
        debounce_present_sec=0.25,
        debounce_absent_sec=0.5,
        median_window=5,
        max_spike_mm=600,
        stale_timeout_sec=2.0,
        consec_agree=2,
    )

    # Feed several far/absent readings (no one there)
    for i in range(10):
        snap = make_snap(left=-1, center=-1, right=-1, lv=False, cv=False, rv=False)
        p = tracker.update(snap)

    assert not p.any_present, "Should start with no presence"

    # One spike reading on left at 500mm
    snap = make_snap(left=500, center=-1, right=-1, lv=True, cv=False, rv=False)
    p = tracker.update(snap)
    assert not p.left, "Single spike should NOT trigger left presence"

    # Back to no reading
    snap = make_snap(left=-1, center=-1, right=-1, lv=False, cv=False, rv=False)
    p = tracker.update(snap)
    assert not p.any_present, "Should return to no presence after spike"
    print("  ✓ Single spike does not trigger false presence")


def test_presence_triggers_on_sustained_readings():
    """Sustained close readings should eventually trigger presence."""
    tracker = TofPresenceTracker(
        present_max_mm=1500,
        absent_min_mm=2000,
        debounce_present_sec=0.05,  # very short for testing
        debounce_absent_sec=0.5,
        median_window=3,
        max_spike_mm=600,
        stale_timeout_sec=2.0,
        consec_agree=2,
    )

    triggered = False

    # Feed sustained close readings with real time gaps.
    # The tracker calls time.time() internally, so we need actual elapsed time
    # for the debounce timers to expire.
    for i in range(30):
        snap = TofSnapshot(
            left_mm=800, center_mm=-1, right_mm=-1,
            left_valid=True, center_valid=False, right_valid=False,
            timestamp=time.time(),
        )
        p = tracker.update(snap)
        if p.left:
            triggered = True
            break
        time.sleep(0.03)  # 30ms real gaps

    assert triggered, "Sustained close readings should trigger presence"
    print("  ✓ Sustained readings correctly trigger presence")


# ── Confirmed rising edge tests ─────────────────────────────────────────────

def test_confirmed_edge_rejects_transient():
    """ConfirmedRisingEdge should not trigger on transient presence."""
    from tof_approach import ConfirmedRisingEdge

    edge = ConfirmedRisingEdge(confirm_sec=0.4, sector_cooldown_sec=1.5)
    now = time.time()

    prev = TofPresence(False, False, False, False, 0)

    # One frame of presence
    pres = TofPresence(True, False, False, True, 1)
    snap = make_snap(left=500)
    result = edge.check(pres, prev, snap, left_right_only=True, present_max_mm=1500, now=now)
    assert result is None, "Should not trigger immediately"

    # Presence disappears before confirm_sec
    prev = pres
    pres = TofPresence(False, False, False, False, 0)
    result = edge.check(pres, prev, snap, left_right_only=True, present_max_mm=1500, now=now + 0.2)
    assert result is None, "Should not trigger after lost presence"
    print("  ✓ ConfirmedRisingEdge rejects transient presence")


def test_confirmed_edge_triggers_sustained():
    """ConfirmedRisingEdge should trigger after sustained presence."""
    from tof_approach import ConfirmedRisingEdge

    edge = ConfirmedRisingEdge(confirm_sec=0.3, sector_cooldown_sec=1.5)
    now = time.time()

    prev = TofPresence(False, False, False, False, 0)
    snap = make_snap(left=500)

    # First frame: rising edge detected
    pres = TofPresence(True, False, False, True, 1)
    result = edge.check(pres, prev, snap, left_right_only=True, present_max_mm=1500, now=now)
    assert result is None, "Should not trigger immediately"

    # Second frame: still present at now+0.35s (past confirm_sec)
    prev = pres
    result = edge.check(pres, prev, snap, left_right_only=True, present_max_mm=1500, now=now + 0.35)
    assert result == "left", f"Should trigger 'left' after sustained presence, got {result}"
    print("  ✓ ConfirmedRisingEdge triggers on sustained presence")


# ── Run all tests ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing ToF reliability fixes...\n")

    print("Rolling median filter:")
    test_median_rejects_spike()
    test_median_accepts_gradual_change()
    test_hold_last_good_during_dropout()
    test_stale_timeout()

    print("\nPresence tracker:")
    test_presence_no_false_trigger_on_spike()
    test_presence_triggers_on_sustained_readings()

    print("\nConfirmed rising edge:")
    test_confirmed_edge_rejects_transient()
    test_confirmed_edge_triggers_sustained()

    print("\n✅ All ToF reliability tests passed!")
