"""Unit tests for one-shot ToF approach state machine."""

from __future__ import annotations

import time

import _bootstrap  # noqa: F401

from tof_approach import (
    ApproachPhase,
    TofApproachController,
    detect_rising_edge,
    pick_latch_sector,
    sector_pan_offset,
)
from tof_presence import TofPresence, TofSnapshot


def _snap(
    left_mm: int = -1,
    center_mm: int = -1,
    right_mm: int = -1,
    *,
    left_valid: bool = False,
    center_valid: bool = False,
    right_valid: bool = False,
) -> TofSnapshot:
    return TofSnapshot(
        left_mm=left_mm,
        center_mm=center_mm,
        right_mm=right_mm,
        left_valid=left_valid,
        center_valid=center_valid,
        right_valid=right_valid,
        timestamp=time.time(),
    )


def _pres(left=False, center=False, right=False) -> TofPresence:
    count = int(left) + int(center) + int(right)
    return TofPresence(left, center, right, count > 0, count)


def _arm(ctrl: TofApproachController, now: float = 100.0) -> None:
    ctrl.sync_presence(_pres(), now - ctrl.startup_grace_sec - 1.0)


def test_sector_pan_offset():
    assert sector_pan_offset("left", 30.0) == -30.0
    assert sector_pan_offset("right", 30.0) == 30.0
    assert sector_pan_offset("center", 30.0) == 0.0


def test_rising_edge_right_only():
    prev = _pres()
    cur = _pres(right=True)
    assert detect_rising_edge(cur, prev, left_right_only=True) == "right"
    assert detect_rising_edge(cur, cur, left_right_only=True) is None


def test_startup_grace_blocks_rising_edge():
    ctrl = TofApproachController(startup_grace_sec=2.0)
    ctrl.sync_presence(_pres(), now=0.0)
    action = ctrl.tick(
        _snap(right_mm=400, right_valid=True),
        _pres(right=True),
        face_locked=False,
        pan_current=80.0,
        skip_motion=False,
        now=0.5,
    )
    assert action is None
    assert ctrl.phase == ApproachPhase.IDLE


def test_sync_presence_avoids_immediate_rising():
    ctrl = TofApproachController(startup_grace_sec=0.0, boot_orient=False)
    pres = _pres(left=True, right=True)
    ctrl.sync_presence(pres, now=1.0)
    action = ctrl.tick(
        _snap(left_mm=300, right_mm=800, left_valid=True, right_valid=True),
        pres,
        face_locked=False,
        pan_current=80.0,
        skip_motion=False,
        now=1.0,
    )
    assert action is None


def test_orient_on_rising_edge_once():
    ctrl = TofApproachController(
        pan_min=40.0,
        pan_max=120.0,
        head_turn_deg=30.0,
        pan_step_deg=30.0,
        use_base=False,
        startup_grace_sec=0.0,
    )
    _arm(ctrl, now=10.0)
    empty = _pres()
    ctrl._prev_presence = empty
    snap = _snap(right_mm=400, right_valid=True)
    pres = _pres(right=True)

    action = ctrl.tick(
        snap, pres, face_locked=False, pan_current=80.0, skip_motion=False, now=10.0
    )
    assert action is not None
    assert action.pan_delta_deg > 0
    assert ctrl.latched_sector == "right"

    ctrl._prev_presence = pres
    pres2 = _pres(right=True, left=True)
    snap2 = _snap(right_mm=400, left_mm=300, right_valid=True, left_valid=True)
    ctrl.tick(
        snap2, pres2, face_locked=False, pan_current=110.0, skip_motion=False, now=10.1
    )
    assert ctrl.latched_sector == "right"


def test_boot_orient_once_after_grace():
    ctrl = TofApproachController(
        pan_min=40.0,
        pan_max=120.0,
        boot_orient=True,
        startup_grace_sec=1.0,
        use_base=False,
    )
    ctrl.sync_presence(_pres(right=True), now=0.0)
    assert (
        ctrl.tick(
            _snap(right_mm=400, right_valid=True),
            _pres(right=True),
            face_locked=False,
            pan_current=80.0,
            skip_motion=False,
            now=0.5,
        )
        is None
    )
    action = ctrl.tick(
        _snap(right_mm=400, right_valid=True),
        _pres(right=True),
        face_locked=False,
        pan_current=80.0,
        skip_motion=False,
        now=2.0,
    )
    assert action is not None
    assert ctrl._boot_orient_done is True


def test_one_base_nudge_per_event():
    ctrl = TofApproachController(
        pan_min=40.0,
        pan_max=120.0,
        use_base=True,
        base_nudge_deg=12.0,
        startup_grace_sec=0.0,
    )
    _arm(ctrl)
    ctrl._start_event("right", now=0.0)
    action = ctrl.tick(
        _snap(right_mm=400, right_valid=True),
        _pres(right=True),
        face_locked=False,
        pan_current=80.0,
        pan_target=80.0,
        skip_motion=False,
        now=0.1,
    )
    assert action is not None
    assert action.base_nudge_deg == -12.0


def test_base_fires_on_first_orient_tick():
    ctrl = TofApproachController(
        pan_min=40.0,
        pan_max=120.0,
        use_base=True,
        base_nudge_deg=12.0,
        startup_grace_sec=0.0,
    )
    _arm(ctrl, now=1.0)
    empty = _pres()
    ctrl._prev_presence = empty
    action = ctrl.tick(
        _snap(left_mm=400, left_valid=True),
        _pres(left=True),
        face_locked=False,
        pan_current=80.0,
        pan_target=80.0,
        skip_motion=False,
        now=1.0,
    )
    assert action is not None
    assert action.base_nudge_deg == 12.0
