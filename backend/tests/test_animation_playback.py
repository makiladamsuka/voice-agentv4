#!/usr/bin/env python3
"""Offline tests for Botango animation loading and playback sampling."""

from __future__ import annotations

import time

import _bootstrap  # noqa: F401
from _bootstrap import BACKEND_ROOT

from animation_player import AnimationPlayer
from botango_loader import load_botango_commands_file

ANIMATIONS_JSON = BACKEND_ROOT / "animations" / "AnimationCommands.json"


def test_botango_export_loads_four_clips() -> None:
    src = ANIMATIONS_JSON
    clips = load_botango_commands_file(src)
    ids = {c.clip_id for c in clips}
    assert len(clips) == 4, f"expected 4 clips, got {len(clips)}: {ids}"
    for expected in (
        "Left_hand_bye",
        "Right_hand_bye",
        "Display_showing",
        "location_showing",
    ):
        assert expected in ids, f"missing clip {expected}"


def test_head_and_arm_tracks_present() -> None:
    clips = load_botango_commands_file(ANIMATIONS_JSON)
    display = next(c for c in clips if c.clip_id == "Display_showing")
    servos = {tr.servo for tr in display.tracks}
    assert "head_pan" in servos
    assert "head_tilt" in servos
    assert any(s.startswith("arm_") for s in servos)


def test_player_samples_active_clip() -> None:
    player = AnimationPlayer()
    for clip in load_botango_commands_file(ANIMATIONS_JSON):
        player.register_clip(clip)

    t0 = 1000.0
    ok = player.play("Display_showing", loop=False, now=t0)
    assert ok

    sample = player.sample(now=t0 + 0.5)
    assert sample, "expected non-empty sample mid-clip"

    after_end = player.sample(now=t0 + 30.0)
    assert after_end == {}, "non-loop clip should end"


if __name__ == "__main__":
    test_botango_export_loads_four_clips()
    test_head_and_arm_tracks_present()
    test_player_samples_active_clip()
    print("animation playback tests passed")
