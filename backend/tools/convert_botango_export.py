#!/usr/bin/env python3
"""Convert Bottango AnimationCommands.json into per-clip runtime JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botango_loader import load_botango_commands_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Bottango AnimationCommands export")
    parser.add_argument(
        "input_json",
        nargs="?",
        default="animations/AnimationCommands.json",
        help="Botango export JSON (default: animations/AnimationCommands.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="animations/converted",
        help="Output directory for runtime clip JSON files",
    )
    args = parser.parse_args()

    src = Path(args.input_json)
    if not src.is_absolute():
        src = Path(__file__).resolve().parents[1] / src
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[1] / out_dir

    clips = load_botango_commands_file(src)
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        payload = {
            "version": 1,
            "clip_id": clip.clip_id,
            "duration_ms": clip.duration_ms,
            "fps": 30,
            "tracks": [
                {
                    "servo": tr.servo,
                    "units": tr.units,
                    "keys": [
                        {"t_ms": int(k.t_ms), "v": round(k.v, 2), "ease": k.ease}
                        for k in tr.keys
                    ],
                }
                for tr in clip.tracks
            ],
            "blend": {
                servo: {"mode": b.mode, "weight": b.weight}
                for servo, b in clip.blends.items()
            },
            "meta": {"source": "botango-converted"},
        }
        dst = out_dir / f"{clip.clip_id}.json"
        with dst.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"Wrote {dst}")

    print(f"Converted {len(clips)} clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
