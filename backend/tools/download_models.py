#!/usr/bin/env python3
"""Download ONNX vision models used by robot_eyes (YuNet face + YOLOv8n person)."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

MODELS = {
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
    "yolov8n.onnx": {
        "url": "https://github.com/CVHub520/X-AnyLabeling/releases/download/v0.1.0/yolov8n.onnx",
        "sha1": "68f864475d06e2ec4037181052739f268eeac38d",
    },
}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, *, force: bool = False) -> Path:
    info = MODELS[name]
    dest = BACKEND_DIR / name
    if dest.is_file() and not force:
        print(f"OK  {name} ({dest.stat().st_size // 1024} KiB)")
        return dest

    print(f"GET {info['url']}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(info["url"], tmp)
    digest = _sha1(tmp)
    expected = info.get("sha1")
    if expected and digest != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{name}: sha1 mismatch (got {digest}, expected {expected})")
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size // 1024} KiB)")
    return dest


def main() -> int:
    force = "--force" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not names:
        names = list(MODELS)
    for name in names:
        if name not in MODELS:
            print(f"Unknown model: {name}", file=sys.stderr)
            return 1
        download(name, force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
