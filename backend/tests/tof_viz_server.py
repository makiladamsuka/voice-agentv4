#!/usr/bin/env python3
"""
Live ToF radar visualization (3× VL53L0X at front, ±45° left/right).

  cd backend
  python tests/tof_viz_server.py --port /dev/ttyUSB0

Open in browser: http://<pi-ip>:8091/

Stop tests/test_tof_sensors.py / start_robot.py first — one serial client at a time.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import _bootstrap  # noqa: F401
from _bootstrap import BACKEND_ROOT

from arduino_servo import ArduinoServoLink
from esp32_serial import connect_esp32
from robot_config import load_config
from tof_presence import TofSnapshot, sanitize_tof_snapshot

STATIC_DIR = BACKEND_ROOT / "static"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8091
DEFAULT_POLL_HZ = 8.0

_lock = threading.Lock()
_latest: dict = {
    "left_mm": -1,
    "center_mm": -1,
    "right_mm": -1,
    "left_valid": False,
    "center_valid": False,
    "right_valid": False,
    "timestamp": 0.0,
    "connected": False,
}


def _snapshot_to_api(snap: TofSnapshot) -> dict:
    return {
        "left_mm": snap.left_mm,
        "center_mm": snap.center_mm,
        "right_mm": snap.right_mm,
        "left_valid": snap.left_valid,
        "center_valid": snap.center_valid,
        "right_valid": snap.right_valid,
        "timestamp": snap.timestamp,
        "connected": True,
    }


def _poll_loop(
    link: ArduinoServoLink,
    hz: float,
    stop: threading.Event,
    *,
    min_valid_mm: int,
) -> None:
    interval = 1.0 / max(0.5, hz)
    while not stop.is_set():
        snap = link.poll_tof(timeout=2.0)
        if snap is not None:
            snap = sanitize_tof_snapshot(snap, min_valid_mm=min_valid_mm)
            with _lock:
                _latest.update(_snapshot_to_api(snap))
        else:
            with _lock:
                _latest["connected"] = link.connected
        stop.wait(interval)


class TofVizHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        pass

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/tof", "/tof_viz.html"):
            html_path = STATIC_DIR / "tof_viz.html"
            if not html_path.is_file():
                self._send_bytes(b"tof_viz.html missing", "text/plain", 404)
                return
            self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/tof":
            with _lock:
                payload = dict(_latest)
            self._send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        self.send_error(404)


def main() -> int:
    parser = argparse.ArgumentParser(description="ToF radar web visualization")
    parser.add_argument("--port", default="", help="ESP32 serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--hz", type=float, default=DEFAULT_POLL_HZ, help="ToF poll rate")
    parser.add_argument("--host", default=DEFAULT_HOST, help="HTTP bind address")
    parser.add_argument(
        "--http-port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help="HTTP port for browser UI",
    )
    args = parser.parse_args()

    cfg = load_config()
    min_valid_mm = int(cfg.tof.min_valid_mm)

    link = connect_esp32(port=args.port, baud=args.baud, prepare=False)
    if link is None:
        print("Could not connect to ESP32. Check USB and stop other serial users.")
        return 1

    stop = threading.Event()
    worker = threading.Thread(
        target=_poll_loop,
        args=(link, args.hz, stop),
        kwargs={"min_valid_mm": min_valid_mm},
        daemon=True,
    )
    worker.start()

    server = ThreadingHTTPServer((args.host, args.http_port), TofVizHandler)
    url = f"http://127.0.0.1:{args.http_port}/"
    if args.host == "0.0.0.0":
        print(f"ToF viz server on port {args.http_port}")
        print(f"  Local:   {url}")
        print(f"  Network: http://<pi-ip>:{args.http_port}/")
    else:
        print(f"Open {url}")
    print("Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop.set()
        worker.join(timeout=2.0)
        server.shutdown()
        link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
