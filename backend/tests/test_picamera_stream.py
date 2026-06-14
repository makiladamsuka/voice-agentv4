#!/usr/bin/env python3
"""
Picamera2 + YuNet face detection + MJPEG stream — standalone hardware test.

  cd backend
  python tests/test_picamera_stream.py

Stop start_robot.py / robot_eyes.py first — only one process can own the camera.

Uses the same preview/sRGB ISP pipeline as rpicam-hello --qt-preview (1640x1232 full sensor).
MJPEG compression still softens vs native Qt preview, but image quality should be much closer.

Browser:
  http://<pi-ip>:8092/          live preview page
  http://<pi-ip>:8092/stream    raw MJPEG feed
  http://<pi-ip>:8092/api/state face/camera JSON status
"""

from __future__ import annotations

import argparse
import io
import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401
from _bootstrap import BACKEND_ROOT

from camera_color import (
    DETECTION_BGR_MODE,
    apply_camera_controls,
    configure_picamera,
    configure_wide_fov_camera,
    detect_faces_yunet,
    detect_faces_yunet_fast,
    frame_to_rgb,
    log_color_pipeline_verification,
    probe_yunet_bgr_mode,
    verify_color_pipeline,
)
from robot_config import load_config

DEFAULT_HTTP_PORT = 8092

_frame_lock = threading.Lock()
_latest_frame: np.ndarray | None = None

_state_lock = threading.Lock()
_state: dict = {
    "camera_ok": False,
    "detector_ok": False,
    "face_count": 0,
    "detection_bgr_mode": "",
    "main_res": [],
    "detect_res": [],
    "stream_res": [],
    "preview_pipeline": True,
    "fps_actual": 0.0,
    "frames": 0,
    "last_error": "",
}

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Picamera + Face Detect Test</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
    h1 { font-size: 18px; margin: 0 0 12px; }
    .wrap { background: #000; border-radius: 8px; overflow: hidden; max-width: min(960px, 100vw - 32px); }
    img { display: block; width: 100%; height: auto; image-rendering: auto; }
    pre { background: #1a1a1a; padding: 12px; border-radius: 8px; margin-top: 12px; font-size: 12px; }
    .ok { color: #3dd68c; }
    .warn { color: #f5a623; }
  </style>
</head>
<body>
  <h1>Picamera + YuNet test stream</h1>
  <div class="wrap"><img src="/stream" alt="camera feed"></div>
  <pre id="status">Loading…</pre>
  <script>
    async function poll() {
      try {
        const r = await fetch('/api/state');
        const s = await r.json();
        const cls = s.face_count > 0 ? 'ok' : 'warn';
        document.getElementById('status').innerHTML =
          `<span class="${cls}">faces=${s.face_count}</span>  ` +
          `mode=${s.detection_bgr_mode}  fps=${s.fps_actual.toFixed(1)}  ` +
          `frames=${s.frames}\\n` +
          `camera=${s.camera_ok}  detector=${s.detector_ok}  ` +
          `main=${s.main_res}  detect=${s.detect_res}  stream=${s.stream_res}` +
          (s.last_error ? `\\nerror: ${s.last_error}` : '');
      } catch (e) {
        document.getElementById('status').textContent = 'API error: ' + e;
      }
    }
    poll();
    setInterval(poll, 500);
  </script>
</body>
</html>
"""


def _map_coords_to_stream_preview(
    fx: float,
    fy: float,
    fw: float,
    fh: float,
    re_x: float,
    re_y: float,
    le_x: float,
    le_y: float,
    *,
    detect_res: tuple[int, int],
    stream_res: tuple[int, int],
    rotate_180: bool,
    wide_fov: bool = False,
) -> tuple[int, int, int, int, int, int, int, int]:
    w, h = detect_res
    if not wide_fov and rotate_180:
        fx = w - fx - fw
        fy = h - fy - fh
        re_x, re_y = w - re_x, h - re_y
        le_x, le_y = w - le_x, h - le_y
    scale_x = stream_res[0] / w
    scale_y = stream_res[1] / h
    return (
        int(fx * scale_x),
        int(fy * scale_y),
        int(fw * scale_x),
        int(fh * scale_y),
        int(re_x * scale_x),
        int(re_y * scale_y),
        int(le_x * scale_x),
        int(le_y * scale_y),
    )


def _draw_faces_on_stream(
    stream_frame: np.ndarray,
    faces: np.ndarray,
    *,
    detect_res: tuple[int, int],
    stream_res: tuple[int, int],
    rotate_180: bool,
    wide_fov: bool = False,
) -> None:
    for face in faces:
        fx, fy, fw, fh = face[0:4]
        re_x, re_y = face[4], face[5]
        le_x, le_y = face[6], face[7]
        fx_s, fy_s, fw_s, fh_s, re_x_s, re_y_s, le_x_s, le_y_s = _map_coords_to_stream_preview(
            fx, fy, fw, fh, re_x, re_y, le_x, le_y,
            detect_res=detect_res,
            stream_res=stream_res,
            rotate_180=rotate_180,
            wide_fov=wide_fov,
        )
        cv2.rectangle(stream_frame, (fx_s, fy_s), (fx_s + fw_s, fy_s + fh_s), (0, 255, 0), 2)
        cv2.circle(stream_frame, (re_x_s, re_y_s), 4, (255, 0, 0), -1)
        cv2.circle(stream_frame, (le_x_s, le_y_s), 4, (255, 0, 0), -1)


def _vision_worker(
    picam2,
    detector,
    cfg,
    stop: threading.Event,
    *,
    flip_stream: bool = False,
) -> None:
    global _latest_frame

    main_res = tuple(cfg.camera.main_res)
    detect_res = tuple(cfg.camera.detect_res)
    stream_res = tuple(cfg.camera.stream_res)
    rotate_180 = bool(cfg.camera.rotate_180)
    stream_swap_rb = bool(cfg.camera.stream_swap_rb)
    wide_fov = bool(getattr(cfg.camera, "wide_fov", False))
    use_preview = bool(getattr(cfg.camera, "use_preview_pipeline", True)) and not wide_fov
    raw_sensor_res = tuple(getattr(cfg.camera, "raw_sensor_res", [3280, 2464]))
    interval = 1.0 / max(1.0, float(cfg.stream.vision_fps))

    frame_count = 0
    fps_window_start = time.perf_counter()
    fps_window_frames = 0

    while not stop.is_set():
        loop_start = time.perf_counter()
        try:
            large_frame = picam2.capture_array()
            if wide_fov:
                frame_raw = cv2.resize(large_frame, detect_res, interpolation=cv2.INTER_AREA)
                if rotate_180:
                    frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)
                stream_frame = cv2.resize(frame_raw, stream_res, interpolation=cv2.INTER_AREA)
                if stream_swap_rb:
                    stream_frame = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)
                faces = detect_faces_yunet_fast(
                    detector,
                    frame_raw,
                    input_size=detect_res,
                )
                mode = DETECTION_BGR_MODE
            else:
                rgb_frame = frame_to_rgb(
                    large_frame,
                    legacy_swap_rb=stream_swap_rb and not use_preview,
                )
                frame_raw = cv2.resize(rgb_frame, detect_res, interpolation=cv2.INTER_AREA)
                if stream_res == main_res:
                    stream_frame = rgb_frame.copy()
                else:
                    stream_frame = cv2.resize(rgb_frame, stream_res, interpolation=cv2.INTER_AREA)
                faces, mode = detect_faces_yunet(
                    detector,
                    frame_raw,
                    input_size=detect_res,
                    rotate_180=rotate_180,
                )
            if flip_stream:
                stream_frame = cv2.rotate(stream_frame, cv2.ROTATE_180)

            face_count = 0 if faces is None else len(faces)
            if face_count:
                _draw_faces_on_stream(
                    stream_frame,
                    faces,
                    detect_res=detect_res,
                    stream_res=stream_res,
                    rotate_180=rotate_180,
                    wide_fov=wide_fov,
                )

            with _frame_lock:
                _latest_frame = stream_frame.copy()

            frame_count += 1
            fps_window_frames += 1
            elapsed = time.perf_counter() - fps_window_start
            fps_actual = fps_window_frames / elapsed if elapsed >= 1.0 else 0.0
            if elapsed >= 1.0:
                fps_window_start = time.perf_counter()
                fps_window_frames = 0

            with _state_lock:
                _state["face_count"] = face_count
                _state["detection_bgr_mode"] = mode
                _state["fps_actual"] = fps_actual
                _state["frames"] = frame_count
                _state["last_error"] = ""

        except Exception as exc:
            with _state_lock:
                _state["last_error"] = str(exc)
            print(f"Vision worker error: {exc}")

        sleep_for = interval - (time.perf_counter() - loop_start)
        if sleep_for > 0:
            stop.wait(sleep_for)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class CameraTestHandler(BaseHTTPRequestHandler):
    jpeg_quality: int = 70
    stream_fps: int = 8

    def log_message(self, format: str, *args) -> None:
        pass

    def _send_bytes(self, data: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, code: int = 200) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8", code)

    def _stream_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        interval = 1.0 / max(1, self.stream_fps)
        try:
            while True:
                with _frame_lock:
                    frame = None if _latest_frame is None else _latest_frame.copy()

                if frame is None:
                    time.sleep(0.05)
                    continue

                buf = io.BytesIO()
                jpg = encode_stream_jpeg_rgb(frame, int(self.jpeg_quality))
                if jpg is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/debug"):
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            with _state_lock:
                payload = dict(_state)
            self._send_json(payload)
            return
        if path == "/stream":
            self._stream_mjpeg()
            return
        self.send_error(404)


def main() -> int:
    parser = argparse.ArgumentParser(description="Picamera2 + YuNet + MJPEG stream test")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument(
        "--http-port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"HTTP port (default {DEFAULT_HTTP_PORT}; robot uses 8090)",
    )
    parser.add_argument("--probe-only", action="store_true", help="Init camera+model, probe one frame, exit")
    parser.add_argument("--main-res", nargs=2, type=int, metavar=("W", "H"), help="Override main capture size")
    parser.add_argument("--stream-res", nargs=2, type=int, metavar=("W", "H"), help="Override MJPEG stream size")
    parser.add_argument(
        "--flip-stream",
        action="store_true",
        help="Rotate stream 180° (only if preview is upside down on your mount)",
    )
    parser.add_argument(
        "--legacy-video",
        action="store_true",
        help="Use video/RGB888 pipeline instead of rpicam-hello preview/sRGB",
    )
    args = parser.parse_args()

    cfg = load_config(BACKEND_ROOT / "config.yaml")
    if args.legacy_video:
        cfg.camera.use_preview_pipeline = False
    if args.main_res:
        cfg.camera.main_res = list(args.main_res)
    if args.stream_res:
        cfg.camera.stream_res = list(args.stream_res)
    main_res = tuple(cfg.camera.main_res)
    detect_res = tuple(cfg.camera.detect_res)
    stream_res = tuple(cfg.camera.stream_res)
    model_path = BACKEND_ROOT / cfg.camera.face_model_path

    with _state_lock:
        _state["main_res"] = list(main_res)
        _state["detect_res"] = list(detect_res)
        _state["stream_res"] = list(stream_res)
        _state["preview_pipeline"] = bool(getattr(cfg.camera, "use_preview_pipeline", True))

    try:
        from picamera2 import Picamera2
    except ImportError:
        print("picamera2 not installed — run: sudo apt install python3-picamera2")
        return 1

    if not model_path.is_file():
        print(f"Missing face model: {model_path}")
        print("Download with: python tools/download_models.py")
        return 1

    print("Initializing Picamera2…")
    picam2 = Picamera2()
    wide_fov = bool(getattr(cfg.camera, "wide_fov", False))
    use_preview = bool(getattr(cfg.camera, "use_preview_pipeline", True)) and not wide_fov
    if wide_fov:
        raw_sensor_res = tuple(getattr(cfg.camera, "raw_sensor_res", [3280, 2464]))
        configure_wide_fov_camera(picam2, main_res, raw_sensor_res=raw_sensor_res)
        pipeline = f"wide-FOV video/RGB888 raw {raw_sensor_res[0]}x{raw_sensor_res[1]}"
    else:
        configure_picamera(picam2, main_res, use_preview_pipeline=use_preview)
        pipeline = "preview/sRGB (rpicam-hello)" if use_preview else "video/RGB888"
    picam2.start()
    apply_camera_controls(
        picam2,
        awb_mode=cfg.camera.awb_mode,
        colour_gains=cfg.camera.colour_gains,
        sharpness=getattr(cfg.camera, "sharpness", 1.0),
        noise_reduction=getattr(cfg.camera, "noise_reduction", "high"),
    )
    with _state_lock:
        _state["camera_ok"] = True
        _state["wide_fov"] = wide_fov
    print(f"Camera started ({pipeline}): main {main_res[0]}x{main_res[1]}, detect {detect_res[0]}x{detect_res[1]}, stream {stream_res[0]}x{stream_res[1]}")

    print("Loading YuNet face detector…")
    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=detect_res,
        score_threshold=cfg.camera.confidence_threshold,
        nms_threshold=cfg.camera.nms_threshold,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )
    with _state_lock:
        _state["detector_ok"] = True
    print(f"YuNet loaded from {model_path.name}")

    if wide_fov:
        capture = cv2.resize(picam2.capture_array(), detect_res, interpolation=cv2.INTER_AREA)
        if cfg.camera.rotate_180:
            capture = cv2.rotate(capture, cv2.ROTATE_180)
        color_stats = verify_color_pipeline(
            cv2.cvtColor(capture, cv2.COLOR_BGR2RGB) if cfg.camera.stream_swap_rb else capture
        )
        mode, probe_count = probe_yunet_bgr_mode(
            detector, capture, input_size=detect_res, rotate_180=False
        )
    else:
        capture = cv2.resize(
            frame_to_rgb(
                picam2.capture_array(),
                legacy_swap_rb=cfg.camera.stream_swap_rb and not use_preview,
            ),
            detect_res,
        )
        color_stats = verify_color_pipeline(capture)
        mode, probe_count = probe_yunet_bgr_mode(
            detector, capture, input_size=detect_res, rotate_180=cfg.camera.rotate_180
        )
    log_color_pipeline_verification(color_stats, detection_mode=mode, face_probe_count=probe_count)
    print(f"Detection BGR mode: {mode} (global={DETECTION_BGR_MODE})")

    if args.probe_only:
        picam2.stop()
        picam2.close()
        print("Probe complete.")
        return 0

    stop = threading.Event()
    worker = threading.Thread(
        target=_vision_worker,
        args=(picam2, detector, cfg, stop),
        kwargs={"flip_stream": args.flip_stream},
        daemon=True,
        name="vision-worker",
    )
    worker.start()

    CameraTestHandler.jpeg_quality = cfg.stream.jpeg_quality
    CameraTestHandler.stream_fps = cfg.stream.fps
    server = ThreadingHTTPServer((args.host, args.http_port), CameraTestHandler)

    local_url = f"http://127.0.0.1:{args.http_port}/"
    if args.host == "0.0.0.0":
        print(f"MJPEG test server on port {args.http_port}")
        print(f"  Local:   {local_url}")
        print(f"  Network: http://<pi-ip>:{args.http_port}/")
        print(f"  Stream:  http://<pi-ip>:{args.http_port}/stream")
    else:
        print(f"Open {local_url}")
    print("Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        stop.set()
        worker.join(timeout=3.0)
        server.shutdown()
        picam2.stop()
        picam2.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
