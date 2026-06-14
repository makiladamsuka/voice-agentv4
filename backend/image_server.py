"""HTTP server for frontend media assets (event posters, maps). Runs on port 8080."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class ImageServer:
    """Serve static media from backend/assets for the v2 frontend ImageDisplay."""

    def __init__(self, assets_dir: Path, port: int = 8080, host: str = "0.0.0.0"):
        self.assets_dir = assets_dir
        self.port = port
        self.host = host
        self.server: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self._server_host: str | None = None

    def _get_local_ip(self) -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "127.0.0.1"

    def start(self) -> None:
        if self.server is not None and self.thread is not None and self.thread.is_alive():
            print(f"Media server already running on port {self.port}")
            return

        self._server_host = self._get_local_ip() if self.host == "0.0.0.0" else self.host
        parent_dir = self.assets_dir.parent
        image_server = self

        class AssetHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self._serve_index()
                else:
                    self._serve_static(parent_dir)

            def _serve_index(self):
                html = f"""
                <html><head><title>Voice Agent Assets</title></head>
                <body style="font-family:sans-serif;padding:20px">
                <h1>Voice Agent Media Server</h1>
                <p>Serving event posters and maps for the frontend at port {image_server.port}.</p>
                <p>Robot debug dashboard: <code>http://&lt;pi-ip&gt;:8090/debug</code></p>
                </body></html>
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())

            def _serve_static(self, base_dir: Path):
                try:
                    if self.path.startswith("/assets/"):
                        file_path = base_dir / self.path[1:]
                    else:
                        file_path = base_dir / self.path.lstrip("/")

                    try:
                        file_path.resolve().relative_to(base_dir.resolve())
                    except ValueError:
                        self.send_response(403)
                        self.end_headers()
                        return

                    if not file_path.is_file():
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"File not found")
                        return

                    suffix = file_path.suffix.lower()
                    content_type = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                    }.get(suffix, "application/octet-stream")

                    content = file_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(f"Error: {e}".encode())

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.end_headers()

            def log_message(self, format, *args):
                pass

        try:
            self.server = HTTPServer((self.host, self.port), AssetHandler)
        except OSError as e:
            if e.errno == 98:
                print(f"Port {self.port} already in use (media server)")
                return
            raise

        def serve():
            print(f"Media server started: http://{self._server_host}:{self.port}")
            self.server.serve_forever()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            print("Media server stopped")

    def get_image_url(self, category: str, filename: str) -> str:
        host = self._server_host or self._get_local_ip()
        return f"http://{host}:{self.port}/assets/{category}/{filename}"
