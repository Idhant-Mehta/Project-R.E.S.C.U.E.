"""HTTP server (stdlib only, light enough for the Pi): dashboard, MJPEG stream, JSON API.

GET  /                    dashboard
GET  /stream              MJPEG video (annotated)
GET  /api/state           live + room-wise state (JSON)
GET  /api/report          generate and return the inspection report (HTML)
GET  /api/snapshot/<id>   room snapshot JPEG
POST /api/room/enter      {"name": "Room 2"}   (name optional)
POST /api/room/exit
POST /api/room/count      {"count": 2} or {"count": null}  operator-confirmed people
POST /api/calibrate       capture clear-air reference for smoke severity

Security note: no authentication. Run it only on the rover's own closed Wi-Fi network.
"""

import json
import logging
import os
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from . import PACKAGE_DIR

log = logging.getLogger("server")


def make_handler(app, stream_fps):
    dashboard_path = os.path.join(PACKAGE_DIR, "web", "dashboard.html")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body)
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith(("text", "application/json")) else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                with open(dashboard_path, "rb") as f:
                    return self._send(200, f.read(), "text/html")
            if p == "/api/state":
                return self._send(200, app.state())
            if p == "/api/report":
                doc, path = app.mission.report_html()
                log.info("Report saved: %s", path)
                return self._send(200, doc, "text/html")
            if p.startswith("/api/snapshot/"):
                try:
                    sp = app.mission.snapshot_path(int(p.rsplit("/", 1)[1]))
                except ValueError:
                    sp = None
                if not sp or not os.path.exists(sp):
                    return self._send(404, {"error": "no snapshot"})
                with open(sp, "rb") as f:
                    return self._send(200, f.read(), "image/jpeg")
            if p == "/stream":
                return self._stream()
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})
            if p == "/api/room/enter":
                r = app.enter_room(str(data.get("name") or ""))
                return self._send(200, r.to_dict())
            if p == "/api/room/exit":
                r = app.exit_room()
                return self._send(200, r.to_dict() if r else {"error": "no active room"})
            if p == "/api/room/count":
                c = data.get("count")
                if c is not None and (not isinstance(c, int) or c < 0 or c > 50):
                    return self._send(400, {"error": "count must be 0..50 or null"})
                r = app.mission.set_operator_count(c)
                return self._send(200, r.to_dict() if r else {"error": "no room"})
            if p == "/api/calibrate":
                app.request_calibration()
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            period = 1.0 / max(1, stream_fps)
            last = None
            try:
                while app.running:
                    jpg = app.latest_jpeg()
                    if jpg is None or jpg is last:
                        time.sleep(0.01)
                        continue
                    last = jpg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(period)
            except OSError:
                pass  # client disconnected (BrokenPipe on Linux, ConnectionAborted/Reset on Windows)
            self.close_connection = True

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(app, host, port, stream_fps):
    srv = ThreadingHTTPServer((host, port), make_handler(app, stream_fps))
    srv.daemon_threads = True
    log.info("Dashboard: http://%s:%d", "localhost" if host in ("0.0.0.0", "") else host, port)
    return srv
