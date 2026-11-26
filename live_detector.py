#!/usr/bin/env python3
"""
Real-time side-by-side person & flame detection viewer.
Streams via HTTP MJPEG — open http://localhost:8080 in your browser.
Left = original camera | Right = detection overlay
"""

import sys
import os
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from flame_detector import FlameDetector

# ── Globals ──
latest_frame = None
frame_lock = threading.Lock()
running = True


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves MJPEG stream at / and a simple HTML page."""

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rescue Rover - Live Detection</title>
<style>
  body { margin:0; background:#111; display:flex; justify-content:center;
         align-items:center; height:100vh; font-family:Arial; }
  img { max-width:98vw; max-height:95vh; border:2px solid #333; border-radius:8px; }
  .title { position:fixed; top:10px; left:50%; transform:translateX(-50%);
           color:#0f0; font-size:18px; background:rgba(0,0,0,0.7);
           padding:8px 20px; border-radius:20px; z-index:10; }
</style></head><body>
<div class="title">🔍 RESCUE ROVER — LIVE DETECTION</div>
<img src="/stream" />
</body></html>"""
            self.wfile.write(html.encode())

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while running:
                with frame_lock:
                    frame = latest_frame
                if frame is None:
                    time.sleep(0.03)
                    continue
                _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                except BrokenPipeError:
                    break
                time.sleep(0.033)  # ~30 fps cap
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence HTTP logs


def detection_loop():
    """Capture frames, run detection, update latest_frame."""
    global latest_frame, running

    # Load model
    print("[1/3] Loading YOLOv8n …")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("      ✓ Model loaded")

    # Flame detector
    print("[2/3] Loading flame detector (YOLO + Fire-vs-Light CNN) …")
    flame_det = FlameDetector(sensitivity=0.6)
    mode = "Fire-vs-Light CNN" if flame_det.classifier is not None else "Fallback"
    print(f"      ✓ Ready ({mode})")

    # Camera source: Pi stream (over Wi-Fi) or local webcam fallback
    PI_STREAM_URL = os.environ.get("PI_STREAM_URL", "")
    if PI_STREAM_URL:
        print(f"[3/3] Connecting to Pi stream: {PI_STREAM_URL} …")
        from pi_stream_reader import PiStream
        cap = PiStream(PI_STREAM_URL)
        # Wait up to 5 s for the first frame
        for _ in range(50):
            if cap.connected:
                break
            time.sleep(0.1)
        if not cap.connected:
            print("ERROR: Cannot connect to Pi stream! Is pi_camera_stream.py running?")
            running = False
            return
        print("      ✓ Pi stream connected\n")
    else:
        print("[3/3] Opening webcam (set PI_STREAM_URL to use Pi camera) …")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open webcam!")
            running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.5)
        print("      ✓ Webcam open\n")
    print("=" * 50)
    print("  Open in browser: http://localhost:8080")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    fps_counter = 0
    fps_time = time.time()
    fps_display = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        original = frame.copy()

        # ── Person detection ──
        results = model.predict(frame, conf=0.40, classes=[0], verbose=False, imgsz=320)
        person_count = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"PERSON {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                person_count += 1

        # ── Flame detection (run on clean original frame) ──
        flames = flame_det.detect(original)
        flame_det.draw(frame, flames)

        # ── FPS ──
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # ── Labels on original side ──
        cv2.putText(original, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # ── Labels on detection side ──
        cv2.putText(frame, "DETECTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps_display}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        status = f"Persons: {person_count} | Flames: {len(flames)}"
        cv2.putText(frame, status, (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Alert banner
        if person_count or flames:
            h = frame.shape[0]
            color = (0, 0, 255) if flames else (0, 200, 0)
            msg = ""
            if person_count and flames:
                msg = "!! PERSON + FLAME DETECTED !!"
                color = (0, 0, 255)
            elif person_count:
                msg = "PERSON DETECTED"
            elif flames:
                msg = "!! FLAME DETECTED !!"
            cv2.rectangle(frame, (0, h - 40), (frame.shape[1], h), color, -1)
            cv2.putText(frame, msg, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── Divider line ──
        divider = np.full((original.shape[0], 3, 3), (0, 255, 0), dtype=np.uint8)

        # ── Side by side ──
        combined = np.hstack([original, divider, frame])

        with frame_lock:
            latest_frame = combined

    cap.release()


if __name__ == "__main__":
    # Start detection in background thread
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    # Start HTTP server
    try:
        server = ThreadingHTTPServer(("0.0.0.0", 8080), MJPEGHandler)
        print("HTTP server starting on port 8080 …\n")
        server.serve_forever()
    except KeyboardInterrupt:
        running = False
        print("\nStopped.")
