#!/usr/bin/env python3
"""
Raspberry Pi camera -> MJPEG stream over Wi-Fi.

Runs ON THE RASPBERRY PI. Your laptop (OpenCV + YOLO) reads the stream with:
    cv2.VideoCapture("http://<PI_IP>:8000/stream.mjpg")

The Pi only captures and JPEG-encodes (hardware encoder when available),
so it stays cool and fast; all AI runs on the laptop.

Run:
    python3 pi_camera_stream.py                      # 640x480 @ 30 fps, port 8000
    python3 pi_camera_stream.py --width 1280 --height 720 --fps 20
Then open http://<PI_IP>:8000 in any browser to check the picture.
Find the Pi's IP with:  hostname -I
"""

import argparse
import io
import logging
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.outputs import FileOutput

try:  # hardware MJPEG encoder (Pi 4); falls back to software JPEG
    from picamera2.encoders import MJPEGEncoder as Encoder
except ImportError:
    from picamera2.encoders import JpegEncoder as Encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("pi-stream")

PAGE = b"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Pi Camera</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100vw;max-height:100vh}</style></head>
<body><img src="/stream.mjpg" alt="Pi camera stream"></body></html>"""


class StreamingOutput(io.BufferedIOBase):
    """Holds the latest JPEG frame; wakes up every connected client."""

    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


output = StreamingOutput()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            log.info("Client connected: %s", self.client_address[0])
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as exc:  # client closed the connection
                log.info("Client %s disconnected (%s)", self.client_address[0], exc.__class__.__name__)
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        pass


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--hflip", action="store_true", help="mirror image (camera mounted reversed)")
    ap.add_argument("--vflip", action="store_true", help="flip vertically (camera mounted upside down)")
    ap.add_argument("--noir", action="store_true",
                    help="use the NoIR colour profile (for Pi Camera V2 NoIR, no IR filter)")
    ap.add_argument("--ev", type=float, default=0.0,
                    help="exposure compensation, -8..8 (e.g. -1.5 stops flames blowing out to white)")
    ap.add_argument("--awb", default="auto",
                    choices=["auto", "incandescent", "tungsten", "fluorescent", "indoor", "daylight", "cloudy"],
                    help="white-balance mode")
    args = ap.parse_args()

    from libcamera import controls as lc
    awb = {"auto": lc.AwbModeEnum.Auto, "incandescent": lc.AwbModeEnum.Incandescent,
           "tungsten": lc.AwbModeEnum.Tungsten, "fluorescent": lc.AwbModeEnum.Fluorescent,
           "indoor": lc.AwbModeEnum.Indoor, "daylight": lc.AwbModeEnum.Daylight,
           "cloudy": lc.AwbModeEnum.Cloudy}[args.awb]

    tuning = Picamera2.load_tuning_file("imx219_noir.json") if args.noir else None
    picam2 = Picamera2(tuning=tuning)
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height)},
        controls={"FrameRate": args.fps, "ExposureValue": args.ev, "AwbMode": awb},
    )
    if args.hflip or args.vflip:
        from libcamera import Transform
        config["transform"] = Transform(hflip=args.hflip, vflip=args.vflip)
    picam2.configure(config)
    picam2.start_recording(Encoder(), FileOutput(output))
    log.info("Streaming %dx%d @ %d fps with %s on port %d (noir=%s, ev=%+.1f, awb=%s)",
             args.width, args.height, args.fps, Encoder.__name__, args.port,
             args.noir, args.ev, args.awb)
    log.info("Open http://<PI_IP>:%d  (find the IP with: hostname -I)", args.port)

    try:
        StreamingServer(("", args.port), StreamingHandler).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop_recording()
        log.info("Stopped.")


if __name__ == "__main__":
    main()
