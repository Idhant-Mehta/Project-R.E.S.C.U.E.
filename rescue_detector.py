#!/usr/bin/env python3
"""
Disaster Rescue Rover — Person & Flame Detection
==================================================
Hardware: Raspberry Pi 4 (2GB) + Pi Camera v2

- Person detection : YOLOv8n (pre-trained COCO, class 0 = person)
- Flame detection  : OpenCV HSV color thresholding (flame_detector.py)

The first run will auto-download the YOLOv8n weights (~6 MB).
"""

import sys
import time
import signal
import logging
import argparse
from datetime import datetime

import cv2
import numpy as np

# Preserve native OpenCV imshow BEFORE ultralytics patches it
_native_imshow = cv2.imshow

# ── Try to import Pi-specific camera; fall back to OpenCV VideoCapture ──
PI_CAMERA = False
try:
    from picamera2 import Picamera2
    PI_CAMERA = True
except ImportError:
    pass

from flame_detector import FlameDetector

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rescue_detector.log"),
    ],
)
log = logging.getLogger("rescue")

# ── Globals for clean shutdown ──
running = True

def _signal_handler(sig, frame):
    global running
    log.info("Shutdown signal received.")
    running = False

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ======================================================================
# Camera helpers
# ======================================================================

def open_camera_pi(width, height):
    """Open Raspberry Pi Camera v2 via picamera2."""
    cam = Picamera2()
    # "RGB888" -> numpy array in BGR order (what OpenCV expects). See grab_frame().
    config = cam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    cam.configure(config)
    cam.start()
    time.sleep(1)  # warm-up
    return cam


def open_camera_usb(index, width, height):
    """Fallback: open a USB / laptop webcam via OpenCV."""
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        log.error("Cannot open camera index %d", index)
        sys.exit(1)
    return cap


def grab_frame(cam):
    """Return a BGR numpy frame from whichever camera backend is active."""
    if PI_CAMERA and isinstance(cam, Picamera2):
        # picamera2/libcamera naming: format "RGB888" is stored in memory as
        # [B, G, R] per pixel, which is already OpenCV's BGR order.
        # Do NOT convert it again, or red and blue get swapped (flames turn blue).
        return cam.capture_array()
    else:
        ret, frame = cam.read()
        if not ret:
            return None
        return frame


# ======================================================================
# Person detector wrapper (YOLOv8n via ultralytics)
# ======================================================================

class PersonDetector:
    """Wraps YOLOv8-nano for person-only detection."""

    def __init__(self, conf_threshold=0.45, model_name="yolov8n.pt"):
        from ultralytics import YOLO
        log.info("Loading %s (auto-downloads on first run) …", model_name)
        self.model = YOLO(model_name)
        self.conf = conf_threshold
        # COCO class 0 = person
        self.person_class = 0

    def detect(self, frame):
        """
        Returns list of dicts:
        [{'bbox': (x1,y1,x2,y2), 'confidence': float}, ...]
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=[self.person_class],
            verbose=False,
            imgsz=320,          # smaller = faster on Pi
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': round(conf, 2),
                })
        return detections

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"PERSON {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame


# ======================================================================
# Alert system (GPIO buzzer / on-screen)
# ======================================================================

class AlertSystem:
    """Simple alert manager — plays on-screen warning + optional GPIO buzzer."""

    def __init__(self, buzzer_pin=18, use_gpio=False):
        self.use_gpio = use_gpio
        self.buzzer_pin = buzzer_pin
        self._last_alert = 0
        self._cooldown = 3  # seconds between alerts

        if self.use_gpio:
            try:
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.buzzer_pin, GPIO.OUT)
                self.GPIO = GPIO
            except Exception as e:
                log.warning("GPIO buzzer init failed: %s — alerts will be on-screen only.", e)
                self.use_gpio = False

    def trigger(self, alert_type, frame):
        now = time.time()
        if now - self._last_alert < self._cooldown:
            return

        self._last_alert = now
        timestamp = datetime.now().strftime("%H:%M:%S")

        if alert_type == "person":
            msg = f"[{timestamp}] *** PERSON DETECTED ***"
            color = (0, 255, 0)
        elif alert_type == "flame":
            msg = f"[{timestamp}] *** FLAME DETECTED ***"
            color = (0, 0, 255)
        elif alert_type == "both":
            msg = f"[{timestamp}] *** PERSON + FLAME — URGENT ***"
            color = (0, 0, 255)
        else:
            return

        log.warning(msg)

        # On-screen banner
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), color, -1)
        cv2.putText(frame, msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # GPIO buzz (short pulse)
        if self.use_gpio:
            self.GPIO.output(self.buzzer_pin, self.GPIO.HIGH)
            time.sleep(0.15)
            self.GPIO.output(self.buzzer_pin, self.GPIO.LOW)

    def cleanup(self):
        if self.use_gpio:
            self.GPIO.cleanup()


# ======================================================================
# Main loop
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Rescue Rover Detection")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--camera", type=int, default=0, help="USB cam index (ignored on Pi)")
    parser.add_argument("--conf", type=float, default=0.45, help="Person detection confidence")
    parser.add_argument("--flame-sensitivity", type=float, default=0.6, help="Flame sensitivity 0-1")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (no cv2 window)")
    parser.add_argument("--save-detections", action="store_true", help="Save frames with detections")
    parser.add_argument("--gpio-buzzer", action="store_true", help="Enable GPIO buzzer alerts")
    args = parser.parse_args()

    # ── Init modules ──
    log.info("Initializing person detector (YOLOv8n) …")
    person_det = PersonDetector(conf_threshold=args.conf)

    log.info("Initializing flame detector (HSV) …")
    flame_det = FlameDetector(sensitivity=args.flame_sensitivity)

    alert = AlertSystem(use_gpio=args.gpio_buzzer)

    # ── Open camera ──
    if PI_CAMERA:
        log.info("Using Raspberry Pi Camera v2 via picamera2")
        cam = open_camera_pi(args.width, args.height)
    else:
        log.info("Pi camera not found — using USB camera %d", args.camera)
        cam = open_camera_usb(args.camera, args.width, args.height)

    log.info("Detection running. Press Ctrl+C to stop.")
    fps_time = time.time()
    frame_count = 0
    save_count = 0

    global running
    while running:
        frame = grab_frame(cam)
        if frame is None:
            log.warning("Empty frame, retrying …")
            time.sleep(0.1)
            continue

        # ── Detect ──
        persons = person_det.detect(frame)
        flames = flame_det.detect(frame)

        # ── Draw ──
        person_det.draw(frame, persons)
        flame_det.draw(frame, flames)

        # ── Alerts ──
        if persons and flames:
            alert.trigger("both", frame)
        elif persons:
            alert.trigger("person", frame)
        elif flames:
            alert.trigger("flame", frame)

        # ── FPS overlay ──
        frame_count += 1
        elapsed = time.time() - fps_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_time = time.time()
        else:
            fps = frame_count / max(elapsed, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Status bar
        status = f"Persons: {len(persons)}  |  Flames: {len(flames)}"
        cv2.putText(frame, status, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Save detections ──
        if args.save_detections and (persons or flames):
            fname = f"detection_{save_count:04d}.jpg"
            cv2.imwrite(fname, frame)
            save_count += 1
            log.info("Saved %s", fname)

        # ── Display ──
        if not args.no_display:
            _native_imshow("Rescue Rover", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Cleanup ──
    log.info("Shutting down …")
    if PI_CAMERA and isinstance(cam, Picamera2):
        cam.stop()
    else:
        cam.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    alert.cleanup()
    log.info("Done.")


if __name__ == "__main__":
    main()
