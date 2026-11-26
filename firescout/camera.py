"""Frame source: Raspberry Pi camera (picamera2), USB/laptop webcam, or a video file."""

import logging
import os
import time

import cv2

log = logging.getLogger("camera")


class Camera:
    """Uniform BGR frame source.

    source: "auto" | "picamera" | "<int index>" | "<path to video file>"
    """

    def __init__(self, source="auto", width=640, height=480):
        self.kind = None
        self.width, self.height = width, height
        self._pi = None
        self._cap = None
        self._is_file = False
        src = str(source)

        if src in ("auto", "picamera"):
            if self._try_picamera():
                return
            if src == "picamera":
                raise RuntimeError("picamera2 requested but not available")
            src = "0"

        if src.startswith(("http://", "https://", "rtsp://", "tcp://")):
            # Network stream, e.g. the Pi running pi/pi_camera_stream.py
            import sys
            from . import PROJECT_DIR
            sys.path.insert(0, PROJECT_DIR)
            from pi_stream_reader import PiStream
            self._cap = PiStream(src)
            self.kind = f"stream:{src}"
            log.info("Camera: network stream %s", src)
            return
        if src.isdigit():
            self._cap = cv2.VideoCapture(int(src))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.kind = f"webcam:{src}"
        elif os.path.exists(src):
            self._cap = cv2.VideoCapture(src)
            self._is_file = True
            self.kind = f"file:{os.path.basename(src)}"
        else:
            raise RuntimeError(f"Unknown camera source: {source}")

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")
        log.info("Camera: %s", self.kind)

    def _try_picamera(self):
        try:
            from picamera2 import Picamera2
        except ImportError:
            return False
        try:
            cam = Picamera2()
            # libcamera "RGB888" is stored as [B, G, R] per pixel = OpenCV's BGR.
            # No colour conversion needed (converting again swaps red/blue).
            cfg = cam.create_video_configuration(main={"size": (self.width, self.height), "format": "RGB888"})
            cam.configure(cfg)
            cam.start()
            time.sleep(1.0)
        except Exception as exc:  # camera not connected, busy, etc.
            log.warning("picamera2 present but camera failed to start: %s", exc)
            return False
        self._pi = cam
        self.kind = "picamera2"
        log.info("Camera: Raspberry Pi camera via picamera2")
        return True

    def read(self):
        """Return a BGR frame, or None if no frame is available."""
        if self._pi is not None:
            return self._pi.capture_array()
        ok, frame = self._cap.read()
        if not ok and self._is_file:  # loop video files for repeatable demos
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return None
        if self._is_file and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height))
        return frame

    @property
    def is_file(self):
        return self._is_file

    def fps_hint(self):
        if self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 1 else None
        return None

    def close(self):
        if self._pi is not None:
            self._pi.stop()
        if self._cap is not None:
            self._cap.release()
