"""
Read the Raspberry Pi camera stream on the laptop, with no lag build-up.

OpenCV buffers network frames; if YOLO is slower than the stream, a plain
cv2.VideoCapture falls further and further behind. PiStream reads in a
background thread and always returns the NEWEST frame.

Use in your own script (drop-in replacement for cv2.VideoCapture(0)):

    from pi_stream_reader import PiStream
    cap = PiStream("http://192.168.1.50:8000/stream.mjpg")
    while True:
        ok, frame = cap.read()        # BGR numpy array, same as a webcam
        if not ok:
            continue
        results = model(frame)        # your YOLO / OpenCV code, unchanged
    cap.release()

Quick test:
    python pi_stream_reader.py http://<PI_IP>:8000/stream.mjpg
"""

import sys
import threading
import time

import cv2


class PiStream:
    def __init__(self, url, reconnect_delay=1.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self._frame = None
        self._frame_id = 0
        self._last_id = 0
        self._lock = threading.Lock()
        self._running = True
        self.connected = False
        self._cap = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    self.connected = False
                    time.sleep(self.reconnect_delay)
                    continue
            ok, frame = self._cap.read()
            if not ok:
                # Wi-Fi drop or Pi restarted: reconnect automatically
                self.connected = False
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue
            self.connected = True
            with self._lock:
                self._frame = frame
                self._frame_id += 1
        if self._cap is not None:
            self._cap.release()

    def read(self, wait_new=True, timeout=2.0):
        """Return (ok, frame). With wait_new, waits for a frame not returned before."""
        t0 = time.time()
        while True:
            with self._lock:
                if self._frame is not None and (not wait_new or self._frame_id != self._last_id):
                    self._last_id = self._frame_id
                    return True, self._frame.copy()
            if time.time() - t0 > timeout:
                return False, None
            time.sleep(0.003)

    def isOpened(self):
        return self.connected

    def release(self):
        # Only signal the reader thread; it releases the capture itself, so we
        # never close the stream while that thread is in the middle of a read.
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python pi_stream_reader.py http://<PI_IP>:8000/stream.mjpg")
        sys.exit(1)
    cap = PiStream(sys.argv[1])
    n, t0 = 0, time.time()
    print("Connecting to", sys.argv[1], "... press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame yet (is pi_camera_stream.py running? same Wi-Fi?)")
            continue
        n += 1
        fps = n / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"Pi stream {frame.shape[1]}x{frame.shape[0]}  {fps:.1f} FPS",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Pi camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
