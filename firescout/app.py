"""Vision loop: camera -> person tracking -> smoke/flame -> room records -> annotated stream."""

import logging
import threading
import time

import cv2

from .alarm import Alarm
from .camera import Camera
from .detectors import PersonDetector, HazardDetector
from .mission import Mission
from .smoke import SmokeEstimator
from .tracker import IoUTracker

log = logging.getLogger("app")

SMOKE_COLORS = {"LOW": (80, 200, 80), "MODERATE": (0, 200, 255), "HIGH": (0, 120, 255), "CRITICAL": (0, 0, 255)}


class FireScoutApp:
    def __init__(self, cfg, sensors, verify_flames=True, beep=True):
        self.cfg = cfg
        m = cfg["models"]
        log.info("Loading person detector …")
        self.person = PersonDetector(m["person"], m["person_imgsz"], m["person_conf"])
        log.info("Loading smoke/fire detector …")
        self.hazard = HazardDetector(m["hazard"], m["flame_classifier"], m["hazard_imgsz"],
                                     m["hazard_conf"], verify_flames=verify_flames)
        self.hazard_every = max(1, int(m["hazard_every_n_frames"]))
        t = cfg["tracker"]
        self.tracker = IoUTracker(t["iou_match"], t["confirm_hits"], t["max_missed"])
        self.smoke = SmokeEstimator(cfg["smoke"])
        self.sensors = sensors
        self.alarm = Alarm(enabled=beep)
        self.mission = Mission(cfg["risk"], cfg["mission"]["data_dir"])
        c = cfg["camera"]
        self.camera = Camera(c["source"], c["width"], c["height"])

        self._lock = threading.Lock()
        self._jpeg = None
        self._raw = None
        self._live = {}
        self._calibrate_req = False
        self.auto_calibrate_after = int(cfg["smoke"].get("auto_calibrate_after_frames", 30))
        self.running = True
        self.jpeg_quality = cfg["server"]["jpeg_quality"]

    # ── API used by the web server ──────────────────────────────────────
    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def request_calibration(self):
        self._calibrate_req = True

    def state(self):
        with self._lock:
            live = dict(self._live)
        return {"live": live, **self.mission.state()}

    def enter_room(self, name):
        self.tracker.reset()  # new room: start fresh track IDs so counts are per room
        return self.mission.enter_room(name)

    def exit_room(self):
        self.tracker.reset()
        return self.mission.exit_room()

    # ── main loop ───────────────────────────────────────────────────────
    def run(self):
        frame_i, fps, t_fps, n_fps = 0, 0.0, time.time(), 0
        hazard = {"smoke_boxes": [], "flames": []}
        smoke = None
        file_delay = 1.0 / (self.camera.fps_hint() or 15) if self.camera.is_file else 0
        while self.running:
            t0 = time.time()
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            # Auto-calibrate once after camera warm-up: the rover is started at the
            # staging area in clear air. The operator can re-calibrate any time.
            auto = (not self.smoke.calibrated and frame_i == self.auto_calibrate_after)
            if self._calibrate_req or auto:
                ref = self.smoke.calibrate(frame)
                self._calibrate_req = False
                self.mission.event("AUTO_CALIBRATE" if auto else "CALIBRATE",
                                   "clear-air reference: " +
                                   ", ".join(f"{k}={v:.3f}" for k, v in ref.items()))

            t_inf = time.time()
            people = self.person.detect(frame)
            tracks = self.tracker.update(people)
            if frame_i % self.hazard_every == 0:
                hazard = self.hazard.detect(frame)
                smoke = self.smoke.update(frame, hazard["smoke_boxes"])
                if hazard["flames"] and self.alarm.trigger():
                    log.warning("FIRE DETECTED - alarm (p=%.2f)", hazard["flames"][0]["confidence"])
            infer_ms = (time.time() - t_inf) * 1000

            room = self.mission.current
            sensors = self.sensors.read(room.name if room else None)
            annotated = self._annotate(frame, people, tracks, hazard, smoke, fps)
            self.mission.update(tracks, smoke, hazard["flames"], sensors, annotated)

            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            n_fps += 1
            if time.time() - t_fps >= 1.0:
                fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()
            with self._lock:
                if ok:
                    self._jpeg = jpg.tobytes()
                self._live = {
                    "fps": round(fps, 1),
                    "inference_ms": round(infer_ms, 1),
                    "camera": self.camera.kind,
                    "persons_now": len(tracks),
                    "raw_person_detections": len(people),
                    "smoke": smoke.to_dict() if smoke else None,
                    "flames_now": len(hazard["flames"]),
                    "sensors": sensors.to_dict(),
                    "ts": time.time(),
                }
            frame_i += 1
            if file_delay:
                time.sleep(max(0.0, file_delay - (time.time() - t0)))

    def _annotate(self, frame, raw, tracks, hazard, smoke, fps):
        out = frame.copy()
        confirmed = {t.id for t in tracks}
        for x1, y1, x2, y2, _ in hazard["smoke_boxes"]:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), 1)
        for f in hazard["flames"]:
            x, y, w, h = f["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(out, f"FIRE {f['confidence']:.0%}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # unconfirmed raw detections: thin yellow (not yet counted)
        for x1, y1, x2, y2, c in raw:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 1)
        for t in self.tracker.tracks:
            if t.id in confirmed:
                x1, y1, x2, y2 = (int(v) for v in t.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, f"P#{t.id} {t.conf:.0%}", (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        h = out.shape[0]
        room = self.mission.current
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        hud = f"{room.name if room else 'NO ROOM'} | people {len(tracks)} | {fps:.1f} FPS"
        cv2.putText(out, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if smoke:
            col = SMOKE_COLORS[smoke.level]
            txt = f"SMOKE {smoke.level} {smoke.score:.2f}" + (" LOW-LIGHT" if smoke.low_light else "")
            cv2.rectangle(out, (0, h - 24), (out.shape[1], h), (0, 0, 0), -1)
            cv2.putText(out, txt, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        return out

    def close(self):
        self.running = False
        time.sleep(0.2)
        self.camera.close()
        self.sensors.close()
        self.mission.close()
