"""Model wrappers.

PersonDetector : YOLOv8n (COCO person class). Ultralytics for now; Phase 5 swaps
                 in an ONNX Runtime backend with the same detect() signature.
HazardDetector : one run of fire_yolov8n.pt gives smoke AND fire boxes; fire boxes
                 are verified by the existing fire-vs-light CNN (flame_detector.py).
"""

import logging
import os
import sys

from . import PROJECT_DIR

log = logging.getLogger("detectors")


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


class PersonDetector:
    def __init__(self, weights="yolov8n.pt", imgsz=320, conf=0.35):
        from ultralytics import YOLO
        self.model = YOLO(_resolve(weights))
        self.imgsz, self.conf = imgsz, conf
        self.person_cls = next(i for i, n in self.model.names.items() if n == "person")

    def detect(self, frame):
        """Return [(x1, y1, x2, y2, conf), ...] for people."""
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                               classes=[self.person_cls], verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out


class HazardDetector:
    def __init__(self, weights="fire_yolov8n.pt", classifier="fire_vs_light.pt",
                 imgsz=320, conf=0.25, verify_flames=True):
        from ultralytics import YOLO
        self.model = YOLO(_resolve(weights))
        names = {i: n.lower() for i, n in self.model.names.items()}
        self.smoke_cls = next((i for i, n in names.items() if "smoke" in n), None)
        self.fire_cls = next((i for i, n in names.items() if "fire" in n or "flame" in n), None)
        self.imgsz, self.conf = imgsz, conf
        self.flame = None
        if verify_flames:
            sys.path.insert(0, PROJECT_DIR)
            from flame_detector import FlameDetector
            # yolo_path=None: reuse this class's YOLO run instead of loading the model twice
            self.flame = FlameDetector(yolo_path=None, classifier_path=_resolve(classifier))
        log.info("Hazard model classes: %s (smoke=%s, fire=%s)", self.model.names,
                 self.smoke_cls, self.fire_cls)

    def detect(self, frame):
        """Return dict(smoke_boxes=[(x1,y1,x2,y2,conf)], flames=[flame dicts])."""
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        smoke, fire_xywh = [], []
        for b in r.boxes:
            c = int(b.cls[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            if c == self.smoke_cls:
                smoke.append((x1, y1, x2, y2, float(b.conf[0])))
            elif c == self.fire_cls and (x2 - x1) > 8 and (y2 - y1) > 8:
                fire_xywh.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        flames = self.flame.detect(frame, yolo_boxes=fire_xywh) if self.flame else [
            {"bbox": b, "confidence": 0.5} for b in fire_xywh]
        return {"smoke_boxes": smoke, "flames": flames}
