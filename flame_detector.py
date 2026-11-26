"""
Flame / Fire Detection using a two-stage Neural Pipeline:
  1. Candidate Proposals : YOLOv8n-Fire (D-Fire) + warm luminous region proposals
  2. Neural Verification : Custom-trained PyTorch MobileNetV3-Small (fire_vs_light.pt)
                           trained specifically to distinguish real flames from
                           light bulbs, LEDs, flashlights, glare, and skin.
"""

import os
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

log = logging.getLogger("flame")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_FIRE_PATH = os.path.join(BASE_DIR, "fire_yolov8n.pt")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "fire_vs_light.pt")


def _box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)


class FlameDetector:
    def __init__(
        self,
        min_area=250,
        sensitivity=0.6,
        yolo_path=YOLO_FIRE_PATH,
        classifier_path=CLASSIFIER_PATH,
    ):
        """
        Args:
            min_area: Minimum contour area (pixels^2) for candidate region proposals.
            sensitivity: 0.0-1.0, higher = lower confidence threshold.
            yolo_path: Optional YOLOv8n-Fire weights for bounding-box proposals.
            classifier_path: Trained MobileNetV3-Small fire_vs_light.pt weights.
        """
        self.min_area = min_area
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        # At sensitivity=0.6 -> require P(fire) >= 0.65
        self.fire_thresh = max(0.35, min(0.90, 0.86 - 0.35 * self.sensitivity))

        self.yolo_model = None
        self.fire_class = 1
        self.classifier = None
        self.clf_tf = None
        self.device = torch.device("cpu")

        self._init_classifier(classifier_path)
        self._init_yolo(yolo_path)

    def _init_classifier(self, path):
        if not os.path.exists(path):
            log.warning("fire_vs_light.pt not found at %s — run train_fire_model.py", path)
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            img_size = ckpt.get("img_size", 128)
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, len(ckpt.get("classes", ["fire", "light"])))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.classifier = model
            self.clf_tf = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            log.info(
                "Loaded Fire-vs-Light classifier (%s, thresh=%.2f)",
                os.path.basename(path),
                self.fire_thresh,
            )
        except Exception as exc:
            log.warning("Failed to load Fire-vs-Light classifier: %s", exc)
            self.classifier = None

    def _init_yolo(self, path):
        # path=None: caller supplies YOLO fire boxes via detect(frame, yolo_boxes=...)
        if not path or not os.path.exists(path):
            return
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(path)
            for idx, name in self.yolo_model.names.items():
                if "fire" in str(name).lower() or "flame" in str(name).lower():
                    self.fire_class = int(idx)
                    break
        except Exception as exc:
            log.warning("Could not load YOLO fire model (%s)", exc)
            self.yolo_model = None

    def _propose_warm_regions(self, frame):
        """Fast candidate box proposals for glowing orange/red/yellow regions
        (catches candles/lighters that D-Fire YOLO misses; verified by CNN)."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        b, g, r = cv2.split(blurred)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        # Warm glowing pixels (R >= G >= B, bright enough to be flame/light)
        warm_rgb = (r >= 170) & (r >= g) & (g >= b) & ((r.astype(np.int16) - b.astype(np.int16)) >= 35)
        warm_hsv = ((h_ch <= 35) | (h_ch >= 165)) & (v_ch >= 170) & (s_ch >= 45)
        mask = (warm_rgb & warm_hsv).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        return boxes

    def _propose_yolo(self, frame):
        if self.yolo_model is None:
            return []
        results = self.yolo_model.predict(
            frame,
            conf=0.25,
            classes=[self.fire_class],
            verbose=False,
            imgsz=320,
        )
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = max(0, x2 - x1), max(0, y2 - y1)
                if w > 8 and h > 8:
                    boxes.append((int(x1), int(y1), int(w), int(h)))
        return boxes

    def _classify_crop(self, frame, bbox):
        """Return P(fire) in [0, 1] from the trained Fire-vs-Light CNN."""
        fh, fw = frame.shape[:2]
        x, y, w, h = bbox
        # Pad by 40% so CNN sees flame tips & surrounding context (vs bulb fixture)
        pad_x = max(24, int(w * 0.4))
        pad_y = max(24, int(h * 0.4))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(fw, x + w + pad_x)
        y2 = min(fh, y + h + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.clf_tf(Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.classifier(tensor), dim=1)
        return float(probs[0, 0].item())  # class 0 = 'fire'

    def detect(self, frame, yolo_boxes=None):
        """
        Detect flames in a clean BGR frame.

        Args:
            yolo_boxes: optional precomputed fire boxes [(x, y, w, h), ...] from a
                        shared YOLO run (avoids running the fire model twice).

        Returns:
            list of dicts: [{'bbox': (x,y,w,h), 'confidence': float, 'center': (cx,cy), 'area': int}]
        """
        if yolo_boxes is None:
            yolo_boxes = self._propose_yolo(frame)
        candidates = list(yolo_boxes) + self._propose_warm_regions(frame)
        if not candidates:
            return []

        # Merge overlapping candidate boxes (keep larger box)
        candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
        merged = []
        for box in candidates:
            if all(_box_iou(box, kept) < 0.35 for kept in merged):
                merged.append(box)
        merged = merged[:8]  # cap per-frame crops for Pi speed

        detections = []
        for x, y, w, h in merged:
            if self.classifier is not None:
                p_fire = self._classify_crop(frame, (x, y, w, h))
                if p_fire < self.fire_thresh:
                    continue
                conf = round(p_fire, 2)
            else:
                conf = 0.50

            detections.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": conf,
                    "center": (int(x + w // 2), int(y + h // 2)),
                    "area": int(w * h),
                }
            )

        detections.sort(key=lambda d: d["area"], reverse=True)
        return detections

    def draw(self, frame, detections):
        """Draw flame bounding boxes on the frame (in-place)."""
        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"FIRE {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw, y), (0, 0, 255), -1)
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return frame
