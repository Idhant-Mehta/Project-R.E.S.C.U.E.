"""Visual smoke severity.

IMPORTANT: this measures VISUAL OBSCURATION of the camera image, not gas
toxicity. Theatrical fog can score CRITICAL while being harmless; thin smoke
can score LOW while carrying lethal CO. Gas sensors are mandatory for hazard.

Score (0..1) = max(obscuration, (1 - w) * obscuration + w * detector_score)
  obscuration    : loss of contrast/sharpness and rise in haze (dark channel)
                   relative to a clear-air reference frame (operator "Calibrate").
  detector_score : fraction of the frame covered by 'smoke' boxes from fire_yolov8n.
Thresholds in config.yaml are UNCALIBRATED until the contrast-target experiment.
"""

from dataclasses import dataclass, asdict

import cv2
import numpy as np

LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")

# Fallback reference used until the operator calibrates in clear air.
DEFAULT_REF = {"contrast": 0.20, "sharpness": 300.0, "dark": 0.15, "brightness": 0.45}


def visibility_metrics(frame):
    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32) / 255.0
    dark = cv2.erode(small.min(axis=2), np.ones((7, 7), np.uint8)).astype(np.float32) / 255.0
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    return {
        "contrast": float(g.std()),                                   # RMS contrast
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),    # edge energy
        "dark": float(dark.mean()),                                   # dark-channel haze proxy
        "saturation": float(sat.mean()),
        "brightness": float(g.mean()),
    }


def obscuration(m, ref):
    c_loss = np.clip(1 - m["contrast"] / max(ref["contrast"], 1e-6), 0, 1)
    s_loss = np.clip(1 - np.log1p(m["sharpness"]) / max(np.log1p(ref["sharpness"]), 1e-6), 0, 1)
    d_gain = np.clip((m["dark"] - ref["dark"]) / max(1 - ref["dark"], 1e-6), 0, 1)
    return float((c_loss + s_loss + d_gain) / 3)


def smoke_box_coverage(boxes, w, h):
    """Union area of smoke boxes / frame area (mask-based, so overlaps aren't double counted)."""
    if not boxes:
        return 0.0
    mask = np.zeros((h // 4 + 1, w // 4 + 1), np.uint8)
    for x1, y1, x2, y2, _ in boxes:
        mask[int(y1) // 4:int(y2) // 4 + 1, int(x1) // 4:int(x2) // 4 + 1] = 1
    return float(mask.mean())


@dataclass
class SmokeResult:
    level: str
    score: float
    obscuration: float
    detector_coverage: float
    detector_max_conf: float
    low_light: bool
    calibrated: bool

    def to_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class SmokeEstimator:
    def __init__(self, cfg):
        self.w = cfg.get("detector_weight", 0.4)
        self.alpha = cfg.get("ema_alpha", 0.3)
        self.low_light = cfg.get("low_light_brightness", 0.12)
        lv = cfg.get("levels", {})
        self.thr = (lv.get("moderate", 0.2), lv.get("high", 0.4), lv.get("critical", 0.6))
        self.ref = dict(DEFAULT_REF)
        self.calibrated = False
        self._ema = None

    def calibrate(self, frame):
        """Capture the clear-air reference. Call only when the view is smoke-free."""
        self.ref = visibility_metrics(frame)
        self.calibrated = True
        self._ema = None
        return self.ref

    def level_for(self, score):
        if score >= self.thr[2]:
            return "CRITICAL"
        if score >= self.thr[1]:
            return "HIGH"
        if score >= self.thr[0]:
            return "MODERATE"
        return "LOW"

    def update(self, frame, smoke_boxes):
        h, w = frame.shape[:2]
        m = visibility_metrics(frame)
        dark_scene = m["brightness"] < self.low_light
        obs = obscuration(m, self.ref)
        cov = smoke_box_coverage(smoke_boxes, w, h)
        det = min(1.0, cov / 0.5)  # >= 50 % of frame covered by smoke boxes -> 1.0
        # In a dark scene, contrast loss is caused by lighting, not smoke:
        # only the detector may raise the score.
        # Otherwise the detector can only RAISE the score above pure obscuration,
        # so dense haze still reaches CRITICAL even when the detector sees no boxes.
        # Without a reference from THIS camera in clear air, obscuration is meaningless
        # (a dim room would read as 'smoky'), so then only the detector counts too.
        if dark_scene or not self.calibrated:
            raw = det
        else:
            raw = max(obs, (1 - self.w) * obs + self.w * det)
        self._ema = raw if self._ema is None else self.alpha * raw + (1 - self.alpha) * self._ema
        return SmokeResult(
            level=self.level_for(self._ema),
            score=self._ema,
            obscuration=obs,
            detector_coverage=cov,
            detector_max_conf=max((b[4] for b in smoke_boxes), default=0.0),
            low_light=dark_scene,
            calibrated=self.calibrated,
        )
