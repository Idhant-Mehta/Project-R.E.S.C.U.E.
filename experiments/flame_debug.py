#!/usr/bin/env python3
"""
Flame pipeline diagnostics on a live camera (e.g. the Pi stream).

Records for N seconds and logs, per analysed frame, what each stage of
flame_detector.py saw:
  - YOLO fire boxes at a very low threshold (conf >= 0.05)
  - warm-colour candidate regions (with area, even below min_area)
  - fire-vs-light classifier P(fire) for every candidate
  - final detections from the unchanged pipeline
Saves raw frames (for re-testing fixes offline) + annotated frames + log.jsonl.

    python experiments/flame_debug.py --source http://rover.local:8000/stream.mjpg --seconds 60
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flame_detector import FlameDetector  # noqa: E402
from pi_stream_reader import PiStream  # noqa: E402


def warm_regions_all(det, frame):
    """Same mask as FlameDetector._propose_warm_regions but returns every blob + area."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    b, g, r = cv2.split(blurred)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    warm_rgb = (r >= 170) & (r >= g) & (g >= b) & ((r.astype(np.int16) - b.astype(np.int16)) >= 35)
    warm_hsv = ((h <= 35) | (h >= 165)) & (v >= 170) & (s >= 45)
    mask = (warm_rgb & warm_hsv).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, hh = cv2.boundingRect(c)
        out.append({"bbox": [int(x), int(y), int(w), int(hh)], "area": float(cv2.contourArea(c))})
    # also: very bright (near-white, saturated sensor) blobs that the warm rule rejects
    bright = ((v >= 240) & (s < 45)).astype(np.uint8) * 255
    bcnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white = [cv2.boundingRect(c) for c in bcnts if cv2.contourArea(c) >= 20]
    return out, [list(map(int, w)) for w in white]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="http://rover.local:8000/stream.mjpg")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--every", type=int, default=3, help="analyse every Nth frame")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "flame_debug_out"))
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "raw")
    ann_dir = os.path.join(args.out, "annotated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    det = FlameDetector(sensitivity=0.6)  # same settings as FireScout
    yolo = det.yolo_model
    log = open(os.path.join(args.out, "log.jsonl"), "w")
    cap = PiStream(args.source) if args.source.startswith("http") else cv2.VideoCapture(int(args.source))

    print(f"Recording {args.seconds:.0f}s from {args.source} — hold the lit lighter in view now.")
    t0, i = time.time(), 0
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        i += 1
        if i % args.every:
            continue
        rec = {"i": i, "t": round(time.time() - t0, 2),
               "brightness": round(float(frame.mean()), 1),
               "max_v": int(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].max())}

        yb = []
        if yolo is not None:
            r = yolo.predict(frame, conf=0.05, classes=[det.fire_class], imgsz=320, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                yb.append({"bbox": [x1, y1, x2 - x1, y2 - y1], "conf": round(float(b.conf[0]), 3)})
        warm, white = warm_regions_all(det, frame)

        cands = [("yolo", c["bbox"]) for c in yb] + [("warm", c["bbox"]) for c in warm] + [("white", w) for w in white]
        scored = []
        for src, bb in cands[:12]:
            p = det._classify_crop(frame, tuple(bb)) if det.classifier is not None else None
            scored.append({"src": src, "bbox": bb, "p_fire": None if p is None else round(p, 3)})

        final = det.detect(frame)
        rec.update({"yolo": yb, "warm": warm, "white_blobs": white, "scored": scored,
                    "final": [{"bbox": list(d["bbox"]), "conf": d["confidence"]} for d in final]})
        log.write(json.dumps(rec) + "\n")

        cv2.imwrite(os.path.join(raw_dir, f"{i:05d}.jpg"), frame)
        ann = frame.copy()
        colors = {"yolo": (0, 165, 255), "warm": (0, 255, 255), "white": (255, 255, 255)}
        for s in scored:
            x, y, w, h = s["bbox"]
            cv2.rectangle(ann, (x, y), (x + w, y + h), colors[s["src"]], 1)
            cv2.putText(ann, f"{s['src']} p={s['p_fire']}", (x, max(10, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[s["src"]], 1)
        for d in final:
            x, y, w, h = d["bbox"]
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(ann_dir, f"{i:05d}.jpg"), ann)
        print(f"t={rec['t']:5.1f}s yolo={len(yb)} warm={len(warm)} white={len(white)} final={len(final)}"
              + (f"  best p_fire={max((s['p_fire'] or 0) for s in scored):.2f}" if scored else ""))
    log.close()
    cap.release()
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
