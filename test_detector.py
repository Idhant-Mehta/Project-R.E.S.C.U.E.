#!/usr/bin/env python3
"""
Quick headless test — captures 15 frames from webcam,
runs person + flame detection, saves output images, prints results.
No GUI window needed.
"""

import sys
import os
import time

# Ensure we find flame_detector.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from flame_detector import FlameDetector

print("=" * 50)
print("  RESCUE ROVER — DETECTION TEST")
print("=" * 50)

# ── Load YOLOv8n ──
print("\n[1/3] Loading YOLOv8n model …")
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
print("       Model loaded OK ✓")

# ── Init flame detector ──
print("[2/3] Initializing flame detector …")
flame_det = FlameDetector(sensitivity=0.6)
print("       Flame detector OK ✓")

# ── Open webcam ──
print("[3/3] Opening webcam …")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("       ERROR: Cannot open webcam!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)  # warm-up
print("       Webcam opened OK ✓")

# ── Create output dir ──
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
os.makedirs(out_dir, exist_ok=True)

print(f"\nRunning detection on 15 frames …\n")

persons_total = 0
flames_total = 0
times = []

for i in range(15):
    ret, frame = cap.read()
    if not ret:
        print(f"  Frame {i+1:2d}: SKIP (empty)")
        continue

    t0 = time.time()

    # Person detection
    results = model.predict(frame, conf=0.40, classes=[0], verbose=False, imgsz=320)
    person_count = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"PERSON {conf:.0%}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            person_count += 1

    # Flame detection
    flames = flame_det.detect(frame)
    flame_det.draw(frame, flames)
    flame_count = len(flames)

    dt = time.time() - t0
    times.append(dt)

    persons_total += person_count
    flames_total += flame_count

    status_parts = []
    if person_count:
        status_parts.append(f"PERSON x{person_count}")
    if flame_count:
        status_parts.append(f"FLAME x{flame_count}")
    status = ", ".join(status_parts) if status_parts else "nothing"

    print(f"  Frame {i+1:2d}: {status:30s} ({dt*1000:.0f} ms)")

    # Save frame
    fname = os.path.join(out_dir, f"frame_{i+1:02d}.jpg")
    cv2.imwrite(fname, frame)

cap.release()

# ── Summary ──
avg_ms = (sum(times) / len(times)) * 1000 if times else 0
avg_fps = 1000 / avg_ms if avg_ms > 0 else 0

print("\n" + "=" * 50)
print("  RESULTS")
print("=" * 50)
print(f"  Frames processed  : {len(times)}")
print(f"  Avg inference time: {avg_ms:.0f} ms ({avg_fps:.1f} FPS)")
print(f"  Person detections : {persons_total}")
print(f"  Flame detections  : {flames_total}")
print()

if persons_total > 0:
    print("  ✓ Person detection is WORKING")
else:
    print("  ⚠ No persons found — try sitting in front of camera")

if flames_total > 0:
    print("  ✓ Flame detection is WORKING")
else:
    print("  ℹ No flames found (expected — no fire in view)")

print(f"\n  Output saved to: {out_dir}")
print("=" * 50)
