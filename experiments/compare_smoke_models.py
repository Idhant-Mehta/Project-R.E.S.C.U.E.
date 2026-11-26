#!/usr/bin/env python3
"""
Compare two smoke detectors on the same images (laptop CPU):
  A) fire_yolov8n.pt          -> class 0 'smoke'  (YOLOv8n, already in project)
  B) kittendev YOLOv8m smoke  -> class 0 'smoke'  (Hugging Face, best.pt)

IMPORTANT LIMITATION: the images used here have NO smoke bounding-box labels.
This script therefore measures only:
  - how often each model fires on each image group (detection rate)
  - CPU latency and model size
It does NOT measure precision/recall/mAP. Groups assumed smoke-free
(light, indoor webcam) give a false-alarm rate *under that assumption*.

Usage:
    python experiments/compare_smoke_models.py --per-group 30
"""

import argparse
import json
import os
import random
import statistics
import time

import cv2
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "smoke_compare_out")

HF_REPO = "kittendev/YOLOv8m-smoke-detection"
HF_FILE = "best.pt"

GROUPS = {
    # name: (folder, assumption about smoke content)
    "fire_dataset/fire": (os.path.join(ROOT, "fire_dataset", "fire"), "mixed: flames, some with visible smoke"),
    "fire_dataset/light": (os.path.join(ROOT, "fire_dataset", "light"), "assumed smoke-free (bulbs, lamps)"),
    "test_output (indoor webcam)": (os.path.join(ROOT, "test_output"), "assumed smoke-free (indoor room)"),
}


def list_images(folder, n, seed=42):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.Random(seed).shuffle(files)
    return [os.path.join(folder, f) for f in files[:n]]


def model_stats(model):
    params = sum(p.numel() for p in model.model.parameters())
    return {"params_M": round(params / 1e6, 2)}


def run(model, smoke_cls, img, imgsz, conf):
    t0 = time.perf_counter()
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=[smoke_cls], verbose=False, device="cpu")[0]
    dt = (time.perf_counter() - t0) * 1000
    return r, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    models = {
        "A_fire_yolov8n(smoke cls)": YOLO(os.path.join(ROOT, "fire_yolov8n.pt")),
        "B_hf_yolov8m_smoke": YOLO(hf_hub_download(HF_REPO, HF_FILE)),
    }
    smoke_cls = {k: next(i for i, n in m.names.items() if "smoke" in n.lower()) for k, m in models.items()}

    os.makedirs(OUT, exist_ok=True)
    report = {"conf": args.conf, "models": {}, "groups": {}, "latency_ms_cpu": {}}
    for k, m in models.items():
        report["models"][k] = {"names": m.names, "smoke_class": smoke_cls[k], **model_stats(m)}

    # ── Detection rate per group (imgsz 640, the HF model's training size) ──
    for gname, (folder, assumption) in GROUPS.items():
        if not os.path.isdir(folder):
            continue
        imgs = list_images(folder, args.per_group)
        g = {"n_images": len(imgs), "assumption": assumption}
        for k, m in models.items():
            hits, per_img = 0, []
            for p in imgs:
                img = cv2.imread(p)
                if img is None:
                    continue
                r, _ = run(m, smoke_cls[k], img, 640, args.conf)
                n = len(r.boxes)
                hits += n > 0
                per_img.append((os.path.basename(p), n, [round(float(c), 2) for c in r.boxes.conf]))
                if n > 0:
                    sub = os.path.join(OUT, k, gname.split()[0].replace("/", "_"))
                    os.makedirs(sub, exist_ok=True)
                    cv2.imwrite(os.path.join(sub, os.path.basename(p)), r.plot())
            g[k] = {"images_with_smoke_det": hits, "rate": round(hits / max(1, len(imgs)), 3), "per_image": per_img}
        report["groups"][gname] = g

    # ── CPU latency at 640 and 320 (same 10 images, 3 warm-up runs) ──
    bench = [cv2.imread(p) for p in list_images(GROUPS["fire_dataset/fire"][0], 10, seed=7)]
    for k, m in models.items():
        report["latency_ms_cpu"][k] = {}
        for imgsz in (640, 320):
            for _ in range(3):
                run(m, smoke_cls[k], bench[0], imgsz, args.conf)
            times = [run(m, smoke_cls[k], im, imgsz, args.conf)[1] for im in bench]
            report["latency_ms_cpu"][k][imgsz] = {
                "median": round(statistics.median(times), 1),
                "min": round(min(times), 1),
                "max": round(max(times), 1),
            }

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── Console summary ──
    print("\nMODELS")
    for k, v in report["models"].items():
        print(f"  {k:28s} params={v['params_M']}M  names={v['names']}")
    print(f"\nDETECTION RATE (images with >=1 smoke box, conf>={args.conf}, imgsz=640)")
    for gname, g in report["groups"].items():
        print(f"  {gname}  [n={g['n_images']}, {g['assumption']}]")
        for k in models:
            print(f"     {k:28s} {g[k]['images_with_smoke_det']:3d}/{g['n_images']}  ({g[k]['rate']:.0%})")
    print("\nCPU LATENCY per image, ms (this laptop, NOT the Pi)")
    for k, d in report["latency_ms_cpu"].items():
        print(f"  {k:28s} " + "   ".join(f"{s}px: median {v['median']} (min {v['min']}, max {v['max']})" for s, v in d.items()))
    print(f"\nAnnotated hits + report.json -> {OUT}")


if __name__ == "__main__":
    main()
