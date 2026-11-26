#!/usr/bin/env python3
"""
Train a custom PyTorch Fire-vs-Light classifier (MobileNetV3-Small).
====================================================================
Distinguishes real flames (candle, lighter, fire) from artificial lights
(light bulbs, LEDs, lamps, flashlights, glare, skin).

Usage:
    python train_fire_model.py              # Download dataset + train fire_vs_light.pt
    python train_fire_model.py --capture light  # Capture 40 frames of your room light
    python train_fire_model.py --capture fire   # Capture 40 frames of real fire
    python train_fire_model.py --retrain    # Retrain on existing dataset/ captures
"""

import os
import sys
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "fire_dataset")
MODEL_OUT = os.path.join(BASE_DIR, "fire_vs_light.pt")
IMG_SIZE = 128

FIRE_QUERIES = [
    "candle flame",
    "lighter flame",
    "matchstick flame",
    "campfire flames",
    "bonfire flames",
    "gas burner flame",
    "fire flames",
    "torch flame",
]

LIGHT_QUERIES = [
    "incandescent light bulb",
    "light bulb glowing",
    "led lamp light",
    "ceiling light lamp",
    "desk lamp glowing",
    "street lamp night",
    "flashlight beam",
    "car headlight night",
    "chandelier lights",
    "hand palm",
]

HEADERS = {"User-Agent": "RescueRoverTrainer/1.0 (educational robotics project)"}


def fetch_wikimedia_urls(query, limit=30):
    """Fetch JPEG/PNG thumbnail URLs from Wikimedia Commons."""
    urls = []
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrnamespace": 6,
                "gsrsearch": query,
                "gsrlimit": limit,
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": 320,
                "format": "json",
            },
            headers=HEADERS,
            timeout=15,
        )
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            mime = info.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            url = info.get("thumburl") or info.get("url")
            if url:
                urls.append(url)
    except Exception as exc:
        print(f"  [warn] Query '{query}' failed: {exc}")
    return urls


def download_image(args):
    url, out_path = args
    if os.path.exists(out_path):
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200 or len(r.content) < 2000:
            return False
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 48 or img.shape[1] < 48:
            return False
        cv2.imwrite(out_path, img)
        return True
    except Exception:
        return False


def download_dataset(target_per_class=160):
    """Download balanced Fire vs Light dataset."""
    for cls_name, queries in [("fire", FIRE_QUERIES), ("light", LIGHT_QUERIES)]:
        cls_dir = os.path.join(DATA_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        existing = [f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png"))]
        if len(existing) >= target_per_class:
            print(f"[{cls_name}] Already has {len(existing)} images, skipping download.")
            continue

        print(f"[{cls_name}] Fetching image URLs …")
        all_urls = []
        seen = set()
        for q in queries:
            for u in fetch_wikimedia_urls(q, limit=30):
                if u not in seen:
                    seen.add(u)
                    all_urls.append(u)

        random.seed(42)
        random.shuffle(all_urls)
        all_urls = all_urls[: target_per_class + 40]

        tasks = []
        for idx, url in enumerate(all_urls):
            out_path = os.path.join(cls_dir, f"wiki_{idx:04d}.jpg")
            tasks.append((url, out_path))

        print(f"[{cls_name}] Downloading up to {len(tasks)} thumbnails …")
        ok = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for res in pool.map(download_image, tasks):
                if res:
                    ok += 1
        total = len([f for f in os.listdir(cls_dir) if f.endswith((".jpg", ".png"))])
        print(f"[{cls_name}] Total images ready: {total}")


def capture_from_webcam(cls_name, count=40, camera_idx=0):
    """Capture training images directly from webcam into fire_dataset/<cls_name>/."""
    cls_dir = os.path.join(DATA_DIR, cls_name)
    os.makedirs(cls_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Capturing {count} '{cls_name}' images in 2 seconds …")
    time.sleep(2.0)

    ts = int(time.time())
    saved = 0
    for i in range(count):
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(cls_dir, f"cam_{ts}_{i:03d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
        time.sleep(0.08)
    cap.release()
    print(f"Saved {saved} '{cls_name}' images to {cls_dir}")


class FireLightDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model(num_classes=2, pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def build_hard_crops():
    """Create tight bright-center crops from light images + glare/skin negatives so
    the classifier sees cropped candidate boxes just like at inference time."""
    light_dir = os.path.join(DATA_DIR, "light")
    os.makedirs(light_dir, exist_ok=True)
    # Skip if already generated
    if any(f.startswith("hard_") for f in os.listdir(light_dir)):
        return

    rng = np.random.default_rng(42)
    idx = 0
    # 1. Tight crops of brightest spots inside real light/bulb images
    for fname in sorted(os.listdir(light_dir)):
        if not fname.startswith("wiki_") or idx >= 60:
            continue
        img = cv2.imread(os.path.join(light_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, _, _, max_loc = cv2.minMaxLoc(cv2.GaussianBlur(gray, (15, 15), 0))
        cx, cy = max_loc
        cw, ch = max(48, w // 3), max(48, h // 3)
        x1 = max(0, min(w - cw, cx - cw // 2))
        y1 = max(0, min(h - ch, cy - ch // 2))
        crop = img[y1 : y1 + ch, x1 : x1 + cw]
        if crop.size > 0:
            cv2.imwrite(os.path.join(light_dir, f"hard_crop_{idx:03d}.jpg"), crop)
            idx += 1

    # 2. Synthetic glowing bulbs (cool white, warm yellow, ceiling LED) & skin tones
    for i in range(40):
        canvas = np.zeros((160, 160, 3), dtype=np.uint8)
        bg = int(rng.integers(5, 60))
        canvas[:] = (bg, bg, bg)
        cx, cy = int(rng.integers(50, 110)), int(rng.integers(50, 110))
        rad = int(rng.integers(15, 45))
        if i < 15:
            # White / cool LED bulb
            outer = (230, 245, 255)
            inner = (255, 255, 255)
        elif i < 30:
            # Warm incandescent bulb / ceiling lamp
            outer = (130, 210, 255)
            inner = (235, 250, 255)
        else:
            # Plain bright glare / flashlight / skin tone
            b = int(rng.integers(90, 230))
            g = int(rng.integers(120, 240))
            r = int(rng.integers(170, 255))
            canvas[:] = (b, g, r)
            cv2.imwrite(os.path.join(light_dir, f"hard_synth_{i:03d}.jpg"), canvas)
            continue
        cv2.circle(canvas, (cx, cy), rad, outer, -1)
        cv2.circle(canvas, (cx, cy), max(5, rad // 2), inner, -1)
        canvas = cv2.GaussianBlur(canvas, (19, 19), 0)
        cv2.imwrite(os.path.join(light_dir, f"hard_synth_{i:03d}.jpg"), canvas)
    print(f"[light] Added hard negative crops & bulb/glare samples.")


def train(epochs=14, batch_size=24, lr=4e-4):
    build_hard_crops()
    classes = ["fire", "light"]
    samples = []
    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(DATA_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, fname), label_idx))

    fire_n = sum(1 for _, l in samples if l == 0)
    light_n = sum(1 for _, l in samples if l == 1)
    print(f"\nDataset: fire={fire_n}, light={light_n}, total={len(samples)}")
    if fire_n < 20 or light_n < 20:
        print("ERROR: Not enough images to train.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        FireLightDataset(train_samples, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FireLightDataset(val_samples, val_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MobileNetV3-Small on {device} for {epochs} epochs …")

    model = build_model(num_classes=2, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
        scheduler.step()

        model.eval()
        val_correct = 0
        val_fire_ok = 0
        val_fire_tot = 0
        val_light_ok = 0
        val_light_tot = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                val_correct += (preds == y).sum().item()
                for p, t in zip(preds, y):
                    if t.item() == 0:
                        val_fire_tot += 1
                        val_fire_ok += int(p.item() == 0)
                    else:
                        val_light_tot += 1
                        val_light_ok += int(p.item() == 1)

        tr_acc = train_correct / len(train_samples)
        va_acc = val_correct / len(val_samples)
        print(
            f"  Epoch {epoch:2d}/{epochs} | loss={train_loss/len(train_samples):.4f} "
            f"| train_acc={tr_acc:.1%} | val_acc={va_acc:.1%} "
            f"(fire={val_fire_ok}/{val_fire_tot}, light={val_light_ok}/{val_light_tot})"
        )

        if va_acc >= best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(
        {
            "model_state_dict": best_state,
            "classes": classes,
            "img_size": IMG_SIZE,
            "val_acc": round(best_acc, 4),
        },
        MODEL_OUT,
    )
    print(f"\nSaved trained model to {MODEL_OUT} (best val_acc={best_acc:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Train Fire vs Light classifier")
    parser.add_argument("--capture", choices=["fire", "light"], help="Capture webcam samples")
    parser.add_argument("--count", type=int, default=40, help="Webcam frames to capture")
    parser.add_argument("--retrain", action="store_true", help="Skip download, retrain only")
    parser.add_argument("--epochs", type=int, default=14, help="Training epochs")
    args = parser.parse_args()

    if args.capture:
        capture_from_webcam(args.capture, count=args.count)
        return

    if not args.retrain:
        download_dataset(target_per_class=160)

    train(epochs=args.epochs)


if __name__ == "__main__":
    main()
