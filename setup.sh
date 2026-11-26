#!/bin/bash
# ============================================================
# Rescue Rover — One-time setup on Raspberry Pi 4
# Target: Raspberry Pi OS 64-bit (Bookworm or newer)
# Run:  chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo "=== Updating system ==="
sudo apt update && sudo apt upgrade -y

echo "=== Installing system dependencies ==="
# picamera2 MUST come from apt: it depends on libcamera's Python bindings,
# which pip cannot build. (libtiff5/libopenjp2-7/libatlas-base-dev were
# removed: the pip OpenCV/numpy wheels bundle what they need, and libtiff5
# does not exist on Bookworm+, which made this script abort.)
sudo apt install -y python3-pip python3-venv python3-picamera2

echo "=== Creating Python virtual environment ==="
# --system-site-packages lets the venv see the apt-installed picamera2.
# Without it, 'from picamera2 import Picamera2' fails inside the venv and
# rescue_detector.py silently falls back to a USB camera.
if [ -d venv ] && ! grep -q "include-system-site-packages = true" venv/pyvenv.cfg; then
    echo "Existing venv was created without system site packages — recreating it."
    rm -rf venv
fi
python3 -m venv --system-site-packages venv
source venv/bin/activate

echo "=== Pinning numpy to the system version ==="
# picamera2's compiled helpers (simplejpeg etc.) are built against the
# apt numpy. If pip upgrades numpy inside the venv (newer OpenCV wheels
# want numpy 2.x), picamera2 crashes on import. Pinning numpy makes pip
# choose package versions compatible with the system numpy instead.
SYS_NUMPY=$(/usr/bin/python3 -c "import numpy; print(numpy.__version__)")
echo "numpy==${SYS_NUMPY}" > constraints-pi.txt
echo "System numpy: ${SYS_NUMPY}"

echo "=== Installing Python packages ==="
pip install --upgrade pip
pip install -r requirements.txt -c constraints-pi.txt

echo "=== Verifying camera stack ==="
python -c "from picamera2 import Picamera2; import cv2, numpy; print('picamera2 OK | OpenCV', cv2.__version__, '| numpy', numpy.__version__)"
if command -v rpicam-hello >/dev/null 2>&1; then
    rpicam-hello --list-cameras || true
elif command -v libcamera-hello >/dev/null 2>&1; then
    libcamera-hello --list-cameras || true
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Activate env:  source venv/bin/activate"
echo "  Run detector:  python rescue_detector.py --no-display"
echo "============================================"
