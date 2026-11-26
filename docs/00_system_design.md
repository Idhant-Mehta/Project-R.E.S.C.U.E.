# FireScout: Room-Level Rescue Intelligence Rover
## System Design Baseline (Phase 0), v0.1

> Status: **design baseline, not final.** Items marked **[MISSING]** must be confirmed before the electrical design is frozen. **[TARGET]** = proposed goal, not a measured result. No measured results exist yet.
>
> Safety scope: academic prototype. It is tested only in controlled, simulated environments (fog-machine smoke, heat lamps, supervised calibration gas). It is **not** certified life-safety equipment and must never be deployed into a real, uncontrolled fire.

---

## 1. Narrative

Before a crew commits to a smoke-filled corridor, FireScout goes in first and returns a **room-by-room picture**: *"Room 2: 2 people (range 2–3), visual smoke HIGH, 61 °C at floor level, CO 180 ppm → PRIORITY 1"*. The aim is to shorten blind search time and show which doors to open first. The project is defined by its Rescue Intelligence layer, not by the RC car underneath.

---

## 2. Communication architecture

```
Phone (browser) ──Wi-Fi: HTTP + WebSocket──► Raspberry Pi 4  (vision, fusion, dashboard, logs, MQTT on localhost)
                                                │ UART 115200, 3.3 V          │ CSI ribbon
                                                ▼                             ▼
                                             ESP32 (motors, watchdog,      Pi Camera V2
                                                    sensors)
                                                │ PWM            │ I²C / ADC
                                             BTS7960(s)        VL53L0X, gas, temp, battery voltage
```

| Link | Choice | Why / rejected alternatives |
|---|---|---|
| Phone ↔ Pi | Wi-Fi, browser web app | One link carries video, telemetry and control, with nothing to install. Bluetooth rejected: short range, no video bandwidth, and on the Pi 4 it shares a radio with Wi-Fi. |
| Pi ↔ ESP32 | Wired UART (Pi GPIO14/15 ↔ ESP32 UART2 GPIO16/17) | Deterministic, both 3.3 V logic, no radio. USB serial only for bench work, because it back-powers the ESP32 and ties grounds through the cable. |
| Pi IPC | MQTT (Mosquitto bound to 127.0.0.1) | Each function runs as its own process, so a vision crash can't kill motor control. Easy to inspect live with `mosquitto_sub`. |
| Wi-Fi infrastructure | Pi hotspot for the lab; travel router at the operator for the demo | Range is a measured metric, not a claim. |

### Control-path safety
- Phone joystick at 20 Hz → Pi → `$DRV` over UART with a sequence number.
- **ESP32 dead-man:** no valid command for 300 ms → motors ramp to 0.
- Pi watchdog: WebSocket silent for 500 ms → explicit STOP plus a "COMMS LOST" banner.
- **Obstacle interlock on the ESP32:** forward is blocked under the stop distance, while reverse and turning stay allowed. It does not depend on Linux.
- Hardware E-stop in the motor positive line.

### UART protocol (NMEA-style, XOR checksum)
```
Pi → ESP32:  $DRV,<seq>,<left -1000..1000>,<right>*<cs>   $STP,<seq>*<cs>   $CFG,<seq>,<key>,<val>*<cs>
ESP32 → Pi:  $TEL,<seq>,<tof_mm>,<batt_mV>,<lpwm>,<rpwm>,<flags>*<cs>   (10 Hz)
             $ENV,<seq>,<temp_cC>,<rh_x10>,<co_ppm>,<gas_raw>,<pm25>*<cs>
             $EVT,<seq>,<OBSTACLE_STOP|WATCHDOG_STOP|LOW_BATT|...>*<cs>
```

### Responsibility split
Rule: **anything that must still work when Linux hangs lives on the ESP32.**

| Device | Responsibilities |
|---|---|
| ESP32 | Motors, ramping, watchdog, ToF, battery ADC, analog gas sensors (the Pi has no ADC), I²C environmental sensors |
| Pi | Camera, detection/tracking, smoke, fusion, dashboard, SQLite log, report, optional MLX90640 |

---

## 3. Feature ranking

| Tier | Features | Rationale |
|---|---|---|
| **Core** | Phone joystick + live video; comms-loss alert + auto-stop; person detection + per-room counting; Rescue Intelligence Dashboard; operator-tagged rooms; temperature + CO; visual smoke severity; rule-based priority; battery monitoring; HW+SW E-stop; ToF obstacle *stop* | Highest rescue value per unit of effort; all feasible on current hardware |
| **Should** | Mission log + auto report; detection alerts; floor vs. mast temperature (smoke layer); ArUco door markers for automatic room ID; particulate sensor as a physical check on visual smoke | Cheap, strong engineering signal |
| **Stretch** | MLX90640 thermal (32×24) | Best cue through smoke, but in real fires the background can be hotter than a person |
| **Don't promise** | Autonomous obstacle avoidance; indoor mapping/SLAM | One narrow ToF is not enough; SLAM needs encoders + IMU at minimum |

---

## 4. AI architecture

### 4.1 Person detection
- YOLO11n or YOLOv8n, one class (person), imgsz 320, exported to **ONNX Runtime or NCNN**. MobileNet-SSD serves as the baseline comparison row.
- **No PyTorch on the 2 GB Pi.** Train on the RTX 4050 laptop, export, and run with `onnxruntime`.
- Raspberry Pi OS Lite 64-bit with zram swap.
- FPS is benchmarked at 256/320/416 in Phase 5. No figure is quoted before it's measured.
- Fine-tuning additions:
  - Lying/crawling people (COCO is biased toward upright people)
  - Occlusion-heavy data
  - Synthetic haze augmentation: `I = J·t + A(1−t)`
- Ultralytics is AGPL-3.0, which is fine for academic use. Note it in the report.

### 4.2 Counting without double counting
- ByteTrack. A track is confirmed after about 5 consecutive hits.
- Per room, report:
  - `max_concurrent`: the lower bound
  - `unique_confirmed_tracks`: the upper bound, inflated by ID switches
- The dashboard shows "2 people (range 2–3)", and the operator can confirm it.

### 4.3 Smoke detection and severity
- **Camera severity = visual obscuration, NOT toxicity.** Theatrical fog is dense and harmless; thin smoke can be lethal. Gas sensors are mandatory.
- Hybrid method:
  1. **Visibility metrics:** dark-channel haze estimate, RMS contrast, Laplacian sharpness, saturation, and edge density compared with a clear-air reference.
  2. **Smoke presence:** the existing `fire_yolov8n.pt` already has a `smoke` class (§9), plus an optional small classifier.
  3. **Severity** (Low/Moderate/High/Critical) from thresholds calibrated against a **printed contrast target at a fixed distance**, which gives objective ground truth.
- Public smoke data (e.g. D-Fire) is mostly outdoor, so indoor fog-machine frames must be collected.

### 4.4 Fusion and priority (rule-based, explainable)
```
hazard = w_s·S(smoke) + w_t·T(temp) + w_c·C(CO) + w_g·G(combustible) + w_f·F(flame)   (sub-scores 0..1, thresholds in YAML)
P1: people ≥ 1 and hazard ≥ HIGH   P2: people ≥ 1, hazard < HIGH
P3: no people, hazard ≥ HIGH       P4: otherwise
```
CO reference points: OSHA PEL 50 ppm (8 h TWA), NIOSH ceiling 200 ppm, NIOSH IDLH 1200 ppm.

---

## 5. Electrical (provisional, blocked on §8)

```
LiPo A ─XT60─ fuse ─ E-stop ─┬─ BTS7960 L ─ left motor(s)
                             ├─ BTS7960 R ─ right motor(s)
                             └─ divider ─ ESP32 ADC (battery voltage)
LiPo B ─XT60─ fuse ─ switch ─┬─ 5 V ≥5 A buck ─ Pi (USB-C pigtail) + camera
                             └─ UBEC 5 V/3 A ─ ESP32, sensors
ALL GROUNDS COMMON (star point)
```

- **Separate motor and logic packs.** Motor start/stall spikes brown out the Pi, causing reboots and SD-card corruption.
- **The UBEC (5 V/3 A) is marginal for the Pi 4.** The official supply is 5.1 V/3 A, and a camera, fan and sensors come on top. Use a buck converter rated ≥5 A. Verify with `vcgencmd get_throttled` = `0x0` under load.
- **BTS7960 (IBT-2):** inputs go through a 74HC244. Powered at 5 V, it may not reliably read 3.3 V logic. Power the logic VCC from 3.3 V (verify on your board) or use a level shifter. The IS pins need a divider before the ESP32 ADC.
- **Protection:**
  - Main fuse sized from **measured** stall current
  - Latching E-stop
  - Low-voltage alarm (around 3.5 V/cell under load as a conservative stop)
  - Balance charging in a LiPo bag
  - 100 nF capacitors across the motor terminals
  - Heatsink + fan on the Pi
- **Runtime:** `usable Wh ≈ 0.8 × V_nom × Ah`, divided by the measured average power. Not computed until the cell count and current are known.

---

## 6. Sensors and placement

| Quantity | Budget | Better | Note |
|---|---|---|---|
| CO | MQ-7 | Electrochemical module (ZE07-CO class) | MQ-7 drifts and cross-reacts with H₂ |
| Temp/RH | DHT22 | SHT41/SHT31 | |
| Hot air | — | K-type thermocouple + MAX31855 | |
| Combustible | MQ-2 | MQ-2 | Indicative only; needs burn-in + clean-air calibration |
| Particulates | — | PMS5003-class | Physical cross-check for visual smoke |
| Thermal | AMG8833 8×8 | MLX90640 32×24 | |
| Obstacle backup | HC-SR04 ultrasonic | — | Unaffected by smoke; the VL53L0X is optical |

**Placement**
- **Camera:** low (15–25 cm) with a slight downward tilt. Smoke rises, and victims are often on the floor.
- **ToF:** front-center at bumper height.
- **Gas sensors:** in a shaded, vented pocket away from the motors and the Pi fan.
- **Temperature:** one sensor low on the chassis and one on a 40–50 cm mast.
- **Batteries:** low and centred.
- Keep the CSI ribbon away from the motor leads.

---

## 7. Evaluation plan (metrics defined; no results yet)

| Metric | Method |
|---|---|
| Detection precision / recall / mAP@0.5 | Held-out set incl. own indoor + haze images |
| Counting accuracy | 0–4 people × N trials; MAE, % exact |
| Detection vs. smoke | Recall vs. measured visibility index |
| Smoke presence / severity | Precision/recall; confusion matrix vs. contrast-target ground truth |
| Sensor response | T90 under step exposure |
| Video / control latency | Film a stopwatch alongside the dashboard; high-speed video for wheel motion |
| Comms range | Distance/walls until packet loss > X% |
| Watchdog stop time | Kill Wi-Fi → wheels stop **[TARGET ≤ 0.5 s]** |
| Obstacle stop | 10 approaches at fixed speed **[TARGET: no contact]** |
| Inference speed | FPS, latency, CPU temperature over 5 min |
| Battery runtime | Continuous mission until LV alarm |
| Room report time | Enter room → record complete |

Every result table distinguishes **Target / Measured (n=…) / Assumption**.

---

## 8. MISSING INFORMATION (blocks the electrical design)

1. **Johnson motors:** count, label (voltage, RPM), stall current, encoders?
2. **BTS7960:** how many modules? Differential drive needs two.
3. **LiPos:** cell count, C-rating; are all three identical?
4. **Drive layout** (2WD + caster / 4WD skid / tracks) and wheel diameter.
5. **Chassis:** material and dimensions, target mass.
6. **ESP32:** exact devkit.
7. **Test venue:** distance and number of walls.
8. **Sensor budget** and supplier.
9. **Fog machine** availability and fire-alarm permission.
10. **Provenance of `fire_yolov8n.pt`:** who trained it, and on which dataset (see §9).

---

## 9. Existing code audit (as of 2026-11-25)

What exists:

| File | What it does |
|---|---|
| `rescue_detector.py` | Pi/USB camera → YOLOv8n person + flame detection, alerts, optional buzzer |
| `flame_detector.py` | Two-stage flame pipeline: `fire_yolov8n.pt` proposals + HSV warm regions, verified by a MobileNetV3-Small fire-vs-light CNN |
| `live_detector.py` | Side-by-side MJPEG stream on :8080 (webcam only) |
| `test_detector.py` | 15-frame headless test |
| `train_fire_model.py` | Wikimedia download + webcam capture + classifier training |
| `fire_dataset/` | fire: 198 images, light: 297 images |
| Models | `fire_yolov8n.pt` classes `{0: smoke, 1: fire}`; `fire_vs_light.pt` (img 128, recorded val_acc 0.8687 on its own random split) |

So far it has only been run on the Windows laptop webcam (`rescue_detector.log` shows the "Pi camera not found" fallback).

### Bugs to fix before running on the Pi
> **Fixed in code on 2026-11-25 (not yet verified on Pi hardware):** items 1–3, plus a numpy pin in `setup.sh` so pip cannot upgrade numpy past the version the apt-installed picamera2 was built against.

1. **Colour channels will be swapped on the Pi.** In picamera2, format `"RGB888"` returns pixels in **B,G,R** order, which OpenCV already expects. `grab_frame()` then applies `COLOR_RGB2BGR`, producing red↔blue swapped frames. That breaks the HSV flame proposals and the fire classifier, and degrades YOLO. Fix: remove the conversion (or request `"BGR888"`, which gives R,G,B order, and keep it).
2. **`setup.sh` will make picamera2 invisible inside the venv.** `python3 -m venv venv` excludes the apt-installed `python3-picamera2`, so the import fails and the code silently falls back to a USB camera. Fix: `python3 -m venv --system-site-packages venv`, and remove `picamera2` from `requirements.txt` (pip cannot build it properly).
3. **`setup.sh` apt line fails on Bookworm and later:** `libtiff5` doesn't exist there (`libtiff6`). Combined with `set -e`, the script stops.
4. **Memory on a 2 GB Pi:** three PyTorch models (person YOLO, fire YOLO, MobileNetV3) plus torch/torchvision/ultralytics in one process. Plan to export all three to ONNX and drop torch on the Pi.

### Design gaps relative to the capstone goals
- The **smoke** class of `fire_yolov8n.pt` is loaded but ignored (only class 1, fire, is used). It's a free smoke-presence signal for §4.3.
- No tracking, so the person count is per frame; no room records.
- `AlertSystem.trigger()` sleeps 150 ms in the vision loop when the buzzer is on.
- `test_detector.py` runs flame detection on a frame that already has person boxes drawn on it.
- `fire_vs_light` validation mixes synthetic hard negatives into a random split. Its 86.9% is not an independent test result and should not be reported as accuracy.
- `fire_yolov8n.pt` was trained from a path `/Users/rbh/Downloads/archive/data.yaml`, apparently a third-party model. Its source and dataset licence must be documented, or it must be retrained yourself.

### How this fits the new architecture
- `flame_detector.py` becomes `vision/flame.py`, contributing an `F(flame)` term to the hazard score.
- `fire_yolov8n` smoke boxes feed the smoke-presence stage.
- `live_detector.py`'s MJPEG server becomes the seed of the dashboard video endpoint.
- `PersonDetector` is replaced by an ONNX detector + tracker in Phase 5.

---

## 10. Phase roadmap (exit gates)

| Phase | Exit gate |
|---|---|
| 0 | §8 answered, power design frozen |
| 1 | Power: all rails measured; Pi stable on battery, `get_throttled` = `0x0` under stress |
| 2 | ESP32 motors with wheels off the ground: ramping works, UART silence stops in < 0.5 s, ToF blocks forward |
| 3 | Phone control; killing Wi-Fi stops the rover |
| 4 | Pi camera + stream (with bug fixes 1–3 above); latency measured |
| 5 | ONNX person detector + tracker + room counting; FPS benchmarked |
| 6 | Smoke severity calibrated in fog trials |
| 7 | Environmental sensors; burn-in; response time |
| 8 | Dashboard + fusion + SQLite + report |
| 9 | Full evaluation campaign |
| 10 | Report, slides, demo, viva |

Planned layout:
```
firescout/
├── docs/
├── firmware/esp32_rover/        (Phase 2)
├── pi/rover_link/  pi/dashboard/  pi/vision/  pi/mission/  pi/config/  pi/systemd/
├── training/                    (laptop, RTX 4050 6 GB; AWS only if a run won't fit)
├── models/                      (.onnx / ncnn for the Pi)
└── legacy/                      (current prototype scripts, kept for reference)
```
