"""Environmental / rover sensor input.

Three sources, always labelled so simulated values can never pass as measurements:
  NoSensors        : no hardware connected -> every reading is None (dashboard shows "—").
  SimulatedSensors : deterministic fake values per room, for UI/demo development ONLY.
                     Tagged source="SIMULATED"; never use them in the results chapter.
  SerialSensors    : reads the ESP32 UART protocol from docs §2 ($ENV / $TEL lines).
"""

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, asdict

log = logging.getLogger("sensors")


@dataclass
class SensorReading:
    source: str                 # "none" | "SIMULATED" | "esp32"
    ts: float
    temp_c: float = None
    rh_pct: float = None
    co_ppm: float = None
    gas_lel_pct: float = None
    pm25: float = None
    battery_v: float = None
    tof_mm: float = None
    flags: int = None

    def to_dict(self):
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


class NoSensors:
    source = "none"

    def read(self, room_name=None):
        return SensorReading(source="none", ts=time.time())

    def close(self):
        pass


class SimulatedSensors:
    """Repeatable fake readings: each room name gets its own fixed 'scenario'."""
    source = "SIMULATED"

    def read(self, room_name=None):
        t = time.time()
        if room_name:
            seed = int(hashlib.md5(room_name.encode()).hexdigest()[:8], 16)
            severity = (seed % 100) / 100.0
        else:
            severity = 0.05
        wobble = 0.05 * math.sin(t / 3.0)
        return SensorReading(
            source="SIMULATED", ts=t,
            temp_c=26 + 70 * severity + 2 * wobble * 10,
            rh_pct=55 - 30 * severity,
            co_ppm=max(0.0, 400 * severity ** 2 + 5 * wobble * 10),
            gas_lel_pct=max(0.0, 12 * severity - 3),
            pm25=10 + 500 * severity,
            battery_v=11.4, tof_mm=1200.0, flags=0,
        )

    def close(self):
        pass


def nmea_checksum(body):
    cs = 0
    for ch in body.encode():
        cs ^= ch
    return cs


def parse_line(line):
    """Parse '$TYPE,f1,f2,...*HH'. Returns (type, [fields]) or None if malformed/bad checksum."""
    line = line.strip()
    if not line.startswith("$") or "*" not in line:
        return None
    body, _, cs = line[1:].rpartition("*")
    try:
        if int(cs, 16) != nmea_checksum(body):
            return None
    except ValueError:
        return None
    parts = body.split(",")
    return parts[0], parts[1:]


class SerialSensors:
    """Background reader for the ESP32 telemetry protocol (see docs §2)."""
    source = "esp32"

    def __init__(self, port, baud=115200, stale_after_s=2.0):
        import serial  # pip install pyserial
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.stale_after = stale_after_s
        self.latest = SensorReading(source="esp32", ts=0.0)
        self.bad_frames = 0
        self._lock = threading.Lock()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            try:
                raw = self.ser.readline().decode("ascii", errors="ignore")
            except Exception as exc:
                log.warning("serial read failed: %s", exc)
                time.sleep(0.5)
                continue
            if not raw:
                continue
            msg = parse_line(raw)
            if msg is None:
                self.bad_frames += 1
                continue
            kind, f = msg
            try:
                with self._lock:
                    r = self.latest
                    if kind == "ENV" and len(f) >= 6:
                        # $ENV,<seq>,<temp_cC>,<rh_x10>,<co_ppm>,<gas_raw>,<pm25>
                        r.temp_c = int(f[1]) / 100.0
                        r.rh_pct = int(f[2]) / 10.0
                        r.co_ppm = float(f[3])
                        r.gas_lel_pct = None  # raw MQ value is not %LEL until calibrated
                        r.pm25 = float(f[5])
                        r.ts = time.time()
                    elif kind == "TEL" and len(f) >= 6:
                        # $TEL,<seq>,<tof_mm>,<batt_mV>,<lpwm>,<rpwm>,<flags>
                        r.tof_mm = float(f[1])
                        r.battery_v = int(f[2]) / 1000.0
                        r.flags = int(f[5])
                        r.ts = time.time()
            except ValueError:
                self.bad_frames += 1

    def read(self, room_name=None):
        with self._lock:
            r = SensorReading(**asdict(self.latest))
        if time.time() - r.ts > self.stale_after:
            # stale data must not be shown as current
            return SensorReading(source="esp32 (STALE)", ts=r.ts)
        return r

    def close(self):
        self._run = False
        self.ser.close()
