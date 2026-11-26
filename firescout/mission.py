"""Mission state: room-wise records, event log, SQLite persistence, inspection report."""

import base64
import html
import json
import os
import sqlite3
import threading
import time
from datetime import datetime

import cv2

from . import PROJECT_DIR
from .risk import assess, HAZARD_ORDER
from .smoke import LEVELS as SMOKE_LEVELS

PRIORITY_ORDER = ("P1", "P2", "P3", "P4")


def _ts(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S") if t else "—"


def _max(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


class Room:
    def __init__(self, rid, name):
        self.id, self.name = rid, name
        self.entered_at, self.exited_at = time.time(), None
        self.max_concurrent = 0
        self.track_ids = set()
        self.operator_count = None
        self.smoke_peak = "LOW"
        self.smoke_peak_score = 0.0
        self.flame_seen = False
        self.temp_max = self.co_max = self.gas_max = self.pm25_max = None
        self.sensor_source = "none"
        self.assessment = None
        self.snapshot = None          # path to JPEG

    @property
    def people(self):
        """People used for prioritisation: operator-confirmed count if given, else the lower bound."""
        return self.operator_count if self.operator_count is not None else self.max_concurrent

    def to_dict(self):
        return {
            "id": self.id, "name": self.name,
            "entered_at": self.entered_at, "exited_at": self.exited_at,
            "active": self.exited_at is None,
            "people": self.people,
            "people_min": self.max_concurrent,
            "people_max": max(self.max_concurrent, len(self.track_ids)),
            "operator_count": self.operator_count,
            "smoke_peak": self.smoke_peak, "smoke_peak_score": round(self.smoke_peak_score, 3),
            "flame_seen": self.flame_seen,
            "temp_max": self.temp_max, "co_max": self.co_max,
            "gas_max": self.gas_max, "pm25_max": self.pm25_max,
            "sensor_source": self.sensor_source,
            "has_snapshot": self.snapshot is not None,
            **(self.assessment or {}),
        }


class Mission:
    def __init__(self, risk_cfg, data_dir="missions"):
        self.risk_cfg = risk_cfg
        self.started = time.time()
        self.id = datetime.now().strftime("mission_%Y%m%d_%H%M%S")
        root = data_dir if os.path.isabs(data_dir) else os.path.join(PROJECT_DIR, data_dir)
        self.dir = os.path.join(root, self.id)
        os.makedirs(os.path.join(self.dir, "snapshots"), exist_ok=True)
        self.rooms, self.events = [], []
        self.current = None
        self._lock = threading.RLock()
        self._last_save = 0.0
        self.db = sqlite3.connect(os.path.join(self.dir, "mission.db"), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events(ts REAL, room TEXT, kind TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS rooms(id INTEGER PRIMARY KEY, name TEXT, entered_at REAL,
                exited_at REAL, data TEXT);
        """)
        self.event("MISSION_START", self.id)

    # ── events ───────────────────────────────────────────────────────────
    def event(self, kind, detail="", room=None):
        with self._lock:
            name = room or (self.current.name if self.current else "—")
            e = {"ts": time.time(), "room": name, "kind": kind, "detail": detail}
            self.events.append(e)
            self.events = self.events[-500:]
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (e["ts"], name, kind, detail))
            self.db.commit()

    # ── room lifecycle ───────────────────────────────────────────────────
    def enter_room(self, name=None):
        with self._lock:
            if self.current:
                self.exit_room()
            name = (name or "").strip() or f"Room {len(self.rooms) + 1}"
            room = Room(len(self.rooms) + 1, name)
            self.rooms.append(room)
            self.current = room
            self.event("ROOM_ENTER", name, room=name)
            return room

    def exit_room(self):
        with self._lock:
            if not self.current:
                return None
            room = self.current
            room.exited_at = time.time()
            self._save_room(room)
            a = room.assessment or {}
            self.event("ROOM_EXIT",
                       f"{room.people} people, smoke {room.smoke_peak}, "
                       f"hazard {a.get('hazard_level', '?')} -> {a.get('priority', '?')}", room=room.name)
            self.current = None
            return room

    def set_operator_count(self, n):
        with self._lock:
            room = self.current or (self.rooms[-1] if self.rooms else None)
            if room is None:
                return None
            room.operator_count = None if n is None else int(n)
            self._reassess(room)
            self.event("OPERATOR_COUNT", f"operator confirmed {n} people", room=room.name)
            return room

    # ── per-frame update ─────────────────────────────────────────────────
    def update(self, confirmed_tracks, smoke, flames, sensors, annotated_frame):
        with self._lock:
            room = self.current
            if room is None:
                return None
            n = len(confirmed_tracks)
            new_ids = {t.id for t in confirmed_tracks} - room.track_ids
            room.track_ids |= {t.id for t in confirmed_tracks}
            if n > room.max_concurrent:
                room.max_concurrent = n
                self.event("PERSON_DETECTED", f"{n} people visible (track IDs {sorted(t.id for t in confirmed_tracks)})")
                self._snapshot(room, annotated_frame)
            elif new_ids and room.max_concurrent > 0:
                self.event("NEW_TRACK", f"new track IDs {sorted(new_ids)} (possible re-count)")

            if smoke is not None and SMOKE_LEVELS.index(smoke.level) > SMOKE_LEVELS.index(room.smoke_peak):
                room.smoke_peak = smoke.level
                self.event("SMOKE_LEVEL", f"visual smoke rose to {smoke.level} (score {smoke.score:.2f})")
            if smoke is not None:
                room.smoke_peak_score = max(room.smoke_peak_score, smoke.score)
            if flames and not room.flame_seen:
                room.flame_seen = True
                self.event("FLAME", f"flame detected (p={flames[0]['confidence']:.2f})")
                self._snapshot(room, annotated_frame, force=True)

            room.sensor_source = sensors.source
            room.temp_max = _max(room.temp_max, sensors.temp_c)
            room.co_max = _max(room.co_max, sensors.co_ppm)
            room.gas_max = _max(room.gas_max, sensors.gas_lel_pct)
            room.pm25_max = _max(room.pm25_max, sensors.pm25)

            before = (room.assessment or {}).get("hazard_level")
            self._reassess(room)
            after = room.assessment["hazard_level"]
            if before and HAZARD_ORDER.index(after) > HAZARD_ORDER.index(before):
                self.event("HAZARD", f"hazard rose to {after}: {', '.join(room.assessment['reasons'])}")
                if after in ("HIGH", "CRITICAL"):
                    self._snapshot(room, annotated_frame, force=True)

            if time.time() - self._last_save > 2.0:
                self._save_room(room)
                self._last_save = time.time()
            return room

    def _reassess(self, room):
        room.assessment = assess(self.risk_cfg, room.people, room.smoke_peak, room.flame_seen,
                                 temp_c=room.temp_max, co_ppm=room.co_max, gas_lel_pct=room.gas_max)

    def _snapshot(self, room, frame, force=False):
        if frame is None or (room.snapshot and not force):
            return
        path = os.path.join(self.dir, "snapshots", f"room{room.id:02d}_{int(time.time() * 1000)}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        room.snapshot = path

    def _save_room(self, room):
        self.db.execute("INSERT OR REPLACE INTO rooms VALUES (?,?,?,?,?)",
                        (room.id, room.name, room.entered_at, room.exited_at, json.dumps(room.to_dict())))
        self.db.commit()

    # ── views ────────────────────────────────────────────────────────────
    def snapshot_path(self, rid):
        with self._lock:
            r = next((r for r in self.rooms if r.id == rid), None)
            return r.snapshot if r else None

    def state(self):
        with self._lock:
            return {
                "mission_id": self.id,
                "started": self.started,
                "current_room": self.current.to_dict() if self.current else None,
                "rooms": [r.to_dict() for r in self.rooms],
                "events": self.events[-40:][::-1],
            }

    def report_html(self):
        """Self-contained inspection report (snapshots embedded). Also saved in the mission folder."""
        with self._lock:
            rooms = sorted(self.rooms, key=lambda r: (
                PRIORITY_ORDER.index((r.assessment or {}).get("priority", "P4")),
                -HAZARD_ORDER.index((r.assessment or {}).get("hazard_level", "LOW"))))
            sources = sorted({r.sensor_source for r in self.rooms}) or ["none"]
            rows, cards = [], []
            for r in rooms:
                d = r.to_dict()
                ppl = f"{d['people_min']}" if d["people_min"] == d["people_max"] else f"{d['people_min']}–{d['people_max']}"
                if d["operator_count"] is not None:
                    ppl += f" (operator: {d['operator_count']})"
                fmt = lambda v, u: "—" if v is None else f"{v:.0f} {u}"
                rows.append(
                    f"<tr class='{d.get('priority','')}'><td><b>{d.get('priority','')}</b></td><td>{html.escape(r.name)}</td>"
                    f"<td>{ppl}</td><td>{d['smoke_peak']}</td><td>{'yes' if r.flame_seen else 'no'}</td>"
                    f"<td>{fmt(r.temp_max, '°C')}</td><td>{fmt(r.co_max, 'ppm')}</td>"
                    f"<td>{d.get('hazard_level','')}</td><td>{html.escape(', '.join(d.get('reasons', [])) or '—')}</td>"
                    f"<td>{_ts(r.entered_at)}–{_ts(r.exited_at)}</td></tr>")
                if r.snapshot and os.path.exists(r.snapshot):
                    b64 = base64.b64encode(open(r.snapshot, "rb").read()).decode()
                    cards.append(f"<figure><img src='data:image/jpeg;base64,{b64}'>"
                                 f"<figcaption>{html.escape(r.name)}</figcaption></figure>")
            events = "".join(
                f"<tr><td>{_ts(e['ts'])}</td><td>{html.escape(e['room'])}</td><td>{e['kind']}</td>"
                f"<td>{html.escape(e['detail'])}</td></tr>" for e in self.events)
            sim_warn = ("<p class='warn'>⚠ Sensor values in this report are <b>SIMULATED</b> "
                        "(UI development mode). They are not measurements.</p>"
                        if "SIMULATED" in sources else "")
            doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FireScout Inspection Report – {self.id}</title>
<style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#1b1f24;max-width:1100px}}
h1{{margin-bottom:4px}} .meta{{color:#555}} table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}}
th,td{{border:1px solid #ccd;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#eef}}
tr.P1 td:first-child{{background:#d62828;color:#fff}} tr.P2 td:first-child{{background:#f77f00;color:#fff}}
tr.P3 td:first-child{{background:#fcbf49}} tr.P4 td:first-child{{background:#cfd8dc}}
.warn{{background:#fff3cd;border:1px solid #e0b000;padding:8px 12px;border-radius:6px}}
figure{{display:inline-block;margin:6px;width:300px}} figure img{{width:100%;border-radius:6px}}
.small{{font-size:12px;color:#555}}
</style></head><body>
<h1>FireScout – Room Inspection Report</h1>
<p class="meta">Mission <b>{self.id}</b> · started {datetime.fromtimestamp(self.started):%Y-%m-%d %H:%M:%S} ·
generated {datetime.now():%Y-%m-%d %H:%M:%S} · {len(self.rooms)} room(s) · sensor source: {', '.join(sources)}</p>
{sim_warn}
<h2>Rescue priority (highest first)</h2>
<table><tr><th>Priority</th><th>Room</th><th>People</th><th>Visual smoke (peak)</th><th>Flame</th>
<th>Temp (max)</th><th>CO (max)</th><th>Hazard</th><th>Why</th><th>Time in room</th></tr>
{''.join(rows) or '<tr><td colspan=10>No rooms recorded.</td></tr>'}</table>
<p class="small">People: automatic count given as a range (lower bound = most people visible at once;
upper bound = distinct tracks, which can over-count after occlusion). Priority: P1 people + hazard ≥ HIGH ·
P2 people, lower hazard · P3 no people, hazard ≥ HIGH · P4 other.</p>
<h2>Snapshots</h2>{''.join(cards) or '<p>None.</p>'}
<h2>Event log</h2><table><tr><th>Time</th><th>Room</th><th>Event</th><th>Detail</th></tr>{events}</table>
<h2>Limitations</h2><ul class="small">
<li>Visual smoke level measures camera obscuration, <b>not</b> toxic gas concentration.</li>
<li>Thresholds are uncalibrated placeholders until the calibration experiments are completed.</li>
<li>Absent sensors are reported as "—" and do not lower the hazard level.</li>
<li>Academic prototype tested in simulated environments. Not certified life-safety equipment.</li>
</ul></body></html>"""
            path = os.path.join(self.dir, "report.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
            return doc, path

    def close(self):
        with self._lock:
            if self.current:
                self.exit_room()
            self.event("MISSION_END", "")
            self.db.close()
