"""Rule-based, explainable hazard and rescue-priority classification.

Hazard score = the WORST single factor (0..1), so the reason is always one
readable sentence ("CO 180 ppm"). Missing sensors contribute nothing and are
listed as missing, so a room is never reported 'safe' just because a sensor
is absent.

Priority:
  P1  people >= 1 and hazard >= HIGH     -> rescue immediately
  P2  people >= 1 and hazard <  HIGH     -> rescue soon
  P3  people == 0 and hazard >= HIGH     -> fire-spread / no-entry risk
  P4  otherwise
"""

import numpy as np

HAZARD_ORDER = ("LOW", "ELEVATED", "HIGH", "CRITICAL")


def _lin(v, start, full):
    return float(np.clip((v - start) / max(full - start, 1e-6), 0, 1))


def _piecewise(v, points):
    xs, ys = zip(*points)
    return float(np.interp(v, xs, ys))


def assess(cfg, people, smoke_level, flame_seen, temp_c=None, co_ppm=None, gas_lel_pct=None):
    factors, missing = [], []

    s = cfg["smoke_level_score"].get(smoke_level, 0.0)
    factors.append((s, f"visual smoke {smoke_level}"))
    if flame_seen:
        factors.append((cfg["flame_score"], "flame detected"))

    if temp_c is None:
        missing.append("temperature")
    else:
        factors.append((_lin(temp_c, cfg["temp_c"]["start"], cfg["temp_c"]["full"]),
                        f"temperature {temp_c:.0f} °C"))
    if co_ppm is None:
        missing.append("CO")
    else:
        factors.append((_piecewise(co_ppm, cfg["co_ppm"]), f"CO {co_ppm:.0f} ppm"))
    if gas_lel_pct is not None:
        factors.append((_lin(gas_lel_pct, cfg["gas_lel_pct"]["start"], cfg["gas_lel_pct"]["full"]),
                        f"combustible gas {gas_lel_pct:.0f} % LEL"))

    score, _ = max(factors, key=lambda f: f[0])
    lv = cfg["hazard_levels"]
    if score >= lv["critical"]:
        level = "CRITICAL"
    elif score >= lv["high"]:
        level = "HIGH"
    elif score >= lv["elevated"]:
        level = "ELEVATED"
    else:
        level = "LOW"

    reasons = [txt for v, txt in sorted(factors, reverse=True) if v >= lv["elevated"]]
    high = HAZARD_ORDER.index(level) >= HAZARD_ORDER.index("HIGH")
    if people >= 1:
        priority = "P1" if high else "P2"
    else:
        priority = "P3" if high else "P4"

    return {
        "hazard_score": round(score, 3),
        "hazard_level": level,
        "priority": priority,
        "reasons": reasons,
        "missing_sensors": missing,
    }
