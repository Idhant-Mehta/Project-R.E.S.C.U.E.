"""Entry point:  python -m firescout [--source 0|picamera|video.mp4] [--simulate-sensors | --serial PORT]"""

import argparse
import logging
import os
import signal
import sys
import threading

import yaml

from . import PACKAGE_DIR
from .app import FireScoutApp
from .sensors import NoSensors, SimulatedSensors, SerialSensors
from .server import serve


def main():
    ap = argparse.ArgumentParser(description="FireScout rescue intelligence (vision + dashboard)")
    ap.add_argument("--config", default=os.path.join(PACKAGE_DIR, "config.yaml"))
    ap.add_argument("--source", help="override camera.source: auto | picamera | 0 | path/to/video")
    ap.add_argument("--port", type=int, help="override server.port")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--simulate-sensors", action="store_true",
                   help="fake per-room sensor values for UI development (clearly labelled SIMULATED)")
    g.add_argument("--serial", help="ESP32 serial port, e.g. /dev/serial0 or COM5")
    ap.add_argument("--no-flame-verify", action="store_true",
                    help="skip the fire-vs-light CNN (saves RAM/CPU; more false flames)")
    bg = ap.add_mutually_exclusive_group()
    bg.add_argument("--beep", dest="beep", action="store_true", default=None,
                    help="enable the fire alarm beep on this machine (skip the question)")
    bg.add_argument("--no-beep", dest="beep", action="store_false",
                    help="disable the fire alarm beep on this machine (skip the question)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.source:
        cfg["camera"]["source"] = args.source
    if args.port:
        cfg["server"]["port"] = args.port

    if args.serial:
        sensors = SerialSensors(args.serial)
    elif args.simulate_sensors:
        sensors = SimulatedSensors()
        logging.warning("SENSOR VALUES ARE SIMULATED – for UI development only, never for results.")
    else:
        sensors = NoSensors()

    beep = args.beep
    if beep is None:
        if sys.stdin.isatty():
            ans = input("Beep on this laptop when fire is detected? [Y/n]: ").strip().lower()
            beep = ans not in ("n", "no")
        else:
            beep = True  # not started from a terminal (e.g. a service): default on
    logging.info("Fire alarm beep: %s", "ON" if beep else "OFF")

    app = FireScoutApp(cfg, sensors, verify_flames=not args.no_flame_verify, beep=beep)
    srv = serve(app, cfg["server"]["host"], cfg["server"]["port"], cfg["server"]["stream_fps"])
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def stop(*_):
        app.running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        app.run()
    finally:
        srv.shutdown()
        app.close()
        logging.info("Mission data saved in %s", app.mission.dir)


if __name__ == "__main__":
    main()
