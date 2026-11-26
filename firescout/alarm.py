"""Audible fire alarm on the machine running the model (non-blocking, rate-limited)."""

import sys
import threading
import time


class Alarm:
    def __init__(self, enabled=True, cooldown_s=2.0, freq_hz=2200, beeps=3, beep_ms=150):
        self.enabled = enabled
        self.cooldown = cooldown_s
        self.freq, self.beeps, self.beep_ms = freq_hz, beeps, beep_ms
        self._last = 0.0
        self._busy = threading.Lock()

    def trigger(self):
        """Beep unless disabled, still in cooldown, or already beeping."""
        now = time.time()
        if not self.enabled or now - self._last < self.cooldown:
            return False
        if not self._busy.acquire(blocking=False):
            return False
        self._last = now
        threading.Thread(target=self._beep, daemon=True).start()
        return True

    def _beep(self):
        try:
            try:
                import winsound  # Windows laptop
                for _ in range(self.beeps):
                    winsound.Beep(self.freq, self.beep_ms)
                    time.sleep(0.06)
            except ImportError:
                # Linux / macOS: terminal bell (works in most terminals)
                for _ in range(self.beeps):
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    time.sleep(0.2)
        finally:
            self._busy.release()
