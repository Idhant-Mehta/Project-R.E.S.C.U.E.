"""Lightweight IoU tracker for counting people without double counting.

Deliberately simple (no re-ID network) so it runs on a Pi 4 and behaves the same
whether detections come from Ultralytics or, later, an ONNX model.

Counting semantics per room (see docs §4.2):
  max_concurrent          = most confirmed tracks visible in one frame   (lower bound)
  unique_confirmed_tracks = distinct confirmed IDs during the room visit (upper bound;
                            ID switches after occlusion inflate it)
"""

from dataclasses import dataclass, field


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: tuple
    conf: float
    hits: int = 1
    missed: int = 0
    confirmed: bool = False
    age: int = 1


@dataclass
class IoUTracker:
    iou_match: float = 0.3
    confirm_hits: int = 5
    max_missed: int = 15
    tracks: list = field(default_factory=list)
    _next_id: int = 1

    def update(self, detections):
        """detections: list of (x1, y1, x2, y2, conf). Returns confirmed tracks seen this frame."""
        # Greedy matching by IoU, highest pairs first.
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, d in enumerate(detections):
                v = iou(t.bbox, d[:4])
                if v >= self.iou_match:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t.bbox, t.conf = tuple(detections[di][:4]), detections[di][4]
            t.hits += 1
            t.missed = 0
            if t.hits >= self.confirm_hits:
                t.confirmed = True

        for ti, t in enumerate(self.tracks):
            t.age += 1
            if ti not in used_t:
                t.missed += 1
                # an unconfirmed track that misses is most likely a false positive
                if not t.confirmed:
                    t.hits = max(0, t.hits - 1)

        for di, d in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(self._next_id, tuple(d[:4]), d[4],
                                         confirmed=self.confirm_hits <= 1))
                self._next_id += 1

        self.tracks = [t for t in self.tracks
                       if t.missed <= self.max_missed and (t.confirmed or t.hits > 0)]
        return [t for t in self.tracks if t.confirmed and t.missed == 0]

    def reset(self):
        self.tracks = []
