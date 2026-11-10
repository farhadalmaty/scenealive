"""Runs the segmentation model about once a second, OFF the frame path.

The model takes a few hundred milliseconds on a laptop CPU. Run inline, every third frame would be slow; so it runs in its own thread
with a 'latest frame wins' slot, and the fast loop only submits a picture and reads the newest answer. An answer older than max_age_s
counts as unknown (never present an old picture of the floor as the current one)."""
from __future__ import annotations
import threading
import time
from .camera import Camera
from .freespace import FreeSpace, analyse
from .segment import group_map


class SegMonitor:
    def __init__(self, segmenter, h_cam: float = 1.4, every_s: float = 1.0, sync: bool = False, max_age_s: float = 3.0, log=None):
        self.seg, self.h_cam, self.every_s, self.sync, self.max_age_s = segmenter, h_cam, every_s, sync, max_age_s
        self.log = log or (lambda *a: None); self._last_t = -1e9; self._pending = None; self._busy = False; self._fs: FreeSpace | None = None; self._fs_t = -1e9
        self._ms: list[float] = []; self._lock = threading.Lock(); self._ev = threading.Event(); self._stop = False; self._thread = None; self.errors = 0

    def due(self, t: float) -> bool:
        return (t - self._last_t >= self.every_s) and not self._busy

    def submit(self, frame_bgr, t: float, pitch_down: float | None, cam: Camera) -> None:
        self._last_t = t
        if self.sync: self._run(frame_bgr, t, pitch_down, cam); return
        with self._lock: self._pending = (frame_bgr.copy(), t, pitch_down, cam); self._busy = True
        if self._thread is None: self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()
        self._ev.set()

    def _loop(self):
        while not self._stop:
            self._ev.wait(); self._ev.clear()
            if self._stop: return
            with self._lock: item, self._pending = self._pending, None
            if item is not None: self._run(*item)
            with self._lock: self._busy = self._pending is not None

    def _run(self, frame, t, pitch_down, cam) -> None:
        t0 = time.perf_counter()
        try:
            labels, id2 = self.seg.segment(frame); fs = analyse(group_map(labels, id2), cam, pitch_down, self.h_cam, getattr(self.seg, "last_conf", None))
            with self._lock: self._fs, self._fs_t = fs, t; self._ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception as ex:
            self.errors += 1; self.log("segmentation error:", type(ex).__name__, ex)

    def state(self, now_t: float):
        """-> (FreeSpace or None, age in seconds). None when there is no answer yet or it is too old."""
        with self._lock: fs, ft = self._fs, self._fs_t
        if fs is None or now_t - ft > self.max_age_s: return None, (None if fs is None else now_t - ft)
        return fs, now_t - ft

    def stats(self) -> dict:
        with self._lock: ms = list(self._ms)
        return {"runs": len(ms), "mean_ms": (sum(ms) / len(ms)) if ms else None, "last_ms": ms[-1] if ms else None, "errors": self.errors}

    def close(self) -> None:
        self._stop = True; self._ev.set()
