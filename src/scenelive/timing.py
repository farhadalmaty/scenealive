"""Per-stage timing. Every stage of the pipeline runs inside `timer.stage(name)`; the report is built from these samples."""
from __future__ import annotations
import time
from collections import defaultdict
from contextlib import contextmanager
import numpy as np


class Timer:
    def __init__(self):
        self.samples: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str, sink: dict | None = None):
        """Time a block. `sink` (optional) additionally receives {name: seconds} for per-frame records."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.samples[name].append(dt)
            if sink is not None: sink[name] = sink.get(name, 0.0) + dt

    def add(self, name: str, seconds: float):
        self.samples[name].append(seconds)

    def table(self, order: list[str] | None = None) -> list[dict]:
        names = [n for n in (order or []) if n in self.samples] + [n for n in self.samples if n not in (order or [])]
        rows = []
        for n in names:
            a = np.asarray(self.samples[n]) * 1000.0
            rows.append({"stage": n, "n": len(a), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                         "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()), "total_s": float(a.sum() / 1000.0)})
        return rows
