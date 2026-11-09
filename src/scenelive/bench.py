"""How fast is the detector on THIS machine, at different input sizes? (The slowest stage decides everything.)"""
from __future__ import annotations
import time
import cv2
import numpy as np
from .video_io import open_video


def bench_detector(video, make_detector, imgszs, n_frames: int = 20, warmup: int = 2, progress=None):
    """make_detector(imgsz) -> detector. Returns rows: imgsz, mean/p50/p95 ms, fps, mean detections."""
    rd = open_video(video); total = int(rd.get(cv2.CAP_PROP_FRAME_COUNT) or 0); frames = []
    step = max(1, total // max(n_frames, 1)) if total else 5; i = 0
    while len(frames) < n_frames:
        ok, fr = rd.read()
        if not ok: break
        if i % step == 0: frames.append(fr)
        i += 1
    rd.release()
    if not frames: raise ValueError("no frames read")
    rows = []
    for sz in imgszs:
        det = make_detector(sz); small = []
        for fr in frames:
            s = sz / max(fr.shape[:2]); small.append(cv2.resize(fr, (int(round(fr.shape[1] * s)), int(round(fr.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1 else fr)
        for fr in small[:warmup]: det.detect(fr)
        ts, nd = [], []
        for fr in small:
            t0 = time.perf_counter(); d = det.detect(fr); ts.append((time.perf_counter() - t0) * 1000.0); nd.append(len(d))
        a = np.array(ts); rows.append({"imgsz": sz, "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                                       "fps": float(1000.0 / a.mean()), "detections": float(np.mean(nd))})
        if progress: progress(rows[-1])
    return rows
