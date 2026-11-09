"""Where is the camera pointing? A phone has a gyroscope/compass; in the replay simulator we have only the video.

yaw is in degrees, positive = the camera turned right, cumulative from the first frame.
  ZeroCompass    no heading (directions are camera-relative)
  CsvCompass     heading log exported from a phone: lines "t_seconds,yaw_degrees"
  VisualCompass  yaw from the horizontal optical flow of the background (a rotation-only 'gyro' made from the video itself;
                 drifts slowly, assumes the camera mostly pans/turns)"""
from __future__ import annotations
import math
import cv2
import numpy as np


class ZeroCompass:
    name = "zero"
    def stats(self) -> dict: return {}
    def update(self, frame_bgr, t, cam) -> float: return 0.0


class CsvCompass:
    name = "csv"
    def stats(self) -> dict: return {}
    def __init__(self, path):
        rows = []
        for line in open(path, encoding="utf-8-sig"):
            p = line.strip().replace(";", ",").split(",")
            try: rows.append((float(p[0]), float(p[1])))
            except (ValueError, IndexError): continue                 # header or junk
        if len(rows) < 2: raise ValueError(f"{path}: need at least two 't,yaw' rows")
        rows.sort(); self.t = np.array([r[0] for r in rows]); self.y = np.unwrap(np.radians([r[1] for r in rows]))
        self.y = np.degrees(self.y); self.y -= self.y[0]               # relative to the first sample

    def update(self, frame_bgr, t, cam) -> float: return float(np.interp(t, self.t, self.y))


class VisualCompass:
    name = "visual"
    def __init__(self, max_width: int = 320, max_points: int = 150):
        self.max_width, self.max_points = max_width, max_points
        self.prev = None; self.yaw = 0.0; self.lost = 0; self.steps = 0; self.yaw_min = self.yaw_max = 0.0

    def stats(self) -> dict:
        """How trustworthy was the heading? `lost` = steps where too few background features could be tracked."""
        return {"steps": self.steps, "lost": self.lost, "yaw_min": self.yaw_min, "yaw_max": self.yaw_max}

    def update(self, frame_bgr, t, cam) -> float:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        s = min(1.0, self.max_width / g.shape[1])
        if s < 1.0: g = cv2.resize(g, (int(round(g.shape[1] * s)), int(round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
        if self.prev is None or self.prev.shape != g.shape:
            self.prev = g; return self.yaw
        pts = cv2.goodFeaturesToTrack(self.prev, self.max_points, 0.01, 8)
        dx = None
        if pts is not None and len(pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() >= 12:
                fx_small = cam.f * (g.shape[1] / cam.width); cx_small = g.shape[1] / 2.0
                u0 = pts[ok, 0, 0].astype(float); u1 = nxt[ok, 0, 0].astype(float)
                # Each feature's own azimuth before and after: on a flat image a pure turn shifts the edges more than the centre
                # (factor 1/cos^2), so a plain median pixel shift would overestimate the turn. Angles do not have this bias.
                d = np.arctan((u1 - cx_small) / fx_small) - np.arctan((u0 - cx_small) / fx_small)
                med = float(np.median(d)); mad = float(np.median(np.abs(d - med))) * 1.4826
                keep = np.abs(d - med) <= max(math.radians(0.15), 2.5 * mad)
                if keep.sum() >= 8: dx = float(np.median(d[keep]))          # radians (name kept: "scene shift")
        if dx is None: self.lost += 1
        else:
            self.yaw += -math.degrees(dx)                                # scene moves left <=> camera turned right
        self.prev = g; self.steps += 1; self.yaw_min = min(self.yaw_min, self.yaw); self.yaw_max = max(self.yaw_max, self.yaw)
        return self.yaw


def make_compass(spec: str):
    """'zero' | 'visual' | path to a CSV file."""
    if spec == "zero": return ZeroCompass()
    if spec == "visual": return VisualCompass()
    return CsvCompass(spec)
