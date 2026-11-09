"""2D multi-object tracker (IoU + class weight) with camera-rotation compensation, and per-track name voting.

Compensation matters: a thin pole is 10 px wide; a 1.3 degree pan between two frames moves it by ~20 px and plain IoU loses it.
Knowing the yaw change (compass) we shift the predicted boxes first."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .config import FastConfig


@dataclass
class Obs:
    t: float
    frame: int
    az_rel: float                 # degrees right of the optical axis
    world_az: float               # degrees, heading + az_rel, 0..360
    el: float                     # degrees above the optical axis
    box: tuple
    name: str
    conf: float
    range: tuple | None = None    # (metres, sigma_log) from the size prior
    cam_pos: tuple | None = None  # camera position in metres, when a pose source with translation exists (ARKit)
    edge: bool = False            # the box touches the left/right frame edge: cut by the frame, its centre is biased inwards


class Track:
    def __init__(self, tid: int, det, t: float, frame: int):
        self.id = tid; self.box = np.asarray(det.box, float); self.pred = self.box.copy()
        self.votes: dict[str, float] = {}; self.counts: dict[str, int] = {}
        self.hits = 0; self.misses = 0; self.first_t = self.last_t = t; self.first_frame = frame
        self.obs: list[Obs] = []

    def vote(self, det) -> None:
        self.votes[det.name] = self.votes.get(det.name, 0.0) + det.conf
        self.counts[det.name] = self.counts.get(det.name, 0) + 1
        self.hits += 1

    @property
    def name(self) -> str:
        return max(self.votes, key=self.votes.get) if self.votes else "unknown"

    @property
    def conf(self) -> float:
        """Mean confidence of the winning name times its share of all votes: unsure when the name flickers."""
        if not self.votes: return 0.0
        n = self.name; return (self.votes[n] / self.counts[n]) * (self.votes[n] / sum(self.votes.values()))

    def status(self, cfg: FastConfig) -> str:
        return "confirmed" if self.hits >= cfg.confirm_hits and self.conf >= cfg.confirm_conf else "provisional"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


class Tracker:
    def __init__(self, cfg: FastConfig):
        self.cfg = cfg; self.active: list[Track] = []; self.archive: list[Track] = []; self._next = 1

    def all_tracks(self) -> list[Track]:
        return self.archive + self.active

    def update(self, dets, t: float, frame: int, dx_px: float = 0.0):
        """dx_px: how far the image content moved horizontally because the camera turned (negative = camera turned right).
        Returns [(track, detection)] for every detection of this frame (matched or newly started). Voting for the name is the
        caller's job (a separate, separately timed stage)."""
        c = self.cfg
        dets = [d for d in dets if d.w >= c.min_box_px and d.h >= c.min_box_px]
        for tr in self.active: tr.pred = tr.box + np.array([dx_px, 0.0, dx_px, 0.0])
        matches: list[tuple[Track, object]] = []
        used_t: set[int] = set(); used_d: set[int] = set()
        if self.active and dets:
            M = iou_matrix(np.array([tr.pred for tr in self.active]), np.array([d.box for d in dets]))
            for i, tr in enumerate(self.active):
                for j, d in enumerate(dets):
                    if tr.name != d.name: M[i, j] *= 0.5
            for flat in np.argsort(-M, axis=None):
                i, j = divmod(int(flat), len(dets))
                if M[i, j] < c.iou_match: break
                if i in used_t or j in used_d: continue
                used_t.add(i); used_d.add(j); matches.append((self.active[i], dets[j]))
        for tr, d in matches:
            tr.box = np.asarray(d.box, float); tr.misses = 0; tr.last_t = t
        for i, tr in enumerate(self.active):
            if i not in used_t: tr.misses += 1
        for j, d in enumerate(dets):
            if j in used_d or d.conf < c.start_conf: continue                      # a weak detection may continue a track (above) but never starts one
            tr = Track(self._next, d, t, frame); self._next += 1; self.active.append(tr); matches.append((tr, d))
        keep = []
        for tr in self.active:
            (self.archive if tr.misses > c.max_age_frames else keep).append(tr)
        self.active = keep
        return matches
