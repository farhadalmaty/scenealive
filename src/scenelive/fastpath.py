"""The fast loop: one frame in, a list of named, located objects out. Every stage is timed.

preprocess -> compass -> detect -> track -> name -> locate -> list"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import cv2
import numpy as np
from .camera import Camera, circular_mean, wrap360
from .config import CameraConfig, FastConfig
from .labels import MOVABLE
from .objects import Entry
from .ranging import estimate_range
from .timing import Timer
from .track import Obs, Tracker

STAGES = ["preprocess", "compass", "detect", "track", "name", "locate", "ground", "list"]


@dataclass
class FrameResult:
    t: float
    frame: int
    yaw: float
    size: tuple                              # (w, h) of the frame the detector saw
    detections: list
    tracks: list                             # [{id, box, name, conf, status, az_rel, world_az, range}] for this frame
    entries: list                            # list of Entry (fast, one per live track)
    timings: dict = field(default_factory=dict)    # stage -> seconds, for THIS frame
    ground: dict | None = None               # surface under the feet per zone (left/center/right), when a ground classifier is attached


class FastPath:
    def __init__(self, detector, cam_cfg: CameraConfig, cfg: FastConfig, compass, timer: Timer | None = None, ground=None):
        self.det, self.cam_cfg, self.cfg, self.compass, self.ground = detector, cam_cfg, cfg, compass, ground
        self.timer = timer or Timer(); self.tracker = Tracker(cfg); self._yaw_prev = 0.0; self._cams: dict = {}

    def camera(self, w: int, h: int) -> Camera:
        if (w, h) not in self._cams: self._cams[(w, h)] = Camera(w, h, self.cam_cfg.fov_long_deg)
        return self._cams[(w, h)]

    def process(self, frame_bgr: np.ndarray, t: float, idx: int) -> FrameResult:
        tim: dict[str, float] = {}; st = self.timer.stage
        with st("preprocess", tim):
            h0, w0 = frame_bgr.shape[:2]; s = self.cfg.imgsz / max(h0, w0)
            frame = cv2.resize(frame_bgr, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else frame_bgr
            cam = self.camera(frame.shape[1], frame.shape[0])
        with st("compass", tim):
            yaw = self.compass.update(frame, t, cam); dyaw = yaw - self._yaw_prev; self._yaw_prev = yaw
        with st("detect", tim):
            dets = self.det.detect(frame)
        with st("track", tim):
            dx = -cam.f * math.tan(math.radians(dyaw))                    # camera turned right -> image content moves left
            matches = self.tracker.update(dets, t, idx, dx)
        strong = [(tr, d) for tr, d in matches if d.conf >= self.cfg.start_conf]      # weak matches only keep a track alive
        with st("name", tim):
            for tr, d in strong: tr.vote(d)
        with st("locate", tim):
            snap = []
            for tr, d in strong:
                az = cam.azimuth_deg(d.cx); wa = wrap360(yaw + az)
                edge = d.box[0] <= 2.0 or d.box[2] >= cam.width - 2.0
                tr.obs.append(Obs(t, idx, az, wa, cam.elevation_deg(d.cy), d.box, d.name, d.conf, estimate_range(d.name, d.box, cam, az), None, edge))
                snap.append({"id": tr.id, "box": d.box, "name": tr.name, "conf": tr.conf, "det_conf": d.conf, "hits": tr.hits,
                             "status": tr.status(self.cfg), "az_rel": az, "world_az": wa, "range": tr.obs[-1].range})
        if self.ground is not None and self.ground.due(t):                 # the surface changes slowly: look about once a second, not on every frame
            with st("ground", tim): self.ground.update(frame, t)
        with st("list", tim):
            cfg = self.cfg
            entries = [self._entry(tr) for tr in self.tracker.active if tr.obs and tr.misses <= cfg.publish_misses and tr.conf >= cfg.min_conf_publish and self._sure_enough(tr)]
        return FrameResult(t, idx, yaw, (frame.shape[1], frame.shape[0]), dets, snap, entries, tim, self.ground.state() if self.ground is not None else None)

    def _sure_enough(self, tr) -> bool:
        """One weak detection is flicker, not an object. Classes that fired falsely need more sightings and no single-shot shortcut."""
        need = self.cfg.publish_hits_by_class.get(tr.name)
        if need is not None: return tr.hits >= need
        return tr.hits >= self.cfg.publish_hits or tr.conf >= self.cfg.publish_single_conf

    def _entry(self, tr) -> Entry:
        recent = tr.obs[-5:]; last = ([o for o in recent if not o.edge] or recent)[-3:]; az = circular_mean([o.world_az for o in last])
        rs = [o.range for o in tr.obs[-5:] if o.range is not None]
        rng = sig = None
        if rs: rng = float(np.median([r[0] for r in rs])); sig = float(np.median([r[1] for r in rs]))
        return Entry(f"T{tr.id}", tr.name, tr.conf, tr.status(self.cfg), az, rng, sig, tr.first_t, tr.last_t, "fast", tr.hits,
                     (tr.id,), tr.name in MOVABLE)
