"""One live session = one phone. Frame in -> result out. Independent of the network layer, so it is tested without sockets."""
from __future__ import annotations
import threading
import time
import cv2
import numpy as np
from ..camera import FOV_PRESETS, angdiff
from ..compass import VisualCompass
from ..config import CameraConfig, FastConfig, MapConfig
from ..fastpath import FastPath
from ..mapper import build_map
from ..objects import ObjectList
from ..timing import Timer


class ClientCompass:
    """Heading reported by the phone (degrees clockwise, absolute or arbitrary start). Accumulated from differences,
    so wrapping at 360 never matters, and the session starts at 0."""
    name = "client"
    def __init__(self): self.pending = None; self.last = None; self.yaw = 0.0
    def set(self, yaw_abs): self.pending = yaw_abs
    def update(self, frame, t, cam) -> float:
        if self.pending is not None:
            a = float(self.pending); self.pending = None
            if self.last is not None: self.yaw += angdiff(a, self.last)
            self.last = a
        return self.yaw
    def stats(self) -> dict: return {}


def camera_from_hello(hello: dict) -> CameraConfig:
    fov = FOV_PRESETS.get(str(hello.get("camera", "main")), 69.0)
    v = hello.get("fov_long_deg")
    if isinstance(v, (int, float)) and 20.0 <= float(v) <= 140.0: fov = float(v)
    return CameraConfig(fov)


class LiveSession:
    def __init__(self, detector, hello: dict, cfg, fast_cfg: FastConfig | None = None, map_cfg: MapConfig | None = None, ground_classifier=None):
        self.ground_classifier = ground_classifier; self.cfg = cfg; self.hello = hello; self.cam_cfg = camera_from_hello(hello)
        self.fast_cfg = fast_cfg or FastConfig(imgsz=cfg.imgsz, conf=cfg.conf); self.map_cfg = map_cfg or MapConfig(interval_s=cfg.map_interval_s)
        self.detector = detector; self.lock = threading.RLock(); self.timer = Timer(); self.created = time.time()
        self.has_heading = bool(hello.get("has_heading")); self.reset()

    def reset(self) -> None:
        with self.lock:
            self.compass = ClientCompass() if self.has_heading else VisualCompass()
            from ..ground import GroundMonitor
            gm = GroundMonitor(self.ground_classifier, getattr(self.cfg, 'ground_every_s', 1.0)) if self.ground_classifier is not None else None
            self.fast = FastPath(self.detector, self.cam_cfg, self.fast_cfg, self.compass, self.timer, gm)
            from ..alerts import AlertEngine
            self.alerts = AlertEngine(getattr(self.cfg, 'alert_cooldown_s', 15.0), getattr(self.cfg, 'alert_range_m', 6.0)) if getattr(self.cfg, 'alerts', True) else None
            self.ground_state = None
            self.olist = ObjectList(); self.t = 0.0; self.heading = 0.0; self.frames = 0; self.next_map = self.map_cfg.interval_s

    # ------------------------------------------------------------------ frames
    def process(self, jpeg: bytes, meta: dict) -> dict:
        tim: dict[str, float] = {}
        with self.timer.stage("server_decode", tim):
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("the frame is not a valid JPEG")
        with self.lock:
            seq = int(meta.get("seq", self.frames)); t = float(meta.get("t", self.t + 1.0 / 3.0))
            if isinstance(self.compass, ClientCompass): self.compass.set(meta.get("yaw"))
            res = self.fast.process(frame, t, seq); tim.update(res.timings)
            if res.ground is not None: self.ground_state = res.ground
            self.olist.apply_fast(res.entries, t); self.heading = res.yaw; self.t = max(self.t, t); self.frames += 1
            while t >= self.next_map:
                with self.timer.stage("map_build", tim):
                    self.olist.apply_map(build_map(self.fast.tracker.all_tracks(), t, self.map_cfg, self.fast_cfg), t)
                self._forget(t); self.next_map += self.map_cfg.interval_s
            view = self.view(); alerts = self.alerts.update(view, res.ground, t) if self.alerts is not None else []
        w, h = res.size
        boxes = [{"id": tr["id"], "name": tr["name"], "status": tr["status"], "conf": round(tr["conf"], 2), "x0": round(tr["box"][0] / w, 4),
                  "y0": round(tr["box"][1] / h, 4), "x1": round(tr["box"][2] / w, 4), "y1": round(tr["box"][3] / h, 4)} for tr in res.tracks]
        return {"type": "result", "seq": seq, "t": t, "heading": round(self.heading, 1), "entries": view, "boxes": boxes,
                "timings_ms": {k: round(v * 1000.0, 2) for k, v in tim.items()}, "compute_ms": round(sum(tim.values()) * 1000.0, 1),
                "n_det": len(res.detections), "size": [w, h], "ground": res.ground, "alerts": alerts}

    def view(self) -> list[dict]:
        with self.lock:
            out = self.olist.view(self.heading)
            for v in out: v["age_s"] = round(max(0.0, self.t - v["last_t"]), 1)
            return out

    def _forget(self, t: float) -> None:
        """Keep the map fast in hours-long sessions: drop tracks older than memory_minutes and cap their number."""
        tr = self.fast.tracker; cut = t - self.cfg.memory_minutes * 60.0
        arch = [x for x in tr.archive if x.last_t >= cut]
        tr.archive = arch[-self.cfg.max_tracks:]

    def stats(self) -> dict:
        from ..sim import PHONE_STAGES, SERVER_STAGES, MAP_STAGES
        return {"frames": self.frames, "seconds": round(self.t, 1), "heading": round(self.heading, 1), "entries": len(self.olist.entries()),
                "camera_fov_long_deg": self.cam_cfg.fov_long_deg, "compass": self.compass.name, "stages": self.timer.table(SERVER_STAGES + MAP_STAGES)}
