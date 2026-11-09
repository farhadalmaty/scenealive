"""A synthetic street walk with KNOWN truth, so every number the system prints can be checked.

The camera stands still and turns right at yaw_rate degrees per second over a textured panorama (so optical flow has something to
track). Coloured boxes stand in for a person, a car (moving), a pole, a door and a wall; the detector for this video is
detect.blob.ColorBlobDetector. Geometry is an exact pinhole camera, so a box's size and position follow from bearing and distance."""
from __future__ import annotations
import math
import os
from dataclasses import dataclass
import cv2
import numpy as np
from .camera import Camera, angdiff


@dataclass
class SynthObject:
    name: str
    bearing: float            # world azimuth at t=0, degrees clockwise from the initial heading
    dist: float               # metres (range from the camera)
    h: float
    w: float
    color: tuple              # BGR
    drift_deg_s: float = 0.0  # bearing change per second (a moving object)

    def bearing_at(self, t: float) -> float:
        return self.bearing + self.drift_deg_s * t


def default_world() -> list[SynthObject]:
    return [SynthObject("person", 15, 6, 1.7, 0.5, (30, 30, 220)),
            SynthObject("car", -20, 12, 1.5, 1.8, (220, 60, 30), drift_deg_s=2.0),
            SynthObject("pole", 40, 8, 4.0, 0.3, (30, 220, 220)),
            SynthObject("door", 62, 5, 2.0, 0.9, (40, 200, 40)),
            SynthObject("wall", 88, 9, 3.0, 6.0, (200, 40, 200))]


def _panorama(h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); pw = 3600
    coarse = rng.integers(70, 190, (h // 10 + 2, pw // 10)).astype(np.float32); coarse[:, -1] = coarse[:, 0]
    tex = cv2.resize(coarse, (pw, h), interpolation=cv2.INTER_CUBIC) + rng.normal(0, 7, (h, pw)).astype(np.float32)
    return cv2.cvtColor(np.clip(tex, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def render_frame(t, objects, cam: Camera, yaw_rate: float, pano, cam_height: float = 1.5, map_cache: dict | None = None):
    h, w = cam.height, cam.width; yaw = yaw_rate * t
    cols = np.array([(yaw + math.degrees(math.atan((u - cam.cx) / cam.f))) % 360.0 for u in range(w)], np.float32) / 360.0 * pano.shape[1]
    mx = np.tile(cols, (h, 1)); my = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    img = cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    for o in sorted(objects, key=lambda o: -o.dist):
        rel = angdiff(o.bearing_at(t), yaw)
        if abs(rel) > 75: continue
        z = o.dist * math.cos(math.radians(rel)); x = o.dist * math.sin(math.radians(rel))
        u0 = cam.cx + cam.f * (x - o.w / 2) / z; u1 = cam.cx + cam.f * (x + o.w / 2) / z
        v0 = cam.cy - cam.f * (-cam_height + o.h) / z; v1 = cam.cy - cam.f * (-cam_height) / z
        cv2.rectangle(img, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))), o.color, -1)
    return img, yaw


def render_video(path: str, objects=None, duration: float = 24.0, fps: float = 15.0, size=(640, 360), fov: float = 69.0,
                 yaw_rate: float = 4.0, seed: int = 0) -> dict:
    objects = objects or default_world(); cam = Camera(size[0], size[1], fov); pano = _panorama(size[1], seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not vw.isOpened(): raise RuntimeError(f"cannot write {path}")
    n = int(duration * fps)
    for i in range(n):
        img, _ = render_frame(i / fps, objects, cam, yaw_rate, pano); vw.write(img)
    vw.release()
    return {"objects": objects, "yaw_rate": yaw_rate, "fov": fov, "size": size, "fps": fps, "duration": duration}


def truth_table(truth: dict) -> list[dict]:
    from .labels import RANGE_PRIORS
    return [{"name": o.name, "bearing_deg": o.bearing, "dist_m": o.dist, "moving": o.drift_deg_s != 0, "has_size_prior": o.name in RANGE_PRIORS}
            for o in truth["objects"]]
