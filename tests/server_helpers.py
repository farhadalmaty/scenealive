import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cv2, numpy as np
from scenelive.camera import Camera
from scenelive.detect.blob import ColorBlobDetector
from scenelive.server.config import ServerConfig
from scenelive.server.session import LiveSession
from scenelive.synth import _panorama, default_world, render_frame

PANO = _panorama(360, 0); CAM = Camera(640, 360, 69.0); WORLD = default_world()


def jpeg_at(t, yaw_rate=4.0, quality=80):
    img, yaw = render_frame(t, WORLD, CAM, yaw_rate, PANO)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality]); return buf.tobytes(), yaw


def make_session(has_heading=True, t_end=9.0, fps=3.0, base=350.0, camera="main", cfg=None, **hello):
    cfg = cfg or ServerConfig(detector="blobs")
    s = LiveSession(ColorBlobDetector(), {"camera": camera, "has_heading": has_heading, **hello}, cfg); res = None
    for k in range(int(t_end * fps) + 1):
        t = k / fps; jpg, yaw = jpeg_at(t)
        res = s.process(jpg, {"seq": k, "t": t, "yaw": (base + yaw) % 360.0 if has_heading else None})
    return s, res
