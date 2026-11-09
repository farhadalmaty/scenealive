import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scenelive.track import Track, Obs
from scenelive.detect.base import Detection


def det(name, box, conf=0.9): return Detection(tuple(float(x) for x in box), name, conf, name)


def make_track(tid, name, t0, t1, az, n=5, conf=0.9, rng=None, cam_pos=None, az_end=None):
    """A finished track with n observations between t0 and t1, azimuth az (moving linearly to az_end)."""
    tr = Track(tid, det(name, (0, 0, 20, 40), conf), t0, 0)
    for i in range(n):
        f = i / max(n - 1, 1); a = az if az_end is None else az + (az_end - az) * f
        tr.obs.append(Obs(t0 + (t1 - t0) * f, i, 0.0, a % 360, 0.0, (0, 0, 20, 40), name, conf, rng, cam_pos)); tr.vote(det(name, (0, 0, 20, 40), conf))
    tr.first_t, tr.last_t = t0, t1
    return tr
