"""Where do several bearing rays from different camera positions meet? (Needs camera TRANSLATION, e.g. ARKit.)

Least-squares point closest to all rays:  x = ( sum_i (I - d_i d_i^T) )^-1  sum_i (I - d_i d_i^T) o_i
Direction vectors use x = east, y = north, z = up; azimuth is clockwise from north."""
from __future__ import annotations
import math
import numpy as np


def direction(az_deg: float, el_deg: float = 0.0) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def triangulate(origins, dirs, min_baseline: float = 0.5, min_angle_deg: float = 4.0):
    """-> (point (3,), rms_residual_m) or None when the rays are (nearly) parallel or the camera barely moved."""
    O = np.asarray(origins, float).reshape(-1, 3); D = np.asarray(dirs, float).reshape(-1, 3)
    if len(O) < 2: return None
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    base = max(np.linalg.norm(O[i] - O[j]) for i in range(len(O)) for j in range(i + 1, len(O)))
    ang = max(math.degrees(math.acos(float(np.clip(D[i] @ D[j], -1, 1)))) for i in range(len(O)) for j in range(i + 1, len(O)))
    if base < min_baseline or ang < min_angle_deg: return None
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(O, D):
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ o
    if np.linalg.cond(A) > 1e8: return None
    x = np.linalg.solve(A, b)
    res = math.sqrt(float(np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o)) ** 2 for o, d in zip(O, D)])))
    return x, res
