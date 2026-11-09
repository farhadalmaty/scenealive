"""Pinhole camera from a field of view; bearings, clock face and distance steps."""
from __future__ import annotations
import math
from dataclasses import dataclass

FOV_PRESETS = {"main": 69.0, "ultrawide": 100.0, "tele": 38.0}     # approx., across the long side of a 16:9 phone video
LADDER = (1.0, 2.0, 5.0, 10.0, 15.0)


def wrap360(a: float) -> float:
    r = a % 360.0
    return 0.0 if r >= 360.0 - 1e-9 else r        # -1e-15 % 360 is 359.999...: that is 0 degrees, not 360


def angdiff(a: float, b: float) -> float:
    """Signed difference a - b in (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def circular_mean(angles, weights=None) -> float:
    import numpy as np
    a = np.radians(np.asarray(angles, float)); w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    return wrap360(math.degrees(math.atan2(float((w * np.sin(a)).sum()), float((w * np.cos(a)).sum()))))


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fov_long_deg: float = 69.0

    @property
    def f(self) -> float:
        """Focal length in pixels (square pixels): f = (L/2) / tan(theta/2)."""
        return (max(self.width, self.height) / 2.0) / math.tan(math.radians(self.fov_long_deg) / 2.0)

    @property
    def cx(self) -> float: return self.width / 2.0
    @property
    def cy(self) -> float: return self.height / 2.0

    def azimuth_deg(self, u: float) -> float:
        """Angle to the right of the optical axis for pixel column u (positive = right)."""
        return math.degrees(math.atan2(u - self.cx, self.f))

    def elevation_deg(self, v: float) -> float:
        """Angle above the optical axis for pixel row v (positive = up)."""
        return math.degrees(math.atan2(-(v - self.cy), self.f))

    def project(self, x: float, y: float, z: float):
        """Camera-frame point (x right, y up, z forward) -> pixel (u, v) or None behind the camera."""
        if z <= 1e-6: return None
        return self.cx + self.f * x / z, self.cy - self.f * y / z


def clock(rel_az_deg: float) -> tuple[int, bool]:
    """(hour on a clock face, is_behind). 12 = straight ahead, 3 = right, 6 = behind, 9 = left."""
    h = int(round(rel_az_deg / 30.0)) % 12 or 12
    return h, abs(rel_az_deg) > 90.0


def ladder_bucket(d_m: float):
    """Nearest rung of 1, 2, 5, 10, 15 m on a log scale; None beyond 20 m."""
    if not d_m or d_m <= 0 or d_m != d_m: return LADDER[0]
    if d_m > 20.0: return None
    return min(LADDER, key=lambda r: abs(math.log(d_m / r)))


def range_label(d_m: float | None, sigma_log: float | None = None) -> str:
    """'~2 m', or '2-5 m' when the uncertainty (1 sigma, log units) spans several rungs, or '?' when unknown."""
    if d_m is None: return "?"
    one = lambda r: "15+ m" if r is None else f"~{r:g} m"
    if not sigma_log: return one(ladder_bucket(d_m))
    a, b = ladder_bucket(d_m * math.exp(-sigma_log)), ladder_bucket(d_m * math.exp(sigma_log))
    if a == b: return one(ladder_bucket(d_m))
    return f"{a:g}+ m" if b is None else f"{a:g}-{b:g} m"
