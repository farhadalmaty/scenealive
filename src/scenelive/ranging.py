"""Distance without a depth sensor: from the apparent size of an object of known typical size.

For a box of height h px (or width w px) and a typical size H metres:  depth Z = f * H / h,  range d = Z / cos(azimuth).
Boxes cut by the frame edge are skipped (their size is not the object's size)."""
from __future__ import annotations
import math
from .camera import Camera
from .labels import RANGE_PRIORS


def estimate_range(name: str, box, cam: Camera, az_rel_deg: float = 0.0, edge_px: float = 3.0, min_px: float = 8.0):
    """-> (range_m, sigma_log) or None when the class has no size prior or the box is unusable."""
    pri = RANGE_PRIORS.get(name)
    if not pri: return None
    side, size, sig = pri
    x0, y0, x1, y1 = box
    if side == "h":
        if y0 <= edge_px or y1 >= cam.height - edge_px: return None
        px = y1 - y0
    else:
        if x0 <= edge_px or x1 >= cam.width - edge_px: return None
        px = x1 - x0
    if px < min_px: return None
    z = cam.f * size / px
    d = z / max(math.cos(math.radians(az_rel_deg)), 0.2)
    return d, math.hypot(sig, 1.0 / px)
