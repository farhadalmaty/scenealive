"""A model-free detector for demos and tests: finds saturated coloured rectangles by hue and names them by colour."""
from __future__ import annotations
import cv2
import numpy as np
from .base import Detection

DEFAULT_HUES = {          # OpenCV hue is 0..179
    "person": [(0, 8), (172, 179)],   # red
    "car": [(100, 125)],              # blue
    "pole": [(20, 35)],               # yellow
    "door": [(45, 75)],               # green
    "wall": [(140, 165)],             # magenta
}


class ColorBlobDetector:
    def __init__(self, hues=None, min_sat: int = 110, min_val: int = 60, min_area: int = 30):
        self.hues = hues or DEFAULT_HUES; self.min_sat, self.min_val, self.min_area = min_sat, min_val, min_area
        self.name = "color-blobs"
        self._k = np.ones((3, 3), np.uint8)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV); out = []
        for name, ranges in self.hues.items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(hsv, (lo, self.min_sat, self.min_val), (hi, 255, 255))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k)
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if a >= self.min_area: out.append(Detection((float(x), float(y), float(x + w), float(y + h)), name, 0.9, name))
        return out
