"""Overlay for the simulator's output video (ASCII text only: OpenCV's fonts have no Cyrillic)."""
from __future__ import annotations
import cv2
import numpy as np
from .camera import clock, range_label

COLORS = {"provisional": (0, 165, 255), "confirmed": (60, 200, 60)}      # BGR


def draw(frame_bgr: np.ndarray, res, latency_s: float) -> np.ndarray:
    img = frame_bgr.copy()
    for tr in res.tracks:
        x0, y0, x1, y1 = (int(v) for v in tr["box"]); col = COLORS.get(tr["status"], (200, 200, 200))
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        hour, behind = clock(tr["az_rel"])                      # relative to where the camera pointed in THIS frame
        rng = range_label(tr["range"][0], tr["range"][1]) if tr["range"] else "?"
        line1 = f"#{tr['id']} {tr['name']} {tr['conf']:.2f}"; line2 = f"{'behind' if behind else str(hour) + ' oclock'} {rng}"
        y = max(14, y0 - 6)
        for k, txt in enumerate((line1, line2)):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            yy = y - (1 - k) * 16 if k == 0 else y
            cv2.rectangle(img, (x0, yy - th - 3), (x0 + tw + 4, yy + 3), col, -1)
            cv2.putText(img, txt, (x0 + 2, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    head = f"t={res.t:5.1f}s  yaw={res.yaw:6.1f}  shown {latency_s:4.2f}s after capture"
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, head, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
