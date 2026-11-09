from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from ..labels import canonical


@dataclass
class Detection:
    box: tuple[float, float, float, float]    # x0, y0, x1, y1 in pixels of the frame given to the detector
    name: str                                 # canonical name
    conf: float
    raw_name: str = ""

    @property
    def w(self) -> float: return self.box[2] - self.box[0]
    @property
    def h(self) -> float: return self.box[3] - self.box[1]
    @property
    def cx(self) -> float: return (self.box[0] + self.box[2]) / 2.0
    @property
    def cy(self) -> float: return (self.box[1] + self.box[3]) / 2.0


class Detector(Protocol):
    """detect() takes a BGR uint8 frame (OpenCV convention) and returns boxes with names."""
    name: str
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def parse_ultralytics(result, allowed=None) -> list[Detection]:
    """ultralytics Results -> Detection list. `allowed` (raw names) filters out classes that make no sense here."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0: return []
    xyxy = boxes.xyxy.cpu().numpy(); cls = boxes.cls.cpu().numpy().astype(int); conf = boxes.conf.cpu().numpy()
    names = result.names; out = []
    for b, c, cf in zip(xyxy, cls, conf):
        raw = names[int(c)]
        if allowed is not None and raw not in allowed: continue
        out.append(Detection(tuple(float(x) for x in b), canonical(raw), float(cf), raw))
    return out
