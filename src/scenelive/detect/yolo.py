"""Real detectors via ultralytics (pip install ultralytics). Not exercised in the dev sandbox (no network, no torch):
verify on your machine with `scenelive bench`. License note: ultralytics is AGPL-3.0."""
from __future__ import annotations
import numpy as np
from .base import Detection, parse_ultralytics
from ..labels import COCO_STREET, STREET_VOCAB


def _kw(d) -> dict:
    """predict() arguments. `half` is passed only when switched on: newer ultralytics prints a deprecation warning for the
    mere presence of the argument ("'half' is deprecated ... use 'quantize'") on every single call."""
    kw = dict(imgsz=d.imgsz, conf=d.conf, iou=d.iou, device=d.device, verbose=False)
    if d.half: kw["half"] = True
    return kw


class YoloDetector:
    """Closed-set COCO detector (boxes only: fastest, most accurate for people and vehicles)."""
    def __init__(self, weights: str = "yolov8s.pt", imgsz: int = 640, conf: float = 0.30, iou: float = 0.6,
                 device: str | None = None, classes=COCO_STREET, half: bool = False):
        from ultralytics import YOLO
        self.m, self.imgsz, self.conf, self.iou, self.half = YOLO(weights), imgsz, conf, iou, half
        from ..device import configure, pick_device, ultralytics_device
        dev = pick_device(device); configure(dev); self.device = ultralytics_device(dev); self.torch_device = dev; self.allowed = set(classes) if classes else None
        self.name = f"yolo:{weights}@{imgsz}"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        r = self.m.predict(np.ascontiguousarray(frame_bgr), **_kw(self))[0]
        return parse_ultralytics(r, self.allowed)


class YoloWorldDetector:
    """Open-vocabulary detector (YOLO-World): any words as classes, so poles, walls, doors, fences and obstacles get names
    in the same pass. Boxes only. Needs a recent ultralytics (set_classes API); check your version."""
    def __init__(self, weights: str = "yolov8s-worldv2.pt", vocab=None, imgsz: int = 640, conf: float = 0.25, iou: float = 0.6,
                 device: str | None = None, half: bool = False):
        from ultralytics import YOLOWorld
        self.m = YOLOWorld(weights); self.vocab = list(vocab or STREET_VOCAB); self.m.set_classes(self.vocab)
        from ..device import configure, pick_device, ultralytics_device
        dev = pick_device(device); configure(dev)
        self.imgsz, self.conf, self.iou, self.half, self.device, self.torch_device = imgsz, conf, iou, half, ultralytics_device(dev), dev
        self.name = f"yolo-world:{weights}@{imgsz}"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        r = self.m.predict(np.ascontiguousarray(frame_bgr), **_kw(self))[0]
        return parse_ultralytics(r, None)
