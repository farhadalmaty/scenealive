from .base import Detection, Detector, parse_ultralytics


def make_detector(kind: str, device: str | None = None, imgsz: int = 640, conf: float = 0.30, weights: str | None = None, vocab=None):
    """kind: yolo (COCO, boxes) | world (YOLO-World, open vocabulary, boxes) | blobs (demo, no model)."""
    if kind == "blobs":
        from .blob import ColorBlobDetector
        return ColorBlobDetector()
    if kind == "yolo":
        from .yolo import YoloDetector
        return YoloDetector(weights or "yolov8s.pt", imgsz, conf, device=device)
    if kind == "world":
        from .yolo import YoloWorldDetector
        return YoloWorldDetector(weights or "yolov8s-worldv2.pt", vocab, imgsz, conf, device=device)
    raise ValueError(f"unknown detector {kind!r}: use yolo, world or blobs")
