"""Video reading with correct orientation (phone videos carry a rotation flag; OpenCV builds disagree on honouring it).
Non-ASCII paths are rejected with a clear message: OpenCV cannot open them on Windows."""
from __future__ import annotations
import cv2

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class VideoReader:
    def __init__(self, cap, meta_rotation: int, rotate: int | None):
        self.cap = cap
        auto_on = False
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            try: cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            except Exception: pass
            auto_on = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) > 0.5
        want = meta_rotation if rotate is None else rotate % 360
        already = meta_rotation if auto_on else 0
        self.rotation = (want - already) % 360
        self.rotation_total = want

    def read(self):
        ok, fr = self.cap.read()
        if ok and self.rotation: fr = cv2.rotate(fr, _ROT[self.rotation])
        return ok, fr

    def get(self, prop): return self.cap.get(prop)
    def release(self): self.cap.release()


def open_video(path, rotate: int | None = None) -> VideoReader:
    p = str(path)
    if not p.isascii():
        raise ValueError(f"Video path contains non-ASCII characters: {p!r}. Copy the video to a Latin-only path "
                         "(e.g. D:\\scenelive_videos\\street.mp4) and try again.")
    cap = cv2.VideoCapture(p)
    if not cap.isOpened(): raise FileNotFoundError(f"cannot open video: {p}")
    from .videometa import mp4_rotation
    return VideoReader(cap, mp4_rotation(p), rotate)
