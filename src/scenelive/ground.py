"""What is under the feet? Road, surface type, bad road.

A road is not an object with a box, so it is looked at as a REGION: the lower part of the frame is cut into three zones (left,
centre, right) and each zone is classified into a surface type once a second. The result is smoothed over the last few looks
(one odd frame must not announce a pothole) and each surface has a quality: good / fair / bad.

The classifier is pluggable:  clip (zero-shot CLIP on the zone crops; not run in the dev sandbox, measure it with `replay --ground clip`)
                              colors (crude colour rules: demo and tests only)"""
from __future__ import annotations
from collections import deque
import numpy as np
from .catalog import QUALITY_RU, SURFACES, SURFACE_QUALITY, SURFACE_RU

ZONES = (("left", 0.0, 0.34), ("center", 0.33, 0.67), ("right", 0.66, 1.0))
ZONE_RU = {"left": "слева", "center": "впереди", "right": "справа"}


class ColorSurfaceClassifier:
    """Crude colour rules (demo and tests only; real use is the CLIP classifier)."""
    name = "colors"
    def classify(self, crops):
        out = []
        for c in crops:
            b, g, r = (float(x) for x in c.reshape(-1, 3).mean(0)); lo = min(r, g, b)
            if g > r + 15 and g > b + 15: lab = "grass"
            elif r > g + 15 and g > b + 10: lab = "dirt"
            elif lo > 200: lab = "snow"
            elif abs(r - g) < 15 and abs(g - b) < 15: lab = "asphalt" if (r + g + b) / 3 < 140 else "concrete pavement"
            else: lab = "gravel"
            out.append([(lab, 0.9), ("gravel" if lab != "gravel" else "asphalt", 0.06), ("dirt" if lab != "dirt" else "grass", 0.04)])
        return out


class ClipSurfaceClassifier:
    """Zero-shot CLIP on the zone crops. Uses the `clip` package that ultralytics installs for YOLO-World (or open_clip if present)."""
    name = "clip"
    def __init__(self, device: str | None = None, model: str = "ViT-B/32", topk: int = 3):
        import torch
        from .device import pick_device
        self.torch, self.topk = torch, topk; self.device = pick_device(device)
        try:
            import clip
            self.model, self.preprocess = clip.load(model, device=self.device); self.tokenize = clip.tokenize
        except ImportError as ex:
            raise ImportError("the CLIP package is not installed (it comes with YOLO-World: run windows\\serve.bat once with --detector world, "
                              "or: pip install git+https://github.com/ultralytics/CLIP.git)") from ex
        self.model.eval(); self.labels = [s[0] for s in SURFACES]
        prompts = [f"a photo of {l} on the ground" if l not in ("indoor floor", "carpet", "stairs") else f"a photo of {l}" for l in self.labels]
        with torch.no_grad():
            t = self.model.encode_text(self.tokenize(prompts).to(self.device)); self.text = t / t.norm(dim=-1, keepdim=True)

    def classify(self, crops):
        from PIL import Image
        torch = self.torch
        with torch.no_grad():
            x = torch.stack([self.preprocess(Image.fromarray(c[..., ::-1].copy())) for c in crops]).to(self.device)        # BGR -> RGB
            f = self.model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(dim=-1).cpu().numpy()
        return [[(self.labels[j], float(row[j])) for j in np.argsort(-row)[: self.topk]] for row in p]


def make_ground(kind: str, device: str | None = None):
    """'none' -> None; 'clip' or 'colors' -> a classifier. Raises ImportError with an explanation when CLIP is not available."""
    if kind in (None, "none"): return None
    if kind == "colors": return ColorSurfaceClassifier()
    if kind == "clip": return ClipSurfaceClassifier(device)
    raise ValueError(f"unknown ground classifier {kind!r}: none, clip or colors")


class GroundMonitor:
    """One per session. `due(t)` says whether it is time to look again; `update(frame, t)` classifies the three zones."""
    def __init__(self, classifier, every_s: float = 1.0, y0: float = 0.55, history: int = 3, bad_conf: float = 0.40):
        self.clf, self.every_s, self.y0, self.bad_conf = classifier, every_s, y0, bad_conf
        self.hist = {z: deque(maxlen=history) for z, _, _ in ZONES}; self.last = -1e9; self.looks = 0; self._state: dict | None = None

    def due(self, t: float) -> bool:
        return t - self.last >= self.every_s

    def update(self, frame_bgr: np.ndarray, t: float) -> dict:
        h, w = frame_bgr.shape[:2]; y = int(h * self.y0)
        crops = [frame_bgr[y:, int(w * a): max(int(w * a) + 8, int(w * b))] for _, a, b in ZONES]
        results = self.clf.classify(crops); self.last = t; self.looks += 1; state = {}
        for (zone, _, _), top in zip(ZONES, results):
            self.hist[zone].append(dict(top)); acc: dict[str, float] = {}
            for d in self.hist[zone]:
                for lab, p in d.items(): acc[lab] = acc.get(lab, 0.0) + p
            n = len(self.hist[zone]); lab = max(acc, key=acc.get); conf = acc[lab] / n; q = SURFACE_QUALITY.get(lab, "fair")
            state[zone] = {"surface": lab, "ru": SURFACE_RU.get(lab, lab), "quality": q, "quality_ru": QUALITY_RU[q], "conf": round(conf, 2),
                           "bad": bool(q == "bad" and conf >= self.bad_conf)}
        self._state = state; return state

    def state(self) -> dict | None:
        return None if self._state is None else {z: dict(v) for z, v in self._state.items()}
