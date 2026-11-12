"""Semantic segmentation of the scene in front of the camera: where is the floor, the walls, the doors, the stairs, the glass.

This is step one of "where can I walk": a wall, a floor and a passage are REGIONS, not boxes. Before building anything on top of
a segmentation model, measure its speed and LOOK at its masks on your own videos:   scenelive bench-seg VIDEO

Groups (everything the model knows is folded into these):
  walkable  floor, sidewalk, path, carpet                              road     the carriageway (cars!), kept apart from the sidewalk
  rough     grass, earth, sand, field                                  wall     walls, buildings, fences, railings, columns
  door      doors                                                      stairs   stairs, steps, escalators (a drop-off or a climb)
  glass     windows, mirrors (a person cannot tell glass from air)     dynamic  people, vehicles, animals (they move)
  obstacle  anything else that is not sky / ceiling                    ignore   sky, ceiling"""
from __future__ import annotations
import os
import time
import cv2
import numpy as np

IGNORE, WALK, ROAD, ROUGH, WALL, DOOR, STAIRS, GLASS, DYNAMIC, OBSTACLE = range(10)
GROUP_NAMES = ["ignore", "walkable", "road", "rough", "wall", "door", "stairs", "glass", "dynamic", "obstacle"]
GROUP_RU = ["не учитывается", "можно идти", "проезжая часть", "неровная земля", "стена", "дверь", "лестница", "стекло", "движущееся", "препятствие"]
GROUP_COLOR = {WALK: (60, 200, 60), ROUGH: (60, 200, 190), ROAD: (20, 90, 160), WALL: (200, 120, 40), DOOR: (0, 140, 255), STAIRS: (0, 240, 255), GLASS: (255, 255, 0),
               DYNAMIC: (60, 60, 230), OBSTACLE: (200, 60, 200)}   # BGR

# Road and sidewalk are DIFFERENT for a person on foot: the first is where the cars are.
_SETS = {
    WALK: {"floor", "sidewalk", "path", "rug"},
    ROAD: {"road", "runway"},
    ROUGH: {"grass", "earth", "field", "sand", "dirt track", "land"},
    WALL: {"wall", "building", "house", "skyscraper", "fence", "railing", "bannister", "column", "tower", "hovel"},
    DOOR: {"door", "screen door"},
    STAIRS: {"stairs", "stairway", "step", "escalator"},
    GLASS: {"windowpane", "mirror"},
    DYNAMIC: {"person", "car", "bus", "truck", "van", "bicycle", "minibike", "animal", "boat", "airplane", "ship"},
    IGNORE: {"sky", "ceiling", "unlabeled"},
}


def group_of(label: str) -> int:
    """ADE20K labels look like 'floor, flooring' or 'rug, carpet, carpeting': the first word is the class. 'glass, drinking glass' is an object, not a window."""
    key = label.split(",")[0].strip().lower()
    for g, names in _SETS.items():
        if key in names: return g
    return OBSTACLE


def group_map(labels: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    lut = np.full(max(max(id2label) + 1, int(labels.max()) + 1), OBSTACLE, np.uint8)
    for i, n in id2label.items(): lut[i] = group_of(n)
    return lut[labels]


def _islands(g: np.ndarray, label: int, min_area: float, walk: np.ndarray, to_walk: bool) -> None:
    """Small components of `label`: relabel them from their surroundings (in place)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((g == label).astype(np.uint8), connectivity=8)
    k = np.ones((5, 5), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area: continue
        x, y, w, h = st[i, :4]; x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(g.shape[1], x + w + 3), min(g.shape[0], y + h + 3)
        comp = lab[y0:y1, x0:x1] == i; ring = (cv2.dilate(comp.astype(np.uint8), k) > 0) & ~comp; around = g[y0:y1, x0:x1][ring]
        if around.size == 0: continue
        if to_walk:
            if walk[y0:y1, x0:x1][ring].mean() > 0.7: g[y0:y1, x0:x1][comp] = WALK                # a drain cover or shadow inside the sidewalk
        elif walk[y0:y1, x0:x1][ring].mean() < 0.3: g[y0:y1, x0:x1][comp] = int(np.bincount(around).argmax())   # a 'floor' blob inside a car


def clean_groups(g: np.ndarray, thin_px: int = 5, island_frac: float = 0.003) -> np.ndarray:
    """Remove what the street pictures showed as noise:
      - thin 'walkable' strips next to the road (painted lines) become road;
      - small 'road' patches surrounded by sidewalk (drain covers, shadows) become sidewalk;
      - small 'walkable' patches surrounded by something else (blocks on a car) take the label around them."""
    g = g.astype(np.uint8).copy(); H, W = g.shape; min_area = island_frac * H * W
    walk = ((g == WALK) | (g == ROUGH)).astype(np.uint8)
    opened = cv2.morphologyEx(walk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thin_px, thin_px)))
    near_road = cv2.dilate((g == ROAD).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    g[(walk > 0) & (opened == 0) & near_road] = ROAD
    walk = (g == WALK) | (g == ROUGH)
    _islands(g, ROAD, min_area, walk, to_walk=True)
    walk = (g == WALK) | (g == ROUGH)
    _islands(g, WALK, min_area, walk, to_walk=False)
    return g


def stats(gmap: np.ndarray) -> dict[str, float]:
    n = gmap.size; return {GROUP_NAMES[g]: float((gmap == g).sum()) / n for g in range(len(GROUP_NAMES))}


def overlay(frame_bgr: np.ndarray, gmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = frame_bgr.copy()
    if gmap.shape[:2] != out.shape[:2]: gmap = cv2.resize(gmap, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(out); mask = np.zeros(gmap.shape, bool)
    for g, c in GROUP_COLOR.items(): color[gmap == g] = c; mask |= gmap == g
    out[mask] = (alpha * color[mask] + (1 - alpha) * out[mask]).astype(np.uint8)
    for k, g in enumerate(GROUP_COLOR):                                                         # legend
        cv2.rectangle(out, (4, 4 + 16 * k), (18, 16 + 16 * k), GROUP_COLOR[g], -1); cv2.putText(out, GROUP_NAMES[g], (24, 15 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class SegFormerSegmenter:
    """SegFormer-B0 trained on ADE20K (150 classes). Needs `pip install transformers` (torch comes with ultralytics). The model downloads on first use.
    Check the licence of the weights before any commercial use."""
    name = "segformer-b0-ade20k"
    def __init__(self, device: str | None = None, model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512", size: int = 0):
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as ex:
            raise ImportError("the 'transformers' package is not installed: run  windows\\setup_seg.bat  (pip install transformers)") from ex
        from .device import configure, pick_device
        self.size = size; self.last_conf = None; self.torch = torch; self.device = pick_device(device); configure(self.device)
        self.proc = SegformerImageProcessor.from_pretrained(model_id); self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def segment(self, frame_bgr: np.ndarray):
        """-> (label map, same size as the frame, ADE20K class ids; id2label)"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.size:                                                                           # aspect-preserving input (multiples of 32): fewer pixels than the default 512x512 square
            h, w = rgb.shape[:2]; k = self.size / max(h, w); rgb = cv2.resize(rgb, (max(32, int(round(w * k / 32)) * 32), max(32, int(round(h * k / 32)) * 32)), interpolation=cv2.INTER_AREA)
            inputs = self.proc(images=rgb, return_tensors="pt", do_resize=False)
        else: inputs = self.proc(images=rgb, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(pixel_values=inputs["pixel_values"].to(self.device)).logits
            conf, lab = logits.softmax(1).max(1)
        size = (frame_bgr.shape[1], frame_bgr.shape[0])                                         # the answer is 1/4 of the input size: resize the answer, not the logits (cheaper)
        self.last_conf = cv2.resize((conf[0].cpu().numpy() * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)          # how sure the model was, 0..255 per pixel
        return cv2.resize(lab[0].cpu().numpy().astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(np.int32), self.id2label


class ColorSegmenter:
    """Crude geometry and colour rules: demo and tests only (ceiling on top, wall in the middle, floor below; red = person, green = door)."""
    name = "colors"
    last_conf = None
    id2label = {0: "wall", 1: "floor, flooring", 2: "door", 3: "ceiling", 4: "person, individual"}
    def segment(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]; lab = np.zeros((h, w), np.int32); lab[: int(h * 0.2)] = 3; lab[int(h * 0.58):] = 1
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV); sat = hsv[..., 1] > 110; hue = hsv[..., 0]
        lab[sat & ((hue < 8) | (hue > 172))] = 4; lab[sat & (hue > 45) & (hue < 75)] = 2
        return lab, self.id2label


def make_segmenter(kind: str, device: str | None = None, size: int = 0):
    if kind == "colors": return ColorSegmenter()
    if kind == "segformer": return SegFormerSegmenter(device, size=size)
    raise ValueError(f"unknown segmenter {kind!r}: segformer or colors")


def bench_segmenter(video: str, seg, out_dir: str, n_frames: int = 12, long_side: int = 640, warmup: int = 1, log=print, clean: bool = True) -> dict:
    """Time the segmenter on frames spread over the video and save an overlay picture of each (floor green, walls blue, doors orange ...)."""
    from .video_io import open_video
    rd = open_video(video); total = int(rd.get(cv2.CAP_PROP_FRAME_COUNT) or 0); step = max(1, total // max(n_frames, 1)) if total else 5; frames = []; i = 0
    while len(frames) < n_frames:
        ok, fr = rd.read()
        if not ok: break
        if i % step == 0:
            s = long_side / max(fr.shape[:2]); frames.append(cv2.resize(fr, (round(fr.shape[1] * s), round(fr.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else fr)
        i += 1
    rd.release()
    if not frames: raise ValueError("no frames read from the video")
    os.makedirs(out_dir, exist_ok=True)
    for fr in frames[:warmup]: seg.segment(fr)                                                  # the first call is slow (model warm-up): not counted
    ms, rows = [], []
    for k, fr in enumerate(frames):
        t0 = time.perf_counter(); labels, id2 = seg.segment(fr); ms.append((time.perf_counter() - t0) * 1000.0)
        g = group_map(labels, id2); g = clean_groups(g) if clean else g; st = stats(g); rows.append(st)
        ok, buf = cv2.imencode(".jpg", overlay(fr, g), [cv2.IMWRITE_JPEG_QUALITY, 88]); buf.tofile(os.path.join(out_dir, f"seg_{k:03d}.jpg"))
    a = np.array(ms); summary = {"model": getattr(seg, "name", "?"), "frames": len(frames), "size": f"{frames[0].shape[1]}x{frames[0].shape[0]}", "mean_ms": float(a.mean()),
                                 "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)), "fps": float(1000.0 / a.mean()),
                                 "mean_share": {n: float(np.mean([r[n] for r in rows])) for n in GROUP_NAMES}}
    lines = [f"model: {summary['model']}   frames: {summary['frames']} at {summary['size']}",
             f"time per frame: mean {summary['mean_ms']:.0f} ms, p50 {summary['p50_ms']:.0f}, p95 {summary['p95_ms']:.0f}  ->  {summary['fps']:.1f} frames per second",
             "mean share of the picture: " + ", ".join(f"{n} {100 * v:.0f}%" for n, v in summary["mean_share"].items() if v > 0.005)]
    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    for ln in lines: log(ln)
    return summary
