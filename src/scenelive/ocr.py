"""Reading text: bus route numbers, signs, shop names.

What is read, and when (to stay light):
  - a bus number is read ONLY inside the box of a detected bus, about once a second, until several readings agree;
  - signs, billboards, kiosks: inside their boxes, a few times;
  - "what is written?": the current frame, once, on request.
Everything read goes into a list of texts that the assistant gives on request.
Engine: EasyOCR (Russian + English; `pip install easyocr`, it downloads its models on first use). Not run in the dev sandbox: measure it
with `scenelive bench-ocr`. Readings are hints: a misread bus number must never be presented as certain."""
from __future__ import annotations
import re
import threading
import time
from dataclasses import dataclass, field
import numpy as np

BUS_CLASSES = {"bus"}
SIGN_CLASSES = {"street sign", "sign", "stop sign", "billboard", "kiosk", "traffic sign"}
BUS_ALLOW = "0123456789ABCEHKMOPTXАВЕКМНОРСТХ"                         # route numbers: digits and a few letters that look the same in both alphabets
BUS_RE = re.compile(r"^\d{1,3}[A-ZА-ЯЁ]?$")


@dataclass
class TextHit:
    text: str
    conf: float
    box: tuple                                                          # x0, y0, x1, y1 in the image that was read


class EasyOcrEngine:
    name = "easyocr"
    def __init__(self, langs=("ru", "en"), gpu: bool = False):
        try: import easyocr
        except ImportError as ex: raise ImportError("the 'easyocr' package is not installed: run  windows\\setup_ocr.bat  (pip install easyocr)") from ex
        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)

    def read(self, img_bgr: np.ndarray, allowlist: str | None = None) -> list[TextHit]:
        out = []
        for pts, text, conf in self.reader.readtext(img_bgr, detail=1, paragraph=False, allowlist=allowlist):
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]; out.append(TextHit(str(text), float(conf), (min(xs), min(ys), max(xs), max(ys))))
        return out


def make_ocr(kind: str, langs: str = "ru,en", device: str | None = "auto"):
    if kind in (None, "none"): return None
    if kind == "easyocr":
        from .device import is_cuda, pick_device
        return EasyOcrEngine(tuple(l.strip() for l in langs.split(",") if l.strip()), gpu=is_cuda(pick_device(device)))
    raise ValueError(f"unknown OCR engine {kind!r}: none or easyocr")


def norm_key(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.casefold())


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .,;:-_|")


def bus_number(hits: list[TextHit]):
    """The most likely route number among the readings of a bus: (number, score) or None. Bigger and surer text wins."""
    best = None
    for h in hits:
        t = re.sub(r"\s+", "", h.text.upper()).replace("O", "0").replace("О", "0") if re.search(r"\d", h.text) else ""
        if not BUS_RE.match(t) or h.conf < 0.3: continue
        score = h.conf * max(1.0, h.box[3] - h.box[1])
        if best is None or score > best[1]: best = (t, score)
    return best


def sign_texts(hits: list[TextHit], min_conf: float = 0.35) -> list[tuple[str, float]]:
    out = []
    for h in hits:
        t = clean_text(h.text)
        if h.conf >= min_conf and len(norm_key(t)) >= 2 and re.search(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі0-9]", t): out.append((t, h.conf))
    return out


# ------------------------------------------------------------------------------------------------------------------ the list of texts
@dataclass
class TextItem:
    text: str
    kind: str                # bus | sign | read
    source: str              # what it was written on (bus, street sign, ...) or "frame"
    world_az: float | None
    first_t: float
    last_t: float
    count: int = 1
    conf: float = 0.0


class TextLog:
    """Everything that was read, without duplicates (same letters = same text), newest first on request."""
    def __init__(self, max_items: int = 200): self.items: dict[str, TextItem] = {}; self.max_items = max_items; self.lock = threading.Lock()

    def add(self, text: str, kind: str, source: str, world_az, t: float, conf: float) -> None:
        k = norm_key(text)
        if not k: return
        with self.lock:
            it = self.items.get(k)
            if it is None:
                self.items[k] = TextItem(text, kind, source, world_az, t, t, 1, conf)
                if len(self.items) > self.max_items: del self.items[min(self.items, key=lambda x: self.items[x].last_t)]
            else:
                it.last_t = t; it.count += 1; it.conf = max(it.conf, conf); it.world_az = world_az if world_az is not None else it.world_az
                if conf > it.conf * 0.99 and len(text) >= len(it.text): it.text = text

    def recent(self, n: int = 10) -> list[TextItem]:
        with self.lock: return sorted(self.items.values(), key=lambda i: -i.last_t)[:n]


# ------------------------------------------------------------------------------------------------------------------ the reader
@dataclass
class _Job:
    kind: str
    tries: int = 0
    last_try: float = -1e9
    votes: dict = field(default_factory=dict)            # text -> [count, best confidence]
    done: bool = False


class TextReader:
    """Decides what to read and reads it in a background thread. One job at a time; a busy reader skips, it never queues up."""
    def __init__(self, engine, every_s: float = 1.0, sync: bool = False, max_tries: int = 8, min_bus_px: int = 60, min_sign_px: int = 24, log=None):
        self.engine, self.every_s, self.sync, self.max_tries, self.min_bus_px, self.min_sign_px = engine, every_s, sync, max_tries, min_bus_px, min_sign_px
        self.log_fn = log or (lambda *a: None); self.jobs: dict[int, _Job] = {}; self.texts = TextLog(); self.busy = False; self.engine_lock = threading.Lock()
        self.ms: list[float] = []; self.errors = 0; self._last_any = -1e9

    # ---- results per track
    def reading(self, track_id: int):
        """(text, number of agreeing readings) for a track, or None."""
        j = self.jobs.get(track_id)
        if not j or not j.votes: return None
        text, (n, c) = max(j.votes.items(), key=lambda kv: (kv[1][0], kv[1][1])); return text, n

    # ---- what to read now
    def consider(self, frame: np.ndarray, t: float, tracks: list[dict], size: tuple, world_az_of) -> None:
        if self.busy or t - self._last_any < self.every_s / 2: return
        sx, sy = frame.shape[1] / max(size[0], 1), frame.shape[0] / max(size[1], 1); cands = []
        for tr in tracks:
            kind = "bus" if tr["name"] in BUS_CLASSES else "sign" if tr["name"] in SIGN_CLASSES else None
            if kind is None: continue
            j = self.jobs.setdefault(tr["id"], _Job(kind))
            if j.done or t - j.last_try < self.every_s: continue
            x0, y0, x1, y1 = tr["box"]; x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
            if (y1 - y0) < (self.min_bus_px if kind == "bus" else self.min_sign_px): continue            # too small to read: wait until it is closer
            cands.append(((y1 - y0) * (x1 - x0), tr, j, (x0, y0, x1, y1)))
        if not cands: return
        _, tr, j, (x0, y0, x1, y1) = max(cands, key=lambda c: c[0])                                     # the biggest one first
        mx, my = 0.05 * (x1 - x0), 0.05 * (y1 - y0); H, W = frame.shape[:2]
        crop = frame[int(max(0, y0 - my)): int(min(H, y1 + my)), int(max(0, x0 - mx)): int(min(W, x1 + mx))].copy()
        j.tries += 1; j.last_try = t; self._last_any = t; az = world_az_of(tr)
        if self.sync: self._read(tr["id"], tr["name"], j, crop, t, az)
        else:
            self.busy = True; threading.Thread(target=self._read, args=(tr["id"], tr["name"], j, crop, t, az), daemon=True).start()

    def _read(self, tid, source, j: _Job, crop, t, az) -> None:
        t0 = time.perf_counter()
        try:
            with self.engine_lock: hits = self.engine.read(crop, BUS_ALLOW if j.kind == "bus" else None)
            self.ms.append((time.perf_counter() - t0) * 1000.0)
            found = [bus_number(hits)] if j.kind == "bus" else [(tx, c) for tx, c in sign_texts(hits)]
            for item in found:
                if not item: continue
                text, score = item; conf = min(1.0, score / 100.0) if j.kind == "bus" else score
                v = j.votes.setdefault(text, [0, 0.0]); v[0] += 1; v[1] = max(v[1], conf)
                self.texts.add(("автобус " + text) if j.kind == "bus" else text, j.kind, source, az, t, conf)
            best = self.reading(tid)
            if (best and best[1] >= (2 if j.kind == "bus" else 3)) or j.tries >= self.max_tries: j.done = True               # agreed, or gave up
        except Exception as ex:
            self.errors += 1; self.log_fn("OCR error:", type(ex).__name__, ex)
        finally: self.busy = False

    # ---- "what is written?"
    def read_now(self, frame: np.ndarray, t: float, az_of_x) -> list[tuple[str, float, float]]:
        """Read the whole frame now (on request). -> [(text, confidence, world azimuth)] left to right; also added to the list."""
        t0 = time.perf_counter()
        with self.engine_lock: hits = self.engine.read(frame)
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        good = [(h, tx, c) for h in hits for tx, c in sign_texts([h])]; good.sort(key=lambda x: (x[0].box[0] + x[0].box[2]) / 2)
        out = []
        for h, tx, c in good:
            az = az_of_x((h.box[0] + h.box[2]) / 2); out.append((tx, c, az)); self.texts.add(tx, "read", "frame", az, t, c)
        return out

    def stats(self) -> dict:
        return {"reads": len(self.ms), "mean_ms": (sum(self.ms) / len(self.ms)) if self.ms else None, "errors": self.errors, "texts": len(self.texts.items)}


def bench_ocr(source: str, engine, out_dir: str, n_frames: int = 12, bus: bool = False, log=print) -> dict:
    """Read text on photos / frames of a video; save pictures with numbered boxes and results.txt (Cyrillic cannot be drawn by OpenCV)."""
    import os
    import cv2
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if os.path.isdir(source): pics = [(f, cv2.imdecode(np.fromfile(os.path.join(source, f), np.uint8), cv2.IMREAD_COLOR)) for f in sorted(os.listdir(source)) if f.lower().endswith(exts)]
    elif source.lower().endswith(exts):
        if not os.path.exists(source): raise FileNotFoundError(source)
        pics = [(os.path.basename(source), cv2.imdecode(np.fromfile(source, np.uint8), cv2.IMREAD_COLOR))]
    else:
        from .video_io import open_video
        rd = open_video(source); total = int(rd.get(cv2.CAP_PROP_FRAME_COUNT) or 0); step = max(1, total // max(n_frames, 1)) if total else 10; pics = []; i = 0
        while len(pics) < n_frames:
            ok, fr = rd.read()
            if not ok: break
            if i % step == 0: pics.append((f"frame_{i:05d}", fr))
            i += 1
        rd.release()
    pics = [(n, p) for n, p in pics if p is not None]
    if not pics: raise ValueError(f"no pictures read from {source}")
    os.makedirs(out_dir, exist_ok=True); ms, lines = [], []
    for name, img in pics:
        t0 = time.perf_counter(); hits = engine.read(img, BUS_ALLOW if bus else None); ms.append((time.perf_counter() - t0) * 1000.0)
        vis = img.copy(); lines.append(f"{name}  ({ms[-1]:.0f} ms)")
        for k, h in enumerate(hits, 1):
            x0, y0, x1, y1 = map(int, h.box); cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2); cv2.putText(vis, str(k), (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            lines.append(f"  {k}. {h.text}   (conf {h.conf:.2f})")
        if bus:
            b = bus_number(hits); lines.append(f"  -> bus number: {b[0] if b else 'not found'}")
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88]); buf.tofile(os.path.join(out_dir, os.path.splitext(name)[0] + "_ocr.jpg"))
    a = np.array(ms); summary = {"pictures": len(pics), "mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95))}
    lines = [f"pictures: {len(pics)}   time per picture: mean {summary['mean_ms']:.0f} ms, p95 {summary['p95_ms']:.0f} ms", ""] + lines
    open(os.path.join(out_dir, "results.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    log(lines[0]); return summary
