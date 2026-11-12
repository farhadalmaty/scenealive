"""Where is the free space? Where is the wall, where is the way through, which way to go.

From a segmentation map (floor / sidewalk / road / wall / door ...) and the camera geometry:
for every column of the picture, go up from the feet while the pixels are walkable; the point where that stops is where an obstacle
(or the end of the sidewalk) meets the ground. A pixel row on the ground plane is a distance: with the camera at height h and tilted
down by theta, the ray through a pixel (x, y)  [x = (u - cx) / f, y = (v - cy) / f]  meets the ground at

    forward Y = h (cos theta - y sin theta) / (y cos theta + sin theta),     lateral X = h x / (y cos theta + sin theta)

(only for rays that go downward). Accuracy is rough: the height of a hand-held phone is a guess (default 1.4 m), the tilt comes from the
phone's sensors. So the distances we SAY are smaller than the measured ones (SAFETY), and we say "not sure" when the geometry is not trustworthy.
A picture cannot show glass, a step down or a hole without texture: that is never promised."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import cv2
import numpy as np
from .camera import Camera
from .segment import DOOR, DYNAMIC, GLASS, GROUP_NAMES, IGNORE, OBSTACLE, ROAD, ROUGH, STAIRS, WALK, WALL
from .speech import metres

CAP_M = 12.0                 # beyond this the free space is simply "far"
SAFETY = 0.8                 # we state 80 % of the measured free distance
STOP_M = 1.5                 # nearer than this (as stated): STOP. A walking person plus the delay of a spoken hint need that much room
CAUTION_M = 3.0              # nearer than this: careful
BLOCKER_RU = {"wall": "стена", "dynamic": "препятствие, возможно человек или машина", "door": "дверь", "stairs": "лестница", "glass": "стекло или окно", "obstacle": "препятствие",
              "road": "проезжая часть", "ignore": "что-то неизвестное", "walkable": "препятствие", "rough": "препятствие"}


def ground_point(v: float, u: float, cam: Camera, pitch_down_deg: float, h_cam: float):
    """Pixel (u, v) -> point on the ground plane (lateral X to the right, forward Y), metres; None when the ray does not go down to the ground."""
    th = math.radians(pitch_down_deg); x = (u - cam.cx) / cam.f; y = (v - cam.cy) / cam.f; den = y * math.cos(th) + math.sin(th)
    if den <= 0.02: return None
    return h_cam * x / den, h_cam * (math.cos(th) - y * math.sin(th)) / den


def row_for_forward(Y: float, cam: Camera, pitch_down_deg: float, h_cam: float) -> float:
    """The inverse of ground_point for lateral X = 0: the pixel row where the ground is Y metres ahead (tests and the picture of the horizon)."""
    th = math.radians(pitch_down_deg); y = (h_cam * math.cos(th) - Y * math.sin(th)) / (Y * math.cos(th) + h_cam * math.sin(th)); return cam.cy + cam.f * y


@dataclass
class Sector:
    lo: float                # degrees right of the optical axis (negative = left)
    hi: float
    free_m: float            # measured free distance to the nearest obstacle in the sector (capped)
    blocker: str             # what stops it: wall, dynamic, door, stairs, glass, obstacle, road ...
    capped: bool

    @property
    def free_safe_m(self) -> float: return self.free_m * SAFETY

    @property
    def mid(self) -> float: return (self.lo + self.hi) / 2.0


@dataclass
class FreeSpace:
    sectors: list
    underfoot: str           # safe | rough | road | blocked | unknown
    reliable: bool
    uncertain: bool
    pitch_known: bool
    pitch_down: float
    reasons: list = field(default_factory=list)
    sectors_cross: list = field(default_factory=list)   # the same, with the road counted as a way (used only while the user says they are crossing)
    sidewalk: tuple | None = None                       # (azimuth deg, distance m) of the nearest sidewalk or floor, or None

    def to_dict(self) -> dict:
        return {"sectors": [{"lo": round(s.lo, 1), "hi": round(s.hi, 1), "free_m": round(s.free_m, 1), "free_safe_m": round(s.free_safe_m, 1), "blocker": s.blocker,
                             "capped": s.capped} for s in self.sectors], "underfoot": self.underfoot, "reliable": self.reliable, "uncertain": self.uncertain,
                "pitch_known": self.pitch_known, "pitch_down": round(self.pitch_down, 1), "reasons": list(self.reasons),
                "sidewalk": None if self.sidewalk is None else {"az": round(self.sidewalk[0], 1), "m": round(self.sidewalk[1], 1)}}


def _first_run(col: np.ndarray, n: int):
    """Index (counted from the bottom) of the first run of n True values in a column, or None."""
    c = np.concatenate([[0], np.cumsum(col)]); idx = np.nonzero((c[n:] - c[:-n]) == n)[0]
    return int(idx[0]) if len(idx) else None


def _sectors(walk: np.ndarray, gmap: np.ndarray, cam: Camera, th: float, h_cam: float, n_sectors: int, min_run: int) -> list:
    H, W = walk.shape; step = max(2, W // 160)
    az = lambda u: math.degrees(math.atan((u - cam.cx) / cam.f)); lo_az, hi_az = az(0), az(W - 1)
    cols = []                                                                              # (azimuth, free distance, blocker, capped) per sampled column
    for u in range(step // 2, W, step):
        k = _first_run((~walk[:, u])[::-1], min_run)
        if k is None: cols.append((az(u), CAP_M, "ignore", True)); continue                    # walkable all the way up: free to the horizon
        v_b = H - 1 - k; above = gmap[max(0, v_b - min_run + 1): v_b + 1, u]; blk = GROUP_NAMES[int(np.bincount(above).argmax())]
        if k == 0: cols.append((az(u), 0.0, blk, False)); continue                              # the obstacle is at the feet
        gp = ground_point(min(v_b + 1, H - 1), u, cam, th, h_cam)
        if gp is None: cols.append((az(u), CAP_M, blk, True)); continue
        r = math.hypot(*gp); cols.append((az(u), min(r, CAP_M), blk, r >= CAP_M))
    edges = np.linspace(lo_az, hi_az, n_sectors + 1); sectors = []
    for i in range(n_sectors):
        sel = [c for c in cols if edges[i] <= c[0] < edges[i + 1] or (i == n_sectors - 1 and c[0] == edges[-1])]
        if not sel: sectors.append(Sector(float(edges[i]), float(edges[i + 1]), CAP_M, "ignore", True)); continue
        free = float(np.percentile([c[1] for c in sel], 20)); lim = [c for c in sel if c[1] <= free * 1.15 + 1e-6] or sel
        blk = max(set(c[2] for c in lim), key=[c[2] for c in lim].count); sectors.append(Sector(float(edges[i]), float(edges[i + 1]), free, blk, free >= CAP_M - 1e-6))
    return sectors


def _nearest(mask: np.ndarray, cam: Camera, th: float, h_cam: float, min_run: int):
    """The nearest point on the ground of a region (e.g. the sidewalk): (azimuth deg, distance m) or None."""
    H, W = mask.shape; best = None
    for u in range(2, W, max(2, W // 160)):
        k = _first_run(mask[:, u][::-1], min_run)
        if k is None: continue
        gp = ground_point(H - 1 - k, u, cam, th, h_cam)
        if gp is None: continue
        r = math.hypot(*gp)
        if best is None or r < best[1]: best = (math.degrees(math.atan((u - cam.cx) / cam.f)), r)
    return best


def analyse(gmap: np.ndarray, cam: Camera, pitch_down_deg: float | None, h_cam: float = 1.4, conf: np.ndarray | None = None, n_sectors: int = 5, min_run_frac: float = 0.02) -> FreeSpace:
    H, W = cam.height, cam.width
    if gmap.shape != (H, W): gmap = cv2.resize(gmap, (W, H), interpolation=cv2.INTER_NEAREST)
    pitch_known = pitch_down_deg is not None; th = float(pitch_down_deg or 0.0); reasons = []
    walk = (gmap == WALK) | (gmap == ROUGH); min_run = max(4, int(H * min_run_frac))
    sectors = _sectors(walk, gmap, cam, th, h_cam, n_sectors, min_run)
    sectors_cross = _sectors(walk | (gmap == ROAD), gmap, cam, th, h_cam, n_sectors, min_run)            # while crossing, the road is the way
    band = gmap[int(H * 0.9):, int(W * 0.2): int(W * 0.8)]; g = int(np.bincount(band.ravel(), minlength=10).argmax())
    underfoot = "safe" if g == WALK else "rough" if g == ROUGH else "road" if g == ROAD else "unknown" if g == IGNORE else "blocked"
    sidewalk = _nearest(gmap == WALK, cam, th, h_cam, min_run) if underfoot == "road" else None
    uncertain = bool(conf is not None and float(np.mean(conf[int(H * 0.5):, :])) / 255.0 < 0.5)
    if not pitch_known: reasons.append("the tilt of the phone is unknown")
    elif th < -5 or th > 55: reasons.append(f"the camera is tilted too far ({th:.0f} deg down)")
    if uncertain: reasons.append("the segmentation model is not sure")
    return FreeSpace(sectors, underfoot, pitch_known and -5 <= th <= 55, uncertain, pitch_known, th, reasons, sectors_cross, sidewalk)


# ----------------------------------------------------------------------------------------------------------------- words
def dist_words(d: float, capped: bool = False) -> str:
    if capped: return "больше " + metres(int(CAP_M * SAFETY))                              # nothing seen as far as we look: "more than 9 metres", not an exact figure
    if d < 1.0: return "меньше метра"
    if d >= 10.0: return "больше 10 метров"
    return "около " + metres(int(d))


def turn_words(mid: float) -> str:
    deg = max(5, int(round(abs(mid) / 5.0)) * 5); return f"{'направо' if mid > 0 else 'налево'} примерно на {deg} градусов"


UNSURE = "Я не вижу путь надёжно. Остановитесь и не полагайтесь на подсказки."


def where_words(az: float) -> str:
    a = abs(az)
    if a <= 15: return "прямо впереди"
    side = "справа" if az > 0 else "слева"
    return f"{side}-спереди" if a <= 50 else side if a <= 120 else "сзади"


def _advise_sectors(sectors: list, parts: list, level: str) -> dict:
    rank = {"ok": 0, "caution": 1, "stop": 2}
    centre = [s for s in sectors if s.hi > -12 and s.lo < 12] or [min(sectors, key=lambda s: abs(s.mid))]; c = min(centre, key=lambda s: s.free_safe_m); d = c.free_safe_m
    if d < STOP_M: level = "stop"; parts.append(f"Стоп. Впереди {BLOCKER_RU.get(c.blocker, c.blocker)}, {dist_words(d)}.")
    elif d < CAUTION_M: level = max(level, "caution", key=rank.get); parts.append(f"Осторожно, впереди {BLOCKER_RU.get(c.blocker, c.blocker)}, {dist_words(d)}.")
    else:
        end = f", дальше {BLOCKER_RU.get(c.blocker, c.blocker)}" if (d < 8.0 and c.blocker != "ignore" and not c.capped) else ""             # say what ends the way
        parts.append(f"Прямо свободно {dist_words(d, c.capped)}{end}.")
    turn = None
    if d < CAUTION_M:
        sides = [s for s in sectors if s not in centre and s.free_safe_m >= max(2.0, d + 1.0)]
        if sides:
            best = max(sides, key=lambda s: s.free_safe_m - 0.03 * abs(s.mid)); turn = best.mid
            parts.append(f"Свободнее {'справа' if best.mid > 0 else 'слева'}, {dist_words(best.free_safe_m, best.capped)}. Поверните {turn_words(best.mid)}.")
        else: parts.append("Свободного прохода рядом не вижу.")
    for s in sectors:
        if s not in centre and s.free_safe_m < STOP_M and d >= CAUTION_M: parts.append(f"{'Справа' if s.mid > 0 else 'Слева'} {BLOCKER_RU.get(s.blocker, s.blocker)} близко, {dist_words(s.free_safe_m)}."); break
    return {"level": level, "text": " ".join(parts), "ahead_m": round(d, 1), "turn_deg": None if turn is None else round(turn, 1), "reasons": []}


def advise(fs: FreeSpace, crossing: bool = False) -> dict:
    """Where to go, in short spoken sentences. Never says 'safe'; says less free space than measured; says 'not sure' when it is.
    On the carriageway it points to the nearest sidewalk instead of 'stop' (stopping in the middle of a road is the worse advice);
    while the user says they are crossing, the road counts as the way and only obstacles ahead are reported."""
    if not fs.reliable or fs.uncertain:
        return {"level": "unsure", "text": UNSURE, "ahead_m": None, "turn_deg": None, "reasons": fs.reasons}
    if crossing:
        return _advise_sectors(fs.sectors_cross or fs.sectors, ["Режим перехода."], "ok")
    if fs.underfoot == "road":
        if fs.sidewalk is not None:
            az, m = fs.sidewalk
            return {"level": "caution", "text": f"Похоже, вы на проезжей части. Ближайший тротуар {where_words(az)}, {dist_words(m * SAFETY)}.", "ahead_m": None, "turn_deg": round(az, 1), "reasons": []}
        return {"level": "caution", "text": "Похоже, вы на проезжей части, тротуара в кадре не вижу. Будьте осторожны и медленно поверните телефон, чтобы я его поискал.",
                "ahead_m": None, "turn_deg": None, "reasons": []}
    parts, level = [], "ok"
    if fs.underfoot == "blocked": parts.append("Под ногами не похоже на пол."); level = "caution"
    return _advise_sectors(fs.sectors, parts, level)


def advise_goal(fs: FreeSpace, goal_az: float, goal_ru: str, goal_range_m: float | None = None) -> dict:
    """The way to something we know about (its direction relative to where the camera points now): is the path to it free?"""
    if not fs.reliable or fs.uncertain: return {"level": "unsure", "text": UNSURE}
    lo, hi = fs.sectors[0].lo, fs.sectors[-1].hi
    if not (lo <= goal_az <= hi):
        side = "справа" if goal_az > 0 else "слева"
        return {"level": "ok", "text": f"{goal_ru.capitalize()} {side}, вне поля зрения камеры. Повернитесь {turn_words(goal_az)}, потом спросите снова, и я проверю путь."}
    s = min(fs.sectors, key=lambda x: abs(x.mid - goal_az)); need = (goal_range_m or 0.0) * 0.9
    if s.free_safe_m >= max(need, 2.0) and not (goal_range_m is None and s.free_safe_m < 4.0):
        return {"level": "ok", "text": f"{goal_ru.capitalize()} {('справа' if goal_az > 5 else 'слева' if goal_az < -5 else 'прямо')}. Путь выглядит свободным, {dist_words(s.free_safe_m, s.capped)}."}
    return {"level": "caution", "text": f"{goal_ru.capitalize()}: путь перекрыт, впереди {BLOCKER_RU.get(s.blocker, s.blocker)}, {dist_words(s.free_safe_m)}."}
