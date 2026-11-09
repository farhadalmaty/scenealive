"""Score the system against YOUR labels of a video, and find the confidence threshold that works best.

labels.csv  - one line per object class visible at a time:   t_seconds, name [, count]        (# starts a comment)
              4.0, person, 3        at second 4 there are three people in the frame
              4.0, car              ... and one car
              8.0, pole
List every object of the classes you care about; classes that never appear in the file are ignored (not counted as false alarms).
The run folder must contain frames.csv and tracks.csv (written by `scenelive replay`)."""
from __future__ import annotations
import csv
import os
from collections import Counter, defaultdict
from .labels import canonical


def read_labels(path) -> dict[float, Counter]:
    out: dict[float, Counter] = defaultdict(Counter)
    for line in open(path, encoding="utf-8-sig"):
        line = line.split("#")[0].strip()
        if not line: continue
        p = [x.strip() for x in line.replace(";", ",").split(",")]
        try: t = float(p[0]); name = canonical(p[1]); n = int(p[2]) if len(p) > 2 and p[2] else 1
        except (ValueError, IndexError): continue                       # header or junk
        out[t][name] += n
    return dict(out)


def _read(run_dir):
    frames = [(int(r["k"]), float(r["t"])) for r in csv.DictReader(open(os.path.join(run_dir, "frames.csv"), encoding="utf-8"))]
    tracks = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run_dir, "tracks.csv"), encoding="utf-8")):
        tracks[int(r["k"])].append({"name": r["name"], "conf": float(r["conf"]), "hits": int(r["hits"])})
    return frames, tracks


def evaluate(run_dir: str, labels_path: str, tol_s: float = 0.5, thresholds=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7), classes=None) -> dict:
    labels = read_labels(labels_path)
    if not labels: raise ValueError(f"{labels_path}: no labels found (lines like '4.0, person, 3')")
    frames, tracks = _read(run_dir); classes = set(classes or {n for c in labels.values() for n in c})
    matched, skipped = [], []
    for t, truth in sorted(labels.items()):
        k, tf = min(frames, key=lambda f: abs(f[1] - t))
        if abs(tf - t) > tol_s: skipped.append(t); continue
        matched.append((t, k, {n: c for n, c in truth.items() if n in classes}))
    rows = []
    for thr in thresholds:
        tp = fp = fn = 0; per: dict[str, list] = defaultdict(lambda: [0, 0, 0])
        for t, k, truth in matched:
            pred = Counter(x["name"] for x in tracks.get(k, []) if x["conf"] >= thr and x["name"] in classes)
            for n in set(pred) | set(truth):
                a, b = pred[n], truth.get(n, 0); m = min(a, b)
                tp += m; fp += a - m; fn += b - m; per[n][0] += m; per[n][1] += a - m; per[n][2] += b - m
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"thr": thr, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                     "per_class": {n: {"tp": v[0], "fp": v[1], "fn": v[2]} for n, v in per.items()}})
    best = max(rows, key=lambda r: (r["f1"], -r["thr"]))
    return {"rows": rows, "best": best, "frames_labeled": len(matched), "skipped": skipped, "classes": sorted(classes)}


def format_eval(res: dict) -> str:
    L = [f"labeled moments used: {res['frames_labeled']}" + (f"  (no processed frame near t = {res['skipped']})" if res["skipped"] else ""),
         f"classes scored: {', '.join(res['classes'])}", "",
         f"  {'conf >=':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'precision':>10} {'recall':>8} {'F1':>6}"]
    for r in res["rows"]:
        mark = "  <- best F1" if r is res["best"] else ""
        L.append(f"  {r['thr']:>8.2f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} {r['precision']:>10.2f} {r['recall']:>8.2f} {r['f1']:>6.2f}{mark}")
    b = res["best"]; L += ["", f"per class at conf >= {b['thr']:.2f}:", f"  {'class':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'precision':>10} {'recall':>8}"]
    for n, v in sorted(b["per_class"].items()):
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0; r = v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else 0.0
        L.append(f"  {n:<14} {v['tp']:>4} {v['fp']:>4} {v['fn']:>4} {p:>10.2f} {r:>8.2f}")
    L += ["", f"suggestion: --min-conf-publish {b['thr']:.2f}"]
    return "\n".join(L)


TEMPLATE_HEADER = """# Разметка видео для `scenelive eval`.
# Формат строки:  секунда, класс, сколько штук      (например:  4.0, person, 2)
# Секунда = число из имени кадра: t_0004.0.jpg это секунда 4.0. Класс пишите по-английски:
#   person, car, bus, truck, bicycle, motorcycle, pole, door, wall, building, tree, fence, traffic light, bench, trash can, stairs ...
# Правило: впишите ВСЕ предметы тех классов, которые вас интересуют и которые видны на кадре в эту секунду.
#   Классы, которых нет ни в одной строке файла, не оцениваются.
# Строки, начинающиеся с #, программа пропускает. Если на кадре нужных предметов нет, оставьте секунду без строк.
"""


def make_template(run_dir: str, every: float = 4.0, prefill: bool = False, conf: float = 0.30) -> str:
    """Text of a labels file with one section per moment (every `every` seconds of the video).
    prefill=True also writes what the system found there, as a starting point: CHECK it against the frame, otherwise the
    score only measures how much you agree with the system."""
    frames, tracks = _read(run_dir)
    if not frames: raise ValueError(f"{run_dir}: no frames in frames.csv")
    tmax = max(t for _, t in frames); out = [TEMPLATE_HEADER]; t = 0.0
    while t <= tmax + 1e-9:
        k, tf = min(frames, key=lambda f: abs(f[1] - t))
        out.append(f"# ---------- секунда {t:.1f}   (кадр t_{t:06.1f}.jpg) ----------")
        if prefill:
            c = Counter(x["name"] for x in tracks.get(k, []) if x["conf"] >= conf)
            for name, n in sorted(c.items()): out.append(f"{t:.1f}, {name}, {n}   # ПРОВЕРЬТЕ по кадру: это находка программы")
        else:
            out.append(f"# {t:.1f}, person, 2      <- пример: уберите # в начале и впишите, что видно на кадре")
        out.append("")
        t += every
    return "\n".join(out)
