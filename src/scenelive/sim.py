"""Replay simulator: a recorded video is played as if a phone were filming it live, and the whole system is timed.

Two clocks. Video time is when a frame was captured. Compute time is measured on THIS computer, per stage, with perf_counter().
The simulator replays them like a real deployment:

  phone: resize + JPEG-encode -> uplink -> [server fast worker: decode, preprocess, compass, detect, track, name, locate, list]
                                         -> downlink -> the phone shows the list
  map worker: every `interval_s` of video time, builds the object map from all tracks, result goes back the same way

The fast worker is ONE worker: if it is slower than the frame rate, frames wait (or, with latest_wins, old ones are dropped).
`compute_scale` multiplies the measured compute ("what if the hardware were 3x slower / 10x faster").
With `shared_cpu` the map worker blocks the fast worker, as it would on one processor."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import cv2
import numpy as np
from .camera import Camera
from .config import CameraConfig, FastConfig, MapConfig, SimConfig, to_dict
from .fastpath import FastPath, STAGES
from .mapper import build_map
from .objects import ObjectList
from .timing import Timer
from .video_io import open_video

PHONE_STAGES = ["client_resize", "client_encode"]
SERVER_STAGES = ["server_decode"] + STAGES
MAP_STAGES = ["map_build"]


@dataclass
class FrameRecord:
    k: int
    t: float                 # capture time (video seconds)
    t_arr: float             # arrived at the server
    start: float             # server started working on it
    finish: float            # server finished
    avail: float             # the phone can show the result
    queue_wait_s: float
    compute_s: float
    dropped_before: int
    n_det: int
    n_tracks: int
    yaw: float
    stage_ms: dict
    tracks: list = field(default_factory=list)
    size: tuple = (0, 0)


@dataclass
class MapRecord:
    due: float
    start: float
    finish: float
    avail: float
    compute_s: float
    n_objects: int


@dataclass
class SimResult:
    config: dict
    frames: list
    maps: list
    stage_rows: list
    events: list
    final_view: list
    summary: dict
    first_names: list
    timer: Timer
    video: dict = field(default_factory=dict)


def _pct(a, q): return float(np.percentile(a, q)) if len(a) else 0.0


def run_replay(video, detector, cam_cfg: CameraConfig, fast_cfg: FastConfig, map_cfg: MapConfig, sim_cfg: SimConfig, compass,
               rotate: int | None = None, out_video: str | None = None, progress=None, ground=None, ground_every_s: float = 1.0) -> SimResult:
    timer = Timer()
    # ---- phone side: sample frames, resize, JPEG-encode (this is what goes over the network)
    rd = open_video(video, rotate); fps = float(rd.get(cv2.CAP_PROP_FPS) or 30.0)
    samples = []; step = 1.0 / sim_cfg.sample_fps; next_t = 0.0; i = 0; vsize = None
    while True:
        ok, fr = rd.read()
        if not ok: break
        t = i / fps; i += 1
        if sim_cfg.max_seconds is not None and t > sim_cfg.max_seconds: break
        if t + 1e-9 < next_t: continue
        while next_t <= t + 1e-9: next_t += step
        vsize = vsize or (fr.shape[1], fr.shape[0])
        t0 = time.perf_counter()
        s = sim_cfg.send_long_side / max(fr.shape[:2])
        small = cv2.resize(fr, (int(round(fr.shape[1] * s)), int(round(fr.shape[0] * s))), interpolation=cv2.INTER_AREA) if s < 1.0 else fr
        t1 = time.perf_counter()
        ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, sim_cfg.jpeg_quality])
        t2 = time.perf_counter()
        timer.add("client_resize", t1 - t0); timer.add("client_encode", t2 - t1)
        samples.append((t, buf.tobytes(), (t2 - t0) * sim_cfg.client_factor))
    rd.release()
    if not samples: raise ValueError("no frames were read from the video")
    end_t = samples[-1][0]

    # ---- server side: event loop on virtual time
    from .ground import GroundMonitor
    fast = FastPath(detector, cam_cfg, fast_cfg, compass, timer, GroundMonitor(ground, ground_every_s) if ground is not None else None)
    up, down = sim_cfg.uplink_ms / 1000.0, sim_cfg.downlink_ms / 1000.0
    arr = [t + cs + up for t, _, cs in samples]
    order = sorted(range(len(samples)), key=lambda j: arr[j])
    w = 0.0; map_free = 0.0; map_due = map_cfg.interval_s; ptr = 0; waiting: list[int] = []; dropped_total = 0
    frames: list[FrameRecord] = []; maps: list[MapRecord] = []; updates: list = []
    writer = None; last_yaw = 0.0

    def run_map(due: float):
        nonlocal w, map_free
        start = max(due, w if sim_cfg.shared_cpu else map_free)
        t0 = time.perf_counter(); entries = build_map(fast.tracker.all_tracks(), start, map_cfg, fast_cfg); dt = time.perf_counter() - t0
        timer.add("map_build", dt); c = dt * sim_cfg.map_compute_scale; finish = start + c
        if sim_cfg.shared_cpu: w = finish
        else: map_free = finish
        updates.append((finish + down, 1, "map", entries)); maps.append(MapRecord(due, start, finish, finish + down, c, len(entries)))

    while True:
        while ptr < len(order) and arr[order[ptr]] <= w: waiting.append(order[ptr]); ptr += 1
        if not waiting:
            if ptr >= len(order): break
            w = max(w, arr[order[ptr]]); continue
        dropped = 0
        if sim_cfg.latest_wins and len(waiting) > 1:
            dropped = len(waiting) - 1; chosen = max(waiting, key=lambda j: samples[j][0]); waiting.clear()
        else:
            chosen = waiting.pop(0)
        dropped_total += dropped
        while map_due <= w and map_due <= end_t: run_map(map_due); map_due += map_cfg.interval_s      # map jobs that start before this frame
        start = max(w, arr[chosen]); t_cap, jpg, _ = samples[chosen]
        tim: dict[str, float] = {}
        with timer.stage("server_decode", tim):
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        res = fast.process(frame, t_cap, chosen)
        tim.update(res.timings)
        compute = sum(tim.values()) * sim_cfg.compute_scale; finish = start + compute; w = finish; last_yaw = res.yaw
        updates.append((finish + down, 0, "fast", res.entries))
        frames.append(FrameRecord(chosen, t_cap, arr[chosen], start, finish, finish + down, start - arr[chosen], compute, dropped,
                                  len(res.detections), len(res.tracks), res.yaw, {k: v * 1000.0 for k, v in tim.items()}, res.tracks, res.size))
        if out_video:
            from .render import draw
            small = cv2.resize(frame, res.size) if (frame.shape[1], frame.shape[0]) != res.size else frame
            img = draw(small, res, finish + down - t_cap)
            if writer is None:
                import os
                os.makedirs(os.path.dirname(os.path.abspath(out_video)), exist_ok=True)
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), sim_cfg.sample_fps, (img.shape[1], img.shape[0]))
            writer.write(img)
        if progress and len(frames) % 5 == 0: progress(t_cap, len(frames), len(samples))
    while map_due <= end_t: run_map(map_due); map_due += map_cfg.interval_s
    run_map(max(w, end_t))                                                                    # final map at the end of the video
    if writer: writer.release()

    # ---- the phone's view: apply every update in the order the phone receives it
    olist = ObjectList()
    for avail, _, kind, payload in sorted(updates, key=lambda u: (u[0], u[1])):
        (olist.apply_fast if kind == "fast" else olist.apply_map)(payload, avail)

    first = {}
    for e in olist.events:
        if e.kind in ("new", "map_new") and e.key not in first: first[e.key] = e
    confirmed = {e.key: e for e in olist.events if e.kind == "confirmed"}
    first_names = sorted(({"key": k, "name": e.name, "first_seen_s": e.first_t, "in_list_s": e.t_avail, "latency_s": e.t_avail - e.first_t,
                           "confirmed_s": (confirmed[k].t_avail if k in confirmed else None)} for k, e in first.items() if k.startswith("T")),
                         key=lambda r: r["first_seen_s"])

    lat = np.array([f.avail - f.t for f in frames]); comp = np.array([f.compute_s for f in frames]); qw = np.array([f.queue_wait_s for f in frames])
    server_stage_total = sum(sum(timer.samples[s]) for s in SERVER_STAGES if s in timer.samples)
    summary = {
        "video_seconds": end_t, "frames_sent": len(samples), "frames_processed": len(frames), "frames_dropped": dropped_total,
        "latency_mean_s": float(lat.mean()), "latency_p50_s": _pct(lat, 50), "latency_p95_s": _pct(lat, 95), "latency_max_s": float(lat.max()),
        "queue_wait_mean_s": float(qw.mean()), "compute_mean_s": float(comp.mean()), "compute_p95_s": _pct(comp, 95),
        "sustainable_fps": float(1.0 / comp.mean()) if comp.mean() > 0 else float("inf"), "sample_fps": sim_cfg.sample_fps,
        "keeps_up": bool(comp.mean() <= 1.0 / sim_cfg.sample_fps), "server_compute_total_s": server_stage_total,
        "map_jobs": len(maps), "map_compute_mean_s": float(np.mean([m.compute_s for m in maps])) if maps else 0.0,
        "map_latency_mean_s": float(np.mean([m.avail - m.due for m in maps])) if maps else 0.0,
        "first_name_latency_mean_s": float(np.mean([r["latency_s"] for r in first_names])) if first_names else None,
        "first_name_latency_p95_s": _pct([r["latency_s"] for r in first_names], 95) if first_names else None,
        "objects_in_list": len(olist.entries()), "compass": compass.stats() if hasattr(compass, "stats") else {},
    }
    return SimResult(to_dict(cam_cfg, fast_cfg, map_cfg, sim_cfg) | {"detector": getattr(detector, "name", "?"),
                     "compass": getattr(compass, "name", "?")}, frames, maps, timer.table(PHONE_STAGES + SERVER_STAGES + MAP_STAGES),
                     olist.events, olist.view(last_yaw), summary, first_names, timer, {"fps": fps, "size": vsize})
