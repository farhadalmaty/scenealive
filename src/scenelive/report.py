"""Text report and files (stages.csv, frames.csv, events.csv, first_names.csv, report.json, report.txt)."""
from __future__ import annotations
import csv
import json
import os
from .sim import PHONE_STAGES, SERVER_STAGES, MAP_STAGES, SimResult


def _row(r, total):
    share = f"{100 * r['total_s'] / total:5.1f}%" if total else "     -"
    return f"  {r['stage']:<14}{r['n']:>5}{r['mean_ms']:>9.1f}{r['p50_ms']:>9.1f}{r['p95_ms']:>9.1f}{r['max_ms']:>9.1f}{r['total_s']:>9.2f}  {share}"


def format_report(res: SimResult) -> str:
    s = res.summary; rows = {r["stage"]: r for r in res.stage_rows}; L = []
    c = res.config
    L.append(f"detector: {c['detector']}   compass: {c['compass']}   fov: {c['CameraConfig']['fov_long_deg']:g} deg   "
             f"sample: {c['SimConfig']['sample_fps']:g} fps   compute scale: {c['SimConfig']['compute_scale']:g}")
    L.append(f"video: {s['video_seconds']:.1f} s, {res.video.get('size')}")
    L.append("")
    L.append("STAGE TIMING (milliseconds per call; measured on this computer)")
    L.append(f"  {'stage':<14}{'n':>5}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}{'total s':>9}  share")
    L.append("  phone (resize + JPEG):")
    for n in PHONE_STAGES:
        if n in rows: L.append(_row(rows[n], 0))
    srv = [rows[n] for n in SERVER_STAGES if n in rows]; tot = sum(r["total_s"] for r in srv)
    L.append("  server, fast loop (share of the server total):")
    for r in srv: L.append(_row(r, tot))
    if "map_build" in rows:
        L.append("  server, map loop:"); L.append(_row(rows["map_build"], 0))
    if srv:
        top = max(srv, key=lambda r: r["total_s"]); L.append(f"  -> the slowest stage is '{top['stage']}': {100 * top['total_s'] / tot:.0f}% of the server time")
    L.append("")
    L.append("END TO END (capture on the phone -> list shown on the phone)")
    L.append(f"  frames: sent {s['frames_sent']}, processed {s['frames_processed']}, dropped {s['frames_dropped']}")
    L.append(f"  latency per frame: mean {s['latency_mean_s']:.2f} s, p50 {s['latency_p50_s']:.2f}, p95 {s['latency_p95_s']:.2f}, max {s['latency_max_s']:.2f}")
    L.append(f"  server compute per frame: mean {s['compute_mean_s'] * 1000:.0f} ms, p95 {s['compute_p95_s'] * 1000:.0f} ms "
             f"-> can sustain {s['sustainable_fps']:.1f} fps (the phone sends {s['sample_fps']:g})")
    L.append("  verdict: " + ("keeps up with the stream." if s["keeps_up"] else
             "TOO SLOW for this frame rate: frames queue up or are dropped. Lower --fps, lower --imgsz, or use a faster machine."))
    if s["map_jobs"]: L.append(f"  map loop: {s['map_jobs']} runs, mean {s['map_compute_mean_s'] * 1000:.0f} ms, result reaches the phone {s['map_latency_mean_s']:.2f} s after it was due")
    cs = s.get("compass") or {}
    if cs:
        L.append(f"  compass: flow lost in {cs['lost']} of {cs['steps']} steps ({100 * cs['lost'] / max(cs['steps'], 1):.0f}%), yaw ranged {cs['yaw_min']:.0f}..{cs['yaw_max']:.0f} deg")
    L.append("")
    L.append("FIRST NAMES (video second of the first detection -> second the phone shows the name)")
    if res.first_names:
        L.append(f"  {'track':<6}{'name':<14}{'first seen':>11}{'in list':>9}{'latency':>9}{'confirmed':>11}")
        for r in res.first_names[:40]:
            cf = f"{r['confirmed_s']:.1f}" if r["confirmed_s"] is not None else "-"
            L.append(f"  {r['key']:<6}{r['name']:<14}{r['first_seen_s']:>10.1f}s{r['in_list_s']:>8.1f}s{r['latency_s']:>8.2f}s{cf:>10}s")
        L.append(f"  mean {s['first_name_latency_mean_s']:.2f} s, p95 {s['first_name_latency_p95_s']:.2f} s  ({len(res.first_names)} tracks)")
    else:
        L.append("  nothing was detected")
    L.append("")
    L.append(f"LIST AT THE END ({s['objects_in_list']} entries; heading = last camera yaw)")
    for v in res.final_view[:30]:
        where = "behind" if v["behind"] else f"{v['clock']} o'clock"
        L.append(f"  {v['key']:<6}{v['name']:<14}{v['status']:<12}{where:<12}{v['range']:<10} conf {v['conf']:.2f}  obs {v['n_obs']}  ({v['source']})")
    return "\n".join(L)


def write_outputs(res: SimResult, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f: f.write(format_report(res) + "\n")
    with open(os.path.join(out_dir, "stages.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res.stage_rows[0].keys())); w.writeheader(); w.writerows(res.stage_rows)
    stages = sorted({k for fr in res.frames for k in fr.stage_ms})
    with open(os.path.join(out_dir, "frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["k", "t", "t_arr", "start", "finish", "avail", "latency_s", "queue_wait_s", "compute_s", "dropped_before", "n_det", "n_tracks", "yaw"] + [f"{s}_ms" for s in stages])
        for fr in res.frames:
            w.writerow([fr.k, f"{fr.t:.3f}", f"{fr.t_arr:.3f}", f"{fr.start:.3f}", f"{fr.finish:.3f}", f"{fr.avail:.3f}", f"{fr.avail - fr.t:.3f}",
                        f"{fr.queue_wait_s:.3f}", f"{fr.compute_s:.4f}", fr.dropped_before, fr.n_det, fr.n_tracks, f"{fr.yaw:.2f}"] + [f"{fr.stage_ms.get(s, 0.0):.2f}" for s in stages])
    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="", encoding="utf-8") as f:          # one row per track per processed frame
        w = csv.writer(f); w.writerow(["k", "t", "yaw", "track", "name", "conf", "det_conf", "hits", "status", "x0", "y0", "x1", "y1", "frame_w", "frame_h", "az_rel", "world_az", "range_m", "range_sigma"])
        for fr in res.frames:
            for tr in fr.tracks:
                rg = tr["range"]; x0, y0, x1, y1 = tr["box"]
                w.writerow([fr.k, f"{fr.t:.3f}", f"{fr.yaw:.2f}", tr["id"], tr["name"], f"{tr['conf']:.3f}", f"{tr['det_conf']:.3f}", tr["hits"], tr["status"],
                            f"{x0:.0f}", f"{y0:.0f}", f"{x1:.0f}", f"{y1:.0f}", fr.size[0], fr.size[1], f"{tr['az_rel']:.1f}", f"{tr['world_az']:.1f}",
                            "" if rg is None else f"{rg[0]:.2f}", "" if rg is None else f"{rg[1]:.2f}"])
    with open(os.path.join(out_dir, "events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["t_avail", "kind", "key", "name", "first_seen_s"])
        for e in res.events: w.writerow([f"{e.t_avail:.3f}", e.kind, e.key, e.name, f"{e.first_t:.3f}"])
    with open(os.path.join(out_dir, "first_names.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["track", "name", "first_seen_s", "in_list_s", "latency_s", "confirmed_s"])
        for r in res.first_names: w.writerow([r["key"], r["name"], f"{r['first_seen_s']:.3f}", f"{r['in_list_s']:.3f}", f"{r['latency_s']:.3f}", "" if r["confirmed_s"] is None else f"{r['confirmed_s']:.3f}"])
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"config": res.config, "summary": res.summary, "stages": res.stage_rows, "first_names": res.first_names,
                   "final_view": res.final_view, "video": res.video}, f, ensure_ascii=False, indent=2, default=float)
