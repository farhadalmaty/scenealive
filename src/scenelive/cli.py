"""scenelive command line: demo | replay | bench"""
from __future__ import annotations
import argparse
import os
import sys
import time


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def server_config(args):
    from .server.config import ServerConfig
    cfg = ServerConfig.from_env(host=args.host, port=args.port, cert_dir=args.cert_dir, detector=args.detector, weights=args.weights, device=args.device, imgsz=args.imgsz,
                                conf=args.conf, floor_conf=getattr(args, "floor_conf", 0.15), map_interval_s=args.map_interval, max_sessions=args.max_sessions, agent=args.agent, model=args.model, token=args.token)
    cfg.seg = getattr(args, "seg", "none"); cfg.seg_size = getattr(args, "seg_size", 0); cfg.seg_every_s = getattr(args, "seg_every", 1.0); cfg.camera_height_m = getattr(args, "camera_height", 1.4)
    cfg.https = not args.no_https; cfg.vocab = getattr(args, "vocab", "extended"); cfg.ground = getattr(args, "ground", "none")
    cfg.alerts = not getattr(args, "no_alerts", False); cfg.alert_range_m = getattr(args, "alert_range", 6.0)
    return cfg


def agent_status(cfg) -> str:
    if cfg.agent == "offline": return "offline (keyword answers in Russian; no key needed)"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "offline: ANTHROPIC_API_KEY is not set on this computer. For Claude: set it and run  windows\\setup_server.bat" if cfg.agent == "auto" else "REQUESTED Claude but ANTHROPIC_API_KEY is not set: the chat will fall back to offline"
    try: import anthropic  # noqa: F401
    except ImportError: return "offline: the 'anthropic' package is not installed (pip install anthropic)"
    return f"Claude ({cfg.model})"


def _run_ok(run_dir: str) -> bool:
    """The run folder must hold frames.csv and tracks.csv (older versions did not write tracks.csv)."""
    miss = [f for f in ("frames.csv", "tracks.csv") if not os.path.exists(os.path.join(run_dir, f))]
    if miss:
        print(f"В папке прогона {run_dir} нет файлов: {', '.join(miss)}.\nЭтот прогон сделан старой версией или папка указана неверно.\nСписок прогонов: dir runs. Запустите replay заново новой версией и укажите его папку."); return False
    return True


def _common(p):
    p.add_argument("--detector", choices=["yolo", "world", "blobs"], default="world", help="yolo = COCO boxes (fast), world = open vocabulary (poles, walls, doors), blobs = demo, no model")
    p.add_argument("--weights", default=None); p.add_argument("--device", default=None, help="cpu | cuda | mps")
    p.add_argument("--vocab", default="extended", help="words for the world detector: core (28), extended (the whole catalogue) or a text file")
    p.add_argument("--imgsz", type=int, default=640, help="detector input size (smaller = faster)"); p.add_argument("--conf", type=float, default=0.30, help="a track may START at this confidence")
    p.add_argument("--floor-conf", type=float, default=0.15, help="the detector reports down to this confidence; weaker detections only keep an existing track alive")
    p.add_argument("--fov", type=float, default=None, help="camera field of view in degrees across the long side"); p.add_argument("--camera", choices=["main", "ultrawide", "tele"], default=None)
    p.add_argument("--compass", default="visual", help="visual | zero | path/to/heading.csv (lines: t_seconds,yaw_degrees)")
    p.add_argument("--fps", type=float, default=3.0, help="frames per second sent from the phone")
    p.add_argument("--uplink-ms", type=float, default=100.0); p.add_argument("--downlink-ms", type=float, default=80.0)
    p.add_argument("--map-interval", type=float, default=6.0, help="seconds between map loop runs")
    p.add_argument("--shared-cpu", action="store_true", help="the map loop blocks the fast loop (one processor)")
    p.add_argument("--no-latest-wins", action="store_true", help="queue every frame instead of dropping old ones when behind")
    p.add_argument("--compute-scale", type=float, default=1.0, help="x measured server compute: 3 = three times slower, 0.1 = ten times faster")
    p.add_argument("--map-compute-scale", type=float, default=1.0); p.add_argument("--client-factor", type=float, default=1.0, help="x phone-side time (phones are slower than this computer)")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--ground", choices=["none", "clip", "colors"], default="none", help="surface under the feet (road, asphalt, bad road): clip = zero-shot CLIP once a second")
    p.add_argument("--min-conf-publish", type=float, default=None, help="show a name only above this confidence (default 0.30)")
    p.add_argument("--publish-hits", type=int, default=None, help="detections needed before a name is shown (default 2; kills one-frame flicker)")
    p.add_argument("--publish-single-conf", type=float, default=None, help="a single detection this sure is shown at once (default 0.55)")
    p.add_argument("--confirm-conf", type=float, default=None, help="confidence for status 'confirmed' (default 0.40)")
    p.add_argument("--confirm-hits", type=int, default=None, help="detections for status 'confirmed' (default 3)")
    p.add_argument("--max-age", type=int, default=None, help="frames a track survives without a detection (default 6)")
    p.add_argument("--iou-match", type=float, default=None, help="minimum overlap to continue a track (default 0.25)")
    p.add_argument("--az-merge", type=float, default=None, help="degrees: map merges same-class tracks closer than this (default 6)")
    p.add_argument("-o", "--out", default=None, help="output folder (default runs/<time>)")
    p.add_argument("--out-video", action="store_true", help="write an overlay video of the processed frames")


def _build(args, video):
    from .camera import FOV_PRESETS
    from .compass import make_compass
    from .config import CameraConfig, FastConfig, MapConfig, SimConfig
    from .detect import make_detector
    fov = args.fov or (FOV_PRESETS[args.camera] if args.camera else 69.0)
    from .labels import load_vocab
    det = make_detector(args.detector, args.device, args.imgsz, args.floor_conf, args.weights, load_vocab(args.vocab) if args.detector == "world" else None)
    pick = lambda **kw: {k: v for k, v in kw.items() if v is not None}                 # only the options the user actually gave
    fast = FastConfig(imgsz=args.imgsz, conf=args.floor_conf, start_conf=args.conf, **pick(min_conf_publish=args.min_conf_publish, publish_hits=args.publish_hits,
                      publish_single_conf=args.publish_single_conf, confirm_conf=args.confirm_conf, confirm_hits=args.confirm_hits,
                      max_age_frames=args.max_age, iou_match=args.iou_match))
    mapc = MapConfig(interval_s=args.map_interval, **pick(az_merge_deg=args.az_merge))
    return (det, CameraConfig(fov), fast, mapc,
            SimConfig(sample_fps=args.fps, uplink_ms=args.uplink_ms, downlink_ms=args.downlink_ms, shared_cpu=args.shared_cpu,
                      latest_wins=not args.no_latest_wins, compute_scale=args.compute_scale, map_compute_scale=args.map_compute_scale,
                      client_factor=args.client_factor, max_seconds=args.max_seconds, send_long_side=max(args.imgsz, 480)),
            make_compass(args.compass))


def _run(args, video, out):
    from .report import format_report, write_outputs
    from .sim import run_replay
    det, cam, fast, mp, sim, comp = _build(args, video)
    print(f"detector: {det.name}; loading done. Replaying {video} ...", flush=True)
    from .ground import make_ground
    try: gr = make_ground(args.ground, args.device)
    except ImportError as ex: print(f"--ground {args.ground}: {ex}"); raise SystemExit(2)
    res = run_replay(video, det, cam, fast, mp, sim, comp, getattr(args, "rotate", None), os.path.join(out, "overlay.mp4") if args.out_video else None,
                     lambda t, k, n: print(f"\r  video {t:6.1f}s   processed {k}/{n}", end="", flush=True), ground=gr)
    print("\n")
    write_outputs(res, out)
    print(format_report(res)); print(f"\nfiles written to {out}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scenelive", description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("demo", help="synthetic street video with known truth, no models needed: checks the whole pipeline and prints timings")
    _common(d); d.set_defaults(detector="blobs")
    r = sp.add_parser("replay", help="play a recorded video as a live stream and time every stage")
    r.add_argument("video"); r.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None); _common(r)
    b = sp.add_parser("bench", help="detector speed on this machine at several input sizes")
    b.add_argument("video"); b.add_argument("--detector", choices=["yolo", "world", "blobs"], default="world"); b.add_argument("--weights", default=None)
    b.add_argument("--device", default=None); b.add_argument("--vocab", default="extended", help="core | extended | file: the speed depends on it, measure it"); b.add_argument("--imgsz", default="320,480,640", help="comma-separated sizes"); b.add_argument("--frames", type=int, default=20)
    e = sp.add_parser("eval", help="score a run against your labels of the video and suggest the confidence threshold")
    e.add_argument("run_dir"); e.add_argument("labels"); e.add_argument("--tol", type=float, default=0.5, help="max seconds between a labeled moment and the nearest processed frame")
    sv = sp.add_parser("serve", help="web version: the page for the phone (camera, list, chat) and the server")
    sv.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from the phone on the same Wi-Fi; 127.0.0.1 = this computer only")
    sv.add_argument("--port", type=int, default=8443); sv.add_argument("--no-https", action="store_true", help="plain http: the phone camera will NOT work (localhost only)")
    sv.add_argument("--cert-dir", default="certs"); sv.add_argument("--token", default=None, help="secret in the address (?token=...); generated automatically for LAN use")
    sv.add_argument("--detector", choices=["yolo", "world", "blobs"], default="world"); sv.add_argument("--weights", default=None); sv.add_argument("--device", default=None)
    sv.add_argument("--imgsz", type=int, default=640); sv.add_argument("--conf", type=float, default=0.30, help="a track may START at this confidence"); sv.add_argument("--floor-conf", type=float, default=0.15, help="weaker detections only keep a track alive"); sv.add_argument("--map-interval", type=float, default=1.0)
    sv.add_argument("--vocab", default="extended", help="detector vocabulary: core (28 words), extended (everything in the catalogue: road, trash, weapons, animals ...) or a text file with one word per line")
    sv.add_argument("--ground", choices=["none", "clip", "colors"], default="none", help="surface under the feet (road, asphalt, bad road); clip needs the CLIP package that YOLO-World installs")
    sv.add_argument("--seg", choices=["none", "segformer", "colors"], default="none", help="floor / walls / road segmentation for 'where to go' (segformer needs transformers: windows\\setup_seg.bat)")
    sv.add_argument("--seg-size", type=int, default=0, help="segmentation input, long side in pixels (448 is faster; 0 = model default)"); sv.add_argument("--seg-every", type=float, default=1.0, help="seconds between segmentations")
    sv.add_argument("--camera-height", type=float, default=1.4, help="metres: height of the phone above the floor (distances to obstacles scale with it)")
    sv.add_argument("--no-alerts", action="store_true", help="switch off the spoken/visual warnings"); sv.add_argument("--alert-range", type=float, default=6.0, help="metres: close vehicles ahead are announced within this range")
    sv.add_argument("--max-sessions", type=int, default=2); sv.add_argument("--agent", choices=["auto", "claude", "offline"], default="auto", help="auto: Claude when ANTHROPIC_API_KEY is set")
    sv.add_argument("--model", default=None, help="Claude model for the chat agent (default claude-sonnet-5-5, or SCENELIVE_MODEL)")
    cs = sp.add_parser("client-sim", help="a simulated phone: streams a recorded video to a running server over a real WebSocket and measures round trips")
    cs.add_argument("video"); cs.add_argument("--url", default="https://localhost:8443/", help="server address; add ?token=... if the server printed one")
    cs.add_argument("--fps", type=float, default=3.0); cs.add_argument("--camera", choices=["main", "ultrawide", "tele"], default="main")
    cs.add_argument("--yaw-csv", default=None, help="heading log (t,yaw) to send like a phone compass; without it the server estimates the heading from the video")
    cs.add_argument("--chat", action="append", default=[], help="a question for the agent after the video (can be repeated)")
    cs.add_argument("--fast", action="store_true", help="do not wait in real time"); cs.add_argument("--max-seconds", type=float, default=None); cs.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None)
    bs = sp.add_parser("bench-seg", help="measure a segmentation model (floor, walls, doors, stairs, glass) on your video and save overlay pictures to LOOK at")
    bs.add_argument("video"); bs.add_argument("--model", choices=["segformer", "colors"], default="segformer", help="segformer = SegFormer-B0 (needs transformers); colors = a crude demo")
    bs.add_argument("--device", default=None); bs.add_argument("--size", type=int, default=0, help="model input: long side in pixels, aspect kept (e.g. 448 is faster); 0 = the model default 512x512, measured at 343 ms"); bs.add_argument("--frames", type=int, default=12); bs.add_argument("--long-side", type=int, default=640); bs.add_argument("-o", "--out", default=None)
    vc = sp.add_parser("vocab", help="show the catalogue of everything that can be named; write a vocabulary file to edit (--out)")
    vc.add_argument("--out", default=None, help="write the extended vocabulary to this text file (one word per line, grouped by category) and use it with --vocab file")
    vc.add_argument("--category", default=None, help="only this category (people, vehicle, animal, structure, indoor, nature, furniture, obstacle, hazard, weapon, trash, surface)")
    lt = sp.add_parser("labels-template", help="write a labels.csv skeleton with one section per moment, to fill in by looking at the snapshots")
    lt.add_argument("run_dir"); lt.add_argument("--every", type=float, default=4.0, help="seconds between moments (use the same value as for snapshots)")
    lt.add_argument("-o", "--out", default="labels.csv"); lt.add_argument("--force", action="store_true", help="overwrite an existing file")
    lt.add_argument("--prefill", action="store_true", help="also write what the system found (to be CHECKED against the frames)")
    s = sp.add_parser("snapshots", help="cut frames out of the video with their time in the file name, to label them")
    s.add_argument("video"); s.add_argument("--every", type=float, default=4.0, help="seconds between snapshots"); s.add_argument("-o", "--out", default=None)
    s.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None)
    args = ap.parse_args(argv)

    if args.cmd == "bench-seg":
        from .segment import bench_segmenter, make_segmenter
        out = args.out or os.path.join("runs", "seg_" + _stamp())
        try:
            seg = make_segmenter(args.model, args.device, args.size); print(f"model: {seg.name}; the first start downloads it", flush=True)
            bench_segmenter(args.video, seg, out, args.frames, args.long_side)
        except ImportError as ex: print(f"Cannot run: {ex}"); return 2
        except (FileNotFoundError, ValueError) as ex: print(f"Cannot run: {ex}"); return 2
        print(f"overlay pictures (green = sidewalk or floor, brown = ROAD, light green = grass or earth, blue = wall, orange = door, yellow = stairs, cyan = glass, red = moving, magenta = obstacle) are in {out}")
        return 0
    if args.cmd == "vocab":
        from .catalog import CATEGORY, CATEGORY_RU, ENTRIES, RU_NAME, SEVERITY
        from .labels import EXTENDED_VOCAB
        cats = list(CATEGORY_RU)
        if args.category and args.category not in cats: print(f"Unknown category {args.category!r}. Choose from: {', '.join(cats)}"); return 2
        lines = []
        for cat in ([args.category] if args.category else cats):
            ns = [e for e in ENTRIES if e[2] == cat]
            print(f"\n{CATEGORY_RU[cat]} ({cat}), {len(ns)}:"); lines.append(f"# {CATEGORY_RU[cat]} ({cat})")
            for n, ru, _, sev, mv in ns: print(f"  {n:<22}{ru:<26}{sev}" + ("  moves" if mv else "")); lines.append(n)
        extra = [w for w in EXTENDED_VOCAB if w not in RU_NAME]
        if extra and not args.category: lines.append("# other words of the first vocabulary"); lines += extra
        print(f"\n{len(EXTENDED_VOCAB)} words in the extended vocabulary; the core vocabulary has 28.")
        if args.out: open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print(f"written to {args.out}: delete the words you do not need and run  windows\\serve.bat --vocab {args.out}")
        return 0
    if args.cmd == "serve":
        cfg = server_config(args)
        try:
            import numpy as np
            from .detect import make_detector
            from .server.runner import make_ssl_context, run_server
            from .ground import make_ground
            from .labels import load_vocab
            vocab = load_vocab(cfg.vocab)
            print(f"loading the detector '{cfg.detector}', vocabulary '{cfg.vocab}' ({len(vocab)} words) (the first start downloads the weights: it can take a few minutes) ...", flush=True)
            det = make_detector(cfg.detector, cfg.device, cfg.imgsz, cfg.floor_conf, cfg.weights, vocab)
            gr = make_ground(cfg.ground, cfg.device)
            from .segment import make_segmenter
            sg = make_segmenter(cfg.seg, cfg.device, cfg.seg_size) if cfg.seg != "none" else None
            try: det.detect(np.zeros((360, 640, 3), np.uint8))                      # warm-up: the first real frame must not pay for it
            except Exception: pass
            if cfg.https: make_ssl_context(cfg)                                    # fail early and clearly if https cannot be set up
        except (FileNotFoundError, ValueError) as ex:
            print(f"Cannot start: {ex}"); return 2
        except ImportError as ex:
            print(f"A package is missing: {ex}\nInstall:  windows\\setup_server.bat   (cryptography for https) and  windows\\setup_models.bat  (ultralytics for the detector)"); return 2
        print(f"agent: {agent_status(cfg)}", flush=True); print(f"surface recognition: {cfg.ground}   free space / where to go: {cfg.seg}   warnings: {'on' if cfg.alerts else 'off'}", flush=True)
        run_server(cfg, det, gr, sg); return 0
    if args.cmd == "client-sim":
        import asyncio
        from .server.client_sim import format_summary, run_client
        try: s = asyncio.run(run_client(args.url, args.video, args.fps, args.camera, args.yaw_csv, not args.fast, args.chat, args.max_seconds, rotate=args.rotate, log=print))
        except (ConnectionError, OSError) as ex: print(f"Could not talk to the server: {ex}\nIs it running?  windows\\serve.bat  (and the address, including ?token=..., must match what it printed)"); return 2
        print(format_summary(s)); return 0
    if args.cmd == "labels-template":
        from .evaluate import make_template
        if os.path.exists(args.out) and not args.force:
            print(f"Файл {args.out} уже есть, я его не трогаю (чтобы не стереть вашу разметку). Другое имя: -o имя.csv, перезаписать: --force."); return 2
        if not _run_ok(args.run_dir): return 2
        txt = make_template(args.run_dir, args.every, args.prefill)
        open(args.out, "w", encoding="utf-8").write(txt)
        print(f"Заготовка записана: {os.path.abspath(args.out)}\nОткройте её в Блокноте, посмотрите кадры из `snapshots` и впишите строки вида:  4.0, person, 2\nПотом: windows\\eval.bat {args.run_dir} {args.out}")
        return 0
    if args.cmd == "eval":
        from .evaluate import evaluate, format_eval
        if not os.path.exists(args.labels):
            print(f"Файл разметки не найден: {args.labels}\n(папка, где ищу: {os.getcwd()})\nСоздайте заготовку командой:  windows\\labels_template.bat {args.run_dir}\nи заполните её (подробности в README, раздел «Подбор порогов»)."); return 2
        if not _run_ok(args.run_dir): return 2
        try: res = evaluate(args.run_dir, args.labels, args.tol)
        except ValueError as ex: print(f"Разметка пуста или не прочитана: {ex}\nВ файле должны быть строки вида:  4.0, person, 2  (строки с # пропускаются)."); return 2
        txt = format_eval(res); print(txt)
        open(os.path.join(args.run_dir, "eval.txt"), "w", encoding="utf-8").write(txt + "\n"); return 0
    if args.cmd == "snapshots":
        import cv2
        from .video_io import open_video
        out = args.out or os.path.join("runs", "snapshots_" + _stamp()); os.makedirs(out, exist_ok=True)
        rd = open_video(args.video, args.rotate); fps = float(rd.get(cv2.CAP_PROP_FPS) or 30.0); i = 0; nxt = 0.0; n = 0
        while True:
            ok, fr = rd.read()
            if not ok: break
            t = i / fps; i += 1
            if t + 1e-9 < nxt: continue
            nxt += args.every
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90]); buf.tofile(os.path.join(out, f"t_{t:06.1f}.jpg")); n += 1
        print(f"{n} snapshots written to {out}. Name = second of the video (t_0012.0.jpg is second 12). Write labels like: 12.0, person, 2")
        return 0

    if args.cmd == "demo":
        from .synth import render_video, truth_table
        out = args.out or os.path.join("runs", "demo_" + _stamp()); os.makedirs(out, exist_ok=True)
        video = os.path.join(out, "synthetic_street.mp4"); truth = render_video(video, fov=args.fov or 69.0)
        res = _run(args, video, out)
        from .camera import angdiff
        t_end = res.summary["video_seconds"]; yaw_true = 4.0 * t_end; yaw_est = res.frames[-1].yaw
        print(f"\nCOMPASS: after {t_end:.1f} s the visual compass says {yaw_est:.1f} deg, truth {yaw_true:.1f} deg (error {yaw_est - yaw_true:+.1f} deg, {100 * abs(yaw_est - yaw_true) / yaw_true:.1f}%)")
        print("GROUND TRUTH vs what the phone would show (bearing = degrees clockwise from the initial heading):")
        by = {}
        for v in res.final_view: by.setdefault(v["name"], v)
        for row in truth_table(truth):
            v = by.get(row["name"])
            if v is None: print(f"  {row['name']:<8} truth {row['bearing_deg']:5.0f} deg, {row['dist_m']:4.0f} m   NOT FOUND"); continue
            err = "moving" if row["moving"] else f"bearing error {angdiff(v['world_az_deg'], row['bearing_deg']):+.1f} deg"
            print(f"  {row['name']:<8} truth {row['bearing_deg']:5.0f} deg, {row['dist_m']:4.0f} m   shown: {v['world_az_deg']:6.1f} deg, {v['range']:<9} {err}")
        return 0
    if args.cmd == "replay":
        out = args.out or os.path.join("runs", _stamp()); os.makedirs(out, exist_ok=True)
        try: _run(args, args.video, out)
        except (FileNotFoundError, ValueError) as ex: print(f"Cannot run: {ex}"); return 2
        return 0
    if args.cmd == "bench":
        from .bench import bench_detector
        from .detect import make_detector
        from .labels import load_vocab
        sizes = [int(x) for x in args.imgsz.split(",") if x.strip()]
        print(f"{'imgsz':>6} {'mean ms':>9} {'p50':>8} {'p95':>8} {'fps':>7} {'detections':>11}")
        bench_detector(args.video, lambda sz: make_detector(args.detector, args.device, sz, 0.3, args.weights, load_vocab(args.vocab) if args.detector == "world" else None), sizes, args.frames,
                       progress=lambda r: print(f"{r['imgsz']:>6} {r['mean_ms']:>9.1f} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f} {r['fps']:>7.1f} {r['detections']:>11.1f}", flush=True))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
