"""A 'phone' that streams a recorded video to a running server over a real WebSocket: test the server (and measure real round-trip
times) on your own video without touching a phone. Same protocol as the web page."""
from __future__ import annotations
import asyncio
import base64
import json
import os
import ssl
import time
from urllib.parse import parse_qs, urlsplit
import cv2
import numpy as np
from .ws import WebSocket, accept_key, OP_TEXT


def pack_frame(meta: dict, jpeg: bytes) -> bytes:
    m = json.dumps(meta, separators=(",", ":")).encode(); return len(m).to_bytes(4, "big") + m + jpeg


async def ws_connect(host: str, port: int, path: str, ssl_ctx=None) -> WebSocket:
    r, w = await asyncio.open_connection(host, port, ssl=ssl_ctx, server_hostname=host if ssl_ctx else None, limit=65536)
    key = base64.b64encode(os.urandom(16)).decode()
    w.write((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode()); await w.drain()
    head = (await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 10)).decode("latin-1"); status = int(head.split(" ", 2)[1])
    if status != 101: w.close(); raise ConnectionError(f"the server answered HTTP {status} instead of switching protocols" + (" (wrong or missing token?)" if status == 401 else ""))
    if accept_key(key) not in head: w.close(); raise ConnectionError("bad Sec-WebSocket-Accept")
    return WebSocket(r, w, max_message=10_000_000, is_client=True)


def parse_url(url: str):
    u = urlsplit(url if "://" in url else "https://" + url); https = u.scheme in ("https", "wss")
    token = (parse_qs(u.query).get("token") or [None])[0]
    return u.hostname, u.port or (443 if https else 80), https, token


async def run_client(url: str, video: str, fps: float = 3.0, camera: str = "main", yaw_csv: str | None = None, pace: bool = True, questions=(),
                     max_seconds: float | None = None, quality: int = 70, long_side: int = 640, inflight_max: int = 2, rotate: int | None = None, log=print) -> dict:
    from ..compass import CsvCompass
    from ..video_io import open_video
    host, port, https, token = parse_url(url); ctx = None
    if https: ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE     # self-signed on purpose
    ws = await ws_connect(host, port, "/ws" + (f"?token={token}" if token else ""), ctx)
    heading = CsvCompass(yaw_csv) if yaw_csv else None
    await ws.send_text(json.dumps({"type": "hello", "camera": camera, "has_heading": heading is not None}))
    sent_at: dict[int, float] = {}; rtts: list[float] = []; comp: list[float] = []; last: dict = {}; replies: list = []; errors: list = []; server_dropped = [0]; ready = {}
    got_ready = asyncio.Event()

    async def reader():
        while True:
            msg = await ws.recv()
            if msg is None: return
            m = json.loads(msg[1].decode("utf-8")); typ = m.get("type")
            if typ == "ready": ready.update(m); got_ready.set()
            elif typ == "result":
                s = sent_at.pop(m["seq"], None)
                if s is not None: rtts.append((time.perf_counter() - s) * 1000.0)
                comp.append(m["compute_ms"]); server_dropped[0] += m.get("dropped", 0); last.update(m)
            elif typ == "chat_reply": replies.append(m)
            elif typ == "error": errors.append(m); got_ready.set()

    rt = asyncio.create_task(reader())
    try:
        await asyncio.wait_for(got_ready.wait(), 30)
        if not ready: raise ConnectionError("the server refused the session: " + (errors[-1]["message"] if errors else "unknown reason"))
        log(f"connected: detector {ready.get('detector')}, camera fov {ready.get('fov_long_deg')} deg, compass {ready.get('compass')}, agent {ready.get('agent')}"
            + (f"\n  warning: {ready['agent_warning']}" if ready.get("agent_warning") else ""))
        rd = open_video(video, rotate); vfps = float(rd.get(cv2.CAP_PROP_FPS) or 30.0); i = 0; nxt = 0.0; seq = 0; skipped = 0; t0 = time.perf_counter()
        while True:
            ok, fr = rd.read()
            if not ok: break
            t = i / vfps; i += 1
            if max_seconds is not None and t > max_seconds: break
            if t + 1e-9 < nxt: continue
            while nxt <= t + 1e-9: nxt += 1.0 / fps
            if pace:
                wait = t - (time.perf_counter() - t0)
                if wait > 0: await asyncio.sleep(wait)
            if len(sent_at) >= inflight_max:                                              # like the page: never more than a couple of frames in flight
                if pace: skipped += 1; continue                                           # real time: a late frame is not worth sending
                while len(sent_at) >= inflight_max: await asyncio.sleep(0.005)            # batch mode: send every frame, just wait
            s = long_side / max(fr.shape[:2]); small = cv2.resize(fr, (round(fr.shape[1] * s), round(fr.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else fr
            ok2, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
            meta = {"seq": seq, "t": round(t, 3), "w": small.shape[1], "h": small.shape[0]}
            if heading is not None: meta["yaw"] = heading.update(None, t, None) % 360.0
            sent_at[seq] = time.perf_counter(); await ws.send_binary(pack_frame(meta, buf.tobytes())); seq += 1
        rd.release()
        for _ in range(100):                                                                 # let the last results arrive
            if not sent_at: break
            await asyncio.sleep(0.1)
        for q in questions:
            n = len(replies); await ws.send_text(json.dumps({"type": "chat", "id": f"q{n}", "text": q}))
            for _ in range(300):
                if len(replies) > n: break
                await asyncio.sleep(0.1)
    finally:
        await ws.close(); rt.cancel()
    pct = lambda a, p: float(np.percentile(a, p)) if a else 0.0
    return {"frames_sent": seq, "results": len(rtts), "skipped_by_client": skipped, "dropped_by_server": server_dropped[0],
            "rtt_ms": {"mean": float(np.mean(rtts)) if rtts else 0.0, "p50": pct(rtts, 50), "p95": pct(rtts, 95), "max": max(rtts) if rtts else 0.0},
            "server_compute_ms_mean": float(np.mean(comp)) if comp else 0.0, "final_entries": last.get("entries", []), "final_heading": last.get("heading"),
            "chat": [(r.get("text"), r.get("tools")) for r in replies], "errors": [e["message"] for e in errors], "ready": ready}


def format_summary(s: dict) -> str:
    r = s["rtt_ms"]; L = [f"frames sent {s['frames_sent']}, results {s['results']}, skipped by the client {s['skipped_by_client']}, dropped by the server {s['dropped_by_server']}",
                          f"round trip (frame sent -> list received): mean {r['mean']:.0f} ms, p50 {r['p50']:.0f}, p95 {r['p95']:.0f}, max {r['max']:.0f}",
                          f"server compute per frame: mean {s['server_compute_ms_mean']:.0f} ms"]
    for v in s["final_entries"][:12]: L.append(f"  {v['name']:<14}{v['status']:<12}{('behind' if v['behind'] else str(v['clock']) + ' o' + chr(39) + 'clock'):<12}{v['range']}")
    for q, tools in s["chat"]: L.append(f"agent ({', '.join(tools) or 'no tools'}): {q}")
    for e in s["errors"]: L.append(f"error: {e}")
    return "\n".join(L)
