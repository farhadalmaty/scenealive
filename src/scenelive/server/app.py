"""HTTP + WebSocket server on asyncio streams.

  GET /                 the web page (camera, list, chat)        GET /api/health   status (no secrets)
  GET /app.js /style.css                                          GET /api/stats    timings of live sessions (token)
  GET /ws?token=...     WebSocket: hello, JPEG frames, chat

Frames: the page sends the newest frame only; if the server is still busy with the previous one, older frames are dropped
(latest wins), so the delay never grows. The heavy work runs in ONE worker thread, shared by all sessions."""
from __future__ import annotations
import asyncio
import hmac
import json
import math
import os
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit
from .. import __version__
from ..catalog import SEVERITY
from .agent import EN_RU, RuleAgent, make_agent
from .config import ServerConfig
from .session import LiveSession
from .ws import OP_BIN, OP_TEXT, ProtocolError, WebSocket, accept_key

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "application/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; "
       "connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
REASONS = {200: "OK", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed", 431: "Request Header Fields Too Large", 500: "Internal Server Error"}


def unpack_frame(data: bytes) -> tuple[dict, bytes]:
    """[uint32 big-endian N][N bytes of JSON meta][JPEG bytes]  ->  (meta, jpeg). Raises ValueError on anything malformed."""
    if len(data) < 6: raise ValueError("frame message too short")
    n = int.from_bytes(data[:4], "big")
    if n > 4096 or 4 + n >= len(data): raise ValueError("bad meta length")
    try: meta = json.loads(data[4:4 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError): raise ValueError("meta is not valid JSON")
    if not isinstance(meta, dict): raise ValueError("meta must be an object")
    clean: dict = {}
    for k, kind in (("seq", int), ("t", float), ("yaw", float), ("pitch", float)):
        v = meta.get(k)
        if v is None: continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v): raise ValueError(f"meta.{k} must be a finite number")
        clean[k] = kind(v)
    return clean, data[4 + n:]


class ServerApp:
    def __init__(self, cfg: ServerConfig, detector, agent_client=None, log=None, ground_classifier=None, segmenter=None, ocr_engine=None):
        self.cfg, self.detector, self.agent_client, self.ground_classifier, self.segmenter, self.ocr_engine = cfg, detector, agent_client, ground_classifier, segmenter, ocr_engine
        self.log = log or (lambda *a: print(*a, file=sys.stderr, flush=True))
        self.sessions: dict[str, LiveSession] = {}; self.executor = ThreadPoolExecutor(max_workers=1); self.started = time.time()
        self.assets = {p: (open(os.path.join(WEB_DIR, f), "rb").read(), ct) for p, (f, ct) in ASSETS.items()}

    # ----------------------------------------------------------------------------------------------- HTTP
    async def _respond(self, w, status: int, body: bytes = b"", ctype: str = "text/plain; charset=utf-8", extra: dict | None = None, length: int | None = None):
        h = {"Content-Type": ctype, "Content-Length": str(len(body) if length is None else length), "Connection": "close", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
             "Content-Security-Policy": CSP, "Permissions-Policy": "camera=(self), microphone=(self), geolocation=()", **(extra or {})}
        w.write((f"HTTP/1.1 {status} {REASONS.get(status, 'OK')}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n").encode("latin-1") + body)
        try: await w.drain()
        except ConnectionError: pass

    def _authorised(self, query: dict) -> bool:
        if not self.cfg.token: return True
        given = (query.get("token") or [""])[0]
        return hmac.compare_digest(given.encode(), self.cfg.token.encode())

    async def handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter):
        try:
            try: raw = await asyncio.wait_for(r.readuntil(b"\r\n\r\n"), 10.0)
            except (asyncio.LimitOverrunError, asyncio.IncompleteReadError, asyncio.TimeoutError) as ex:
                if isinstance(ex, asyncio.LimitOverrunError): await self._respond(w, 431, b"header too large")
                return
            lines = raw.decode("latin-1").split("\r\n"); parts = lines[0].split(" ")
            if len(parts) != 3 or not parts[2].startswith("HTTP/"): return await self._respond(w, 400, b"bad request")
            method, target = parts[0], parts[1]; headers = {}
            for ln in lines[1:]:
                if ":" in ln: k, v = ln.split(":", 1); headers[k.strip().lower()] = v.strip()
            if method not in ("GET", "HEAD"): return await self._respond(w, 405, b"method not allowed", extra={"Allow": "GET, HEAD"})
            u = urlsplit(target); path = u.path; query = parse_qs(u.query)
            if path == "/ws": return await self._websocket(r, w, headers, query)
            if path in self.assets:
                body, ct = self.assets[path]; return await self._respond(w, 200, b"" if method == "HEAD" else body, ct, {"Cache-Control": "no-store"}, length=len(body))
            if path == "/favicon.ico": return await self._respond(w, 204)
            if path == "/api/health":
                return await self._respond(w, 200, json.dumps({"ok": True, "version": __version__, "detector": getattr(self.detector, "name", "?"), "sessions": len(self.sessions),
                                                               "max_sessions": self.cfg.max_sessions, "agent": self.cfg.agent, "device": self.cfg.device}).encode(), "application/json")
            if path == "/api/stats":
                if not self._authorised(query): return await self._respond(w, 401, b"token required")
                return await self._respond(w, 200, json.dumps({"uptime_s": round(time.time() - self.started), "sessions": [s.stats() for s in list(self.sessions.values())]}).encode(), "application/json")
            await self._respond(w, 404, b"not found")
        except (ConnectionError, asyncio.CancelledError): pass
        except Exception as ex:
            self.log("http error:", type(ex).__name__, ex)
            try: await self._respond(w, 500, b"server error")
            except Exception: pass
        finally:
            try: w.close()
            except Exception: pass

    # ------------------------------------------------------------------------------------------- WebSocket
    async def _websocket(self, r, w, headers: dict, query: dict):
        key = headers.get("sec-websocket-key", "")
        if headers.get("upgrade", "").lower() != "websocket" or "upgrade" not in headers.get("connection", "").lower() or headers.get("sec-websocket-version") != "13" or not key:
            return await self._respond(w, 400, b"websocket upgrade expected")
        if not self._authorised(query): return await self._respond(w, 401, b"token required")
        w.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + accept_key(key) + "\r\n\r\n").encode())
        await w.drain()
        await self.on_ws(WebSocket(r, w, max_message=self.cfg.max_frame_bytes + 8192))

    async def on_ws(self, ws: WebSocket):
        loop = asyncio.get_running_loop(); cfg = self.cfg; sid = secrets.token_hex(4); session = agent = None
        pending = None; dropped = 0; ev = asyncio.Event(); tasks: set = set(); worker = None
        send = lambda obj: ws.send_text(json.dumps(obj, ensure_ascii=False))

        async def work():
            nonlocal pending, dropped
            while True:
                await ev.wait(); ev.clear()
                while pending is not None:
                    meta, jpeg = pending; pending = None; d, dropped = dropped, 0
                    try:
                        resp = await loop.run_in_executor(self.executor, session.process, jpeg, meta); resp["dropped"] = d; await send(resp)
                    except ValueError as ex: await send({"type": "error", "message": str(ex), "seq": meta.get("seq")})
                    except Exception as ex:
                        self.log("frame error:", type(ex).__name__, ex); await send({"type": "error", "message": "server error while processing the frame", "seq": meta.get("seq")})

        async def chat(m: dict):
            text = str(m.get("text", ""))[:1000]
            try: ans, used = await asyncio.wait_for(asyncio.to_thread(agent.reply, text), 90.0)
            except asyncio.TimeoutError: ans, used = "Агент не ответил вовремя.", []
            except Exception as ex: self.log("agent error:", type(ex).__name__, ex); ans, used = f"Ошибка агента ({type(ex).__name__}).", []
            await send({"type": "chat_reply", "id": m.get("id"), "text": ans, "tools": used, "agent": agent.kind})

        try:
            while True:
                try: msg = await asyncio.wait_for(ws.recv(), cfg.idle_timeout_s)
                except asyncio.TimeoutError: await ws.close(1001, "idle"); break
                except ProtocolError as ex: self.log("ws protocol error:", ex); break
                if msg is None: break
                op, data = msg
                if op == OP_TEXT:
                    try: m = json.loads(data.decode("utf-8")); assert isinstance(m, dict)
                    except Exception: await send({"type": "error", "message": "text messages must be JSON objects"}); continue
                    typ = m.get("type")
                    if typ == "hello":
                        if session is not None: await send({"type": "error", "message": "hello was already sent"}); continue
                        if len(self.sessions) >= cfg.max_sessions: await send({"type": "error", "message": "the server is busy: too many live sessions"}); await ws.close(1013, "busy"); break
                        session = LiveSession(self.detector, m, cfg, ground_classifier=self.ground_classifier, segmenter=self.segmenter, log=self.log, ocr_engine=self.ocr_engine); self.sessions[sid] = session; warning = None
                        try: agent = make_agent(session, cfg, self.agent_client)
                        except Exception as ex: agent = RuleAgent(session); warning = f"Claude agent unavailable ({ex}); using the offline agent"
                        worker = asyncio.create_task(work())
                        await send({"type": "ready", "session": sid, "detector": getattr(self.detector, "name", "?"), "fov_long_deg": session.cam_cfg.fov_long_deg,
                                    "compass": session.compass.name, "agent": agent.kind, "agent_warning": warning, "map_interval_s": cfg.map_interval_s,
                                    "names_ru": EN_RU, "severity": SEVERITY, "ground_enabled": self.ground_classifier is not None, "seg_enabled": self.segmenter is not None, "ocr_enabled": self.ocr_engine is not None, "alerts_enabled": bool(cfg.alerts)})
                    elif session is None: await send({"type": "error", "message": "send hello first"})
                    elif typ == "chat":
                        t = asyncio.create_task(chat(m)); tasks.add(t); t.add_done_callback(tasks.discard)
                    elif typ == "reset": session.reset(); await send({"type": "reset_done"})
                    elif typ == "ping": await send({"type": "pong", "t": m.get("t")})
                    else: await send({"type": "error", "message": f"unknown message type {typ!r}"})
                elif op == OP_BIN:
                    if session is None: await send({"type": "error", "message": "send hello first"}); continue
                    if len(data) > cfg.max_frame_bytes: await send({"type": "error", "message": f"frame larger than {cfg.max_frame_bytes} bytes"}); continue
                    try: meta, jpeg = unpack_frame(data)
                    except ValueError as ex: await send({"type": "error", "message": str(ex)}); continue
                    if pending is not None: dropped += 1                    # the server is behind: the older waiting frame is dropped, the newest wins
                    pending = (meta, jpeg); ev.set()
        finally:
            for t in list(tasks) + ([worker] if worker else []): t.cancel()
            if session is not None and getattr(session, 'seg', None) is not None: session.seg.close()
            self.sessions.pop(sid, None)
