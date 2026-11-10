from __future__ import annotations
import asyncio
import secrets
import ssl
import sys
import threading
from .app import ServerApp
from .config import ServerConfig


def make_ssl_context(cfg: ServerConfig):
    if not cfg.https: return None
    from .tls import ensure_self_signed_cert
    cp, kp = ensure_self_signed_cert(cfg.cert_dir)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.minimum_version = ssl.TLSVersion.TLSv1_2; ctx.load_cert_chain(cp, kp); return ctx


async def start(app: ServerApp, host: str, port: int, ssl_ctx=None):
    return await asyncio.start_server(app.handle, host, port, ssl=ssl_ctx, limit=65536)


def run_server(cfg: ServerConfig, detector, ground_classifier=None, segmenter=None) -> None:
    """Blocking. Prints the addresses to open on the phone."""
    from .tls import lan_ips
    if cfg.token is None and cfg.host not in ("127.0.0.1", "localhost"): cfg.token = secrets.token_urlsafe(9)      # a LAN-visible camera server must not be open
    app = ServerApp(cfg, detector, ground_classifier=ground_classifier, segmenter=segmenter); ctx = make_ssl_context(cfg); scheme = "https" if cfg.https else "http"

    async def main():
        srv = await start(app, cfg.host, cfg.port, ctx)
        q = f"/?token={cfg.token}" if cfg.token else "/"
        print(f"\nscenelive server is running ({getattr(detector, 'name', '?')}, agent: {cfg.agent}).", flush=True)
        print(f"On this computer:  {scheme}://localhost:{cfg.port}{q}", flush=True)
        if cfg.host in ("0.0.0.0", ""):
            for ip in lan_ips(): print(f"On the phone (same Wi-Fi):  {scheme}://{ip}:{cfg.port}{q}", flush=True)
        if cfg.https: print("The certificate is self-signed: the phone shows a warning once, choose 'advanced' and 'continue'.", flush=True)
        print("Stop: Ctrl+C\n", flush=True)
        async with srv: await srv.serve_forever()
    try: asyncio.run(main())
    except KeyboardInterrupt: print("stopped", file=sys.stderr)


class ServerThread:
    """The server in a background thread (tests, client simulator). port=0 picks a free port."""
    def __init__(self, app: ServerApp, host: str = "127.0.0.1", port: int = 0, ssl_ctx=None):
        self.app, self.host, self._port, self.ssl_ctx = app, host, port, ssl_ctx; self.loop = None; self.port = None
        self._ready = threading.Event(); self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.loop = asyncio.new_event_loop(); asyncio.set_event_loop(self.loop)
        self.srv = self.loop.run_until_complete(start(self.app, self.host, self._port, self.ssl_ctx)); self.port = self.srv.sockets[0].getsockname()[1]
        self._ready.set(); self.loop.run_forever()

    def __enter__(self): self._thread.start(); self._ready.wait(10); return self

    def __exit__(self, *a):
        async def stop():
            self.srv.close(); await self.srv.wait_closed()
            for t in asyncio.all_tasks():
                if t is not asyncio.current_task(): t.cancel()
        try: asyncio.run_coroutine_threadsafe(stop(), self.loop).result(5)
        except Exception: pass
        self.loop.call_soon_threadsafe(self.loop.stop); self._thread.join(5)
