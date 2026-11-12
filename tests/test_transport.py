import asyncio, http.client, json, os, socket, ssl, struct, tempfile, unittest
import helpers, server_helpers as H
from scenelive.detect.blob import ColorBlobDetector
from scenelive.server.app import ServerApp, unpack_frame
from scenelive.server.client_sim import pack_frame, parse_url, run_client, ws_connect
from scenelive.server.config import ServerConfig
from scenelive.server.runner import ServerThread, make_ssl_context
from scenelive.server.tls import ensure_self_signed_cert
from scenelive.server.ws import OP_BIN, OP_CLOSE, OP_CONT, OP_PING, OP_PONG, OP_TEXT, ProtocolError, WebSocket, accept_key, apply_mask, encode_frame
from scenelive.synth import render_video


class FakeWriter:
    def __init__(self): self.out = bytearray(); self.closed = False
    def write(self, b): self.out += b
    async def drain(self): pass
    def close(self): self.closed = True


def recv_all(chunks, n=1, client=False, **kw):
    """Feed raw bytes to a server-side WebSocket and read n messages. Everything is created INSIDE the running loop."""
    async def go():
        r = asyncio.StreamReader()
        for c in chunks: r.feed_data(c)
        r.feed_eof(); w = FakeWriter(); ws = WebSocket(r, w, is_client=client, **kw); out = [await ws.recv() for _ in range(n)]
        return out, w, ws
    return asyncio.run(go())


def masked(op, payload, fin=True): return encode_frame(op, payload, fin=fin, mask=True)


class TestFraming(unittest.TestCase):
    def test_rfc6455_handshake_example(self):
        self.assertEqual(accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=")             # the example from the RFC

    def test_frame_lengths_and_masking_round_trip(self):
        for n in (0, 5, 125, 126, 300, 65535, 65536, 70000):
            data = os.urandom(n); self.assertEqual(recv_all([masked(OP_BIN, data)])[0][0], (OP_BIN, data), n)
        k = os.urandom(4); self.assertEqual(apply_mask(apply_mask(b"hello world", k), k), b"hello world")
        self.assertEqual(len(encode_frame(OP_TEXT, b"x" * 126)), 4 + 126); self.assertEqual(len(encode_frame(OP_TEXT, b"x" * 70000)), 10 + 70000)

    def test_fragmented_message_is_reassembled(self):
        out, _, _ = recv_all([masked(OP_TEXT, b"hel", fin=False), masked(OP_CONT, b"lo ", fin=False), masked(OP_CONT, "мир".encode(), fin=True)])
        self.assertEqual(out[0], (OP_TEXT, "hello мир".encode()))

    def test_ping_is_answered_and_close_is_echoed(self):
        out, w, ws = recv_all([masked(OP_PING, b"abc"), masked(OP_TEXT, b"hi"), masked(OP_CLOSE, struct.pack("!H", 1000))], n=2)
        self.assertEqual(out, [(OP_TEXT, b"hi"), None]); self.assertIn(bytes([0x80 | OP_PONG, 3]) + b"abc", bytes(w.out)); self.assertTrue(ws.closed)

    def test_broken_peers_are_refused(self):
        with self.assertRaises(ProtocolError): recv_all([encode_frame(OP_TEXT, b"x", mask=False)])                                   # unmasked client frame
        with self.assertRaises(ProtocolError) as c: recv_all([masked(OP_BIN, b"x" * 100)], max_message=50)
        self.assertEqual(c.exception.code, 1009)
        with self.assertRaises(ProtocolError) as c: recv_all([masked(OP_TEXT, b"\xff\xfe")])
        self.assertEqual(c.exception.code, 1007)
        with self.assertRaises(ProtocolError): recv_all([masked(OP_CONT, b"x")])                                                     # continuation without a start
        with self.assertRaises(ProtocolError): recv_all([bytes([0xC1, 0x80]) + b"\0\0\0\0"])                                       # reserved bit
        with self.assertRaises(ProtocolError): recv_all([masked(OP_PING, b"x" * 200)])                                              # control frame too long
        self.assertIsNone(recv_all([b"\x81"])[0][0])                                                                                # connection cut in the middle: a clean end

    def test_frame_messages_pack_and_unpack(self):
        meta, jpeg = unpack_frame(pack_frame({"seq": 3, "t": 1.5, "yaw": 12.5, "junk": "x"}, b"\xff\xd8abc"))
        self.assertEqual((meta, jpeg), ({"seq": 3, "t": 1.5, "yaw": 12.5}, b"\xff\xd8abc"))                                  # unknown meta keys are dropped
        for bad in (b"", b"abc", (5000).to_bytes(4, "big") + b"x" * 6000, (2).to_bytes(4, "big") + b"{}", (4).to_bytes(4, "big") + b"nope" + b"jpg",
                    pack_frame({"t": float("nan")}, b"j"), pack_frame({"seq": True}, b"j"), pack_frame({"yaw": "12"}, b"j")):
            with self.assertRaises(ValueError): unpack_frame(bad)
        self.assertEqual(parse_url("https://192.168.1.5:8443/?token=abc"), ("192.168.1.5", 8443, True, "abc")); self.assertEqual(parse_url("localhost:9")[:2], ("localhost", 9))


class SlowDetector(ColorBlobDetector):
    name = "slow-blobs"
    def detect(self, frame):
        import time; time.sleep(0.12); return super().detect(frame)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); cls.video = os.path.join(cls.d, "v.mp4"); render_video(cls.video, duration=10.0)
        cls.cfg = ServerConfig(detector="blobs", agent="offline", https=False); cls.cfg.idle_timeout_s = 30
        cls.app = ServerApp(cls.cfg, ColorBlobDetector(), log=lambda *a: None); cls.th = ServerThread(cls.app); cls.th.__enter__(); cls.port = cls.th.port

    @classmethod
    def tearDownClass(cls): cls.th.__exit__(None, None, None)

    def get(self, path, method="GET", port=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=5); c.request(method, path, body=body); r = c.getresponse(); data = r.read(); h = dict(r.getheaders()); c.close(); return r.status, h, data

    def raw(self, payload: bytes) -> bytes:
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5); s.sendall(payload); s.settimeout(3); out = b""
        try:
            while True:
                b = s.recv(4096)
                if not b: break
                out += b
        except (socket.timeout, ConnectionError): pass
        s.close(); return out


class TestHttp(Base):
    def test_page_and_assets(self):
        st, h, body = self.get("/"); self.assertEqual(st, 200); self.assertIn("text/html", h["Content-Type"])
        for s in (b'id="video"', b'id="start"', b'/app.js', b'id="chat-text"', b'playsinline', b'id="mic"', b'id="voice"', b'id="ground"', b'id="alert"'): self.assertIn(s, body)
        self.assertNotIn(b"<script>", body)                                                                                    # CSP forbids inline scripts
        st, h, js = self.get("/app.js"); self.assertEqual(st, 200); self.assertIn("javascript", h["Content-Type"]); self.assertIn(b"compassHeading", js)
        self.assertEqual(self.get("/style.css")[0], 200); self.assertEqual(self.get("/favicon.ico")[0], 204)

    def test_security_headers(self):
        _, h, _ = self.get("/")
        self.assertIn("script-src 'self'", h["Content-Security-Policy"]); self.assertEqual(h["X-Content-Type-Options"], "nosniff"); self.assertIn("camera=(self)", h["Permissions-Policy"])
        self.assertEqual(h["Cache-Control"], "no-store")

    def test_health_has_no_secrets(self):
        st, _, body = self.get("/api/health"); j = json.loads(body); self.assertTrue(j["ok"]); self.assertEqual(j["detector"], "color-blobs"); self.assertNotIn("token", body.decode().lower())

    def test_only_known_paths_are_served(self):
        for p in ("/nope", "/../../etc/passwd", "/%2e%2e/%2e%2e/etc/passwd", "/static/../app.js", "/app.js/", "//etc/passwd", "/web/app.js"): self.assertEqual(self.get(p)[0], 404, p)

    def test_methods_and_head(self):
        st, h, _ = self.get("/", method="POST", body=b"x"); self.assertEqual(st, 405); self.assertIn("GET", h["Allow"])
        st, h, body = self.get("/", method="HEAD"); self.assertEqual((st, body), (200, b"")); self.assertGreater(int(h["Content-Length"]), 100)

    def test_garbage_does_not_kill_the_server(self):
        self.assertIn(b"400", self.raw(b"hello there\r\n\r\n")); self.assertIn(b"431", self.raw(b"GET / HTTP/1.1\r\nX: " + b"a" * 200000 + b"\r\n\r\n"))
        self.assertIn(b"400", self.raw(b"GET /ws HTTP/1.1\r\nHost: x\r\n\r\n"))                                              # /ws without an upgrade
        self.raw(b"\x00\x01\x02\xff\xfe\r\n\r\n"); self.assertEqual(self.get("/api/health")[0], 200)


class TestSessionOverTheWire(Base):
    def test_ready_carries_russian_names_and_severity_for_the_page(self):
        async def go():
            ws = await ws_connect("127.0.0.1", self.port, "/ws"); await ws.send_text(json.dumps({"type": "hello"})); m = json.loads((await ws.recv())[1]); await ws.close(); return m
        m = asyncio.run(go()); self.assertEqual(m["names_ru"]["knife"], "нож"); self.assertEqual(m["names_ru"]["person"], "человек"); self.assertEqual(m["severity"]["knife"], "danger")
        self.assertEqual((m["ground_enabled"], m["alerts_enabled"], m["ocr_enabled"], m["seg_enabled"]), (False, True, False, False))

    def test_a_whole_walk_with_the_phone_simulator(self):
        s = asyncio.run(run_client(f"http://127.0.0.1:{self.port}/", self.video, fps=3.0, pace=False, questions=["Что вокруг?", "Где человек?"]))
        self.assertEqual(s["errors"], []); self.assertEqual(s["frames_sent"], s["results"]); self.assertGreater(s["results"], 25)
        names = {v["name"] for v in s["final_entries"]}; self.assertTrue({"person", "car", "pole", "door"} <= names)
        self.assertEqual(len(s["chat"]), 2); self.assertIn("человек", s["chat"][0][0]); self.assertEqual(s["ready"]["agent"], "offline")
        self.assertLess(s["rtt_ms"]["p50"], 500); self.assertGreater(s["server_compute_ms_mean"], 0)

    def test_a_phone_with_a_compass_and_a_wide_camera(self):
        yaw = os.path.join(self.d, "h.csv"); open(yaw, "w").write("t,yaw\n" + "\n".join(f"{t / 2},{4.0 * t / 2}" for t in range(0, 25)))
        s = asyncio.run(run_client(f"http://127.0.0.1:{self.port}/", self.video, fps=3.0, pace=False, camera="ultrawide", yaw_csv=yaw))
        self.assertEqual(s["ready"]["fov_long_deg"], 100.0); self.assertEqual(s["ready"]["compass"], "client"); self.assertAlmostEqual(s["final_heading"], 36.0, delta=3.0)

    def test_protocol_errors_are_answered_and_the_session_survives(self):
        async def go():
            ws = await ws_connect("127.0.0.1", self.port, "/ws"); rj = lambda: asyncio.wait_for(ws.recv(), 5)
            async def reply(): return json.loads((await rj())[1])
            await ws.send_binary(pack_frame({"seq": 0}, b"x" * 50)); e1 = await reply()                                     # no hello yet
            await ws.send_text("not json"); e2 = await reply(); await ws.send_text(json.dumps({"type": "chat", "text": "?"})); e3 = await reply()
            await ws.send_text(json.dumps({"type": "hello", "camera": "main", "has_heading": True})); ready = await reply()
            await ws.send_text(json.dumps({"type": "hello"})); e4 = await reply(); await ws.send_text(json.dumps({"type": "dance"})); e5 = await reply()
            await ws.send_binary(b"abc"); e6 = await reply(); await ws.send_binary(pack_frame({"seq": 1, "t": float("inf")}, b"j")); e7 = await reply()
            await ws.send_binary(pack_frame({"seq": 2, "t": 0.0}, b"definitely not a jpeg")); e8 = await reply()
            await ws.send_binary(b"\0" * (self.cfg.max_frame_bytes + 10)); e9 = await reply()
            await ws.send_text(json.dumps({"type": "ping", "t": 5})); pong = await reply(); await ws.send_text(json.dumps({"type": "reset"})); rst = await reply()
            jpg, yaw = H.jpeg_at(0.0); await ws.send_binary(pack_frame({"seq": 3, "t": 0.0, "yaw": 10.0}, jpg)); res = await reply(); await ws.close()
            return e1, e2, e3, ready, e4, e5, e6, e7, e8, e9, pong, rst, res
        e1, e2, e3, ready, e4, e5, e6, e7, e8, e9, pong, rst, res = asyncio.run(go())
        for e in (e1, e2, e3, e4, e5, e6, e7, e8, e9): self.assertEqual(e["type"], "error", e)
        self.assertIn("hello", e1["message"]); self.assertIn("JSON", e2["message"]); self.assertEqual(ready["type"], "ready"); self.assertEqual(e8["seq"], 2)
        self.assertIn("larger", e9["message"]); self.assertEqual((pong["type"], pong["t"], rst["type"]), ("pong", 5, "reset_done")); self.assertEqual((res["type"], res["seq"]), ("result", 3))

    def test_a_slow_server_drops_old_frames_and_keeps_the_newest(self):
        app = ServerApp(self.cfg, SlowDetector(), log=lambda *a: None)
        async def go(port):
            ws = await ws_connect("127.0.0.1", port, "/ws"); await ws.send_text(json.dumps({"type": "hello", "camera": "main"})); await ws.recv(); res = []
            for k in range(8):
                jpg, _ = H.jpeg_at(k / 3); await ws.send_binary(pack_frame({"seq": k, "t": k / 3}, jpg))                    # a burst: faster than the server can work
            while not res or res[-1]["seq"] != 7:
                m = json.loads((await asyncio.wait_for(ws.recv(), 10))[1]); res.append(m) if m["type"] == "result" else None
            await ws.close(); return res
        with ServerThread(app) as th: res = asyncio.run(go(th.port))
        self.assertLess(len(res), 8); self.assertEqual(res[-1]["seq"], 7); self.assertEqual(len(res) + sum(r["dropped"] for r in res), 8)

    def test_session_limit_and_slot_release(self):
        cfg = ServerConfig(detector="blobs", agent="offline", https=False, max_sessions=1); app = ServerApp(cfg, ColorBlobDetector(), log=lambda *a: None)
        async def go(port):
            hello = json.dumps({"type": "hello"}); a = await ws_connect("127.0.0.1", port, "/ws"); await a.send_text(hello); ra = json.loads((await a.recv())[1])
            b = await ws_connect("127.0.0.1", port, "/ws"); await b.send_text(hello); rb = json.loads((await b.recv())[1]); closed = await asyncio.wait_for(b.recv(), 5)
            await a.close(); await asyncio.sleep(0.3)
            c = await ws_connect("127.0.0.1", port, "/ws"); await c.send_text(hello); rc = json.loads((await c.recv())[1]); await c.close(); return ra, rb, closed, rc
        with ServerThread(app) as th: ra, rb, closed, rc = asyncio.run(go(th.port))
        self.assertEqual(ra["type"], "ready"); self.assertEqual(rb["type"], "error"); self.assertIn("busy", rb["message"]); self.assertIsNone(closed); self.assertEqual(rc["type"], "ready")

    def test_idle_connections_are_closed(self):
        cfg = ServerConfig(detector="blobs", agent="offline", https=False, idle_timeout_s=0.4); app = ServerApp(cfg, ColorBlobDetector(), log=lambda *a: None)
        async def go(port):
            ws = await ws_connect("127.0.0.1", port, "/ws"); await ws.send_text(json.dumps({"type": "hello"})); await ws.recv(); return await asyncio.wait_for(ws.recv(), 5)
        with ServerThread(app) as th: self.assertIsNone(asyncio.run(go(th.port)))

    def test_the_stats_page_shows_live_sessions(self):
        async def go():
            ws = await ws_connect("127.0.0.1", self.port, "/ws"); await ws.send_text(json.dumps({"type": "hello"})); await ws.recv()
            jpg, _ = H.jpeg_at(0.0); await ws.send_binary(pack_frame({"seq": 0, "t": 0.0}, jpg)); await ws.recv(); st = self.get("/api/stats"); await ws.close(); return st
        st, _, body = asyncio.run(go()); j = json.loads(body); self.assertEqual(st, 200); self.assertEqual(len(j["sessions"]), 1); self.assertTrue(any(r["stage"] == "detect" for r in j["sessions"][0]["stages"]))


class TestToken(unittest.TestCase):
    def test_the_token_protects_the_socket_and_the_stats_but_not_the_page(self):
        cfg = ServerConfig(detector="blobs", agent="offline", https=False, token="s3cret"); app = ServerApp(cfg, ColorBlobDetector(), log=lambda *a: None)
        with ServerThread(app) as th:
            def get(path):
                c = http.client.HTTPConnection("127.0.0.1", th.port, timeout=5); c.request("GET", path); r = c.getresponse(); r.read(); c.close(); return r.status
            self.assertEqual((get("/"), get("/api/health"), get("/api/stats"), get("/api/stats?token=bad"), get("/api/stats?token=s3cret")), (200, 200, 401, 401, 200))
            for path in ("/ws", "/ws?token=wrong", "/ws?token="):
                with self.assertRaises(ConnectionError) as c: asyncio.run(ws_connect("127.0.0.1", th.port, path))
                self.assertIn("401", str(c.exception))
            async def ok():
                ws = await ws_connect("127.0.0.1", th.port, "/ws?token=s3cret"); await ws.send_text(json.dumps({"type": "hello"})); m = json.loads((await ws.recv())[1]); await ws.close(); return m
            self.assertEqual(asyncio.run(ok())["type"], "ready")


class TestTls(unittest.TestCase):
    def test_self_signed_certificate_and_https_session(self):
        d = tempfile.mkdtemp(); cert_dir = os.path.join(d, "certs"); cp, kp = ensure_self_signed_cert(cert_dir); m1 = os.path.getmtime(cp)
        self.assertEqual(ensure_self_signed_cert(cert_dir), (cp, kp)); self.assertEqual(os.path.getmtime(cp), m1)             # reused while valid
        from cryptography import x509
        san = x509.load_pem_x509_certificate(open(cp, "rb").read()).extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        self.assertIn("localhost", san.get_values_for_type(x509.DNSName)); self.assertIn("127.0.0.1", {str(i) for i in san.get_values_for_type(x509.IPAddress)})
        cfg = ServerConfig(detector="blobs", agent="offline", https=True, cert_dir=cert_dir); app = ServerApp(cfg, ColorBlobDetector(), log=lambda *a: None)
        v = os.path.join(d, "v.mp4"); render_video(v, duration=4.0)
        with ServerThread(app, ssl_ctx=make_ssl_context(cfg)) as th:
            ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
            c = http.client.HTTPSConnection("127.0.0.1", th.port, context=ctx, timeout=5); c.request("GET", "/api/health"); self.assertEqual(c.getresponse().status, 200); c.close()
            s = asyncio.run(run_client(f"https://127.0.0.1:{th.port}/", v, fps=3.0, pace=False)); self.assertEqual(s["errors"], []); self.assertGreater(s["results"], 5)
            with self.assertRaises(Exception): c2 = http.client.HTTPConnection("127.0.0.1", th.port, timeout=3); c2.request("GET", "/"); c2.getresponse().read()      # plain http to a TLS port fails


if __name__ == "__main__": unittest.main()
