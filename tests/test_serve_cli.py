import contextlib, http.client, io, os, re, socket, ssl, subprocess, sys, tempfile, time, unittest
import helpers
from scenelive.cli import agent_status, main, server_config
from scenelive.detect.blob import ColorBlobDetector
from scenelive.server.app import ServerApp
from scenelive.server.config import ServerConfig
from scenelive.server.runner import ServerThread
from scenelive.synth import render_video

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Args:
    def __init__(self, **kw):
        self.__dict__.update(dict(host="0.0.0.0", port=8443, no_https=False, cert_dir="certs", token=None, detector="world", weights=None, device=None, imgsz=640, conf=0.3,
                                  map_interval=6.0, max_sessions=2, agent="auto", model=None, vocab="extended", ground="none", no_alerts=False, alert_range=6.0, seg="none", seg_size=0, seg_every=1.0, camera_height=1.4, ocr="none", ocr_langs="ru,en")); self.__dict__.update(kw)


class TestConfig(unittest.TestCase):
    def test_flags_become_the_server_config(self):
        c = server_config(Args(port=9000, no_https=True, detector="yolo", imgsz=480, max_sessions=3, agent="offline", token="t", map_interval=4.0))
        self.assertEqual((c.port, c.https, c.detector, c.imgsz, c.max_sessions, c.agent, c.token, c.map_interval_s), (9000, False, "yolo", 480, 3, "offline", "t", 4.0))
        self.assertTrue(server_config(Args()).https); self.assertEqual(server_config(Args()).model, "claude-sonnet-5-5")

    def test_vocabulary_ground_and_alert_flags(self):
        c = server_config(Args(vocab="core", ground="clip", no_alerts=True, alert_range=9.0)); self.assertEqual((c.vocab, c.ground, c.alerts, c.alert_range_m), ("core", "clip", False, 9.0))
        d = server_config(Args()); self.assertEqual((d.vocab, d.ground, d.alerts, d.alert_range_m), ("extended", "none", True, 6.0))

    def test_ocr_flags(self):
        c = server_config(Args(ocr="easyocr", ocr_langs="ru,en,kk")); self.assertEqual((c.ocr, c.ocr_langs), ("easyocr", "ru,en,kk")); self.assertEqual(server_config(Args()).ocr, "none")

    def test_segmentation_flags(self):
        c = server_config(Args(seg="segformer", seg_size=448, seg_every=0.5, camera_height=1.2)); self.assertEqual((c.seg, c.seg_size, c.seg_every_s, c.camera_height_m), ("segformer", 448, 0.5, 1.2))
        d = server_config(Args()); self.assertEqual((d.seg, d.seg_size, d.seg_every_s, d.camera_height_m), ("none", 0, 1.0, 1.4))

    def test_token_and_model_from_the_environment(self):
        old = {k: os.environ.get(k) for k in ("SCENELIVE_TOKEN", "SCENELIVE_MODEL")}
        try:
            os.environ["SCENELIVE_TOKEN"] = "envtoken"; os.environ["SCENELIVE_MODEL"] = "some-model"; c = server_config(Args())
            self.assertEqual((c.token, c.model), ("envtoken", "some-model")); self.assertEqual(server_config(Args(token="flag")).token, "flag")      # the flag wins
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

    def test_agent_status_explains_what_is_missing(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertIn("offline", agent_status(ServerConfig(agent="offline"))); self.assertIn("ANTHROPIC_API_KEY is not set", agent_status(ServerConfig(agent="auto")))
            self.assertIn("REQUESTED Claude", agent_status(ServerConfig(agent="claude"))); os.environ["ANTHROPIC_API_KEY"] = "x"
            self.assertIn("anthropic", agent_status(ServerConfig(agent="auto")))                                                  # the package is not installed here
        finally:
            if old is None: os.environ.pop("ANTHROPIC_API_KEY", None)
            else: os.environ["ANTHROPIC_API_KEY"] = old


class TestClientSimCommand(unittest.TestCase):
    def test_client_sim_against_a_server(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=6.0)
        app = ServerApp(ServerConfig(detector="blobs", agent="offline", https=False), ColorBlobDetector(), log=lambda *a: None); buf = io.StringIO()
        with ServerThread(app) as th, contextlib.redirect_stdout(buf):
            code = main(["client-sim", v, "--url", f"http://127.0.0.1:{th.port}/", "--fast", "--chat", "Где человек?"])
        out = buf.getvalue(); self.assertEqual(code, 0); self.assertIn("round trip", out); self.assertIn("agent (where_is)", out); self.assertIn("человек", out)

    def test_a_dead_server_is_a_clear_message(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=2.0); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = main(["client-sim", v, "--url", f"http://127.0.0.1:{free_port()}/", "--fast"])
        self.assertEqual(code, 2); self.assertIn("Is it running", buf.getvalue())


class TestRealServeProcess(unittest.TestCase):
    """The real `scenelive serve` command in a separate process."""
    def run_serve(self, *extra, https=False):
        port = free_port(); env = dict(os.environ, PYTHONPATH=SRC, PYTHONUTF8="1"); env.pop("ANTHROPIC_API_KEY", None); env.pop("SCENELIVE_TOKEN", None)
        p = subprocess.Popen([sys.executable, "-m", "scenelive.cli", "serve", "--detector", "blobs", "--port", str(port), "--agent", "offline", *extra], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        def get(path):
            c = (http.client.HTTPSConnection("127.0.0.1", port, context=ctx, timeout=3) if https else http.client.HTTPConnection("127.0.0.1", port, timeout=3)); c.request("GET", path); r = c.getresponse(); r.read(); c.close(); return r.status
        for _ in range(60):
            if p.poll() is not None: break
            try: get("/api/health"); break
            except Exception: time.sleep(0.25)
        return p, port, get

    def stop(self, p):
        p.terminate()
        try: return p.communicate(timeout=8)[0]
        except subprocess.TimeoutExpired: p.kill(); return p.communicate()[0]

    def test_lan_mode_generates_a_token_and_prints_the_phone_address(self):
        p, port, get = self.run_serve("--no-https")
        try:
            self.assertEqual(get("/api/health"), 200); self.assertEqual(get("/api/stats"), 401)                              # a LAN-visible server is not open
        finally: out = self.stop(p)
        m = re.search(r"token=([\w-]+)", out); self.assertTrue(m, out); self.assertIn(f"http://localhost:{port}/?token=", out); self.assertIn("agent: offline", out)

    def test_localhost_mode_needs_no_token(self):
        p, port, get = self.run_serve("--no-https", "--host", "127.0.0.1")
        try: self.assertEqual(get("/api/stats"), 200)
        finally: out = self.stop(p)
        self.assertNotIn("token=", out)

    def test_a_missing_vocabulary_file_is_a_clear_message_not_a_crash(self):
        port = free_port(); env = dict(os.environ, PYTHONPATH=SRC, PYTHONUTF8="1")
        r = subprocess.run([sys.executable, "-m", "scenelive.cli", "serve", "--detector", "blobs", "--no-https", "--port", str(port), "--vocab", "nope.txt"], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 2); self.assertIn("Cannot start", r.stdout); self.assertNotIn("Traceback", r.stdout + r.stderr)

    def test_https_mode_makes_a_certificate_and_serves(self):
        d = tempfile.mkdtemp(); p, port, get = self.run_serve("--cert-dir", os.path.join(d, "c"), "--host", "127.0.0.1", https=True)
        try: self.assertEqual(get("/api/health"), 200)
        finally: out = self.stop(p)
        self.assertTrue(os.path.exists(os.path.join(d, "c", "cert.pem"))); self.assertIn("https://localhost", out); self.assertIn("self-signed", out)


if __name__ == "__main__": unittest.main()
