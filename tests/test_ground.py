import json, os, tempfile, unittest
import numpy as np
import helpers, server_helpers as H
from scenelive.catalog import SURFACES, SURFACE_QUALITY
from scenelive.config import CameraConfig, FastConfig, MapConfig, SimConfig
from scenelive.compass import ZeroCompass
from scenelive.detect.blob import ColorBlobDetector
from scenelive.fastpath import FastPath
from scenelive.ground import ClipSurfaceClassifier, ColorSurfaceClassifier, GroundMonitor, make_ground
from scenelive.report import format_report
from scenelive.server.config import ServerConfig
from scenelive.server.session import LiveSession
from scenelive.sim import run_replay
from scenelive.synth import render_video


def frame_with_ground(left, center, right, h=360, w=640):
    f = np.full((h, w, 3), 120, np.uint8); y = int(h * 0.55)
    f[y:, : w // 3] = left; f[y:, w // 3: 2 * w // 3] = center; f[y:, 2 * w // 3:] = right; return f


ASPHALT, GRASS, DIRT = (90, 90, 90), (40, 160, 50), (60, 100, 150)                    # BGR


class Scripted:
    """A classifier that returns prepared answers (one list of per-zone top-k lists per call) and counts calls."""
    name = "scripted"
    def __init__(self, answers): self.answers, self.calls = answers, 0
    def classify(self, crops):
        a = self.answers[min(self.calls, len(self.answers) - 1)]; self.calls += 1; return a


def zones(l, c, r): return [[(l[0], l[1])], [(c[0], c[1])], [(r[0], r[1])]]


class TestGroundMonitor(unittest.TestCase):
    def test_three_zones_get_their_own_surface(self):
        m = GroundMonitor(ColorSurfaceClassifier()); st = m.update(frame_with_ground(ASPHALT, GRASS, DIRT), 0.0)
        self.assertEqual((st["left"]["surface"], st["center"]["surface"], st["right"]["surface"]), ("asphalt", "grass", "dirt"))
        self.assertEqual((st["left"]["ru"], st["center"]["ru"]), ("асфальт", "трава")); self.assertEqual((st["left"]["quality"], st["center"]["quality"]), ("good", "fair"))
        json.dumps(st)

    def test_it_looks_about_once_a_second_not_on_every_frame(self):
        clf = Scripted([zones(("asphalt", 0.9), ("asphalt", 0.9), ("asphalt", 0.9))]); m = GroundMonitor(clf, every_s=1.0); f = frame_with_ground(ASPHALT, ASPHALT, ASPHALT)
        for k in range(10):                                                                  # 3 frames per second for 3.3 s
            t = k / 3.0
            if m.due(t): m.update(f, t)
        self.assertEqual(clf.calls, 4)                                                       # t = 0, 1, 2, 3

    def test_one_odd_look_does_not_announce_a_pothole(self):
        A, P = zones(("asphalt", 0.9), ("asphalt", 0.9), ("asphalt", 0.9)), zones(("asphalt", 0.9), ("pothole", 0.9), ("asphalt", 0.9))
        m = GroundMonitor(Scripted([A, A, P, A, A]), history=3); f = frame_with_ground(ASPHALT, ASPHALT, ASPHALT); seen = []
        for k in range(5): seen.append(m.update(f, float(k))["center"]["surface"])
        self.assertEqual(seen, ["asphalt"] * 5)                                              # the single pothole look is outvoted

    def test_a_lasting_change_gets_through_and_is_flagged_bad(self):
        A, P = zones(("asphalt", 0.9), ("asphalt", 0.9), ("asphalt", 0.9)), zones(("asphalt", 0.9), ("pothole", 0.9), ("asphalt", 0.9))
        m = GroundMonitor(Scripted([A, A, P, P, P]), history=3); f = frame_with_ground(ASPHALT, ASPHALT, ASPHALT); out = [m.update(f, float(k))["center"] for k in range(5)]
        self.assertEqual([o["surface"] for o in out], ["asphalt", "asphalt", "asphalt", "pothole", "pothole"]); self.assertTrue(out[-1]["bad"]); self.assertFalse(out[0]["bad"])
        self.assertEqual(out[-1]["quality_ru"], "плохое")

    def test_a_doubtful_bad_surface_is_not_called_bad(self):
        m = GroundMonitor(Scripted([zones(("asphalt", 0.9), ("ice", 0.3), ("asphalt", 0.9))])); st = m.update(frame_with_ground(ASPHALT, ASPHALT, ASPHALT), 0.0)
        self.assertEqual(st["center"]["surface"], "ice"); self.assertFalse(st["center"]["bad"])                  # quality bad, but confidence 0.3 < 0.4

    def test_state_is_a_copy(self):
        m = GroundMonitor(ColorSurfaceClassifier()); self.assertIsNone(m.state()); m.update(frame_with_ground(ASPHALT, ASPHALT, ASPHALT), 0.0)
        s = m.state(); s["left"]["surface"] = "x"; self.assertEqual(m.state()["left"]["surface"], "asphalt")

    def test_every_surface_has_a_known_quality(self):
        self.assertTrue(all(q in ("good", "fair", "bad") for q in SURFACE_QUALITY.values())); self.assertEqual(len(SURFACES), len(SURFACE_QUALITY))
        for s in ("pothole", "ice", "mud", "puddle", "stairs", "cracked asphalt", "snow"): self.assertEqual(SURFACE_QUALITY[s], "bad")

    def test_factory_and_a_missing_clip_package_is_explained(self):
        self.assertIsNone(make_ground("none")); self.assertEqual(make_ground("colors").name, "colors")
        with self.assertRaises(ValueError): make_ground("magic")
        with self.assertRaises(ImportError): ClipSurfaceClassifier()                          # neither torch nor clip here: a clear ImportError, not a crash later


class TestGroundInThePipeline(unittest.TestCase):
    def test_fastpath_runs_the_ground_stage_only_when_due(self):
        clf = Scripted([zones(("asphalt", 0.9), ("asphalt", 0.9), ("asphalt", 0.9))]); fp = FastPath(ColorBlobDetector(), CameraConfig(), FastConfig(), ZeroCompass(), ground=GroundMonitor(clf, 1.0))
        frames = [fp.process(H.jpeg_at(k / 3)[0] and __import__("cv2").imdecode(np.frombuffer(H.jpeg_at(k / 3)[0], np.uint8), 1), k / 3, k) for k in range(7)]
        ran = [("ground" in r.timings) for r in frames]; self.assertEqual(ran, [True, False, False, True, False, False, True]); self.assertTrue(all(r.ground is not None for r in frames))
        self.assertIsNone(FastPath(ColorBlobDetector(), CameraConfig(), FastConfig(), ZeroCompass()).process(frames[0] and np.zeros((360, 640, 3), np.uint8), 0.0, 0).ground)

    def test_the_session_result_carries_the_surface(self):
        s = LiveSession(ColorBlobDetector(), {"camera": "main"}, ServerConfig(detector="blobs"), ground_classifier=ColorSurfaceClassifier())
        r = s.process(H.jpeg_at(0.0)[0], {"seq": 0, "t": 0.0}); self.assertIn(r["ground"]["center"]["surface"], {a[0] for a in SURFACES}); json.dumps(r)
        self.assertIsNone(LiveSession(ColorBlobDetector(), {}, ServerConfig(detector="blobs")).process(H.jpeg_at(0.0)[0], {"seq": 0, "t": 0.0})["ground"])

    def test_the_replay_report_shows_the_cost_of_the_ground_stage(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=8.0)
        res = run_replay(v, ColorBlobDetector(), CameraConfig(), FastConfig(), MapConfig(), SimConfig(), ZeroCompass(), ground=ColorSurfaceClassifier())
        rows = {r["stage"]: r for r in res.stage_rows}; self.assertIn("ground", rows); self.assertLess(rows["ground"]["n"], rows["detect"]["n"]); self.assertIn("ground", format_report(res))


if __name__ == "__main__": unittest.main()
