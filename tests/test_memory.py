import unittest
import cv2, numpy as np
import helpers, server_helpers as H
from helpers import det
from scenelive.camera import Camera
from scenelive.compass import ZeroCompass
from scenelive.config import CameraConfig, FastConfig
from scenelive.detect.base import Detection
from scenelive.detect.blob import ColorBlobDetector
from scenelive.fastpath import FastPath
from scenelive.server.agent import RuleAgent
from scenelive.server.config import ServerConfig
from scenelive.server.session import LiveSession
from scenelive.synth import _panorama, default_world, render_frame
from scenelive.track import Tracker

BLANK = np.zeros((360, 640, 3), np.uint8)
BOX = (300.0, 100.0, 340.0, 250.0)


class Scripted:
    name = "scripted"
    def __init__(self, script): self.script, self.i = script, 0
    def detect(self, frame): out = self.script[min(self.i, len(self.script) - 1)]; self.i += 1; return out


def run(script, **cfg):
    fp = FastPath(Scripted(script), CameraConfig(), FastConfig(**cfg), ZeroCompass()); return [fp.process(BLANK, k / 3, k) for k in range(len(script))], fp


class TestTheLostDoor(unittest.TestCase):
    """Your report, as a test: a door seen for a moment, the camera turns away, the assistant must still know the door."""
    WORLD = [o for o in default_world() if o.drift_deg_s == 0]; PANO = _panorama(360, 0); CAM = Camera(640, 360, 69.0)

    @staticmethod
    def yaw_at(t): return 4.0 * t if t <= 8.0 else max(0.0, 32.0 - 16.0 * (t - 8.0))      # right until t=8 (the door is visible 6.9..8.3 s), then back fast

    def frame(self, t):
        y = self.yaw_at(t); img, _ = render_frame(y / 4.0, self.WORLD, self.CAM, 4.0, self.PANO); ok, buf = cv2.imencode(".jpg", img); return buf.tobytes(), y

    def test_there_is_no_gap_in_which_the_door_is_forgotten(self):
        s = LiveSession(ColorBlobDetector(), {"camera": "main", "has_heading": True}, ServerConfig(detector="blobs", alerts=False)); seen = False; gaps = []
        for k in range(int(20 * 3) + 1):
            t = k / 3.0; jpg, y = self.frame(t); r = s.process(jpg, {"seq": k, "t": t, "yaw": y}); has = any(v["name"] == "door" for v in r["entries"])
            seen = seen or has
            if seen and not has: gaps.append(round(t, 1))
        self.assertTrue(seen); self.assertEqual(gaps, [], f"the door vanished from the list at t = {gaps}")

    def test_the_assistant_answers_right_after_the_door_leaves_the_view(self):
        for tq in (9.0, 10.0, 11.0, 12.5):
            s = LiveSession(ColorBlobDetector(), {"camera": "main", "has_heading": True}, ServerConfig(detector="blobs", alerts=False))
            for k in range(int(tq * 3) + 1):
                t = k / 3.0; jpg, y = self.frame(t); s.process(jpg, {"seq": k, "t": t, "yaw": y})
            txt = RuleAgent(s).reply("где дверь?")[0]; self.assertIn("дверь", txt, tq); self.assertNotIn("Не вижу", txt, (tq, txt))
            if tq >= 10.0: self.assertIn("сейчас вне кадра", txt); self.assertIn("назад", txt)               # and says how long ago it was last seen


class TestMapOnEvent(unittest.TestCase):
    def test_the_map_is_rebuilt_the_moment_an_object_leaves_the_live_list(self):
        script = [[Detection(BOX, "door", 0.8, "door")]] * 4 + [[]] * 10
        cfg = ServerConfig(detector="blobs", alerts=False, map_interval_s=1000.0)                              # no timer at all: only the event can build the map
        s = LiveSession(Scripted(script), {"camera": "main"}, cfg); results = [s.process(H.jpeg_at(k / 3)[0], {"seq": k, "t": k / 3}) for k in range(14)]
        door = [[v for v in r["entries"] if v["name"] == "door"] for r in results]
        self.assertTrue(all(door[k] for k in range(4, 14)), [bool(d) for d in door]); self.assertTrue(all(d[0]["source"] == "map" for d in door[8:]))
        self.assertEqual(door[13][0]["status"], "remembered"); self.assertGreater(door[13][0]["age_s"], 3.0)

    def test_the_map_also_runs_on_a_short_timer(self):
        self.assertEqual(ServerConfig().map_interval_s, 1.0)


class TestHysteresis(unittest.TestCase):
    def test_a_weak_detection_cannot_start_a_track(self):
        self.assertEqual(Tracker(FastConfig()).update([det("person", BOX, 0.2)], 0.0, 0), []); self.assertEqual(len(Tracker(FastConfig()).update([det("person", BOX, 0.3)], 0.0, 0)), 1)

    def test_a_weak_detection_keeps_an_existing_track_alive_without_changing_its_name_confidence(self):
        script = [[Detection(BOX, "person", 0.8, "person")]] * 3 + [[Detection(BOX, "person", 0.18, "person")]] * 6 + [[]] * 2
        res, fp = run(script); ids = {tr["id"] for r in res for tr in r.tracks}
        self.assertEqual(ids, {1}); self.assertEqual([len(r.entries) for r in res[:9]], [1] * 9)                      # published all the way, one track
        self.assertAlmostEqual(res[8].entries[0].conf, 0.8, places=2)                                                  # weak looks did not dilute the confidence
        self.assertEqual(len(fp.tracker.active[0].obs), 3)

    def test_weak_detections_alone_never_produce_anything(self):
        res, _ = run([[Detection(BOX, "person", 0.2, "person")]] * 8); self.assertTrue(all(not r.entries and not r.tracks for r in res))

    def test_tracks_survive_a_long_gap_with_the_same_number(self):
        script = [[Detection(BOX, "door", 0.8, "door")]] * 2 + [[]] * 9 + [[Detection(BOX, "door", 0.8, "door")]] * 2
        res, fp = run(script); self.assertEqual({tr["id"] for r in res for tr in r.tracks}, {1})                            # 9 missed frames: still the same door


class TestPublishRules(unittest.TestCase):
    def test_an_entry_is_live_for_two_missed_frames_then_leaves_the_live_list(self):
        res, _ = run([[Detection(BOX, "person", 0.9, "person")]] * 3 + [[]] * 5); self.assertEqual([len(r.entries) for r in res], [1, 1, 1, 1, 1, 0, 0, 0])      # 3 sightings, then misses 1 and 2 still live, from the third miss on the live list lets go

    def test_classes_that_fired_falsely_need_more_sightings_and_have_no_shortcut(self):
        res, _ = run([[Detection(BOX, "billboard", 0.9, "billboard")]] * 5); self.assertEqual([len(r.entries) for r in res], [0, 0, 0, 1, 1])                    # 4th sighting
        res, _ = run([[Detection(BOX, "person", 0.9, "person")]] * 2); self.assertEqual([len(r.entries) for r in res], [1, 1])                                    # sure and ordinary: at once


if __name__ == "__main__": unittest.main()
