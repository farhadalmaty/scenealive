import asyncio, json, time, unittest
import numpy as np
import helpers, server_helpers as H
from scenelive.camera import Camera
from scenelive.detect.base import Detection
from scenelive.detect.blob import ColorBlobDetector
from scenelive.freespace import UNSURE, row_for_forward
from scenelive.seg_monitor import SegMonitor
from scenelive.server.agent import LiveTools, RuleAgent, TOOL_SCHEMAS
from scenelive.server.app import ServerApp, unpack_frame
from scenelive.server.client_sim import pack_frame, ws_connect
from scenelive.server.config import ServerConfig
from scenelive.server.runner import ServerThread
from scenelive.server.session import LiveSession

IDS = {"sky": 0, "floor": 1, "wall": 2, "road": 3, "person": 4, "door": 5, "stairs": 6, "glass": 7, "grass": 8, "chair": 9}
ID2 = {0: "sky", 1: "floor, flooring", 2: "wall", 3: "road, route", 4: "person, individual", 5: "door", 6: "stairs, steps", 7: "windowpane, window", 8: "grass", 9: "chair"}


class GeoSegmenter:
    """A segmentation 'model' that paints what the exact geometry says: floor, with a wall (or anything) at a chosen distance."""
    name = "geo"; last_conf = None
    def __init__(self, wall_forward=None, blocker="wall", fill="floor", pitch=15.0, h_cam=1.4, fail=False):
        self.wall_forward, self.blocker, self.fill, self.pitch, self.h_cam, self.fail, self.calls = wall_forward, blocker, fill, pitch, h_cam, fail, 0
    def segment(self, frame):
        self.calls += 1
        if self.fail: raise RuntimeError("model crashed")
        h, w = frame.shape[:2]; cam = Camera(w, h, 69.0); lab = np.full((h, w), IDS[self.fill], np.int32)
        lab[: int(row_for_forward(40.0, cam, self.pitch, self.h_cam))] = IDS["sky"]
        if self.wall_forward is not None: lab[: int(np.floor(row_for_forward(self.wall_forward, cam, self.pitch, self.h_cam)))] = IDS[self.blocker]
        return lab, ID2


class Door:
    name = "door-detector"
    def detect(self, frame): return [Detection((300.0, 100.0, 340.0, 250.0), "door", 0.8, "door")]


def session(seg, detector=None, frames=8, pitch=15.0, **cfg):
    c = ServerConfig(detector="blobs", seg_sync=True, **cfg); s = LiveSession(detector or ColorBlobDetector(), {"camera": "main"}, c, segmenter=seg); res = []
    for k in range(frames):
        meta = {"seq": k, "t": k / 3}
        if pitch is not None: meta["pitch"] = pitch
        res.append(s.process(H.jpeg_at(k / 3)[0], meta))
    return s, res


class TestSegMonitor(unittest.TestCase):
    def cam(self): return Camera(640, 360, 69.0)

    def test_it_looks_about_once_a_second_and_reports_the_age(self):
        m = SegMonitor(GeoSegmenter(), every_s=1.0, sync=True); frame = np.zeros((360, 640, 3), np.uint8); ran = []
        for k in range(10):
            t = k / 3.0
            if m.due(t): m.submit(frame, t, 15.0, self.cam()); ran.append(round(t, 2))
        self.assertEqual(ran, [0.0, 1.0, 2.0, 3.0]); fs, age = m.state(3.2); self.assertIsNotNone(fs); self.assertAlmostEqual(age, 0.2, places=6)

    def test_an_old_picture_of_the_floor_counts_as_unknown(self):
        m = SegMonitor(GeoSegmenter(), sync=True, max_age_s=3.0); m.submit(np.zeros((360, 640, 3), np.uint8), 0.0, 15.0, self.cam())
        self.assertIsNotNone(m.state(2.9)[0]); fs, age = m.state(3.5); self.assertIsNone(fs); self.assertAlmostEqual(age, 3.5)
        self.assertEqual(SegMonitor(GeoSegmenter(), sync=True).state(1.0), (None, None))

    def test_a_crashing_model_is_counted_not_fatal(self):
        m = SegMonitor(GeoSegmenter(fail=True), sync=True, log=lambda *a: None); m.submit(np.zeros((360, 640, 3), np.uint8), 0.0, 15.0, self.cam())
        self.assertEqual(m.errors, 1); self.assertIsNone(m.state(0.1)[0]); self.assertEqual(m.stats()["runs"], 0)

    def test_the_background_thread_delivers_without_blocking_the_caller(self):
        class Slow(GeoSegmenter):
            def segment(self, f): time.sleep(0.25); return super().segment(f)
        m = SegMonitor(Slow(wall_forward=4.0), every_s=1.0, sync=False); t0 = time.perf_counter(); m.submit(np.zeros((360, 640, 3), np.uint8), 0.0, 15.0, self.cam())
        self.assertLess(time.perf_counter() - t0, 0.1); self.assertFalse(m.due(0.5)); self.assertEqual(m.state(0.5)[0], None)                       # busy: not due, no answer yet
        for _ in range(40):
            if m.state(0.6)[0] is not None: break
            time.sleep(0.05)
        self.assertIsNotNone(m.state(0.6)[0]); self.assertGreater(m.stats()["mean_ms"], 200); m.close()


class TestSession(unittest.TestCase):
    def test_the_result_carries_the_free_space_and_the_advice(self):
        s, res = session(GeoSegmenter(wall_forward=4.0)); fs = res[-1]["free_space"]
        self.assertTrue(fs["available"]); self.assertEqual(len(fs["sectors"]), 5); self.assertTrue(fs["reliable"]); self.assertEqual(fs["underfoot"], "safe"); json.dumps(res[-1])
        self.assertIn("дальше стена", fs["advice"]["text"]); self.assertEqual(fs["advice"]["level"], "ok"); self.assertEqual(fs["sectors"][2]["blocker"], "wall")

    def test_without_the_tilt_of_the_phone_it_says_it_is_not_sure(self):
        s, res = session(GeoSegmenter(wall_forward=4.0), pitch=None); self.assertEqual(res[-1]["free_space"]["advice"]["text"], UNSURE); self.assertFalse(res[-1]["free_space"]["reliable"])

    def test_before_the_first_picture_and_without_segmentation(self):
        self.assertFalse(session(GeoSegmenter(), frames=1, pitch=15.0)[1][0]["free_space"]["available"] is False)                       # sync: ready at once
        s = LiveSession(ColorBlobDetector(), {}, ServerConfig(detector="blobs")); self.assertIsNone(s.process(H.jpeg_at(0.0)[0], {"seq": 0, "t": 0.0})["free_space"])

    def test_the_picture_goes_stale_when_the_model_stops_answering(self):
        seg = GeoSegmenter(wall_forward=4.0); s, res = session(seg, frames=3); seg.fail = True
        out = [s.process(H.jpeg_at(k / 3)[0], {"seq": k, "t": k / 3, "pitch": 15.0}) for k in range(3, 24)]
        self.assertTrue(out[0]["free_space"]["available"]); self.assertFalse(out[-1]["free_space"]["available"])                                       # more than 3 s since the last good picture

    def test_stats_show_the_cost_of_the_model(self):
        s, _ = session(GeoSegmenter(), frames=8); self.assertEqual(s.stats()["seg"]["errors"], 0); self.assertGreater(s.stats()["seg"]["runs"], 1)


class TestAlertsFromTheFloor(unittest.TestCase):
    def alerts(self, seg, pitch=15.0):
        s, res = session(seg, pitch=pitch, frames=12); return [a for r in res for a in r["alerts"] if a["name"] in ("stop", "road")]

    def test_walking_into_a_wall_is_a_stop(self):
        a = self.alerts(GeoSegmenter(fill="wall")); self.assertEqual(len(a), 1); self.assertEqual(a[0]["level"], "danger"); self.assertTrue(a[0]["text"].startswith("Под ногами") or "Стоп" in a[0]["text"]); self.assertIn("Стоп", a[0]["text"])

    def test_standing_on_the_road_is_announced(self):
        a = self.alerts(GeoSegmenter(fill="road")); self.assertEqual([x["text"] for x in a], ["Внимание! Похоже, вы на проезжей части."])

    def test_a_free_floor_and_an_unsure_picture_stay_quiet(self):
        self.assertEqual(self.alerts(GeoSegmenter(wall_forward=6.0)), []); self.assertEqual(self.alerts(GeoSegmenter(fill="wall"), pitch=None), [])        # no tilt: no warnings from geometry


class TestAssistant(unittest.TestCase):
    def test_which_way_to_go(self):
        s, _ = session(GeoSegmenter(wall_forward=4.0)); txt, used = RuleAgent(s).reply("Куда идти?")
        self.assertEqual(used, ["where_to_go"]); self.assertTrue(txt.startswith("Прямо свободно")); self.assertIn("дальше стена", txt); self.assertIn("Камера может не заметить стекло, ступеньку вниз или яму", txt); self.assertNotIn("безопасно", txt.lower())

    def test_where_is_the_wall(self):
        s, _ = session(GeoSegmenter(wall_forward=4.0)); txt, used = RuleAgent(s).reply("где стена?"); self.assertEqual(used, ["free_space"]); self.assertTrue(txt.startswith("Стена: впереди около 3 метров"), txt)
        s2, _ = session(GeoSegmenter()); self.assertIn("Стен рядом не вижу", RuleAgent(s2).reply("где стена?")[0])

    def test_take_me_to_the_door(self):
        s, _ = session(GeoSegmenter(wall_forward=9.0), detector=Door()); txt, used = RuleAgent(s).reply("веди к двери"); self.assertEqual(used, ["where_to_go"]); self.assertTrue(txt.startswith("Дверь"), txt)
        self.assertIn("свободным", txt); self.assertIn("Камера может не заметить", txt)
        blocked, _ = session(GeoSegmenter(wall_forward=3.0, blocker="person"), detector=Door()); self.assertIn("На пути к дверь человек или транспорт", RuleAgent(blocked).reply("как пройти к двери")[0])

    def test_it_does_not_pretend_when_it_cannot_see(self):
        s, _ = session(GeoSegmenter(wall_forward=4.0), pitch=None); txt, _ = RuleAgent(s).reply("можно ли идти?"); self.assertIn("Остановитесь и не полагайтесь", txt); self.assertNotIn("Камера может не заметить", txt)
        off = LiveSession(ColorBlobDetector(), {}, ServerConfig(detector="blobs")); self.assertIn("--seg segformer", RuleAgent(off).reply("куда идти?")[0])

    def test_the_tools_for_the_model(self):
        s, _ = session(GeoSegmenter(wall_forward=4.0)); t = LiveTools(s); r = t.where_to_go(); self.assertTrue(r["available"]); self.assertIn("caveat", r); self.assertEqual(r["level"], "ok")
        f = t.free_space(); self.assertTrue(f["known"]); self.assertEqual(len(f["sectors"]), 5); self.assertFalse(LiveTools(LiveSession(ColorBlobDetector(), {}, ServerConfig(detector="blobs"))).free_space()["available"])
        self.assertTrue({"where_to_go", "free_space"} <= {x["name"] for x in TOOL_SCHEMAS}); self.assertFalse(t.where_to_go("zebra")["found"])


class TestOverTheWire(unittest.TestCase):
    def test_pitch_travels_in_the_frame_and_the_free_space_comes_back(self):
        meta, _ = unpack_frame(pack_frame({"seq": 1, "t": 0.5, "pitch": 12.5}, b"jpeg")); self.assertEqual(meta["pitch"], 12.5)
        with self.assertRaises(ValueError): unpack_frame(pack_frame({"pitch": float("nan")}, b"jpeg"))
        cfg = ServerConfig(detector="blobs", agent="offline", https=False, seg_sync=True); app = ServerApp(cfg, ColorBlobDetector(), log=lambda *a: None, segmenter=GeoSegmenter(wall_forward=4.0))
        async def go(port):
            ws = await ws_connect("127.0.0.1", port, "/ws"); await ws.send_text(json.dumps({"type": "hello", "camera": "main"})); ready = json.loads((await ws.recv())[1]); out = []
            for k in range(4):
                await ws.send_binary(pack_frame({"seq": k, "t": k / 3, "pitch": 15.0}, H.jpeg_at(k / 3)[0])); out.append(json.loads((await asyncio.wait_for(ws.recv(), 5))[1]))
            await ws.send_text(json.dumps({"type": "chat", "id": "1", "text": "куда идти?"})); chat = json.loads((await asyncio.wait_for(ws.recv(), 5))[1]); await ws.close(); return ready, out, chat
        with ServerThread(app) as th: ready, out, chat = asyncio.run(go(th.port))
        self.assertTrue(ready["seg_enabled"]); self.assertTrue(out[-1]["free_space"]["available"]); self.assertIn("дальше стена", out[-1]["free_space"]["advice"]["text"]); self.assertEqual(chat["tools"], ["where_to_go"])


if __name__ == "__main__": unittest.main()
