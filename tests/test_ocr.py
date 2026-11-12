import contextlib, io, os, tempfile, time, unittest
import cv2, numpy as np
import helpers, server_helpers as H
from scenelive.cli import main
from scenelive.detect.base import Detection
from scenelive.ocr import (BUS_ALLOW, EasyOcrEngine, TextHit, TextLog, TextReader, bench_ocr, bus_number, make_ocr, norm_key, sign_texts)
from scenelive.server.agent import LiveTools, RuleAgent, TOOL_SCHEMAS
from scenelive.server.config import ServerConfig
from scenelive.server.session import LiveSession
from scenelive.synth import render_video


class FakeOcr:
    """Returns prepared readings: for a bus crop (allowlist given) the route number, for anything else the sign text."""
    name = "fake"
    def __init__(self, bus=("12",), sign=("Аптека",), delay=0.0): self.bus, self.sign, self.delay, self.calls = list(bus), list(sign), delay, []
    def read(self, img, allowlist=None):
        self.calls.append(("bus" if allowlist else "text", img.shape)); time.sleep(self.delay); h = img.shape[0]
        if allowlist: return [TextHit(self.bus[min(len(self.calls) - 1, len(self.bus) - 1)], 0.9, (10, 10, 60, 10 + h * 0.3)), TextHit("ГОРОД", 0.8, (70, 10, 200, 25))]
        return [TextHit(t, 0.85, (20 + 100 * i, 20, 90 + 100 * i, 50)) for i, t in enumerate(self.sign)]


def tr(i, name, box): return {"id": i, "name": name, "box": box, "world_az": 30.0}


FRAME = np.zeros((360, 640, 3), np.uint8)


class TestParsing(unittest.TestCase):
    def test_bus_numbers(self):
        hits = [TextHit("12", 0.9, (0, 0, 40, 40)), TextHit("ГОРОД", 0.95, (0, 0, 200, 60)), TextHit("7", 0.5, (0, 0, 10, 10))]
        self.assertEqual(bus_number(hits)[0], "12")                                                         # the big, sure number; words are not numbers
        self.assertEqual(bus_number([TextHit("1O", 0.8, (0, 0, 10, 30))])[0], "10"); self.assertEqual(bus_number([TextHit("37А", 0.8, (0, 0, 10, 30))])[0], "37А")
        self.assertIsNone(bus_number([TextHit("1234", 0.9, (0, 0, 10, 30))])); self.assertIsNone(bus_number([TextHit("12", 0.1, (0, 0, 10, 30))])); self.assertIsNone(bus_number([]))
        for ch in "0123456789": self.assertIn(ch, BUS_ALLOW)

    def test_sign_texts_drop_noise(self):
        out = sign_texts([TextHit(" Аптека. ", 0.9, (0, 0, 1, 1)), TextHit("x", 0.9, (0, 0, 1, 1)), TextHit("-- ..", 0.9, (0, 0, 1, 1)), TextHit("Дәріхана", 0.8, (0, 0, 1, 1)), TextHit("Магазин", 0.1, (0, 0, 1, 1))])
        self.assertEqual([t for t, _ in out], ["Аптека", "Дәріхана"])                                       # Kazakh letters are kept

    def test_the_list_of_texts(self):
        log = TextLog(max_items=3); log.add("Аптека", "sign", "street sign", 10.0, 1.0, 0.8); log.add("АПТЕКА!", "sign", "street sign", 12.0, 5.0, 0.9)
        self.assertEqual(len(log.items), 1); it = log.recent(1)[0]; self.assertEqual((it.count, it.last_t, it.conf), (2, 5.0, 0.9))       # same letters = same text
        for k, t in enumerate(("Хлеб", "Банк", "Кафе")): log.add(t, "sign", "sign", None, 10.0 + k, 0.7)
        self.assertEqual(len(log.items), 3); self.assertEqual([i.text for i in log.recent(5)], ["Кафе", "Банк", "Хлеб"]); self.assertEqual(norm_key(" A-b c "), "abc")


class TestReader(unittest.TestCase):
    def test_nothing_is_read_until_a_bus_or_a_sign_is_found(self):
        ocr = FakeOcr(); r = TextReader(ocr, sync=True)
        for k in range(10): r.consider(FRAME, k / 3, [tr(1, "person", (100, 50, 300, 300)), tr(2, "car", (300, 100, 500, 300))], (640, 360), lambda t: 0.0)
        self.assertEqual(ocr.calls, [])

    def test_a_bus_is_read_inside_its_box_until_two_readings_agree(self):
        ocr = FakeOcr(bus=("12", "12", "12")); r = TextReader(ocr, every_s=1.0, sync=True)
        for k in range(15): r.consider(FRAME, k / 3, [tr(7, "bus", (100, 60, 500, 300))], (640, 360), lambda t: 30.0)
        self.assertEqual([c[0] for c in ocr.calls], ["bus", "bus"]); self.assertEqual(r.reading(7), ("12", 2)); self.assertTrue(r.jobs[7].done)       # agreed: no more reading
        self.assertLessEqual(ocr.calls[0][1][0], 300); self.assertLessEqual(ocr.calls[0][1][1], 450)                                                 # a crop, not the frame
        self.assertIn("автобус12", {norm_key(i.text) for i in r.texts.recent()})

    def test_a_far_bus_waits_until_it_is_bigger(self):
        ocr = FakeOcr(); r = TextReader(ocr, sync=True); r.consider(FRAME, 0.0, [tr(1, "bus", (100, 100, 140, 130))], (640, 360), lambda t: 0.0); self.assertEqual(ocr.calls, [])

    def test_it_gives_up_after_a_few_tries(self):
        ocr = FakeOcr(bus=("1", "22", "333", "4", "55", "6", "77", "8", "99", "10")); r = TextReader(ocr, every_s=1.0, sync=True, max_tries=4)
        for k in range(40): r.consider(FRAME, k / 3, [tr(3, "bus", (100, 60, 500, 300))], (640, 360), lambda t: 0.0)
        self.assertEqual(len(ocr.calls), 4); self.assertTrue(r.jobs[3].done)

    def test_signs_go_to_the_list(self):
        ocr = FakeOcr(sign=("Аптека", "24 часа")); r = TextReader(ocr, sync=True); r.consider(FRAME, 0.0, [tr(4, "billboard", (100, 60, 300, 120))], (640, 360), lambda t: 0.0)
        self.assertEqual(ocr.calls[0][0], "text"); self.assertEqual({i.text for i in r.texts.recent()}, {"Аптека", "24 часа"})

    def test_reading_in_the_background_does_not_block_and_never_queues(self):
        ocr = FakeOcr(delay=0.3); r = TextReader(ocr, sync=False); t0 = time.perf_counter()
        r.consider(FRAME, 0.0, [tr(1, "bus", (100, 60, 500, 300))], (640, 360), lambda t: 0.0); self.assertLess(time.perf_counter() - t0, 0.1); self.assertTrue(r.busy)
        r.consider(FRAME, 2.0, [tr(1, "bus", (100, 60, 500, 300))], (640, 360), lambda t: 0.0)                                                     # busy: skipped, not queued
        for _ in range(30):
            if not r.busy: break
            time.sleep(0.05)
        self.assertEqual(len(ocr.calls), 1); self.assertEqual(r.reading(1), ("12", 1))

    def test_what_is_written_reads_the_frame_left_to_right(self):
        ocr = FakeOcr(sign=("Аптека", "Хлеб")); r = TextReader(ocr, sync=True); out = r.read_now(FRAME, 5.0, lambda x: x / 10.0)
        self.assertEqual([t for t, _, _ in out], ["Аптека", "Хлеб"]); self.assertEqual(ocr.calls[0], ("text", (360, 640, 3))); self.assertEqual(len(r.texts.items), 2)


class BusDetector:
    name = "bus-detector"
    def detect(self, frame): return [Detection((100.0, 60.0, 500.0, 300.0), "bus", 0.8, "bus")]


def session(ocr, detector=None, frames=10):
    s = LiveSession(detector or BusDetector(), {"camera": "main"}, ServerConfig(detector="blobs", alerts=False, ocr_sync=True), ocr_engine=ocr); res = []
    for k in range(frames): res.append(s.process(H.jpeg_at(k / 3)[0], {"seq": k, "t": k / 3}))
    return s, res


class TestAssistant(unittest.TestCase):
    def test_the_bus_in_the_list_carries_its_number(self):
        s, res = session(FakeOcr(bus=("12",))); bus = [v for v in res[-1]["entries"] if v["name"] == "bus"][0]; self.assertEqual((bus["text"], bus["text_reads"]), ("12", 2))

    def test_which_bus_is_it(self):
        s, _ = session(FakeOcr(bus=("12",))); txt, used = RuleAgent(s).reply("Какой номер автобуса?"); self.assertEqual(used, ["bus_numbers"])
        self.assertTrue(txt.startswith("Автобус 12"), txt); self.assertIn("уточнить у водителя", txt)
        once, _ = session(FakeOcr(bus=("12",)), frames=2); self.assertTrue(RuleAgent(once).reply("какой автобус?")[0].startswith("Возможно, автобус 12"))       # one reading: 'possibly'

    def test_no_bus_no_reading(self):
        from scenelive.detect.blob import ColorBlobDetector
        ocr = FakeOcr(); s, _ = session(ocr, detector=ColorBlobDetector()); self.assertEqual(ocr.calls, []); self.assertEqual(RuleAgent(s).reply("какой номер автобуса?")[0], "Автобусов сейчас не вижу.")

    def test_what_is_written_and_the_list_of_texts(self):
        s, _ = session(FakeOcr(sign=("Аптека",))); a = RuleAgent(s)
        txt, used = a.reply("Что написано?"); self.assertEqual(used, ["read_text"]); self.assertTrue(txt.startswith("Написано: «Аптека»")); self.assertIn("Возможны ошибки", txt)
        lst, used = a.reply("какие надписи вокруг?"); self.assertEqual(used, ["list_texts"]); self.assertIn("«Аптека»", lst); self.assertIn("«автобус 12»", lst)

    def test_reading_switched_off(self):
        from scenelive.detect.blob import ColorBlobDetector
        s = LiveSession(ColorBlobDetector(), {}, ServerConfig(detector="blobs")); a = RuleAgent(s)
        for q in ("что написано?", "какие надписи?", "номер автобуса?"): self.assertIn("--ocr easyocr", a.reply(q)[0])
        self.assertFalse(LiveTools(s).read_text()["available"])

    def test_tools_for_the_model(self):
        s, _ = session(FakeOcr()); t = LiveTools(s)
        self.assertEqual(t.bus_numbers()["buses"][0]["number"], "12"); self.assertTrue(t.list_texts()["count"] >= 1); self.assertEqual(t.read_text()["texts"][0]["text"], "Аптека")
        self.assertTrue({"read_text", "list_texts", "bus_numbers"} <= {x["name"] for x in TOOL_SCHEMAS})


class TestTools(unittest.TestCase):
    def test_a_missing_easyocr_is_explained(self):
        with self.assertRaises(ImportError) as c: EasyOcrEngine()
        self.assertIn("setup_ocr", str(c.exception)); self.assertIsNone(make_ocr("none"))
        with self.assertRaises(ValueError): make_ocr("magic")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = main(["bench-ocr", "x.jpg"])
        self.assertEqual(code, 2); self.assertIn("easyocr", buf.getvalue())

    def test_bench_on_photos_and_on_a_video(self):
        d = tempfile.mkdtemp(); pics = os.path.join(d, "pics"); os.makedirs(pics)
        for k in range(3): cv2.imencode(".jpg", np.full((200, 300, 3), 80 * k, np.uint8))[1].tofile(os.path.join(pics, f"p{k}.jpg"))
        out = os.path.join(d, "o"); s = bench_ocr(pics, FakeOcr(sign=("Аптека",)), out, log=lambda *a: None)
        self.assertEqual(s["pictures"], 3); res = open(os.path.join(out, "results.txt"), encoding="utf-8").read(); self.assertIn("Аптека", res); self.assertTrue(os.path.exists(os.path.join(out, "p0_ocr.jpg")))
        v = os.path.join(d, "v.mp4"); render_video(v, duration=3.0); s2 = bench_ocr(v, FakeOcr(), os.path.join(d, "o2"), n_frames=4, bus=True, log=lambda *a: None)
        self.assertEqual(s2["pictures"], 4); self.assertIn("bus number: 12", open(os.path.join(d, "o2", "results.txt"), encoding="utf-8").read())


if __name__ == "__main__": unittest.main()
