import json, unittest
import helpers, server_helpers as H
from scenelive.alerts import AlertEngine
from scenelive.detect.base import Detection
from scenelive.server.config import ServerConfig
from scenelive.server.session import LiveSession
from scenelive.speech import direction_words, metres, range_speech


def v(name, status="confirmed", rel=0.0, rng=None, conf=0.8, n=5, age=0.0, clock=12):
    return {"name": name, "status": status, "rel_az_deg": rel, "range_m": rng, "range": "?" if rng is None else f"~{rng:g} m", "conf": conf, "n_obs": n, "age_s": age, "clock": clock, "behind": False}


class TestSpeechWords(unittest.TestCase):
    def test_metres_declension(self):
        self.assertEqual([metres(n) for n in (1, 2, 5, 10, 11, 15, 21)], ["1 метра", "2 метров", "5 метров", "10 метров", "11 метров", "15 метров", "21 метра"]); self.assertEqual(metres(2.5), "2.5 метра")

    def test_range_phrases(self):
        self.assertEqual([range_speech(x) for x in ("~5 m", "~1 m", "5-10 m", "15+ m", "?", None)], ["около 5 метров", "около 1 метра", "от 5 до 10 метров", "дальше 15 метров", "", ""])

    def test_directions(self):
        self.assertEqual((direction_words(12), direction_words(3), direction_words(11), direction_words(6)), ("прямо по курсу", "справа", "слева-спереди", "сзади"))


class TestAlertRules(unittest.TestCase):
    def test_a_dangerous_object_is_announced_with_direction_and_distance(self):
        a = AlertEngine(); out = a.update([v("knife", clock=2, rel=60, rng=3.0, conf=0.8)], None, 10.0)
        self.assertEqual(len(out), 1); self.assertEqual(out[0]["level"], "danger"); self.assertTrue(out[0]["text"].startswith("Осторожно! Опасность: нож, справа-спереди")); self.assertIn("метров", out[0]["text"]); json.dumps(out)

    def test_an_unsure_detection_says_possibly(self):
        self.assertIn("возможно, нож", AlertEngine().update([v("knife", conf=0.4)], None, 10.0)[0]["text"])
        self.assertNotIn("возможно", AlertEngine().update([v("knife", conf=0.8)], None, 10.0)[0]["text"])

    def test_weak_remembered_or_barely_seen_dangers_stay_silent(self):
        a = AlertEngine()
        for item in (v("knife", n=2), v("knife", conf=0.2), v("knife", status="remembered"), v("knife", age=4.0)): self.assertEqual(a.update([item], None, 100.0 + id(item) % 7 * 100), [], item)

    def test_the_same_thing_is_not_repeated_but_other_directions_are_new(self):
        a = AlertEngine(cooldown_s=15.0, min_gap_s=2.5); k = v("knife", rel=0.0)
        self.assertEqual(len(a.update([k], None, 10.0)), 1); self.assertEqual(a.update([k], None, 20.0), []); self.assertEqual(len(a.update([k], None, 26.0)), 1)      # again after the cooldown
        left = v("knife", rel=-90.0, clock=9); self.assertEqual(len(a.update([left], None, 29.0)), 1)                                                         # another sector: a separate alert

    def test_never_more_than_one_alert_per_gap_and_danger_goes_first(self):
        a = AlertEngine(min_gap_s=2.5); both = [v("car", rng=4.0), v("knife", rel=90.0, clock=3)]
        first = a.update(both, None, 10.0); self.assertEqual([x["name"] for x in first], ["knife"]); self.assertEqual(a.update(both, None, 11.0), [])          # the gap
        second = a.update(both, None, 13.0); self.assertEqual([x["name"] for x in second], ["car"])                                                           # then the next one

    def test_close_vehicles_ahead_only(self):
        a = AlertEngine()
        self.assertEqual(len(a.update([v("car", rng=5.0, rel=20.0)], None, 10.0)), 1)
        for it in (v("car", rng=10.0), v("car", rng=4.0, rel=100.0), v("person", rng=2.0), v("car", rng=None)): self.assertEqual(AlertEngine().update([it], None, 10.0), [], it)

    def test_holes_and_ice_straight_ahead_without_a_distance(self):
        self.assertEqual(len(AlertEngine().update([v("pothole", rel=10.0)], None, 10.0)), 1)
        self.assertEqual(AlertEngine().update([v("pothole", rel=70.0)], None, 10.0), []); self.assertEqual(AlertEngine().update([v("pothole", n=2)], None, 10.0), [])

    def test_bad_surface_straight_ahead(self):
        bad = {"center": {"surface": "pothole", "ru": "яма на дороге", "quality_ru": "плохое", "bad": True}, "left": {"bad": True, "surface": "mud", "ru": "грязь", "quality_ru": "плохое"}}
        a = AlertEngine(); out = a.update([], bad, 10.0); self.assertEqual(out[0]["text"], "Внимание: впереди яма на дороге, покрытие плохое."); self.assertEqual(a.update([], bad, 12.0), [])
        self.assertEqual(AlertEngine().update([], {"center": {"bad": False, "surface": "asphalt", "ru": "асфальт", "quality_ru": "хорошее"}}, 10.0), [])      # the sides alone do not alert

    def test_quiet_when_nothing_is_wrong(self):
        self.assertEqual(AlertEngine().update([v("person", rng=2.0), v("tree"), v("wall"), v("car", rng=20.0)], None, 10.0), [])


class KnifeDetector:
    name = "knife-detector"
    def detect(self, frame): return [Detection((300.0, 100.0, 340.0, 250.0), "knife", 0.8, "knife")]


class TestAlertsInTheSession(unittest.TestCase):
    def test_the_session_announces_a_knife_once_it_has_been_seen_enough_and_not_again(self):
        s = LiveSession(KnifeDetector(), {"camera": "main"}, ServerConfig(detector="blobs")); got = []
        for k in range(12):
            r = s.process(H.jpeg_at(k / 3)[0], {"seq": k, "t": k / 3}); got.append(r["alerts"])
        self.assertEqual([len(x) for x in got[:2]], [0, 0]); self.assertEqual(sum(len(x) for x in got), 1)                      # 3 sightings needed; one alert in 4 s thanks to the cooldown
        alert = next(x for g in got for x in g); self.assertEqual((alert["level"], alert["name"]), ("danger", "knife")); self.assertIn("нож", alert["text"])

    def test_alerts_can_be_switched_off(self):
        cfg = ServerConfig(detector="blobs"); cfg.alerts = False; s = LiveSession(KnifeDetector(), {}, cfg)
        self.assertTrue(all(s.process(H.jpeg_at(k / 3)[0], {"seq": k, "t": k / 3})["alerts"] == [] for k in range(8)))


if __name__ == "__main__": unittest.main()
