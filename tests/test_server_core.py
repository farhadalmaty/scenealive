import json, os, types, unittest
import helpers, server_helpers as H
from scenelive.camera import angdiff
from scenelive.server.agent import (ChatAgent, LiveTools, RuleAgent, TOOL_SCHEMAS, direction_center, make_agent, resolve_names)
from scenelive.server.config import ServerConfig
from scenelive.server.session import ClientCompass, LiveSession, camera_from_hello
from scenelive.detect.blob import ColorBlobDetector


class TestSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, cls.res = H.make_session(has_heading=True, t_end=9.0, base=350.0)        # the phone reports a compass heading that wraps through 0

    def test_result_shape_and_timings(self):
        r = self.res
        for k in ("type", "seq", "t", "heading", "entries", "boxes", "timings_ms", "compute_ms", "n_det", "size"): self.assertIn(k, r)
        for st in ("server_decode", "preprocess", "compass", "detect", "track", "name", "locate", "list"): self.assertIn(st, r["timings_ms"], st)
        self.assertAlmostEqual(sum(r["timings_ms"].values()), r["compute_ms"], delta=0.5); json.dumps(r)             # must be JSON-serialisable

    def test_heading_is_relative_and_survives_the_360_wrap(self):
        self.assertAlmostEqual(self.res["heading"], 36.0, delta=0.5)                       # 9 s * 4 deg/s; absolute headings went 350 -> 26

    def test_the_list_has_the_objects_in_the_right_directions(self):
        by = {v["name"]: v for v in self.res["entries"]}
        for n in ("person", "car", "pole", "door"): self.assertIn(n, by)
        self.assertLess(abs(angdiff(by["pole"]["world_az_deg"], 40.0)), 3.0); self.assertLess(abs(angdiff(by["door"]["world_az_deg"], 62.0)), 3.0)
        self.assertEqual(by["door"]["clock"], 1)                                          # 62-36 = 26 deg right of where the camera points: 1 o'clock
        self.assertTrue(4.3 <= by["door"]["range_m"] <= 6.0); self.assertIsNone(by["pole"]["range_m"])
        self.assertEqual((by["person"]["status"], by["person"]["age_s"]), ("confirmed", 0.0))     # in view now: shown live, not as of the last map run

    def test_the_map_loop_ran_and_merged_tracks(self):
        self.assertTrue(any(v["source"] == "map" for v in self.res["entries"]))
        self.assertTrue(all(v["age_s"] >= 0 for v in self.res["entries"]))

    def test_boxes_are_normalised(self):
        self.assertTrue(self.res["boxes"])
        for b in self.res["boxes"]: self.assertTrue(0.0 <= b["x0"] < b["x1"] <= 1.0 and 0.0 <= b["y0"] < b["y1"] <= 1.0, b)

    def test_without_a_phone_compass_the_visual_one_takes_over(self):
        s, r = H.make_session(has_heading=False, t_end=9.0)
        self.assertEqual(s.compass.name, "visual"); self.assertAlmostEqual(r["heading"], 36.0, delta=1.5)

    def test_reset_starts_a_new_walk(self):
        s, _ = H.make_session(t_end=4.0); s.reset(); self.assertEqual((s.frames, s.t, s.heading, s.view()), (0, 0.0, 0.0, []))

    def test_a_broken_frame_is_an_error_not_a_crash(self):
        with self.assertRaises(ValueError): self.s.process(b"not a jpeg", {"seq": 1, "t": 1.0})

    def test_old_tracks_are_forgotten_and_their_number_is_capped(self):
        cfg = ServerConfig(detector="blobs", memory_minutes=1.0, max_tracks=5); s, _ = H.make_session(t_end=3.0, cfg=cfg)
        for i in range(20):
            tr = helpers.make_track(100 + i, "pole", 0.0, 1.0, 10.0); s.fast.tracker.archive.append(tr)
        s._forget(30.0); self.assertEqual(len(s.fast.tracker.archive), 5)                 # capped
        s._forget(500.0); self.assertEqual(len(s.fast.tracker.archive), 0)                # older than a minute: gone

    def test_camera_from_the_hello_message(self):
        self.assertEqual(camera_from_hello({"camera": "ultrawide"}).fov_long_deg, 100.0); self.assertEqual(camera_from_hello({"camera": "tele"}).fov_long_deg, 38.0)
        self.assertEqual(camera_from_hello({}).fov_long_deg, 69.0); self.assertEqual(camera_from_hello({"camera": "main", "fov_long_deg": 85}).fov_long_deg, 85.0)
        self.assertEqual(camera_from_hello({"camera": "main", "fov_long_deg": 3}).fov_long_deg, 69.0)    # absurd values are ignored
        self.assertEqual(camera_from_hello({"camera": "nonsense"}).fov_long_deg, 69.0)

    def test_the_camera_choice_changes_the_distance_by_the_focal_ratio(self):
        a = {v["name"]: v for v in H.make_session(t_end=3.0, camera="main")[1]["entries"]}["person"]["range_m"]
        b = {v["name"]: v for v in H.make_session(t_end=3.0, camera="ultrawide")[1]["entries"]}["person"]["range_m"]
        import math; self.assertAlmostEqual(b / a, math.tan(math.radians(34.5)) / math.tan(math.radians(50.0)), delta=0.02)

    def test_client_compass_accumulates_differences(self):
        c = ClientCompass(); out = []
        for a in (350.0, 355.0, 5.0, 30.0): c.set(a); out.append(c.update(None, 0, None))
        self.assertEqual(out, [0.0, 5.0, 15.0, 40.0]); c.set(None); self.assertEqual(c.update(None, 0, None), 40.0)

    def test_stats_for_the_status_page(self):
        st = self.s.stats(); self.assertEqual(st["frames"], self.s.frames); self.assertTrue(any(r["stage"] == "detect" for r in st["stages"]))


class TestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, _ = H.make_session(t_end=9.0); cls.t = LiveTools(cls.s)

    def test_summary(self):
        r = self.t.scene_summary(); self.assertGreaterEqual(r["objects"], 4); self.assertIn("person", r["by_name"]); self.assertIn("ESTIMATE", r["note"])

    def test_where_is_in_russian_and_english(self):
        for q in ("человек", "где человек", "person", "People"):
            r = self.t.where_is(q); self.assertTrue(r["found"], q); self.assertEqual(r["objects"][0]["name"], "person")
        r = self.t.where_is("автобус"); self.assertFalse(r["found"]); self.assertIn("person", r["known_names_now"])

    def test_directions(self):
        right = self.t.objects_in_direction("right", 180.0); self.assertTrue(any(o["name"] == "door" for o in right["objects"]))        # the door is 26 deg right of ahead
        ahead = self.t.objects_in_direction("ahead", 90.0); self.assertTrue({"door", "person"} <= {o["name"] for o in ahead["objects"]}); self.assertEqual(self.t.objects_in_direction("справа")["count"], 0)
        self.assertIn("error", self.t.objects_in_direction("sideways")); self.assertEqual(self.t.objects_in_direction("clock 3")["direction_deg"], 90.0)

    def test_nearest_skips_unknown_ranges_and_remembered_objects(self):
        r = self.t.nearest_objects(5); self.assertTrue(r["objects"])
        self.assertTrue(all(o["range_m"] is not None and o["status"] != "remembered" for o in r["objects"]))
        self.assertEqual([o["range_m"] for o in r["objects"]], sorted(o["range_m"] for o in r["objects"]))

    def test_unknown_tool_is_an_error_dict(self):
        self.assertIn("error", self.t.call("fly", {})); self.assertEqual(self.t.call("list_objects", {"limit": 2})["count"], len(self.s.view()))

    def test_name_and_direction_parsing(self):
        self.assertEqual(resolve_names("где машина?"), {"car", "van"}); self.assertEqual(resolve_names("кот"), {"cat"}); self.assertEqual(resolve_names("который час"), set())
        self.assertIn("traffic light", resolve_names("is there a traffic light")); self.assertIn("pole", resolve_names("фонарный столб"))
        for d, c in (("слева", -90.0), ("right", 90.0), ("впереди", 0.0), ("сзади", 180.0), ("clock 9", -90.0), ("clock 6", 180.0), (45, 45.0), ("-30", -30.0)): self.assertEqual(direction_center(d), c, d)
        self.assertIsNone(direction_center("вверх"))

    def test_every_tool_has_a_schema_and_exists(self):
        for sch in TOOL_SCHEMAS: self.assertTrue(callable(getattr(self.t, sch["name"]))); self.assertEqual(sch["input_schema"]["type"], "object")


class TestAgents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s, _ = H.make_session(t_end=9.0)

    def test_offline_agent_in_russian(self):
        a = RuleAgent(self.s)
        txt, used = a.reply("Что вокруг?"); self.assertIn("человек", txt); self.assertIn("scene_summary", used)
        txt, used = a.reply("Где человек?"); self.assertIn("человек", txt); self.assertIn("where_is", used); self.assertIn("ч)", txt)
        self.assertIn("Не вижу", a.reply("где автобус")[0]); self.assertIn("дверь", a.reply("что впереди?")[0]); self.assertIn("ничего", a.reply("что сзади?")[0])
        self.assertIn("Спросите", a.reply("расскажи анекдот")[0]); self.assertTrue(a.reply("что ближайшее?")[0].startswith("Ближайшие"))
        self.assertIn("Сейчас в списке", a.reply("сколько людей")[0])

    def test_offline_agent_on_an_empty_session(self):
        s = LiveSession(ColorBlobDetector(), {}, ServerConfig(detector="blobs")); self.assertIn("ничего не нашла", RuleAgent(s).reply("что вокруг")[0])

    def _client(self, script):
        calls = []
        class Msgs:
            def create(_, **kw):
                calls.append(json.loads(json.dumps(kw, default=lambda o: str(o)))); return script.pop(0)
        return types.SimpleNamespace(messages=Msgs()), calls

    def _tool_use(self, name, args, id="tu1"):
        return types.SimpleNamespace(stop_reason="tool_use", content=[types.SimpleNamespace(type="tool_use", name=name, input=args, id=id)])

    def _text(self, text): return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(type="text", text=text)])

    def test_claude_agent_runs_the_tool_loop(self):
        client, calls = self._client([self._tool_use("where_is", {"name": "person"}), self._text("Человек справа, около 5 м.")])
        a = ChatAgent(self.s, client, "m"); txt, used = a.reply("где человек?")
        self.assertEqual((txt, used), ("Человек справа, около 5 м.", ["where_is"]))
        self.assertEqual(len(calls), 2); self.assertEqual(calls[0]["model"], "m"); self.assertEqual(len(calls[0]["tools"]), len(TOOL_SCHEMAS)); self.assertIn("ESTIMATES", calls[0]["system"])
        tr = calls[1]["messages"][-1]["content"][0]; self.assertEqual((tr["type"], tr["tool_use_id"]), ("tool_result", "tu1")); self.assertIn("person", tr["content"])

    def test_a_failing_tool_is_reported_to_the_model_not_raised(self):
        client, calls = self._client([self._tool_use("objects_in_direction", {"direction": "sideways"}), self._tool_use("nope", {}, "tu2"), self._text("Не понял направление.")])
        a = ChatAgent(self.s, client, "m"); txt, used = a.reply("что сбоку"); self.assertEqual(used, ["objects_in_direction", "nope"])
        self.assertTrue(calls[1]["messages"][-1]["content"][0].get("is_error")); self.assertTrue(calls[2]["messages"][-1]["content"][0].get("is_error"))

    def test_an_endless_tool_loop_is_cut_off(self):
        client, _ = self._client([self._tool_use("scene_summary", {}, f"t{i}") for i in range(20)])
        txt, used = ChatAgent(self.s, client, "m", max_steps=3).reply("?"); self.assertIn("Слишком много шагов", txt); self.assertEqual(len(used), 3)

    def test_history_is_kept_between_questions(self):
        client, calls = self._client([self._text("Первый ответ."), self._text("Второй ответ.")]); a = ChatAgent(self.s, client, "m")
        a.reply("первый вопрос"); a.reply("второй вопрос"); self.assertEqual(len(calls[1]["messages"]), 3)
        self.assertEqual([m["role"] for m in calls[1]["messages"]], ["user", "assistant", "user"])

    def test_agent_selection(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertEqual(make_agent(self.s, ServerConfig(agent="auto")).kind, "offline"); self.assertEqual(make_agent(self.s, ServerConfig(agent="offline")).kind, "offline")
            with self.assertRaises(RuntimeError): make_agent(self.s, ServerConfig(agent="claude"))
            self.assertEqual(make_agent(self.s, ServerConfig(agent="auto"), client=object()).kind, "claude")
            os.environ["ANTHROPIC_API_KEY"] = "x"
            self.assertEqual(make_agent(self.s, ServerConfig(agent="auto")).kind, "offline")            # key set but the package is not installed here: falls back
        finally:
            if old is not None: os.environ["ANTHROPIC_API_KEY"] = old
            else: os.environ.pop("ANTHROPIC_API_KEY", None)


if __name__ == "__main__": unittest.main()
