import json, math, unittest
import numpy as np
import helpers
from scenelive.camera import Camera
from scenelive.freespace import (CAP_M, SAFETY, UNSURE, advise, advise_goal, analyse, ground_point, row_for_forward)
from scenelive.segment import DOOR, DYNAMIC, GLASS, IGNORE, ROAD, ROUGH, STAIRS, WALK, WALL

CAM = Camera(360, 640, 100.0)         # a portrait phone, ultra-wide: about 67 degrees across
H_CAM, PITCH = 1.4, 15.0


def floor(rows_above_blocked=None, blocker=WALL, wall_forward=None, left_only=False, fill=WALK):
    """A label map built from EXACT geometry: the floor everywhere below the row where the ground is `wall_forward` metres ahead."""
    g = np.full((CAM.height, CAM.width), fill, np.uint8)
    far = int(row_for_forward(40.0, CAM, PITCH, H_CAM)); g[:far] = IGNORE                          # beyond 40 m: sky
    if wall_forward is not None:
        v = int(math.floor(row_for_forward(wall_forward, CAM, PITCH, H_CAM)))
        if left_only: g[:v, : CAM.width // 2] = blocker
        else: g[:v] = blocker
    return g


class TestGeometry(unittest.TestCase):
    def test_a_pixel_row_is_a_distance_and_back(self):
        for th in (0.0, 15.0, 30.0):
            for Y in (1.5, 3.0, 6.0, 10.0):
                v = row_for_forward(Y, CAM, th, H_CAM)
                if v >= CAM.height or v <= CAM.cy: continue
                X, Yb = ground_point(v, CAM.cx, CAM, th, H_CAM); self.assertAlmostEqual(Yb, Y, delta=0.01); self.assertAlmostEqual(X, 0.0, places=6)
        X, _ = ground_point(500, CAM.cx + 100, CAM, PITCH, H_CAM); self.assertGreater(X, 0); X, _ = ground_point(500, CAM.cx - 100, CAM, PITCH, H_CAM); self.assertLess(X, 0)
        self.assertIsNone(ground_point(10, CAM.cx, CAM, 0.0, H_CAM))                              # above the horizon: no ground there

    def test_the_wall_is_found_where_it_stands(self):
        for Yw in (2.0, 3.0, 5.0, 8.0):
            fs = analyse(floor(wall_forward=Yw), CAM, PITCH, H_CAM); centre = fs.sectors[2]
            self.assertAlmostEqual(centre.free_m, Yw, delta=0.06 * Yw + 0.1, msg=Yw); self.assertEqual(centre.blocker, "wall")
            for s in fs.sectors: self.assertGreaterEqual(s.free_m, centre.free_m - 0.05)         # a flat wall is farther off to the sides (slant distance)

    def test_open_space_is_far_not_infinite(self):
        fs = analyse(floor(), CAM, PITCH, H_CAM); self.assertTrue(all(s.capped and s.free_m == CAP_M for s in fs.sectors)); self.assertTrue(fs.reliable)

    def test_sectors_cover_the_picture(self):
        fs = analyse(floor(), CAM, PITCH, H_CAM, n_sectors=5); self.assertEqual(len(fs.sectors), 5)
        for a, b in zip(fs.sectors, fs.sectors[1:]): self.assertAlmostEqual(a.hi, b.lo, places=6)
        self.assertAlmostEqual(fs.sectors[0].lo, -fs.sectors[-1].hi, delta=0.5); self.assertAlmostEqual(fs.sectors[-1].hi, math.degrees(math.atan(179.5 / CAM.f)), delta=0.6)

    def test_what_is_underfoot(self):
        for fill, want in ((WALK, "safe"), (ROUGH, "rough"), (ROAD, "road"), (WALL, "blocked"), (IGNORE, "unknown")):
            g = floor(); g[int(CAM.height * 0.85):] = fill; self.assertEqual(analyse(g, CAM, PITCH, H_CAM).underfoot, want, fill)

    def test_a_stated_distance_is_smaller_than_the_measured_one(self):
        s = analyse(floor(wall_forward=4.0), CAM, PITCH, H_CAM).sectors[2]; self.assertAlmostEqual(s.free_safe_m, SAFETY * s.free_m); self.assertLess(s.free_safe_m, s.free_m)

    def test_it_says_when_it_cannot_be_trusted(self):
        for pitch, why in ((None, "unknown"), (-20.0, "tilted"), (70.0, "tilted")):
            fs = analyse(floor(wall_forward=5.0), CAM, pitch, H_CAM); self.assertFalse(fs.reliable, pitch); self.assertEqual(advise(fs)["text"], UNSURE); self.assertTrue(any(why in r for r in fs.reasons), fs.reasons)
        fs = analyse(floor(wall_forward=5.0), CAM, PITCH, H_CAM, conf=np.full((CAM.height, CAM.width), 60, np.uint8)); self.assertTrue(fs.uncertain); self.assertEqual(advise(fs)["level"], "unsure")
        ok = analyse(floor(wall_forward=5.0), CAM, PITCH, H_CAM, conf=np.full((CAM.height, CAM.width), 230, np.uint8)); self.assertFalse(ok.uncertain)

    def test_a_small_hole_in_the_floor_is_not_an_obstacle(self):
        g = floor(); g[500:503, 150:200] = WALL                                                  # 3 px: a manhole cover, a line, a speck
        self.assertTrue(all(s.capped for s in analyse(g, CAM, PITCH, H_CAM).sectors))

    def test_json(self):
        d = analyse(floor(wall_forward=3.0), CAM, PITCH, H_CAM).to_dict(); json.dumps(d); self.assertEqual(len(d["sectors"]), 5); self.assertIn("free_safe_m", d["sectors"][0])


class TestAdvice(unittest.TestCase):
    def adv(self, g, pitch=PITCH): return advise(analyse(g, CAM, pitch, H_CAM))

    def test_open_way(self):
        a = self.adv(floor()); self.assertEqual(a["level"], "ok"); self.assertTrue(a["text"].startswith("Прямо свободно больше 9 метров"), a["text"]); self.assertIsNone(a["turn_deg"])

    def test_a_wall_ahead_says_so_and_what_ends_the_way(self):
        a = self.adv(floor(wall_forward=5.0)); self.assertEqual(a["level"], "ok"); self.assertIn("дальше стена", a["text"]); self.assertTrue(4.0 >= a["ahead_m"] >= 3.0)             # 80 % of ~5 m

    def test_a_wall_close_ahead_means_stop(self):
        a = self.adv(floor(wall_forward=1.3)); self.assertEqual(a["level"], "stop"); self.assertTrue(a["text"].startswith("Стоп. Впереди стена")); self.assertIn("метр", a["text"])

    def test_something_at_the_feet_is_a_stop(self):
        g = floor(); g[int(CAM.height * 0.92):] = DYNAMIC; a = self.adv(g); self.assertEqual(a["level"], "stop"); self.assertIn("человек или транспорт", a["text"]); self.assertEqual(a["ahead_m"], 0.0)

    def test_it_points_to_the_free_side(self):
        a = self.adv(floor(wall_forward=2.0, blocker=DYNAMIC, left_only=True)); self.assertEqual(a["level"], "caution")
        self.assertIn("Осторожно, впереди человек или транспорт", a["text"]); self.assertIn("Свободнее справа", a["text"]); self.assertIn("Поверните направо примерно на", a["text"]); self.assertGreater(a["turn_deg"], 0)
        b = floor(); v = int(math.floor(row_for_forward(2.0, CAM, PITCH, H_CAM))); b[:v, CAM.width // 2:] = DYNAMIC; c = self.adv(b)
        self.assertIn("Свободнее слева", c["text"]); self.assertIn("налево", c["text"]); self.assertLess(c["turn_deg"], 0)

    def test_a_dead_end_says_there_is_no_way(self):
        a = self.adv(floor(wall_forward=1.8)); self.assertIn("Свободного прохода рядом не вижу", a["text"]); self.assertIsNone(a["turn_deg"])

    def test_the_road_is_not_a_way_to_walk(self):
        a = self.adv(floor(wall_forward=5.0, blocker=ROAD)); self.assertIn("дальше проезжая часть", a["text"])
        g = floor(fill=ROAD); b = self.adv(g); self.assertIn("Похоже, вы на проезжей части", b["text"]); self.assertIn("Остановитесь", b["text"]); self.assertEqual(b["level"], "stop")
        close = self.adv(floor(wall_forward=1.5, blocker=ROAD)); self.assertIn("проезжая часть", close["text"]); self.assertEqual(close["level"], "stop")                    # the kerb ahead

    def test_stairs_and_glass_are_named(self):
        for blk, word in ((STAIRS, "лестница"), (GLASS, "стекло или окно"), (DOOR, "дверь")): self.assertIn(word, self.adv(floor(wall_forward=1.4, blocker=blk))["text"], blk)

    def test_a_close_wall_on_one_side_is_mentioned(self):
        g = floor(); v = int(math.floor(row_for_forward(1.2, CAM, PITCH, H_CAM))); g[:v, : int(CAM.width * 0.15)] = WALL
        a = self.adv(g); self.assertEqual(a["level"], "ok"); self.assertIn("Слева стена близко", a["text"])

    def test_the_advice_never_claims_safety(self):
        for g in (floor(), floor(wall_forward=5.0), floor(wall_forward=1.3), floor(fill=ROAD)):
            t = self.adv(g)["text"].lower(); self.assertNotIn("безопасно", t); self.assertNotIn("путь свободен", t)


class TestGoal(unittest.TestCase):
    def test_a_free_path_to_a_known_object(self):
        fs = analyse(floor(wall_forward=9.0), CAM, PITCH, H_CAM); a = advise_goal(fs, 8.0, "дверь", 4.0); self.assertEqual(a["level"], "ok"); self.assertIn("Путь к ней выглядит свободным", a["text"]); self.assertTrue(a["text"].startswith("Дверь"))

    def test_a_blocked_path_names_the_blocker(self):
        fs = analyse(floor(wall_forward=2.0, blocker=DYNAMIC), CAM, PITCH, H_CAM); a = advise_goal(fs, 0.0, "дверь", 6.0); self.assertEqual(a["level"], "caution"); self.assertIn("На пути к дверь человек или транспорт", a["text"])

    def test_a_goal_outside_the_picture_means_turn_first(self):
        fs = analyse(floor(), CAM, PITCH, H_CAM); a = advise_goal(fs, 80.0, "дверь"); self.assertIn("вне поля зрения", a["text"]); self.assertIn("направо примерно на 80 градусов", a["text"]); self.assertIn("слева", advise_goal(fs, -90.0, "дверь")["text"])

    def test_unsure_stays_unsure(self):
        self.assertEqual(advise_goal(analyse(floor(), CAM, None, H_CAM), 5.0, "дверь")["level"], "unsure")


if __name__ == "__main__": unittest.main()
