import unittest
import numpy as np
import helpers
from scenelive.camera import Camera
from scenelive.freespace import CAP_M, Sector, analyse
from scenelive.seg_monitor import smooth
from scenelive.segment import DYNAMIC, ROAD, WALK, WALL, clean_groups

H, W = 640, 360


class TestCleanUp(unittest.TestCase):
    def test_a_painted_line_on_the_road_is_road(self):
        g = np.full((H, W), ROAD, np.uint8); g[300:640, 100:103] = WALK; g[300:640, 108:111] = WALK; g[500:, 250:] = WALK            # two 3-px lines + a real sidewalk
        c = clean_groups(g); self.assertTrue((c[300:640, 100:111] == ROAD).all()); self.assertTrue((c[520:, 270:] == WALK).all())

    def test_a_drain_cover_on_the_sidewalk_is_sidewalk(self):
        g = np.full((H, W), WALK, np.uint8); g[:200] = WALL; g[500:515, 150:165] = ROAD                                                  # 225 px of 'road' in the sidewalk
        c = clean_groups(g); self.assertTrue((c[500:515, 150:165] == WALK).all())

    def test_a_real_road_is_not_eaten(self):
        g = np.full((H, W), WALK, np.uint8); g[300:, :180] = ROAD; c = clean_groups(g); self.assertTrue((c[320:, :170] == ROAD).all())

    def test_floor_blobs_inside_a_car_take_the_car(self):
        g = np.full((H, W), ROAD, np.uint8); g[200:400, 50:300] = DYNAMIC; g[300:312, 100:112] = WALK
        c = clean_groups(g); self.assertTrue((c[300:312, 100:112] == DYNAMIC).all())

    def test_a_real_sidewalk_island_is_kept(self):
        g = np.full((H, W), ROAD, np.uint8); g[400:600, 100:250] = WALK; c = clean_groups(g); self.assertTrue((c[410:590, 110:240] == WALK).all())


class TestSmoothing(unittest.TestCase):
    def fs(self, frees, under="safe"):
        cam = Camera(W, H, 100.0); f = analyse(np.full((H, W), WALK, np.uint8), cam, 15.0); f.underfoot = under
        f.sectors = [Sector(-30 + 12 * i, -18 + 12 * i, v, "wall", v >= CAP_M) for i, v in enumerate(frees)]; f.sectors_cross = list(f.sectors); return f

    def test_a_new_obstacle_shows_at_once(self):
        s = smooth([self.fs([8, 8, 8, 8, 8]), self.fs([8, 8, 8, 8, 8]), self.fs([8, 8, 1.2, 8, 8])]); self.assertEqual(s.sectors[2].free_m, 1.2)

    def test_a_single_spurious_free_look_is_ignored(self):
        s = smooth([self.fs([2, 2, 2, 2, 2]), self.fs([2, 2, 2, 2, 2]), self.fs([2, 2, 12, 2, 2])]); self.assertEqual(s.sectors[2].free_m, 2.0); self.assertFalse(s.sectors[2].capped)

    def test_underfoot_by_majority(self):
        self.assertEqual(smooth([self.fs([5] * 5, "safe"), self.fs([5] * 5, "safe"), self.fs([5] * 5, "road")]).underfoot, "safe")
        self.assertEqual(smooth([self.fs([5] * 5, "safe"), self.fs([5] * 5, "road"), self.fs([5] * 5, "road")]).underfoot, "road")
        one = self.fs([5] * 5); self.assertIs(smooth([one]), one)


if __name__ == "__main__": unittest.main()
