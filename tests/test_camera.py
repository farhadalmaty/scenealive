import math, unittest
import helpers
from scenelive.camera import Camera, angdiff, wrap360, circular_mean, clock, ladder_bucket, range_label


class TestCamera(unittest.TestCase):
    def test_focal_and_bearings(self):
        c = Camera(640, 360, 69.0)
        self.assertAlmostEqual(c.f, 320 / math.tan(math.radians(34.5)), places=6)
        self.assertAlmostEqual(c.azimuth_deg(320), 0.0, places=9); self.assertAlmostEqual(c.azimuth_deg(640), 34.5, places=6)
        self.assertAlmostEqual(c.azimuth_deg(0), -34.5, places=6)
        self.assertGreater(c.elevation_deg(0), 0); self.assertLess(c.elevation_deg(359), 0)           # up in the image = positive

    def test_portrait_uses_the_long_side(self):
        self.assertAlmostEqual(Camera(360, 640, 69.0).f, Camera(640, 360, 69.0).f, places=9)

    def test_project_is_the_inverse_of_azimuth(self):
        c = Camera(640, 360, 69.0); u, v = c.project(2.0, 0.5, 10.0)
        self.assertAlmostEqual(c.azimuth_deg(u), math.degrees(math.atan2(2.0, 10.0)), places=6)
        self.assertAlmostEqual(c.elevation_deg(v), math.degrees(math.atan2(0.5, 10.0)), places=6); self.assertIsNone(c.project(0, 0, -1))

    def test_angles(self):
        self.assertAlmostEqual(angdiff(350, 10), -20); self.assertAlmostEqual(angdiff(10, 350), 20); self.assertEqual(angdiff(190, 10), 180)
        self.assertAlmostEqual(wrap360(-10), 350); self.assertAlmostEqual(circular_mean([350, 10]), 0.0, places=6)

    def test_clock_face(self):
        self.assertEqual(clock(0), (12, False)); self.assertEqual(clock(90), (3, False)); self.assertEqual(clock(-90), (9, False))
        self.assertEqual(clock(180), (6, True)); self.assertEqual(clock(31), (1, False)); self.assertEqual(clock(-14), (12, False))

    def test_distance_steps(self):
        for d, want in {0.3: 1, 1.4: 1, 1.5: 2, 3.1: 2, 3.2: 5, 7.0: 5, 7.2: 10, 12.2: 10, 12.4: 15}.items(): self.assertEqual(ladder_bucket(d), want, d)
        self.assertIsNone(ladder_bucket(25)); self.assertEqual(range_label(2.2), "~2 m"); self.assertEqual(range_label(3.0, 0.35), "2-5 m")
        self.assertEqual(range_label(None), "?"); self.assertEqual(range_label(40), "15+ m")


if __name__ == "__main__": unittest.main()
