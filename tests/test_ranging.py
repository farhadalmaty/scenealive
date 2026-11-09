import math, unittest
import helpers
from scenelive.camera import Camera
from scenelive.ranging import estimate_range


def box_for(cam, side_px, h=True, cx=None, cy=None):
    cx = cam.cx if cx is None else cx; cy = cam.cy if cy is None else cy
    return (cx - 10, cy - side_px / 2, cx + 10, cy + side_px / 2) if h else (cx - side_px / 2, cy - 10, cx + side_px / 2, cy + 10)


class TestRanging(unittest.TestCase):
    cam = Camera(640, 360, 69.0)

    def test_person_at_known_distance(self):
        for d in (4.0, 6.0, 12.0):
            px = self.cam.f * 1.7 / d; r = estimate_range("person", box_for(self.cam, px), self.cam)
            self.assertAlmostEqual(r[0], d, delta=0.01 * d)

    def test_off_axis_range_is_euclidean_not_depth(self):
        d, az = 8.0, 30.0; Z = d * math.cos(math.radians(az)); px = self.cam.f * 1.5 / Z
        self.assertAlmostEqual(estimate_range("car", box_for(self.cam, px), self.cam, az_rel_deg=az)[0], d, delta=0.01 * d)

    def test_width_based_class(self):
        d = 5.0; px = self.cam.f * 1.6 / d
        self.assertAlmostEqual(estimate_range("bench", box_for(self.cam, px, h=False), self.cam)[0], d, delta=0.05)

    def test_unknown_class_and_unusable_boxes(self):
        self.assertIsNone(estimate_range("pole", box_for(self.cam, 100), self.cam))               # no size prior
        self.assertIsNone(estimate_range("person", (100, 0, 130, 120), self.cam))                  # touches the top: cut by the frame
        self.assertIsNone(estimate_range("person", (100, 200, 130, 359), self.cam))                # touches the bottom
        self.assertIsNone(estimate_range("person", box_for(self.cam, 5), self.cam))                # too small to trust
        self.assertIsNotNone(estimate_range("person", (0, 100, 40, 220), self.cam))               # touching a SIDE does not matter for a height prior

    def test_uncertainty_has_a_floor_from_the_prior(self):
        self.assertGreaterEqual(estimate_range("person", box_for(self.cam, 200), self.cam)[1], 0.12)


if __name__ == "__main__": unittest.main()
