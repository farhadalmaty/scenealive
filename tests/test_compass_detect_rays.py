import math, os, tempfile, unittest
import numpy as np, cv2
import helpers
from helpers import det
from scenelive.camera import Camera
from scenelive.compass import ZeroCompass, CsvCompass, VisualCompass, make_compass
from scenelive.detect import make_detector
from scenelive.detect.base import parse_ultralytics
from scenelive.detect.blob import ColorBlobDetector
from scenelive.detect.yolo import YoloDetector
from scenelive.labels import COCO_STREET, canonical
from scenelive.rays import triangulate, direction
from scenelive.synth import _panorama, render_frame


class TestCompass(unittest.TestCase):
    def test_zero_and_factory(self):
        self.assertEqual(ZeroCompass().update(None, 1.0, None), 0.0); self.assertIsInstance(make_compass("visual"), VisualCompass)

    def test_csv_compass_interpolates_unwraps_and_skips_junk(self):
        p = os.path.join(tempfile.mkdtemp(), "h.csv")
        open(p, "w", encoding="utf-8").write("t,yaw\n0,350\n1;10\njunk\n2,30\n")
        c = CsvCompass(p)
        self.assertAlmostEqual(c.update(None, 0, None), 0.0); self.assertAlmostEqual(c.update(None, 1, None), 20.0)     # 350 -> 10 is +20, not -340
        self.assertAlmostEqual(c.update(None, 1.5, None), 30.0); self.assertAlmostEqual(c.update(None, 9, None), 40.0)  # clamped after the last row
        with open(p, "w", encoding="utf-8") as f: f.write("only,one\n1,2\n")
        with self.assertRaises(ValueError): CsvCompass(p)                                          # a single row cannot define a heading

    def test_visual_compass_follows_a_pan(self):
        cam = Camera(640, 360, 69.0); pano = _panorama(360, 3); vc = VisualCompass(); yaw = []
        for k in range(30):                                       # 3 fps, 4 deg/s
            img, _ = render_frame(k / 3, [], cam, 4.0, pano); yaw.append(vc.update(img, k / 3, cam))
        self.assertAlmostEqual(yaw[-1], 4.0 * 29 / 3, delta=0.02 * 4.0 * 29 / 3 + 0.5)

    def test_visual_compass_survives_an_empty_scene(self):
        vc = VisualCompass(); cam = Camera(640, 360, 69.0); blank = np.full((360, 640, 3), 120, np.uint8)
        for k in range(5): vc.update(blank, k, cam)
        self.assertEqual(vc.yaw, 0.0); self.assertGreaterEqual(vc.lost, 1)


class TestDetectors(unittest.TestCase):
    def test_color_blobs_find_boxes_and_names(self):
        img = np.full((200, 300, 3), 128, np.uint8)
        cv2.rectangle(img, (40, 50), (79, 149), (30, 30, 220), -1); cv2.rectangle(img, (150, 80), (229, 139), (220, 60, 30), -1)
        d = {x.name: x for x in ColorBlobDetector().detect(img)}
        self.assertEqual(set(d), {"person", "car"}); self.assertEqual([round(v) for v in d["person"].box], [40, 50, 80, 150])
        self.assertEqual([round(v) for v in d["car"].box], [150, 80, 230, 140])

    def test_factory(self):
        self.assertEqual(make_detector("blobs").name, "color-blobs")
        with self.assertRaises(ValueError): make_detector("nope")

    def test_canonical_names(self):
        self.assertEqual(canonical("Lamp Post"), "pole"); self.assertEqual(canonical("handbag"), "bag"); self.assertEqual(canonical("person"), "person")

    def _fake_result(self):
        class T:
            def __init__(s, a): s.a = np.asarray(a)
            def cpu(s): return s
            def numpy(s): return s.a
        class Boxes:
            xyxy = T([[10, 20, 50, 120], [60, 60, 90, 80], [0, 0, 5, 5]]); cls = T([0, 1, 2]); conf = T([0.9, 0.5, 0.7])
            def __len__(s): return 3
        class Res: names = {0: "person", 1: "handbag", 2: "toilet"}; boxes = Boxes()
        return Res()

    def test_parse_ultralytics_filters_and_renames(self):
        out = parse_ultralytics(self._fake_result(), set(COCO_STREET))
        self.assertEqual([(d.name, d.raw_name) for d in out], [("person", "person"), ("bag", "handbag")])          # toilet is not a street class
        self.assertEqual(out[0].box, (10.0, 20.0, 50.0, 120.0)); self.assertEqual(len(parse_ultralytics(self._fake_result(), None)), 3)
        class Empty: boxes = None
        self.assertEqual(parse_ultralytics(Empty()), [])

    def test_yolo_wrapper_passes_the_frame_unchanged_in_bgr(self):
        seen = {}
        class Model:
            def predict(s, img, **kw): seen["img"], seen["kw"] = img, kw; return [self_.fake]
        self_ = self; self.fake = self._fake_result()
        y = YoloDetector.__new__(YoloDetector); y.m = Model(); y.imgsz, y.conf, y.iou, y.half, y.device, y.allowed = 640, 0.3, 0.6, False, "cpu", set(COCO_STREET)
        frame = np.zeros((60, 80, 3), np.uint8); frame[..., 0] = 200                       # blue channel only (BGR)
        out = y.detect(frame); self.assertEqual(len(out), 2); self.assertEqual(seen["img"][..., 0].max(), 200)
        self.assertTrue(seen["img"].flags["C_CONTIGUOUS"]); self.assertEqual(seen["kw"]["imgsz"], 640); self.assertFalse(seen["kw"]["verbose"])


class TestRays(unittest.TestCase):
    def test_three_cameras_find_the_point(self):
        P = np.array([40.0, 25.0, 1.0]); O = np.array([[0, 0, 1.5], [10, 0, 1.5], [20, 5, 1.5]], float)
        D = [(P - o) for o in O]; x, res = triangulate(O, D)
        np.testing.assert_allclose(x, P, atol=1e-6); self.assertLess(res, 1e-6)

    def test_azimuth_and_elevation_make_the_same_direction(self):
        d = direction(90, 0); np.testing.assert_allclose(d, [1, 0, 0], atol=1e-9); d = direction(0, 45); np.testing.assert_allclose(d, [0, math.cos(math.pi / 4), math.sin(math.pi / 4)], atol=1e-9)

    def test_degenerate_geometry_is_refused(self):
        D = [(0, 1, 0), (0, 1, 0)]
        self.assertIsNone(triangulate([[0, 0, 0], [3, 0, 0]], D))                              # parallel rays
        self.assertIsNone(triangulate([[0, 0, 0], [0.1, 0, 0]], [(0, 1, 0), (0.2, 1, 0)]))     # the camera barely moved
        self.assertIsNone(triangulate([[0, 0, 0]], [(0, 1, 0)]))

    def test_noisy_rays_have_a_residual(self):
        P = np.array([30.0, 20.0, 1.0]); rng = np.random.default_rng(0); O = rng.uniform(0, 10, (6, 3)); O[:, 2] = 1.5
        D = [(P - o) + rng.normal(0, 0.3, 3) for o in O]; x, res = triangulate(O, D)
        self.assertLess(np.linalg.norm(x - P), 3.0); self.assertGreater(res, 1e-6)


if __name__ == "__main__": unittest.main()
