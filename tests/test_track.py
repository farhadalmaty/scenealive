import unittest
import numpy as np
import helpers
from helpers import det
from scenelive.config import FastConfig
from scenelive.track import Tracker, iou_matrix


class TestTracker(unittest.TestCase):
    def test_iou(self):
        a = np.array([[0, 0, 10, 10.]]); b = np.array([[0, 0, 10, 10.], [5, 0, 15, 10.], [20, 20, 30, 30.]])
        np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]]); self.assertEqual(iou_matrix(a[:0], b).shape, (0, 3))

    def test_a_moving_object_keeps_its_id(self):
        tk = Tracker(FastConfig()); ids = []
        for i in range(8):
            m = tk.update([det("car", (100 + 8 * i, 100, 160 + 8 * i, 140))], i / 3, i)
            for tr, d in m: tr.vote(d)
            ids.append(m[0][0].id)
        self.assertEqual(set(ids), {1}); self.assertEqual(tk.active[0].hits, 8)

    def test_two_objects_stay_two(self):
        tk = Tracker(FastConfig())
        for i in range(4):
            m = tk.update([det("person", (50, 100, 80, 200)), det("person", (400, 100, 430, 200))], i, i)
            for tr, d in m: tr.vote(d)
        self.assertEqual(sorted(t.id for t in tk.active), [1, 2])

    def test_camera_turn_compensation_keeps_a_thin_pole(self):
        """A 10 px pole shifts 25 px when the camera turns: without compensation IoU is 0 and the track is lost."""
        def run(dx):
            tk = Tracker(FastConfig()); tk.update([det("pole", (300, 50, 310, 250))], 0, 0)
            m = tk.update([det("pole", (275, 50, 285, 250))], 1, 1, dx_px=dx); return {tr.id for tr, _ in m}
        self.assertEqual(run(-25.0), {1}); self.assertEqual(run(0.0), {2})

    def test_class_mismatch_needs_more_overlap(self):
        tk = Tracker(FastConfig()); m = tk.update([det("car", (100, 100, 200, 160))], 0, 0)
        for tr, d in m: tr.vote(d)
        m2 = tk.update([det("truck", (130, 100, 230, 160))], 1, 1)                        # IoU ~0.54 -> x0.5 = 0.27 > 0.25: still matched
        self.assertEqual(m2[0][0].id, 1)
        m3 = tk.update([det("bus", (170, 100, 270, 160))], 2, 2)                           # the track is now at x=130..230: IoU 0.43 x0.5 = 0.21 < 0.25: new track
        self.assertEqual(m3[0][0].id, 2)

    def test_lost_tracks_go_to_the_archive(self):
        cfg = FastConfig(max_age_frames=2); tk = Tracker(cfg); tk.update([det("person", (10, 10, 40, 100))], 0, 0)
        for i in range(1, 5): tk.update([], i, i)
        self.assertEqual(len(tk.active), 0); self.assertEqual(len(tk.archive), 1); self.assertEqual(len(tk.all_tracks()), 1)

    def test_tiny_boxes_are_ignored(self):
        self.assertEqual(Tracker(FastConfig()).update([det("person", (10, 10, 14, 14))], 0, 0), [])

    def test_voting_name_confidence_and_status(self):
        cfg = FastConfig(); tk = Tracker(cfg)
        for i, name in enumerate(["car", "car", "truck", "car"]):
            m = tk.update([det(name, (100, 100, 200, 160), 0.8)], i, i)
            for tr, d in m: tr.vote(d)
        tr = tk.active[0]; self.assertEqual(tr.name, "car"); self.assertEqual(tr.hits, 4)
        self.assertAlmostEqual(tr.conf, 0.8 * (2.4 / 3.2), places=6)                        # mean conf x share of votes: a flickering name is less sure
        self.assertEqual(tr.status(cfg), "confirmed")
        t2 = Tracker(cfg); m = t2.update([det("car", (0, 0, 50, 50), 0.9)], 0, 0); m[0][0].vote(m[0][1])
        self.assertEqual(m[0][0].status(cfg), "provisional")                                  # one detection is not enough


if __name__ == "__main__": unittest.main()
