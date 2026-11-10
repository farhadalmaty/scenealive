import contextlib, io, os, tempfile, unittest
import numpy as np
import helpers
from scenelive.cli import main
from scenelive.segment import (DOOR, DYNAMIC, GLASS, IGNORE, OBSTACLE, ROAD, ROUGH, STAIRS, WALK, WALL, ColorSegmenter, SegFormerSegmenter, bench_segmenter, group_map, group_of, make_segmenter, overlay, stats)
from scenelive.synth import render_video


class TestGroups(unittest.TestCase):
    def test_real_ade20k_label_strings(self):
        want = {"floor, flooring": WALK, "rug, carpet, carpeting": WALK, "sidewalk, pavement": WALK, "path": WALK, "road, route": ROAD, "runway": ROAD,
                "earth, ground": ROUGH, "grass": ROUGH, "sand": ROUGH, "field": ROUGH,
                "wall": WALL, "building, edifice": WALL, "fence, fencing": WALL, "railing, rail": WALL, "column, pillar": WALL, "door, double door": DOOR, "screen door, screen": DOOR,
                "stairs, steps": STAIRS, "stairway, staircase": STAIRS, "escalator, moving staircase, moving stairway": STAIRS, "windowpane, window": GLASS, "mirror": GLASS,
                "person, individual, someone, somebody, mortal, soul": DYNAMIC, "car, auto, automobile, machine, motorcar": DYNAMIC, "animal, animate being, beast, brute, creature, fauna": DYNAMIC,
                "sky": IGNORE, "ceiling": IGNORE, "pole": OBSTACLE, "chair": OBSTACLE, "table": OBSTACLE, "plant, flora, plant life": OBSTACLE, "curtain, drape, drapery, mantle, pall": OBSTACLE,
                "glass, drinking glass": OBSTACLE, "something unseen": OBSTACLE}                                          # a drinking glass is not a window
        for label, group in want.items(): self.assertEqual(group_of(label), group, label)

    def test_group_map_and_stats(self):
        id2 = {0: "wall", 1: "floor, flooring", 2: "door", 3: "sky", 4: "pole"}; lab = np.array([[0, 0, 3, 3], [1, 1, 2, 4], [1, 1, 1, 1], [1, 1, 1, 1]])
        g = group_map(lab, id2); self.assertEqual(g[0, 0], WALL); self.assertEqual(g[1, 2], DOOR); self.assertEqual(g[0, 2], IGNORE); self.assertEqual(g[1, 3], OBSTACLE)
        st = stats(g); self.assertAlmostEqual(sum(st.values()), 1.0); self.assertAlmostEqual(st["walkable"], 10 / 16); self.assertAlmostEqual(st["door"], 1 / 16)

    def test_overlay_colours_the_regions_and_leaves_the_ignored_alone(self):
        frame = np.full((240, 360, 3), 100, np.uint8); g = np.zeros((240, 360), np.uint8); g[120:, 150:300] = WALK; g[:, 300:] = WALL        # the legend occupies the top-left corner
        out = overlay(frame, g); self.assertEqual(out.shape, frame.shape)
        floor, wall = out[200, 200].astype(int), out[60, 330].astype(int)
        self.assertTrue(floor[1] > floor[0] and floor[1] > floor[2], floor)                                                                  # the floor looks green
        self.assertTrue(wall[0] > wall[2], wall)                                                                                             # the wall looks blue
        self.assertTrue((out[10:100, 150:290] == frame[10:100, 150:290]).all())                                                              # ignored pixels are untouched
        small = overlay(frame, np.full((10, 15), WALK, np.uint8)); self.assertEqual(small.shape, frame.shape)                               # a mask of another size is resized


class TestBench(unittest.TestCase):
    def test_demo_segmenter_and_bench_files(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=6.0); out = os.path.join(d, "seg"); lines = []
        s = bench_segmenter(v, ColorSegmenter(), out, n_frames=5, log=lines.append)
        self.assertEqual(s["frames"], 5); self.assertEqual(sorted(f for f in os.listdir(out) if f.startswith("seg_")), [f"seg_{k:03d}.jpg" for k in range(5)]); self.assertTrue(os.path.exists(os.path.join(out, "summary.txt")))
        self.assertGreater(s["mean_share"]["walkable"], 0.3); self.assertGreater(s["fps"], 1.0); self.assertTrue(any("time per frame" in l for l in lines))

    def test_cli_bench_seg(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=4.0); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = main(["bench-seg", v, "--model", "colors", "--frames", "4", "-o", os.path.join(d, "o")])
        self.assertEqual(code, 0); self.assertIn("frames per second", buf.getvalue()); self.assertIn("brown = ROAD", buf.getvalue())

    def test_a_missing_transformers_package_is_explained(self):
        with self.assertRaises(ImportError) as c: SegFormerSegmenter()
        self.assertIn("setup_seg", str(c.exception)); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = main(["bench-seg", "x.mp4"])
        self.assertEqual(code, 2); self.assertIn("transformers", buf.getvalue()); self.assertNotIn("Traceback", buf.getvalue())
        with self.assertRaises(ValueError): make_segmenter("magic")

    def test_the_road_is_not_the_sidewalk(self):
        self.assertNotEqual(group_of("road, route"), group_of("sidewalk, pavement")); self.assertNotEqual(group_of("grass"), group_of("floor, flooring"))
        from scenelive.segment import GROUP_COLOR, GROUP_NAMES
        self.assertEqual(len(set(GROUP_COLOR.values())), len(GROUP_COLOR)); self.assertEqual(GROUP_NAMES[ROAD], "road")

    def test_a_missing_video_is_a_message(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = main(["bench-seg", os.path.join(tempfile.mkdtemp(), "nope.mp4"), "--model", "colors"])
        self.assertEqual(code, 2); self.assertIn("Cannot run", buf.getvalue())


if __name__ == "__main__": unittest.main()
