import contextlib, io, os, tempfile, unittest
import helpers
from scenelive.cli import main
from scenelive.synth import render_video


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): code = main(argv)
    return code, buf.getvalue()


class TestCli(unittest.TestCase):
    def test_demo_prints_timings_and_checks_against_truth(self):
        out = os.path.join(tempfile.mkdtemp(), "demo"); code, txt = run(["demo", "-o", out])
        self.assertEqual(code, 0)
        for s in ("STAGE TIMING", "COMPASS: after", "GROUND TRUTH", "person", "door", "bearing error"): self.assertIn(s, txt)
        self.assertTrue(os.path.exists(os.path.join(out, "report.json"))); self.assertTrue(os.path.exists(os.path.join(out, "synthetic_street.mp4")))

    def test_replay_with_options(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=8.0)
        code, txt = run(["replay", v, "--detector", "blobs", "--compass", "zero", "--fps", "2", "--compute-scale", "2", "--camera", "main", "--out-video", "-o", os.path.join(d, "r")])
        self.assertEqual(code, 0); self.assertIn("sample: 2 fps", txt); self.assertIn("compute scale: 2", txt); self.assertTrue(os.path.exists(os.path.join(d, "r", "overlay.mp4")))

    def test_bench_prints_one_row_per_size(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=4.0)
        code, txt = run(["bench", v, "--detector", "blobs", "--imgsz", "320,640", "--frames", "6"])
        self.assertEqual(code, 0); rows = [l for l in txt.splitlines() if l.strip()[:3] in ("320", "640")]; self.assertEqual(len(rows), 2)


if __name__ == "__main__": unittest.main()
