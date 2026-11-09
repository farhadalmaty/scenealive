import contextlib, io, os, tempfile, unittest
import helpers
from scenelive.cli import main
from scenelive.compass import VisualCompass
from scenelive.config import CameraConfig, FastConfig, MapConfig, SimConfig
from scenelive.detect.blob import ColorBlobDetector
from scenelive.evaluate import make_template, read_labels
from scenelive.report import write_outputs
from scenelive.sim import run_replay
from scenelive.synth import render_video


def cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): code = main(argv)
    return code, buf.getvalue()


class TestLabelsHelp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); v = os.path.join(cls.d, "v.mp4"); render_video(v, duration=12.0)
        cls.rd = os.path.join(cls.d, "run")
        write_outputs(run_replay(v, ColorBlobDetector(), CameraConfig(), FastConfig(), MapConfig(), SimConfig(), VisualCompass()), cls.rd)

    def test_the_empty_template_explains_itself_and_has_a_section_per_moment(self):
        t = make_template(self.rd, every=4.0)
        for s in ("Формат строки", "секунда 0.0", "секунда 4.0", "секунда 8.0", "t_0004.0.jpg", "# 4.0, person, 2"): self.assertIn(s, t)
        self.assertEqual(read_labels_from_text(t), {})                               # nothing to score until the user writes lines

    def test_prefill_lists_what_the_system_found_and_asks_to_check_it(self):
        t = make_template(self.rd, every=4.0, prefill=True); self.assertIn("ПРОВЕРЬТЕ", t)
        L = read_labels_from_text(t); self.assertTrue(L); self.assertTrue(any("person" in c for c in L.values()))

    def test_cli_writes_the_file_and_never_overwrites_the_users_work(self):
        out = os.path.join(self.d, "labels.csv"); code, txt = cli(["labels-template", self.rd, "-o", out])
        self.assertEqual(code, 0); self.assertTrue(os.path.exists(out)); self.assertIn("Заготовка записана", txt)
        open(out, "a", encoding="utf-8").write("4.0, person, 2\n")
        code2, txt2 = cli(["labels-template", self.rd, "-o", out]); self.assertEqual(code2, 2); self.assertIn("не трогаю", txt2)
        self.assertIn("4.0, person, 2", open(out, encoding="utf-8").read())             # still there
        self.assertEqual(cli(["labels-template", self.rd, "-o", out, "--force"])[0], 0)

    def test_eval_without_a_labels_file_explains_what_to_do(self):
        code, txt = cli(["eval", self.rd, os.path.join(self.d, "nope.csv")])
        self.assertEqual(code, 2); self.assertIn("Файл разметки не найден", txt); self.assertIn("labels_template", txt)

    def test_eval_on_an_old_run_without_tracks_says_so(self):
        old = os.path.join(self.d, "old"); os.makedirs(old); open(os.path.join(old, "frames.csv"), "w").write("k,t\n0,0\n")
        lab = os.path.join(self.d, "l.csv"); open(lab, "w").write("0.0,person\n")
        code, txt = cli(["eval", old, lab]); self.assertEqual(code, 2); self.assertIn("tracks.csv", txt); self.assertIn("старой версией", txt)

    def test_eval_with_an_empty_labels_file_explains_the_format(self):
        lab = os.path.join(self.d, "empty.csv"); open(lab, "w", encoding="utf-8").write(make_template(self.rd))
        code, txt = cli(["eval", self.rd, lab]); self.assertEqual(code, 2); self.assertIn("4.0, person, 2", txt)

    def test_a_filled_template_scores(self):
        lab = os.path.join(self.d, "filled.csv"); open(lab, "w", encoding="utf-8").write(make_template(self.rd) + "\n4.0, person, 1\n4.0, car, 1\n")
        code, txt = cli(["eval", self.rd, lab]); self.assertEqual(code, 0); self.assertIn("precision", txt)


def read_labels_from_text(text):
    p = os.path.join(tempfile.mkdtemp(), "x.csv"); open(p, "w", encoding="utf-8").write(text); return read_labels(p)


if __name__ == "__main__": unittest.main()
