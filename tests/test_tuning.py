import contextlib, csv, io, json, os, tempfile, unittest
import numpy as np
import helpers
from helpers import det
from scenelive.cli import main
from scenelive.compass import ZeroCompass, VisualCompass
from scenelive.config import CameraConfig, FastConfig, MapConfig, SimConfig
from scenelive.detect.blob import ColorBlobDetector
from scenelive.detect.yolo import YoloDetector, YoloWorldDetector, _kw
from scenelive.evaluate import read_labels, evaluate, format_eval
from scenelive.fastpath import FastPath
from scenelive.report import write_outputs, format_report
from scenelive.sim import run_replay
from scenelive.synth import render_video

FRAME = np.zeros((360, 640, 3), np.uint8)


class Scripted:
    name = "scripted"
    def __init__(self, script): self.script, self.i = script, 0
    def detect(self, frame): out = self.script[min(self.i, len(self.script) - 1)]; self.i += 1; return out


def run_script(script, **cfg):
    fp = FastPath(Scripted(script), CameraConfig(), FastConfig(**cfg), ZeroCompass()); return [fp.process(FRAME, i / 3, i).entries for i in range(len(script))]


class TestFlickerGuard(unittest.TestCase):
    box = (300, 100, 340, 250)

    def test_one_weak_detection_is_not_shown_two_are(self):
        e = run_script([[det("person", self.box, 0.4)], [det("person", self.box, 0.4)]])
        self.assertEqual(len(e[0]), 0); self.assertEqual(len(e[1]), 1)

    def test_one_sure_detection_is_shown_at_once(self):
        self.assertEqual(len(run_script([[det("person", self.box, 0.7)]])[0]), 1)

    def test_below_the_floor_is_never_shown(self):
        e = run_script([[det("person", self.box, 0.2)]] * 5); self.assertTrue(all(len(x) == 0 for x in e))

    def test_a_single_frame_flash_among_nothing_never_reaches_the_list(self):
        e = run_script([[det("pole", self.box, 0.4)], [], [], []]); self.assertTrue(all(len(x) == 0 for x in e))

    def test_the_guard_can_be_switched_off(self):
        self.assertEqual(len(run_script([[det("person", self.box, 0.4)]], publish_hits=1)[0]), 1)


class TestNoWarnings(unittest.TestCase):
    def test_half_is_passed_only_when_asked_for(self):
        class D: imgsz, conf, iou, device, half = 640, 0.3, 0.6, "cpu", False
        self.assertNotIn("half", _kw(D)); D.half = True; self.assertTrue(_kw(D)["half"]); self.assertFalse(_kw(D)["verbose"])


class TestOutputsForAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); cls.video = os.path.join(cls.d, "v.mp4"); render_video(cls.video, duration=12.0)
        cls.res = run_replay(cls.video, ColorBlobDetector(), CameraConfig(69.0), FastConfig(), MapConfig(), SimConfig(), VisualCompass())
        cls.out = os.path.join(cls.d, "run"); write_outputs(cls.res, cls.out)

    def test_tracks_csv_has_a_row_per_track_per_frame(self):
        rows = list(csv.DictReader(open(os.path.join(self.out, "tracks.csv"))))
        self.assertEqual(len(rows), sum(f.n_tracks for f in self.res.frames))
        for k in ("k", "t", "track", "name", "conf", "det_conf", "hits", "x0", "x1", "az_rel", "world_az", "range_m"): self.assertIn(k, rows[0])
        self.assertEqual({r["name"] for r in rows}, {"person", "car", "pole", "door", "wall"})      # the wide wall enters the view early: its edge is at 70 deg

    def test_compass_quality_is_reported(self):
        cs = self.res.summary["compass"]; self.assertEqual(cs["steps"], len(self.res.frames) - 1); self.assertEqual(cs["lost"], 0)
        self.assertIn("compass: flow lost in 0 of", format_report(self.res)); self.assertGreater(cs["yaw_max"], 40)

    def test_zero_compass_reports_nothing_and_does_not_break_the_report(self):
        r = run_replay(self.video, ColorBlobDetector(), CameraConfig(), FastConfig(), MapConfig(), SimConfig(), ZeroCompass())
        self.assertEqual(r.summary["compass"], {}); self.assertNotIn("flow lost", format_report(r))


class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TestOutputsForAnalysis.setUpClass.__func__(TestOutputsForAnalysis); cls.out = TestOutputsForAnalysis.out; cls.d = TestOutputsForAnalysis.d

    def labels(self, text):
        p = os.path.join(self.d, "labels.csv"); open(p, "w", encoding="utf-8").write(text); return p

    def test_label_file_parsing(self):
        p = self.labels("t,name,count\n# a comment\n4.0, person, 3\n4.0, Lamp Post\n8,car # trailing comment\njunk line\n")
        L = read_labels(p); self.assertEqual(L[4.0]["person"], 3); self.assertEqual(L[4.0]["pole"], 1); self.assertEqual(L[8.0]["car"], 1); self.assertEqual(len(L), 2)

    def test_perfect_labels_give_perfect_scores(self):
        # at t=3 and t=6 the camera (4 deg/s) sees exactly: person (15 deg), car (-20 -> -8 deg, drifting) and pole (40 deg); door and wall are still out of view
        r = evaluate(self.out, self.labels("3.0,person\n3.0,car\n3.0,pole\n6.0,person\n6.0,car\n6.0,pole\n"))
        b = [x for x in r["rows"] if x["thr"] == 0.5][0]; self.assertEqual((b["tp"], b["fp"], b["fn"]), (6, 0, 0)); self.assertEqual(b["precision"], 1.0); self.assertEqual(b["recall"], 1.0)
        miss = evaluate(self.out, self.labels("3.0,person\n3.0,car\n6.0,person\n6.0,car\n6.0,pole\n"))                 # the pole at t=3 was not written down
        self.assertEqual([x for x in miss["rows"] if x["thr"] == 0.5][0]["fp"], 1)

    def test_a_missed_object_lowers_recall_only(self):
        r = evaluate(self.out, self.labels("3.0,person\n3.0,car\n3.0,bus\n")); b = [x for x in r["rows"] if x["thr"] == 0.5][0]
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (2, 0, 1)); self.assertAlmostEqual(b["recall"], 2 / 3)

    def test_an_extra_detection_lowers_precision_only(self):
        r = evaluate(self.out, self.labels("3.0,person\n")); b = [x for x in r["rows"] if x["thr"] == 0.5][0]
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (1, 0, 0))                          # the car was detected too, but 'car' is not a labeled class: ignored
        r2 = evaluate(self.out, self.labels("3.0,person\n"), classes={"person", "car"}); b2 = [x for x in r2["rows"] if x["thr"] == 0.5][0]
        self.assertEqual((b2["tp"], b2["fp"]), (1, 1))                                      # now the car counts as a false alarm

    def test_threshold_sweep_and_suggestion(self):
        r = evaluate(self.out, self.labels("3.0,person\n6.0,pole\n"), thresholds=(0.5, 0.95))
        self.assertEqual(r["rows"][1]["tp"], 0); self.assertEqual(r["rows"][1]["fn"], 2); self.assertEqual(r["best"]["thr"], 0.5)
        txt = format_eval(r); self.assertIn("best F1", txt); self.assertIn("suggestion: --min-conf-publish 0.50", txt); self.assertIn("person", txt)

    def test_moments_without_a_processed_frame_are_reported_not_scored(self):
        r = evaluate(self.out, self.labels("3.0,person\n500.0,person\n")); self.assertEqual(r["skipped"], [500.0]); self.assertEqual(r["frames_labeled"], 1)

    def test_empty_labels_are_an_error(self):
        with self.assertRaises(ValueError): evaluate(self.out, self.labels("# nothing\n"))

    def test_cli_eval_writes_a_file(self):
        p = self.labels("3.0,person\n"); buf = io.StringIO()
        with contextlib.redirect_stdout(buf): code = main(["eval", self.out, p])
        self.assertEqual(code, 0); self.assertIn("precision", buf.getvalue()); self.assertTrue(os.path.exists(os.path.join(self.out, "eval.txt")))


class TestCliAdditions(unittest.TestCase):
    def test_tuning_flags_reach_the_config(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=6.0)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["replay", v, "--detector", "blobs", "--compass", "zero", "--publish-hits", "4", "--min-conf-publish", "0.45", "--confirm-conf", "0.3",
                  "--confirm-hits", "2", "--max-age", "9", "--iou-match", "0.2", "--az-merge", "12", "-o", os.path.join(d, "r")])
        c = json.load(open(os.path.join(d, "r", "report.json")))["config"]
        self.assertEqual((c["FastConfig"]["publish_hits"], c["FastConfig"]["min_conf_publish"], c["FastConfig"]["confirm_conf"]), (4, 0.45, 0.3))
        self.assertEqual((c["FastConfig"]["confirm_hits"], c["FastConfig"]["max_age_frames"], c["FastConfig"]["iou_match"], c["MapConfig"]["az_merge_deg"]), (2, 9, 0.2, 12.0))

    def test_snapshots_are_named_by_the_second(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=6.0); out = os.path.join(d, "snaps")
        with contextlib.redirect_stdout(io.StringIO()): main(["snapshots", v, "--every", "2", "-o", out])
        self.assertEqual(sorted(os.listdir(out)), ["t_0000.0.jpg", "t_0002.0.jpg", "t_0004.0.jpg"])


if __name__ == "__main__": unittest.main()
