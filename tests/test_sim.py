import contextlib, io, json, os, tempfile, unittest
import numpy as np, cv2
import helpers
from scenelive.camera import angdiff
from scenelive.compass import VisualCompass, ZeroCompass, CsvCompass
from scenelive.config import CameraConfig, FastConfig, MapConfig, SimConfig
from scenelive.detect.blob import ColorBlobDetector
from scenelive.report import format_report, write_outputs
from scenelive.sim import run_replay, PHONE_STAGES, SERVER_STAGES
from scenelive.synth import render_video, render_frame, _panorama, SynthObject, truth_table
from scenelive.camera import Camera


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp(); cls.video = os.path.join(cls.d, "street.mp4"); cls.truth = render_video(cls.video, duration=16.0)
        cls.static = {o.name: o for o in cls.truth["objects"] if o.drift_deg_s == 0}

    def run_sim(self, compass=None, map_cfg=None, **kw):
        sim = SimConfig(**kw)
        return run_replay(self.video, ColorBlobDetector(), CameraConfig(69.0), FastConfig(), map_cfg or MapConfig(), sim, compass or VisualCompass())


class TestResult(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass(); cls.res = Base.run_sim(cls, ) if False else None

    def test_names_bearings_and_distances(self):
        res = self.run_sim(); by = {v["name"]: v for v in res.final_view}
        for n in ("person", "car", "pole", "door"): self.assertIn(n, by)
        for n in ("person", "pole", "door"):
            self.assertLess(abs(angdiff(by[n]["world_az_deg"], self.static[n].bearing)), 4.0, n)
        self.assertTrue(4.5 <= by["person"]["range_m"] <= 7.8, by["person"]); self.assertTrue(4.3 <= by["door"]["range_m"] <= 6.0, by["door"])
        self.assertIsNone(by["pole"]["range_m"])                                      # no size prior: the system says "?" instead of guessing
        self.assertEqual(by["door"]["status"], "confirmed"); self.assertEqual(by["person"]["status"], "remembered")      # the camera has turned away

    def test_csv_compass_gives_exact_bearings(self):
        p = os.path.join(self.d, "h.csv"); open(p, "w").write("t,yaw\n" + "\n".join(f"{t / 2},{4.0 * t / 2}" for t in range(0, 40)))
        res = self.run_sim(compass=CsvCompass(p)); by = {v["name"]: v for v in res.final_view}
        for n in ("person", "pole", "door"): self.assertLess(abs(angdiff(by[n]["world_az_deg"], self.static[n].bearing)), 1.5, n)

    def test_without_a_compass_bearings_are_camera_relative_and_wrong(self):
        res = self.run_sim(compass=ZeroCompass()); by = {v["name"]: v for v in res.final_view}
        self.assertGreater(max(abs(angdiff(by[n]["world_az_deg"], self.static[n].bearing)) for n in ("person", "pole", "door")), 5.0)

    def test_first_names_come_fast_and_confirmation_later(self):
        res = self.run_sim(); fn = {r["name"]: r for r in res.first_names}
        self.assertLess(fn["person"]["latency_s"], 0.6); self.assertGreater(fn["person"]["latency_s"], 0.18)        # at least the two network legs
        self.assertGreater(fn["person"]["confirmed_s"], fn["person"]["in_list_s"])                                    # names first, confirmation after a few frames


class TestTimingAccounting(Base):
    def test_every_stage_is_timed_and_adds_up(self):
        res = self.run_sim(); rows = {r["stage"]: r for r in res.stage_rows}
        for n in PHONE_STAGES + SERVER_STAGES + ["map_build"]:
            if n != "ground": self.assertIn(n, rows, n)                                    # "ground" exists only when a surface classifier is attached
        self.assertNotIn("ground", rows)
        for f in res.frames:
            self.assertAlmostEqual(sum(f.stage_ms.values()) / 1000.0, f.compute_s, places=6)                       # compute = sum of the stages
            self.assertAlmostEqual(f.avail - f.t, (f.avail - f.finish) + f.compute_s + f.queue_wait_s + (f.t_arr - f.t), places=6)
        srv = sum(rows[n]["total_s"] for n in SERVER_STAGES if n in rows); self.assertAlmostEqual(srv, res.summary["server_compute_total_s"], places=6)
        self.assertEqual(rows["detect"]["n"], res.summary["frames_processed"]); self.assertGreater(rows["detect"]["mean_ms"], 0)

    def test_network_legs_add_to_latency(self):
        base = self.run_sim(uplink_ms=100, downlink_ms=80).summary["latency_mean_s"]
        slow = self.run_sim(uplink_ms=1000, downlink_ms=80).summary["latency_mean_s"]
        self.assertAlmostEqual(slow - base, 0.9, delta=0.12)

    def test_compute_scale_multiplies_measured_time(self):
        a = self.run_sim(compute_scale=1.0).summary["compute_mean_s"]; b = self.run_sim(compute_scale=10.0).summary["compute_mean_s"]
        self.assertAlmostEqual(b / a, 10.0, delta=3.0)                                                              # two separate runs: timings jitter

    def test_a_server_that_is_too_slow_drops_frames_and_delays_everything(self):
        ok = self.run_sim().summary; slow = self.run_sim(compute_scale=300.0).summary
        self.assertTrue(ok["keeps_up"]); self.assertFalse(slow["keeps_up"])
        self.assertGreater(slow["frames_dropped"], 0); self.assertLess(slow["frames_processed"], slow["frames_sent"])
        self.assertEqual(slow["frames_processed"] + slow["frames_dropped"], slow["frames_sent"])
        self.assertGreater(slow["latency_mean_s"], ok["latency_mean_s"] + 1.0)

    def test_without_dropping_the_queue_just_keeps_growing(self):
        res = self.run_sim(compute_scale=300.0, latest_wins=False)
        self.assertEqual(res.summary["frames_dropped"], 0); self.assertEqual(res.summary["frames_processed"], res.summary["frames_sent"])
        lat = [f.avail - f.t for f in res.frames]; self.assertGreater(lat[-1], lat[0] + 5.0)
        dropping = self.run_sim(compute_scale=300.0, latest_wins=True).summary
        self.assertLess(dropping["latency_max_s"], max(lat))                                                         # dropping old frames bounds the delay

    def test_shared_cpu_lets_the_map_loop_block_the_fast_loop(self):
        own = self.run_sim(map_compute_scale=20000.0, shared_cpu=False); shared = self.run_sim(map_compute_scale=20000.0, shared_cpu=True)
        self.assertLess(max(f.queue_wait_s for f in own.frames), 0.3); self.assertGreater(max(f.queue_wait_s for f in shared.frames), 1.0)

    def test_the_map_runs_on_schedule_and_once_more_at_the_end(self):
        res = self.run_sim(map_cfg=MapConfig(interval_s=5.0)); dues = [m.due for m in res.maps]
        self.assertEqual(dues[:3], [5.0, 10.0, 15.0]); self.assertGreaterEqual(len(dues), 4)
        for m in res.maps: self.assertGreaterEqual(m.avail, m.due + 0.08 - 1e-9)

    def test_fewer_frames_per_second_means_fewer_frames(self):
        n3 = self.run_sim(sample_fps=3.0).summary["frames_sent"]; n15 = self.run_sim(sample_fps=1.5).summary["frames_sent"]
        self.assertAlmostEqual(n15 / n3, 0.5, delta=0.06); self.assertEqual(self.run_sim(max_seconds=4.0).summary["frames_sent"], 13)


class TestOutputs(Base):
    def test_report_text_and_files(self):
        res = self.run_sim(); txt = format_report(res)
        for s in ("STAGE TIMING", "END TO END", "FIRST NAMES", "LIST AT THE END", "slowest stage", "detect", "compass", "verdict: keeps up"): self.assertIn(s, txt)
        out = os.path.join(self.d, "out"); write_outputs(res, out)
        for f in ("report.txt", "report.json", "stages.csv", "frames.csv", "events.csv", "first_names.csv"): self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        j = json.load(open(os.path.join(out, "report.json"), encoding="utf-8")); self.assertEqual(j["summary"]["frames_processed"], res.summary["frames_processed"])
        self.assertEqual(j["config"]["detector"], "color-blobs")
        self.assertEqual(len(open(os.path.join(out, "frames.csv")).read().strip().splitlines()), res.summary["frames_processed"] + 1)

    def test_overlay_video_has_one_frame_per_processed_frame(self):
        out = os.path.join(self.d, "ov", "o.mp4")
        res = run_replay(self.video, ColorBlobDetector(), CameraConfig(69.0), FastConfig(), MapConfig(), SimConfig(), VisualCompass(), out_video=out)
        cap = cv2.VideoCapture(out); n = 0
        while cap.read()[0]: n += 1
        self.assertEqual(n, res.summary["frames_processed"])

    def test_a_missing_video_is_a_clear_error(self):
        with self.assertRaises(FileNotFoundError): run_replay(os.path.join(self.d, "nope.mp4"), ColorBlobDetector(), CameraConfig(), FastConfig(), MapConfig(), SimConfig(), ZeroCompass())


class TestSynth(unittest.TestCase):
    def test_box_size_and_position_follow_the_pinhole_model(self):
        cam = Camera(640, 360, 69.0); pano = _panorama(360, 0)
        img, _ = render_frame(0.0, [SynthObject("person", 0, 10, 1.7, 0.5, (30, 30, 220))], cam, 4.0, pano)
        d = ColorBlobDetector().detect(img)[0]
        self.assertAlmostEqual(d.h, cam.f * 1.7 / 10, delta=2.0); self.assertAlmostEqual(d.cx, cam.cx, delta=1.5)
        self.assertAlmostEqual(d.box[3], cam.cy + cam.f * 1.5 / 10, delta=2.0)                  # feet on the ground, 1.5 m below the camera

    def test_objects_behind_the_camera_are_not_drawn_and_truth_flags_priors(self):
        cam = Camera(640, 360, 69.0); img, _ = render_frame(0.0, [SynthObject("car", 170, 10, 1.5, 1.8, (220, 60, 30))], cam, 4.0, _panorama(360, 0))
        self.assertEqual(ColorBlobDetector().detect(img), [])
        t = {r["name"]: r for r in truth_table({"objects": __import__("scenelive.synth", fromlist=["x"]).default_world()})}
        self.assertTrue(t["person"]["has_size_prior"]); self.assertFalse(t["pole"]["has_size_prior"]); self.assertTrue(t["car"]["moving"])


if __name__ == "__main__": unittest.main()
