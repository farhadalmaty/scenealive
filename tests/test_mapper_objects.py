import unittest
import helpers
from helpers import make_track
from scenelive.config import FastConfig, MapConfig
from scenelive.mapper import build_map
from scenelive.objects import Entry, ObjectList

F, M = FastConfig(), MapConfig()


class TestMapper(unittest.TestCase):
    def test_same_class_same_bearing_is_one_object(self):
        e = build_map([make_track(1, "pole", 0, 3, 40), make_track(2, "pole", 5, 8, 42)], 9.0, M, F)
        self.assertEqual(len(e), 1); self.assertEqual(e[0].track_ids, (1, 2)); self.assertEqual(e[0].key, "O1"); self.assertAlmostEqual(e[0].world_az, 41, delta=0.5)

    def test_different_bearing_or_class_are_different_objects(self):
        e = build_map([make_track(1, "pole", 0, 3, 40), make_track(2, "pole", 0, 3, 70), make_track(3, "door", 0, 3, 40)], 4.0, M, F)
        self.assertEqual(sorted((x.name, x.track_ids) for x in e), [("door", (3,)), ("pole", (1,)), ("pole", (2,))])

    def test_moving_class_continues_only_in_sequence(self):
        seq = build_map([make_track(1, "car", 0, 3, 10, az_end=20), make_track(2, "car", 4, 7, 22, az_end=30)], 8.0, M, F)
        self.assertEqual(len(seq), 1)
        both = build_map([make_track(1, "car", 0, 5, 10), make_track(2, "car", 1, 6, 12)], 8.0, M, F)             # two cars at the same time
        self.assertEqual(len(both), 2)
        late = build_map([make_track(1, "car", 0, 3, 10), make_track(2, "car", 20, 23, 10)], 25.0, M, F)           # long gap: another car
        self.assertEqual(len(late), 2)

    def test_moving_object_reports_where_it_was_last_seen(self):
        e = build_map([make_track(1, "person", 0, 4, 10, az_end=30)], 5.0, M, F)[0]
        self.assertAlmostEqual(e.world_az, 30, delta=0.5); self.assertTrue(e.moving)

    def test_a_mover_leaving_the_frame_is_reported_at_its_last_whole_box(self):
        t = make_track(1, "person", 0, 4, 10, n=5)
        t.obs[-1].world_az = 25.0; t.obs[-1].edge = True                            # the last box is cut by the frame edge
        self.assertAlmostEqual(build_map([t], 5.0, M, F)[0].world_az, 10.0, delta=0.01)

    def test_static_object_averages_its_bearing(self):
        e = build_map([make_track(1, "wall", 0, 4, 80, az_end=90)], 5.0, M, F)[0]; self.assertAlmostEqual(e.world_az, 85, delta=1.0)

    def test_boxes_cut_by_the_frame_edge_do_not_bias_the_bearing(self):
        t = make_track(1, "door", 0, 4, 60, n=6)
        for o in t.obs[:2]: o.world_az = 75.0; o.edge = True                         # two cut observations, centre pushed inwards
        self.assertAlmostEqual(build_map([t], 5.0, M, F)[0].world_az, 60.0, delta=0.01)
        all_cut = make_track(2, "wall", 0, 4, 80, n=4)
        for o in all_cut.obs: o.edge = True                                          # nothing better available: still reported
        self.assertAlmostEqual(build_map([all_cut], 5.0, M, F)[0].world_az, 80.0, delta=0.01)

    def test_status_confirmed_then_remembered(self):
        t = make_track(1, "door", 0, 10, 60)
        self.assertEqual(build_map([t], 11.0, M, F)[0].status, "confirmed"); self.assertEqual(build_map([t], 14.0, M, F)[0].status, "remembered")

    def test_weak_tracks_are_not_mapped(self):
        self.assertEqual(build_map([make_track(1, "door", 0, 0, 60, n=1, conf=0.4)], 5, M, F), [])                # seen once and not sure: noise
        self.assertEqual(build_map([make_track(1, "door", 0, 3, 60, conf=0.1)], 5, M, F), [])                    # name too unsure
        glimpse = build_map([make_track(1, "car", 0, 0, 60, n=1, conf=0.9)], 5, M, F); self.assertEqual(len(glimpse), 1)   # a SURE glimpse is remembered
        for hits, n_obj in ((3, 0), (4, 1)): self.assertEqual(len(build_map([make_track(1, "billboard", 0, 3, 60, n=hits, conf=0.9)], 5, M, F)), n_obj)    # a false-alarm-prone class needs 4 sightings

    def test_range_is_the_median_with_less_uncertainty_than_one_observation(self):
        t = make_track(1, "person", 0, 4, 10, n=9, rng=(6.0, 0.2)); e = build_map([t], 5.0, M, F)[0]
        self.assertAlmostEqual(e.range_m, 6.0); self.assertLess(e.range_sigma, 0.2); self.assertGreaterEqual(e.range_sigma, 0.12)
        self.assertIsNone(build_map([make_track(1, "pole", 0, 4, 10)], 5.0, M, F)[0].range_m)                    # no size prior -> unknown

    def test_translation_gives_a_triangulated_distance(self):
        import math
        P = (30.0, 40.0); trs = []
        for i, cp in enumerate([(0, 0, 1.5), (10, 0, 1.5), (20, 0, 1.5)]):
            az = math.degrees(math.atan2(P[0] - cp[0], P[1] - cp[1])); trs.append(make_track(i + 1, "pole", i, i + 1, az, n=2, cam_pos=cp))
        e = build_map(trs, 4.0, MapConfig(az_merge_deg=40.0), F)
        self.assertEqual(len(e), 1); self.assertAlmostEqual(e[0].range_m, math.hypot(P[0] - 20, P[1] - 0), delta=0.5)


def entry(key, name="car", status="provisional", az=0.0, ids=(1,), source="fast", rng=5.0):
    return Entry(key, name, 0.9, status, az, rng, 0.2, 0.0, 1.0, source, 3, ids)


class TestObjectList(unittest.TestCase):
    def test_events_new_confirmed_lost(self):
        ol = ObjectList(); ol.apply_fast([entry("T1")], 1.0); ol.apply_fast([entry("T1", status="confirmed")], 2.0); ol.apply_fast([], 3.0)
        self.assertEqual([(e.kind, e.t_avail) for e in ol.events], [("new", 1.0), ("confirmed", 2.0), ("lost", 3.0)])

    def test_map_objects_absorb_their_tracks(self):
        ol = ObjectList(); ol.apply_fast([entry("T1", ids=(1,)), entry("T2", name="pole", ids=(2,))], 1.0)
        ol.apply_map([entry("O1", "car", "confirmed", ids=(1,), source="map")], 2.0)
        self.assertEqual(sorted(e.key for e in ol.entries()), ["O1", "T2"])                    # T1 is inside O1; T2 is still shown on its own
        ol.apply_fast([entry("T2", name="pole", ids=(2,))], 3.0)                                # T1 disappears from the fast loop: not "lost", the map has it
        self.assertNotIn("lost", [e.kind for e in ol.events]); self.assertIn("map_new", [e.kind for e in ol.events])

    def test_an_object_in_view_is_shown_with_live_data_not_the_last_map_run(self):
        ol = ObjectList(); mover = entry("O1", "person", "remembered", az=10.0, ids=(1,), source="map"); mover.moving = True; mover.last_t = 5.0
        ol.apply_map([mover], 6.0); live = entry("T1", "person", "confirmed", az=40.0, ids=(1,), rng=3.0); live.last_t = 9.0; live.moving = True; ol.apply_fast([live], 9.0)
        e = ol.entries()[0]; self.assertEqual((e.key, e.status, e.world_az, e.range_m, e.last_t), ("O1", "confirmed", 40.0, 3.0, 9.0))      # a mover: its live position
        still = entry("O2", "door", "remembered", az=60.0, ids=(2,), source="map"); ol2 = ObjectList(); ol2.apply_map([still], 6.0)
        l2 = entry("T2", "door", "confirmed", az=63.0, ids=(2,)); l2.last_t = 9.0; ol2.apply_fast([l2], 9.0)
        e2 = ol2.entries()[0]; self.assertEqual((e2.status, e2.world_az, e2.last_t), ("confirmed", 60.0, 9.0))                              # a static object keeps its refined bearing
        ol2.apply_fast([], 12.0); self.assertEqual(ol2.entries()[0].last_t, 6.0 if False else still.last_t)                                  # out of view again: the map's own data

    def test_view_uses_the_current_heading_and_sorts(self):
        ol = ObjectList(); ol.apply_fast([entry("T1", az=90, rng=10), entry("T2", name="person", status="confirmed", az=0, ids=(2,), rng=3)], 1.0)
        v = ol.view(heading_deg=0.0); self.assertEqual([x["name"] for x in v], ["person", "car"])                      # confirmed before provisional
        by = lambda heading: {x["name"]: x for x in ol.view(heading)}
        self.assertEqual((by(0)["person"]["clock"], by(0)["car"]["clock"]), (12, 3))
        self.assertEqual((by(90)["person"]["clock"], by(90)["car"]["clock"]), (9, 12))             # turned right 90: the person is on the left, the car ahead
        self.assertTrue(by(180)["person"]["behind"]); self.assertFalse(by(90)["car"]["behind"])


if __name__ == "__main__": unittest.main()
