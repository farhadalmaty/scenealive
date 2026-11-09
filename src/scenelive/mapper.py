"""The map loop: merge tracks into objects, vote names, fuse bearings and distances. Runs every few seconds, in the background.

Two tracks of the same class are one object when their world azimuths agree (static classes) or when one ends where the other begins
(moving classes). With camera translation (ARKit) the rays from different positions are intersected for a real distance."""
from __future__ import annotations
import math
import numpy as np
from .camera import angdiff, circular_mean
from .config import FastConfig, MapConfig
from .labels import MOVABLE
from .objects import Entry
from .rays import direction, triangulate


def _find(p, i):
    while p[i] != i: p[i] = p[p[i]]; i = p[i]
    return i


def build_map(tracks, now_t: float, cfg: MapConfig, fast_cfg: FastConfig) -> list[Entry]:
    by = fast_cfg.publish_hits_by_class
    def eligible(tr) -> bool:
        if not tr.obs or tr.conf < fast_cfg.min_conf_publish: return False
        if tr.name in by: return tr.hits >= by[tr.name]                                   # false-alarm-prone classes: more sightings, no shortcut
        return tr.hits >= cfg.min_obs or tr.conf >= cfg.single_conf                       # a sure glimpse is remembered too
    elig = [tr for tr in tracks if eligible(tr)]
    groups: dict[str, list] = {}
    for tr in elig: groups.setdefault(tr.name, []).append(tr)
    out: list[Entry] = []
    for name, trs in groups.items():
        moving = name in MOVABLE; n = len(trs); p = list(range(n))
        if moving:                                              # pairwise, but with everything precomputed (no work inside the loop)
            first_t = [t.first_t for t in trs]; last_t = [t.last_t for t in trs]
            first_az = [t.obs[0].world_az for t in trs]; last_az = [t.obs[-1].world_az for t in trs]
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = (i, j) if first_t[i] <= first_t[j] else (j, i)
                    gap = first_t[b] - last_t[a]
                    if 0.0 <= gap <= cfg.moving_gap_s and abs(angdiff(last_az[a], first_az[b])) <= cfg.az_merge_deg * 2:
                        p[_find(p, i)] = _find(p, j)
        else:                                                   # 1-D: tracks within the threshold form chains of neighbours on a circle
            mean_az = [circular_mean([o.world_az for o in t.obs]) for t in trs]
            order = sorted(range(n), key=lambda i: mean_az[i])
            for k in range(n - 1):
                i, j = order[k], order[k + 1]
                if abs(angdiff(mean_az[i], mean_az[j])) <= cfg.az_merge_deg: p[_find(p, i)] = _find(p, j)
            if n > 2 and abs(angdiff(mean_az[order[-1]], mean_az[order[0]])) <= cfg.az_merge_deg: p[_find(p, order[-1])] = _find(p, order[0])
        comps: dict[int, list] = {}
        for i in range(n): comps.setdefault(_find(p, i), []).append(trs[i])
        for members in comps.values(): out.append(_object(name, members, moving, now_t, cfg))
    return sorted(out, key=lambda e: e.first_t)


def _object(name, members, moving, now_t, cfg: MapConfig) -> Entry:
    hits = sum(m.hits for m in members); latest = max(members, key=lambda m: m.last_t)
    obs = latest.obs[-3:] if moving else [o for m in members for o in m.obs]
    pool = [o for m in members for o in m.obs]; whole = [o for o in pool if not o.edge] or pool       # boxes cut by the frame edge have biased centres
    if moving:                                                  # where it was last seen, preferring the last box not cut by the frame
        cand = [o for o in latest.obs if not o.edge]; az = (cand[-1] if cand else latest.obs[-1]).world_az
    else:
        az = circular_mean([o.world_az for o in whole])
    rng, sig = None, None
    rs = [o.range for o in obs if o.range is not None]
    if rs:
        rng = float(np.median([r[0] for r in rs])); sig = max(0.12, float(np.median([r[1] for r in rs])) / math.sqrt(len(rs)))
    if not moving:                                              # translation available: intersect rays for a real distance
        withpos = [o for m in members for o in m.obs if o.cam_pos is not None]
        if len(withpos) >= 2:
            tri = triangulate([o.cam_pos for o in withpos], [direction(o.world_az, o.el) for o in withpos])
            if tri is not None:
                x, res = tri; rng = float(np.linalg.norm(x - np.asarray(withpos[-1].cam_pos))); sig = max(0.08, res / max(rng, 1e-6))
    conf = sum(m.conf * m.hits for m in members) / hits
    last_t = max(m.last_t for m in members)
    status = "confirmed" if now_t - last_t <= cfg.remembered_after_s else "remembered"
    ids = tuple(sorted(m.id for m in members))
    return Entry(f"O{ids[0]}", name, conf, status, az, rng, sig, min(m.first_t for m in members), last_t, "map", hits, ids, moving)
