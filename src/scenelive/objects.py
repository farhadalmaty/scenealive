"""The list the phone shows. Two sources feed it: the fast loop (one entry per track, provisional names, right now) and the map loop
(one entry per merged object, confirmed names and better distances, every few seconds). Map entries cover the tracks they merged."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from .camera import angdiff, clock, range_label


@dataclass
class Entry:
    key: str                      # "T12" (a track from the fast loop) or "O3" (an object from the map loop)
    name: str
    conf: float
    status: str                   # provisional | confirmed | remembered
    world_az: float
    range_m: float | None
    range_sigma: float | None
    first_t: float                # video time of the first detection
    last_t: float
    source: str                   # fast | map
    n_obs: int
    track_ids: tuple = ()
    moving: bool = False

    def view(self, heading_deg: float) -> dict:
        rel = angdiff(self.world_az, heading_deg); hour, behind = clock(rel)
        return {"key": self.key, "name": self.name, "conf": round(self.conf, 2), "status": self.status, "world_az_deg": round(self.world_az, 1), "rel_az_deg": round(rel, 1),
                "clock": hour, "behind": behind, "range_m": None if self.range_m is None else round(self.range_m, 1),
                "range": range_label(self.range_m, self.range_sigma), "source": self.source, "n_obs": self.n_obs,
                "first_t": round(self.first_t, 2), "last_t": round(self.last_t, 2), "moving": self.moving, "track_ids": list(self.track_ids)}


@dataclass
class ListEvent:
    t_avail: float                # when the phone could first see this change (server finish + network)
    kind: str                     # new | confirmed | lost | map_new | map_remembered
    key: str
    name: str
    first_t: float                # video time of the first detection behind it


class ObjectList:
    def __init__(self):
        self.fast: dict[str, Entry] = {}; self.map: dict[str, Entry] = {}; self.events: list[ListEvent] = []; self.version = 0

    def apply_fast(self, entries: list[Entry], t_avail: float) -> None:
        new = {e.key: e for e in entries}
        for k, e in new.items():
            old = self.fast.get(k)
            if old is None: self.events.append(ListEvent(t_avail, "new", k, e.name, e.first_t))
            elif old.status != "confirmed" and e.status == "confirmed": self.events.append(ListEvent(t_avail, "confirmed", k, e.name, e.first_t))
        ids = self._covered_ids()
        for k, e in self.fast.items():
            if k not in new and not self._covered(e, ids): self.events.append(ListEvent(t_avail, "lost", k, e.name, e.first_t))
        self.fast = new; self.version += 1
        if len(self.events) > 5000: del self.events[:1000]                         # a session may run for hours

    def apply_map(self, entries: list[Entry], t_avail: float) -> None:
        new = {e.key: e for e in entries}
        for k, e in new.items():
            old = self.map.get(k)
            if old is None: self.events.append(ListEvent(t_avail, "map_new", k, e.name, e.first_t))
            elif old.status != "remembered" and e.status == "remembered": self.events.append(ListEvent(t_avail, "map_remembered", k, e.name, e.first_t))
        self.map = new; self.version += 1

    def _covered_ids(self) -> set:
        return {i for m in self.map.values() for i in m.track_ids}

    def _covered(self, e: Entry, ids: set | None = None) -> bool:
        ids = self._covered_ids() if ids is None else ids
        return bool(e.track_ids) and all(i in ids for i in e.track_ids)

    def entries(self) -> list[Entry]:
        """What the phone shows: map objects first, then fast tracks that no map object has absorbed yet."""
        ids = self._covered_ids(); live = {i: e for e in self.fast.values() for i in e.track_ids}; out = []
        for m in self.map.values():
            cur = [live[i] for i in m.track_ids if i in live]
            if not cur: out.append(m); continue
            f = max(cur, key=lambda e: e.last_t)                    # the object is in view right now: do not show data from the last map run
            upd = dict(last_t=max(m.last_t, f.last_t), status="confirmed")
            if m.moving: upd.update(world_az=f.world_az, range_m=f.range_m if f.range_m is not None else m.range_m, range_sigma=f.range_sigma if f.range_m is not None else m.range_sigma)
            out.append(replace(m, **upd))
        return out + [e for e in self.fast.values() if not self._covered(e, ids)]

    def view(self, heading_deg: float) -> list[dict]:
        order = {"confirmed": 0, "provisional": 1, "remembered": 2}
        return sorted((e.view(heading_deg) for e in self.entries()), key=lambda v: (order[v["status"]], v["range_m"] is None, v["range_m"] or 0))
