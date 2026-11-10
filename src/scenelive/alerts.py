"""Spoken warnings. The list answers questions; alerts speak up by themselves, but sparingly:
  danger  - a dangerous object (knife, weapon, fire, open manhole, predator ...) is in view: warn at once, whatever the direction
  caution - a vehicle, big animal or obstacle is CLOSE and AHEAD; a hole, ice or puddle is straight ahead
  ground  - the surface straight ahead is bad (pothole, ice, mud, cracked road ...)
Protection against noise: the same thing is not repeated for `cooldown_s`, at most one alert per update and one per `min_gap_s`.
These warnings are hints from a camera, not a safety system: detections of weapons and hazards are often wrong in both directions."""
from __future__ import annotations
from .catalog import CATEGORY, RU_NAME, SEVERITY
from .speech import direction_words, range_speech


class AlertEngine:
    def __init__(self, cooldown_s: float = 15.0, caution_range_m: float = 6.0, min_gap_s: float = 2.5, ahead_deg: float = 60.0,
                 straight_deg: float = 30.0, min_hits: int = 3, danger_conf: float = 0.35, sure_conf: float = 0.55):
        self.cooldown_s, self.caution_range_m, self.min_gap_s, self.ahead_deg, self.straight_deg = cooldown_s, caution_range_m, min_gap_s, ahead_deg, straight_deg
        self.min_hits, self.danger_conf, self.sure_conf = min_hits, danger_conf, sure_conf
        self.last: dict = {}; self.last_any = -1e9

    @staticmethod
    def _sector(rel: float) -> str:
        return "ahead" if abs(rel) <= 45 else ("right" if rel > 0 else "left") if abs(rel) <= 135 else "behind"

    def _text(self, level: str, name: str, v: dict) -> str:
        ru = RU_NAME.get(name, name); where = "" if v["clock"] is None else direction_words(v["clock"]); dist = range_speech(v.get("range"))
        tail = ", ".join(x for x in (where, dist) if x)
        maybe = "возможно, " if v["conf"] < self.sure_conf else ""
        head = "Осторожно! Опасность: " if level == "danger" else "Внимание: "
        return f"{head}{maybe}{ru}" + (f", {tail}" if tail else "") + "."

    def update(self, view: list[dict], ground: dict | None, t: float, free=None) -> list[dict]:
        cands = []                                                                           # (priority, distance, alert dict, key)
        for v in view:
            if v["status"] == "remembered" or v.get("age_s", 0) > 2.5: continue
            name = v["name"]; sev = SEVERITY.get(name, "info"); rel = v["rel_az_deg"]
            if sev == "danger":
                if v["n_obs"] >= self.min_hits and v["conf"] >= self.danger_conf:
                    cands.append((0, v["range_m"] or 99.0, {"level": "danger", "text": self._text("danger", name, v), "name": name}, (name, self._sector(rel))))
            elif sev == "caution" and v["n_obs"] >= 2:
                near_ahead = v["range_m"] is not None and v["range_m"] <= self.caution_range_m and abs(rel) <= self.ahead_deg
                straight = v["range_m"] is None and CATEGORY.get(name) in ("hazard", "obstacle") and v["n_obs"] >= self.min_hits and abs(rel) <= self.straight_deg
                if near_ahead or straight:
                    cands.append((1, v["range_m"] or 99.0, {"level": "caution", "text": self._text("caution", name, v), "name": name}, (name, self._sector(rel))))
        c = (ground or {}).get("center")
        if c and c.get("bad"):
            cands.append((1, 0.0, {"level": "caution", "text": f"Внимание: впереди {c['ru']}, покрытие {c['quality_ru']}.", "name": "ground:" + c["surface"]}, ("ground", c["surface"])))
        if free is not None and free.reliable and not free.uncertain:                          # free space: standing on the road, or about to walk into something
            from .freespace import advise
            if free.underfoot == "road":
                cands.append((0, 0.0, {"level": "danger", "text": "Внимание! Похоже, вы на проезжей части.", "name": "road"}, ("road", "underfoot")))
            else:
                a = advise(free)
                if a["level"] == "stop":
                    centre = min((s for s in free.sectors if s.hi > -12 and s.lo < 12), key=lambda s: s.free_safe_m, default=None)
                    cands.append((0, 0.0, {"level": "danger", "text": a["text"], "name": "stop"}, ("stop", centre.blocker if centre else "?")))
        out = []
        if t - self.last_any < self.min_gap_s: return out
        for prio, dist, alert, key in sorted(cands, key=lambda x: (x[0], x[1])):
            if t - self.last.get(key, -1e9) < self.cooldown_s: continue
            self.last[key] = t; self.last_any = t; out.append({**alert, "t": round(t, 2)}); break                  # at most one per update
        return out
