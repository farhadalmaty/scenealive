"""The chat agent over the live list. Tools read the session; the agent turns questions into tool calls and answers briefly
(short answers now, because they will be read aloud when the voice mode arrives).
Two agents share one interface `reply(text) -> (answer, [tool names])`:
  ChatAgent  - Claude with tool use (needs the `anthropic` package and ANTHROPIC_API_KEY on the SERVER)
  RuleAgent  - offline, keyword based, Russian and English (works without any key)"""
from __future__ import annotations
import json
import os
import re
from ..camera import angdiff
from ..catalog import CATEGORY, CATEGORY_RU, NAMES, RU_NAME, SEVERITY, names_in_category, names_with_severity
from ..freespace import advise, advise_goal, dist_words
from ..labels import CANONICAL, COCO_STREET, STREET_VOCAB

# Russian word stems -> canonical names. Stems shorter than 4 letters must match almost exactly ("кот" must not match "который"),
# and ambiguous short words ("стол" vs "столб") are listed with all their forms in RU_EXACT.
RU_EN = {
    "человек": ["person"], "люди": ["person"], "людей": ["person"], "пешеход": ["person"], "прохожи": ["person"],
    "машин": ["car", "van"], "автомобил": ["car", "van"], "легков": ["car"], "такси": ["car"], "автобус": ["bus"], "грузовик": ["truck"], "фургон": ["van"],
    "велосипед": ["bicycle"], "велик": ["bicycle"], "мотоцикл": ["motorcycle"], "мопед": ["motorcycle"], "самокат": ["scooter"], "трамва": ["tram"],
    "поезд": ["train"], "коляск": ["stroller"], "инвалидн": ["wheelchair"], "тележк": ["shopping cart"],
    "собак": ["dog"], "кошк": ["cat"], "птиц": ["bird"], "куриц": ["chicken"], "утк": ["duck"], "лошад": ["horse"], "коров": ["cow"], "свин": ["pig"],
    "олен": ["deer"], "крыс": ["rat"], "змея": ["snake"], "змеи": ["snake"], "змею": ["snake"], "змей": ["snake"], "медвед": ["bear"], "волк": ["wolf"],
    "светофор": ["traffic light"], "знак": ["sign", "stop sign", "street sign"], "гидрант": ["fire hydrant"], "скамей": ["bench"], "лавочк": ["bench"],
    "урн": ["trash can"], "столбик": ["bollard"], "ограждени": ["barrier", "fence"], "барьер": ["barrier"], "заграждени": ["barrier", "barricade"],
    "баррикад": ["barricade"], "забор": ["fence"], "калитк": ["fence"], "ворота": ["fence"], "столб": ["pole"], "фонар": ["pole"], "опор": ["pole"],
    "паркомат": ["parking meter"], "киоск": ["kiosk"], "рекламн": ["billboard"], "билборд": ["billboard"], "почтов": ["mailbox"], "конус": ["construction cone"],
    "контейнер": ["dumpster"], "кабел": ["cable"], "провод": ["cable", "exposed wire"], "оголённ": ["exposed wire"], "оголенн": ["exposed wire"],
    "сугроб": ["snow pile"], "лежач": ["speed bump"],
    "дерев": ["tree"], "куст": ["bush"], "изгород": ["hedge"], "ветк": ["branch"], "ветви": ["branch"], "камн": ["rock"], "камен": ["rock"], "бревн": ["log"], "клумб": ["flower bed"],
    "стен": ["wall"], "здани": ["building"], "дома": ["building"], "двер": ["door"], "стеклянн": ["glass door"], "окн": ["window"], "лестниц": ["stairs"], "ступен": ["stairs"],
    "бордюр": ["curb"], "поребрик": ["curb"], "пандус": ["ramp"], "перил": ["railing"], "колонн": ["pillar"], "мост": ["bridge"], "лифт": ["elevator"], "эскалатор": ["escalator"],
    "штор": ["curtain"], "занавес": ["curtain"], "гардин": ["curtain"], "диван": ["sofa"], "кроват": ["bed"], "стул": ["chair"],
    "рюкзак": ["backpack"], "зонт": ["umbrella"], "сумк": ["bag"], "чемодан": ["suitcase"], "растени": ["plant"],
    "выбоин": ["pothole"], "ухаб": ["pothole"], "канав": ["ditch"], "ледян": ["ice"], "разбит": ["broken glass"], "осколк": ["broken glass"], "шприц": ["syringe"],
    "огонь": ["fire"], "огня": ["fire"], "пожар": ["fire"], "пламя": ["fire"], "молоток": ["hammer"], "молотк": ["hammer"], "ножниц": ["scissors"],
    "оружи": ["weapon", "gun", "knife"], "пистолет": ["gun"], "автомат": ["gun"], "ружь": ["gun"], "ружей": ["gun"], "винтовк": ["gun"], "огнестрел": ["gun"],
    "топор": ["axe"], "бензопил": ["chainsaw"], "мусорный": ["garbage bag", "dumpster", "trash can"], "пакет": ["plastic bag", "garbage bag"], "бутылк": ["bottle"],
    "картон": ["cardboard box"], "обломк": ["debris"], "строительн": ["construction waste", "scaffolding"], "дорог": ["road", "cracked road", "dirt road"],
    "тротуар": ["sidewalk"], "асфальт": ["asphalt", "cracked road"], "плитк": ["paving tiles", "tactile paving"], "тактильн": ["tactile paving"], "переход": ["crosswalk"],
    "зебр": ["crosswalk"], "гравий": ["gravel"], "щебен": ["gravel"], "щебн": ["gravel"], "грунт": ["dirt road"], "песок": ["sand"], "песк": ["sand"],
    "трав": ["grass"], "рельс": ["tram tracks"], "выброс": ["trash"],
}
RU_SHORT = {"кот": ["cat"], "дом": ["building"], "коза": ["goat"], "коз": ["goat"], "овц": ["sheep"], "цеп": ["chain"], "люк": ["open manhole", "manhole cover"], "луж": ["puddle"],
            "дым": ["smoke"], "нож": ["knife"], "бит": ["baseball bat"], "лом": ["crowbar"]}
RU_EXACT = {"стол": ["table"], "стола": ["table"], "столу": ["table"], "столом": ["table"], "столе": ["table"], "столы": ["table"], "столов": ["table"],
            "яма": ["hole", "pothole"], "ямы": ["hole", "pothole"], "яму": ["hole", "pothole"], "ямой": ["hole", "pothole"], "ям": ["hole", "pothole"],
            "лёд": ["ice"], "лед": ["ice"], "льда": ["ice"], "льду": ["ice"], "льдом": ["ice"], "грязь": ["mud"], "грязи": ["mud"], "грязью": ["mud"],
            "банка": ["can"], "банки": ["can"], "банку": ["can"], "банкой": ["can"], "банок": ["can"], "лес": [], "ножа": ["knife"], "ножи": ["knife"], "ножом": ["knife"], "ножей": ["knife"]}
RU_CATS = {"животн": ["animal"], "зверь": ["animal"], "зверей": ["animal"], "мусор": ["trash"], "отход": ["trash"], "оружи": ["weapon"], "острые": ["weapon"], "острых": ["weapon"],
           "транспорт": ["vehicle"], "препятств": ["obstacle", "furniture"], "опасн": ["danger"], "угроз": ["danger"]}
EN_CATS = {"animals": ["animal"], "animal": ["animal"], "trash": ["trash"], "garbage": ["trash"], "weapons": ["weapon"], "weapon": ["weapon"], "vehicles": ["vehicle"],
           "obstacles": ["obstacle", "furniture"], "dangerous": ["danger"], "danger": ["danger"], "hazards": ["danger"]}
EN_SYN = {"people": ["person"], "pedestrian": ["person"], "pedestrians": ["person"], "man": ["person"], "woman": ["person"],
          "vehicle": ["car", "van", "bus", "truck"], "vehicles": ["car", "van", "bus", "truck"], "lamp": ["pole"], "lamppost": ["pole"], "obstacle": ["barrier", "bollard", "pole", "fence"]}
EN_RU = {"person": "человек", "car": "машина", "van": "фургон", "bus": "автобус", "truck": "грузовик", "bicycle": "велосипед",
         "motorcycle": "мотоцикл", "scooter": "самокат", "dog": "собака", "cat": "кошка", "bird": "птица", "traffic light": "светофор",
         "stop sign": "знак «стоп»", "sign": "знак", "fire hydrant": "гидрант", "bench": "скамейка", "trash can": "урна",
         "bollard": "столбик", "barrier": "ограждение", "fence": "забор", "pole": "столб", "tree": "дерево", "wall": "стена",
         "building": "здание", "door": "дверь", "window": "окно", "stairs": "лестница", "backpack": "рюкзак", "umbrella": "зонт",
         "bag": "сумка", "suitcase": "чемодан", "chair": "стул", "plant": "растение", "skateboard": "скейт", "boat": "лодка", "train": "поезд"}
EN_RU.update(RU_NAME)
DIRECTIONS = [("слева", -90.0), ("налево", -90.0), ("левее", -90.0), ("справа", 90.0), ("направо", 90.0), ("правее", 90.0),
              ("впереди", 0.0), ("спереди", 0.0), ("прямо", 0.0), ("сзади", 180.0), ("позади", 180.0),
              ("left", -90.0), ("right", 90.0), ("ahead", 0.0), ("front", 0.0), ("behind", 180.0)]
KNOWN = set(STREET_VOCAB) | set(COCO_STREET) | set(CANONICAL.values()) | set(EN_RU) | set(NAMES)


def _cat_names(cats) -> list[str]:
    out: list[str] = []
    for c in cats: out += names_with_severity("danger") if c == "danger" else names_in_category(c)
    return out


def resolve_names(query: str) -> set[str]:
    """Words of a question (Russian or English) -> canonical object names. Empty set if nothing is recognised."""
    q = (query or "").lower(); names: set[str] = set()
    for tok in re.findall(r"[a-zа-яё]+", q):
        for stem, ns in RU_EN.items():
            if tok.startswith(stem): names.update(ns)
        for stem, ns in RU_SHORT.items():
            if tok == stem or (tok.startswith(stem) and len(tok) <= len(stem) + 2): names.update(ns)
        names.update(RU_EXACT.get(tok, [])); names.update(EN_SYN.get(tok, []))
        for stem, cats in RU_CATS.items():
            if tok.startswith(stem): names.update(_cat_names(cats))
        for cats in (EN_CATS.get(tok),):
            if cats: names.update(_cat_names(cats))
    for n in KNOWN:
        if re.search(r"\b" + re.escape(n) + r"s?\b", q): names.add(n)
    for raw, canon in CANONICAL.items():
        if re.search(r"\b" + re.escape(raw) + r"s?\b", q): names.add(canon)
    return names


def direction_center(text) -> float | None:
    """'слева' / 'right' / 90 / '-45' -> degrees relative to where the camera points (clockwise positive); None if unknown."""
    if isinstance(text, (int, float)): return float(text)
    s = str(text).strip().lower()
    try: return float(s)
    except ValueError: pass
    m = re.fullmatch(r"clock[: ]*(\d{1,2})", s)
    if m: h = int(m.group(1)) % 12; return float(h * 30 if h <= 6 else h * 30 - 360)
    for w, c in DIRECTIONS:
        if w in s: return c
    return None


def name_ru(n: str) -> str: return EN_RU.get(n, n)


def clock_ru(clock: int, behind: bool) -> str:
    where = {12: "прямо по курсу", 1: "справа-спереди", 2: "справа-спереди", 3: "справа", 4: "справа-сзади", 5: "справа-сзади", 6: "сзади",
             7: "слева-сзади", 8: "слева-сзади", 9: "слева", 10: "слева-спереди", 11: "слева-спереди"}[clock]
    return f"{where} ({clock} ч)"


def range_ru(label: str) -> str:
    if label == "?": return "расстояние неизвестно"
    if label.startswith("~"): return "около " + label[1:].replace(" m", " м")
    if label.endswith("+ m"): return "дальше " + label[:-3] + " м"
    return label.replace("-", "–").replace(" m", " м")


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n)); return one if (n % 10 == 1 and n % 100 != 11) else (few if (2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14) else many)


def ago_ru(seconds: float) -> str:
    """'7 секунд назад', '1 минуту назад' ..."""
    s = int(round(seconds))
    if s < 60: return f"{s} {plural(s, 'секунду', 'секунды', 'секунд')} назад"
    m = s // 60
    return f"{m} {plural(m, 'минуту', 'минуты', 'минут')} назад"


def describe(v: dict) -> str:
    tail = {"provisional": ", предварительно", "confirmed": "", "remembered": ", сейчас вне кадра"}[v["status"]]
    if v["status"] == "remembered" and (v.get("age_s") or 0) >= 1: tail += f", видел {ago_ru(v['age_s'])}"
    return f"{name_ru(v['name'])}: {clock_ru(v['clock'], v['behind'])}, {range_ru(v['range'])}{tail}"


# --------------------------------------------------------------------------------------------------- tools
class LiveTools:
    NOTE = "range is an ESTIMATE from the typical size of the object class ('?' = this class has no typical size); direction is relative to where the camera points now; status provisional = just seen, remembered = out of view now"

    def __init__(self, session): self.s = session

    def _all(self) -> list[dict]: return self.s.view()

    @staticmethod
    def _slim(v: dict) -> dict:
        return {"name": v["name"], "status": v["status"], "clock": v["clock"], "behind": v["behind"], "rel_az_deg": v["rel_az_deg"], "range": v["range"],
                "range_m": v["range_m"], "conf": v["conf"], "age_s": v.get("age_s"), "moving": v.get("moving", False)}

    def scene_summary(self) -> dict:
        vs = self._all(); by: dict[str, int] = {}; st: dict[str, int] = {}
        for v in vs: by[v["name"]] = by.get(v["name"], 0) + 1; st[v["status"]] = st.get(v["status"], 0) + 1
        return {"objects": len(vs), "by_name": dict(sorted(by.items(), key=lambda kv: -kv[1])[:10]), "by_status": st,
                "session_seconds": round(self.s.t, 1), "frames": self.s.frames, "note": self.NOTE}

    def list_objects(self, status: str | None = None, name: str | None = None, limit: int = 15, category: str | None = None) -> dict:
        vs = self._all()
        if status: vs = [v for v in vs if v["status"] == status]
        if category:
            ns = set(_cat_names(["danger"] if category == "danger" else [category])); vs = [v for v in vs if v["name"] in ns]
        if name:
            ns = resolve_names(name) or {name.lower()}; vs = [v for v in vs if v["name"] in ns]
        return {"count": len(vs), "objects": [self._slim(v) for v in vs[: max(1, min(int(limit), 40))]], "note": self.NOTE}

    def where_is(self, name: str) -> dict:
        ns = resolve_names(name) or {str(name).lower()}; vs = [v for v in self._all() if v["name"] in ns]
        vs.sort(key=lambda v: (v["status"] == "remembered", v["range_m"] is None, v["range_m"] or 0.0))
        known = sorted({v["name"] for v in self._all()})
        return {"query": name, "resolved_names": sorted(ns), "found": bool(vs), "count": len(vs), "objects": [self._slim(v) for v in vs[:5]],
                "known_names_now": known, "note": self.NOTE}

    def objects_in_direction(self, direction, width_deg: float = 40.0) -> dict:
        c = direction_center(direction)
        if c is None: return {"error": f"unknown direction {direction!r}: use left, right, ahead, behind, 'clock 3' or degrees"}
        half = max(5.0, min(float(width_deg), 180.0)) / 2.0
        vs = [v for v in self._all() if abs(angdiff(v["rel_az_deg"], c)) <= half]
        vs.sort(key=lambda v: (v["range_m"] is None, v["range_m"] or 0.0))
        return {"direction_deg": c, "width_deg": half * 2, "count": len(vs), "objects": [self._slim(v) for v in vs[:10]], "note": self.NOTE}

    def nearest_objects(self, limit: int = 5, max_range_m: float | None = None) -> dict:
        vs = self._all(); known = [v for v in vs if v["range_m"] is not None and v["status"] != "remembered"]
        if max_range_m is not None: known = [v for v in known if v["range_m"] <= float(max_range_m)]
        known.sort(key=lambda v: v["range_m"])
        return {"count": len(known), "objects": [self._slim(v) for v in known[: max(1, min(int(limit), 20))]],
                "objects_without_range": sum(1 for v in vs if v["range_m"] is None and v["status"] != "remembered"), "note": self.NOTE}

    def ground_info(self) -> dict:
        s = self.s; g = getattr(s, "ground_state", None)
        if getattr(s, "ground_classifier", None) is None:
            return {"available": False, "note": "surface recognition is switched off on this server (start it with --ground clip)"}
        if g is None: return {"available": True, "zones": {}, "bad_zones": [], "note": "no surface reading yet"}
        zones = {z: {"surface": v["surface"], "quality": v["quality"], "conf": v["conf"], "bad": v["bad"]} for z, v in g.items()}
        return {"available": True, "zones": zones, "bad_zones": [z for z, v in g.items() if v["bad"]],
                "note": "surface type is a guess from the picture: good / fair / bad quality; zones are left, center (straight ahead) and right of the ground in front of the camera"}

    WAY_CAVEAT = "Камера может не заметить стекло, ступеньку вниз или яму."

    def free_space(self) -> dict:
        s = self.s
        if getattr(s, "seg", None) is None: return {"available": False, "note": "free-space recognition is switched off on this server (start it with --seg segformer)"}
        if s.free_space is None: return {"available": True, "known": False, "note": "no reliable picture of the floor yet"}
        d = s.free_space.to_dict(); d.update(available=True, known=True, advice=s.free_advice); return d

    def where_to_go(self, goal: str | None = None) -> dict:
        """Which way to go. Without a goal: the direction with the most free floor. With a goal (door, bench ...): is the path to it free? Always approximate."""
        fi = self.free_space()
        if not fi.get("available"): return fi
        if not fi.get("known"): return {"available": True, "level": "unsure", "text": "Я пока не вижу пол надёжно. Остановитесь и не полагайтесь на подсказки.", "caveat": self.WAY_CAVEAT}
        fs = self.s.free_space
        if not goal: return {"available": True, **advise(fs), "caveat": self.WAY_CAVEAT}
        r = self.where_is(goal)
        if not r["objects"]: return {"available": True, "found": False, "level": "ok", "text": f"{goal} сейчас в списке нет: не знаю, куда идти.", "known_names_now": r["known_names_now"]}
        o = r["objects"][0]; a = advise_goal(fs, o["rel_az_deg"], name_ru(o["name"]), o["range_m"])
        return {"available": True, "found": True, "goal": o["name"], **a, "caveat": self.WAY_CAVEAT}

    def hazards(self) -> dict:
        vs = [v for v in self._all() if v["status"] != "remembered"]
        danger = [v for v in vs if SEVERITY.get(v["name"]) == "danger"]; caution = [v for v in vs if SEVERITY.get(v["name"]) == "caution"]
        close = [v for v in caution if v["range_m"] is not None and v["range_m"] <= 8.0 or (v["range_m"] is None and CATEGORY.get(v["name"]) in ("hazard", "obstacle"))]
        gi = self.ground_info()
        return {"danger": [self._slim(v) for v in danger[:6]], "caution": [self._slim(v) for v in close[:8]], "bad_ground": gi.get("bad_zones", []),
                "note": self.NOTE + "; weapon and hazard detections are often wrong in both directions: never tell the user it is safe"}

    def call(self, name: str, args: dict) -> dict:
        if name not in TOOL_NAMES: return {"error": f"unknown tool {name!r}"}
        return getattr(self, name)(**(args or {}))


TOOL_SCHEMAS = [
    {"name": "scene_summary", "description": "Overview of what is known now: how many objects, counts by name and by status.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "list_objects", "description": "List objects (nearest first). Optional filters by status (provisional, confirmed, remembered), by name, or by category (people, vehicle, animal, structure, indoor, nature, furniture, obstacle, hazard, weapon, trash, surface, danger).",
     "input_schema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["provisional", "confirmed", "remembered"]}, "name": {"type": "string"},
                                                       "category": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "where_to_go", "description": "Which way to go, from the free floor in front of the camera (sectors with how far the floor is free and what stops it: wall, person or vehicle, door, stairs, glass, road). Without a goal: the direction with the most free floor. With a goal (an object such as a door): whether the path to it is free. Approximate; always pass on its caveat.",
     "input_schema": {"type": "object", "properties": {"goal": {"type": "string"}}}},
    {"name": "free_space", "description": "Raw free-space picture: five sectors left to right with the free distance and the blocker, what is underfoot (safe floor, road, rough ground), and whether the picture is reliable. Use for 'where is the wall', 'how wide is the passage'.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "ground_info", "description": "The surface under the feet in three zones (left, center, right): asphalt, gravel, grass, ice, pothole ... and its quality (good, fair, bad). Use for questions about the road, the surface, a bad road.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "hazards", "description": "Dangerous or cautionary things right now: dangerous objects (knife, weapon, fire, open manhole, predators ...), close vehicles and obstacles, bad surface ahead. Use for 'is it safe', 'what is dangerous'.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "where_is", "description": "Where is an object of this kind (English name such as person, car, pole, door, wall, tree; Russian words also work)? Returns direction as a clock hour (12 = ahead, 3 = right, 9 = left, 6 = behind) and a distance estimate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "objects_in_direction", "description": "What is in a direction relative to where the camera points: left, right, ahead, behind, 'clock 3', or degrees (clockwise positive).",
     "input_schema": {"type": "object", "properties": {"direction": {"type": "string"}, "width_deg": {"type": "number"}}, "required": ["direction"]}},
    {"name": "nearest_objects", "description": "The nearest objects that have a distance estimate (not remembered ones).",
     "input_schema": {"type": "object", "properties": {"limit": {"type": "integer"}, "max_range_m": {"type": "number"}}}},
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}

SYSTEM_PROMPT = """You are the assistant of a live camera app. A phone camera streams video; the app keeps a list of objects with a name, a direction and a distance estimate. You answer questions about the surroundings using ONLY the tools; never invent objects, directions or distances.
Rules:
- The user may be blind or have low vision and may be walking: be calm, concrete and brief; give directions as left / right / ahead / behind with a rough distance; put danger first.
- Answer in the user's language (usually Russian). Keep answers to one to three short sentences: they may be read aloud.
- For an object whose status is 'remembered' say that it is out of view now and how long ago it was last seen (age_s).
- Directions are relative to where the camera points now: 12 o'clock = straight ahead, 3 = right, 6 = behind, 9 = left. Say it naturally ("справа", "слева-спереди").
- Distances are ESTIMATES from typical object size: say "около 5 м" or a range, never an exact number; if the range is '?' say the distance is unknown. Never claim precision.
- status 'provisional' means just seen, not yet confirmed; 'remembered' means it is out of view right now. Mention this when it matters.
- If a tool finds nothing, say so and mention what is known (known_names_now). Do not guess. If the list is empty, say the camera has not found anything yet.
- Weapons, knives and other hazards are detected unreliably, in BOTH directions. Say 'возможно' for them. Never say that something is safe or that nothing dangerous is there; say only that the camera did not notice anything and that it must not be relied on.
- Road and surface questions (road, asphalt, bad road, puddle, ice, pothole): use ground_info; surface type is a guess.
- 'Which way to go', 'is the way free', 'where is the wall / the passage', 'take me to the door': use where_to_go or free_space and repeat only what they say, with their caveat. Distances there are rough and already reduced for safety. Never say the way is safe; if the level is 'unsure', say you cannot see the way reliably and the user must be careful.
- Your answers may be read aloud: plain sentences only, no lists, no markdown, no emoji, no parentheses."""


# --------------------------------------------------------------------------------------------------- agents
GROUND_RE = re.compile(r"дорог|покрыт|асфальт|под ногами|тротуар|плитк|гравий|щебен|щебн|грунт|песок|песк|\bтрав|скольз|грязь|грязи|\bлуж|\bлёд|\bлед|снег|\bям[аыуо]|выбоин|ухаб|рельс|переход|зебр")
WAY_RE = re.compile(r"куда\s+(мне\s+)?(идти|пойти|повернуть|двигаться)|можно ли идти|свободен ли путь|свободно ли|где проход|есть ли проход|веди|как (мне )?(пройти|дойти|добраться|идти)")
WALL_RE = re.compile(r"\bстен")
GOAL_RE = re.compile(r"\bк\s|\bко\s|\bдо\s|веди|дойти|пройти|добраться")
HAZARD_RE = re.compile(r"опасн|угроз|осторожн|безопасн|что не так")
DISCLAIMER = "Это только подсказка камеры, на неё нельзя полагаться."
CAT_GEN = {"animal": "животных", "trash": "мусора", "weapon": "оружия и острых предметов", "vehicle": "транспорта", "obstacle": "препятствий", "danger": "опасных предметов"}


def categories_in(text: str) -> list[str]:
    cats: list[str] = []
    for tok in re.findall(r"[a-zа-яё]+", (text or "").lower()):
        for stem, cs in RU_CATS.items():
            if tok.startswith(stem): cats += cs
        cats += EN_CATS.get(tok, [])
    return list(dict.fromkeys(cats))


class RuleAgent:
    kind = "offline"

    def __init__(self, session): self.tools = LiveTools(session)

    # ---- phrases
    def _ground_phrase(self, gi: dict) -> str:
        if not gi.get("available"): return "Распознавание покрытия на сервере не включено: запустите сервер с ключом --ground clip."
        z = gi.get("zones") or {}
        if not z: return "Покрытие пока не распознано."
        from ..ground import ZONE_RU
        from ..catalog import SURFACE_RU
        parts = [f"{ZONE_RU[k]} {SURFACE_RU.get(z[k]['surface'], z[k]['surface'])}" for k in ("left", "center", "right") if k in z]
        out = "Под ногами: " + ", ".join(parts) + "."
        bad = [k for k in ("center", "left", "right") if k in z and z[k]["bad"]]
        if bad: out += " Осторожно, плохое покрытие " + ", ".join(ZONE_RU[k] for k in bad) + "."
        return out

    def _hazard_phrase(self) -> str:
        h = self.tools.hazards(); parts = []
        if h["danger"]: parts.append("Осторожно, возможно: " + "; ".join(describe(o) for o in h["danger"][:3]) + ".")
        if h["caution"]: parts.append("Внимание: " + "; ".join(describe(o) for o in h["caution"][:3]) + ".")
        if h["bad_ground"]:
            from ..ground import ZONE_RU
            parts.append("Осторожно, плохое покрытие " + ", ".join(ZONE_RU[z] for z in h["bad_ground"]) + ".")
        return " ".join(parts) if parts else "Камера не заметила явных опасностей. " + DISCLAIMER

    def _walls_phrase(self, fi: dict) -> str:
        groups = {"впереди": [], "слева": [], "справа": []}                                       # what is ahead first
        for s in fi["sectors"]:
            if s["blocker"] == "wall" and s["free_safe_m"] < 8.0: groups["слева" if (s["lo"] + s["hi"]) / 2 < -12 else "справа" if (s["lo"] + s["hi"]) / 2 > 12 else "впереди"].append(s["free_safe_m"])
        parts = [f"{k} {dist_words(min(v))}" for k, v in groups.items() if v]
        return ("Стена: " + ", ".join(parts) + ". " + LiveTools.WAY_CAVEAT) if parts else "Стен рядом не вижу. Это только подсказка камеры."

    # ---- the answer
    def reply(self, text: str) -> tuple[str, list[str]]:
        t = (text or "").lower(); names = resolve_names(t); used: list[str] = []; cats = categories_in(t)
        dirs = [(w, c) for w, c in DIRECTIONS if re.search(r"\b" + w, t)]
        if HAZARD_RE.search(t) and not (set(names) - set(names_with_severity("danger"))):
            used += ["hazards"]; return self._hazard_phrase(), used
        if WAY_RE.search(t):
            goal_names = sorted(n for n in names if CATEGORY.get(n) != "surface"); goal = goal_names[0] if (goal_names and GOAL_RE.search(t)) else None
            used.append("where_to_go"); r = self.tools.where_to_go(goal)
            if not r.get("available"): return "Распознавание свободного пространства на сервере не включено: запустите сервер с ключом --seg segformer.", used
            return r["text"] + ("" if r.get("level") == "unsure" else " " + r.get("caveat", "")), used
        if WALL_RE.search(t) and "wall" in names:
            fi = self.tools.free_space()
            if fi.get("available") and fi.get("known") and fi.get("reliable") and not fi.get("uncertain"): used.append("free_space"); return self._walls_phrase(fi), used
        if GROUND_RE.search(t):
            used.append("ground_info"); ans = self._ground_phrase(self.tools.ground_info()); obj = [n for n in names if CATEGORY.get(n) not in ("surface",)]
            if obj and re.search(r"\bям|\bлуж|\bлёд|\bлед|выбоин|ухаб", t):
                used.append("where_is"); r = self.tools.where_is(" ".join(obj))
                if r["objects"]: ans += " На изображении: " + "; ".join(describe(self._full(o)) for o in r["objects"][:2]) + "."
            return ans, used
        if re.search(r"сколько", t) and names:
            used.append("list_objects"); n = sum(len(self.tools.list_objects(name=x)["objects"]) for x in names)
            return (f"Сейчас в списке: {n}." if n else "Таких предметов в списке нет."), used
        if names:
            used.append("where_is"); r = self.tools.where_is(" ".join(sorted(names))); objs = r["objects"]
            if dirs:
                used.append("objects_in_direction"); c = dirs[0][1]; objs = [o for o in objs if abs(angdiff(o["rel_az_deg"], c)) <= 60]
            risky = bool(set(names) & set(names_with_severity("danger"))) or "danger" in cats or "weapon" in cats
            if not objs:
                known = ", ".join(name_ru(n) for n in r["known_names_now"][:8]) or "пока ничего"
                what = ", ".join(CAT_GEN[c] for c in cats if c in CAT_GEN) or ", ".join(name_ru(n) for n in sorted(names)[:4])
                msg = f"Не вижу: {what}. Сейчас в списке: {known}."
                return (msg + " Но камера могла не заметить. " + DISCLAIMER if risky else msg), used
            ans = "; ".join(describe(self._full(o)) for o in objs[:4]) + "."
            return (("Возможно: " + ans + " " + DISCLAIMER) if risky else ans), used
        if dirs:
            used.append("objects_in_direction"); c = dirs[0][1]; r = self.tools.objects_in_direction(c, 90.0 if c == 0.0 else 120.0)
            if not r["objects"]: return "В этом направлении ничего не найдено.", used
            return "; ".join(describe(self._full(o)) for o in r["objects"][:4]) + ".", used
        if re.search(r"ближайш|близк|ближе", t):
            used.append("nearest_objects"); r = self.tools.nearest_objects(4)
            if not r["objects"]: return "Предметов с известным расстоянием пока нет.", used
            return "Ближайшие: " + "; ".join(describe(self._full(o)) for o in r["objects"]) + ".", used
        if re.search(r"что|вокруг|видишь|видно|опиши|осмотр|есть|what|see|around", t) or not t.strip():
            used.append("scene_summary"); s = self.tools.scene_summary()
            if not s["objects"]: return "Камера пока ничего не нашла.", used
            used.append("list_objects"); objs = self.tools.list_objects(limit=6)["objects"]
            return "Вижу: " + "; ".join(describe(self._full(o)) for o in objs) + f". Всего в списке {s['objects']}.", used
        return "Спросите, например: «что вокруг?», «где машина?», «что слева?», «что ближайшее?», «какая дорога?», «что опасного?».", used

    @staticmethod
    def _full(o: dict) -> dict:
        return {**o, "range": o["range"]}


class ChatAgent:
    kind = "claude"

    def __init__(self, session, client, model: str, max_steps: int = 8):
        self.tools = LiveTools(session); self.client, self.model, self.max_steps = client, model, max_steps; self.history: list[dict] = []

    def reply(self, text: str) -> tuple[str, list[str]]:
        msgs = self.history[-12:] + [{"role": "user", "content": text}]; used: list[str] = []
        for _ in range(self.max_steps):
            resp = self.client.messages.create(model=self.model, max_tokens=500, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, messages=msgs)
            if getattr(resp, "stop_reason", None) != "tool_use":
                answer = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip() or "Не удалось ответить."
                self.history = (msgs + [{"role": "assistant", "content": answer}])[-12:]; return answer, used
            msgs.append({"role": "assistant", "content": resp.content}); results = []
            for b in resp.content:
                if getattr(b, "type", "") != "tool_use": continue
                used.append(b.name)
                try: out = self.tools.call(b.name, dict(b.input or {})); err = "error" in out
                except Exception as ex: out = {"error": f"{type(ex).__name__}: {ex}"}; err = True
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, ensure_ascii=False), **({"is_error": True} if err else {})})
            msgs.append({"role": "user", "content": results})
        return "Слишком много шагов, попробуйте спросить проще.", used


def make_agent(session, cfg, client=None):
    """claude (when asked for, or auto with a key) or the offline agent. Never raises: a missing package or key means offline."""
    want = cfg.agent
    if want == "offline": return RuleAgent(session)
    try:
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                if want == "claude": raise RuntimeError("ANTHROPIC_API_KEY is not set on the server")
                return RuleAgent(session)
            import anthropic
            client = anthropic.Anthropic()
        return ChatAgent(session, client, cfg.model)
    except Exception:
        if want == "claude": raise
        return RuleAgent(session)
