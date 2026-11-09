import unittest
import helpers, server_helpers as H
from scenelive.catalog import NAMES, RU_NAME
from scenelive.detect.base import Detection
from scenelive.ground import ColorSurfaceClassifier
from scenelive.server.agent import DISCLAIMER, SYSTEM_PROMPT, TOOL_SCHEMAS, LiveTools, RuleAgent, name_ru, resolve_names
from scenelive.server.config import ServerConfig
from scenelive.server.session import LiveSession

BOXES = {"knife": (300.0, 100.0, 340.0, 250.0), "dog": (60.0, 150.0, 160.0, 250.0), "snake": (480.0, 220.0, 600.0, 260.0), "car": (200.0, 120.0, 320.0, 230.0),
         "curtain": (10.0, 10.0, 120.0, 300.0), "trash": (400.0, 200.0, 450.0, 240.0)}


class Scripted:
    name = "scripted"
    def __init__(self, names): self.names = names
    def detect(self, frame): return [Detection(BOXES[n], n, 0.8, n) for n in self.names]


class ScriptedGround:
    name = "scripted-ground"
    def __init__(self, center): self.center = center
    def classify(self, crops): return [[("asphalt", 0.9)], [(self.center, 0.9)], [("asphalt", 0.9)]]


def session(names, ground=None, frames=7):
    s = LiveSession(Scripted(names), {"camera": "main"}, ServerConfig(detector="blobs", alerts=False), ground_classifier=ground)
    for k in range(frames): s.process(H.jpeg_at(k / 3)[0], {"seq": k, "t": k / 3})
    return s


class TestRussianWords(unittest.TestCase):
    def test_new_words_resolve(self):
        cases = {"есть ли нож": "knife", "там пистолет": "gun", "штора": "curtain", "большая яма": "pothole", "лёд на дороге": "ice", "лужа": "puddle", "змея": "snake",
                 "медведь": "bear", "бордюр": "curb", "урна": "trash can", "оголённый провод": "exposed wire", "топор": "axe", "пожар": "fire", "трамвай": "tram", "грязь": "mud"}
        for text, want in cases.items(): self.assertIn(want, resolve_names(text), text)

    def test_ambiguous_short_words_are_kept_apart(self):
        self.assertIn("pole", resolve_names("столб")); self.assertNotIn("table", resolve_names("столб")); self.assertIn("table", resolve_names("стол")); self.assertNotIn("pole", resolve_names("стол"))
        self.assertEqual(resolve_names("который час"), set()); self.assertEqual(resolve_names("банк"), set()); self.assertIn("can", resolve_names("банка"))

    def test_category_words_expand_to_every_member(self):
        self.assertTrue({"dog", "snake", "horse", "cow"} <= resolve_names("есть ли животные")); self.assertTrue({"trash", "garbage bag", "bottle"} <= resolve_names("где мусор"))
        self.assertTrue({"knife", "gun", "axe"} <= resolve_names("оружие")); self.assertTrue({"knife", "snake", "fire", "open manhole"} <= resolve_names("опасные предметы"))

    def test_every_catalogue_name_has_a_russian_name(self):
        for n in NAMES: self.assertNotEqual(name_ru(n), n, n); self.assertEqual(name_ru(n), RU_NAME[n])


class TestExtendedAgent(unittest.TestCase):
    def test_a_weapon_is_reported_cautiously(self):
        txt, used = RuleAgent(session(["knife"])).reply("есть ли оружие?"); self.assertTrue(txt.startswith("Возможно:")); self.assertIn("нож", txt); self.assertIn(DISCLAIMER, txt); self.assertIn("where_is", used)

    def test_not_seeing_a_weapon_is_not_a_guarantee(self):
        txt, _ = RuleAgent(session(["dog"])).reply("есть ли нож или оружие?"); self.assertIn("Не вижу", txt); self.assertIn("могла не заметить", txt); self.assertIn(DISCLAIMER, txt)

    def test_animals_by_category(self):
        a = RuleAgent(session(["dog"])); self.assertIn("собака", a.reply("какие животные рядом")[0]); self.assertIn("Не вижу: животных", RuleAgent(session(["car"])).reply("есть животные?")[0])

    def test_trash_and_curtains(self):
        self.assertIn("мусор", RuleAgent(session(["trash"])).reply("где мусор")[0]); self.assertIn("штора", RuleAgent(session(["curtain"])).reply("видишь штору?")[0])

    def test_what_is_dangerous(self):
        txt, used = RuleAgent(session(["knife", "car"])).reply("что опасного?"); self.assertIn("Осторожно, возможно: нож", txt); self.assertEqual(used, ["hazards"])
        quiet, _ = RuleAgent(session(["curtain"])).reply("безопасно ли тут?"); self.assertIn("не заметила явных опасностей", quiet); self.assertIn(DISCLAIMER, quiet)

    def test_a_bad_surface_counts_as_a_hazard(self):
        txt, _ = RuleAgent(session(["curtain"], ground=ScriptedGround("pothole"))).reply("что опасного"); self.assertIn("плохое покрытие впереди", txt)

    def test_road_and_surface_questions(self):
        s = session(["curtain"], ground=ColorSurfaceClassifier()); txt, used = RuleAgent(s).reply("какая дорога?")
        self.assertTrue(txt.startswith("Под ногами:")); self.assertIn("асфальт", txt); self.assertEqual(used, ["ground_info"])
        bad, _ = RuleAgent(session(["curtain"], ground=ScriptedGround("pothole"))).reply("что с покрытием, есть ямы?"); self.assertIn("яма на дороге", bad); self.assertIn("Осторожно, плохое покрытие впереди", bad)
        off, _ = RuleAgent(session(["curtain"])).reply("какое покрытие под ногами"); self.assertIn("--ground clip", off)

    def test_tools(self):
        s = session(["dog", "knife"]); t = LiveTools(s)
        self.assertEqual({o["name"] for o in t.list_objects(category="animal")["objects"]}, {"dog"}); self.assertEqual({o["name"] for o in t.list_objects(category="danger")["objects"]}, {"knife"})
        h = t.hazards(); self.assertEqual([o["name"] for o in h["danger"]], ["knife"]); self.assertIn("never tell the user it is safe", h["note"])
        self.assertFalse(t.ground_info()["available"]); g = LiveTools(session(["dog"], ground=ScriptedGround("ice"))).ground_info()
        self.assertTrue(g["available"]); self.assertEqual(g["bad_zones"], ["center"]); self.assertEqual(g["zones"]["center"]["quality"], "bad")

    def test_the_model_gets_the_new_tools_and_the_safety_rules(self):
        names = {t["name"] for t in TOOL_SCHEMAS}; self.assertTrue({"ground_info", "hazards", "list_objects"} <= names)
        for phrase in ("Never say that something is safe", "unreliably", "ground_info", "no markdown", "возможно"): self.assertIn(phrase, SYSTEM_PROMPT)
        self.assertIn("category", next(t for t in TOOL_SCHEMAS if t["name"] == "list_objects")["input_schema"]["properties"])


if __name__ == "__main__": unittest.main()
