import contextlib, io, os, tempfile, unittest
import helpers
from scenelive.cli import main
from scenelive.labels import CORE_VOCAB, EXTENDED_VOCAB, load_vocab
from scenelive.synth import render_video


def cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): code = main(argv)
    return code, buf.getvalue()


class TestVocab(unittest.TestCase):
    def test_core_is_inside_extended_and_extended_has_what_was_asked_for(self):
        self.assertTrue(set(CORE_VOCAB) <= set(EXTENDED_VOCAB)); self.assertEqual(EXTENDED_VOCAB[: len(CORE_VOCAB)], CORE_VOCAB)
        for w in ("road", "asphalt", "pothole", "cracked road", "trash", "garbage bag", "knife", "gun", "axe", "snake", "dog", "horse", "curtain", "tree", "pole", "wall", "puddle", "ice", "open manhole", "broken glass"):
            self.assertIn(w, EXTENDED_VOCAB, w)
        self.assertEqual(len(EXTENDED_VOCAB), len(set(EXTENDED_VOCAB)))

    def test_load_vocab(self):
        self.assertEqual(load_vocab("core"), CORE_VOCAB); self.assertEqual(load_vocab("extended"), EXTENDED_VOCAB)
        p = os.path.join(tempfile.mkdtemp(), "v.txt"); open(p, "w", encoding="utf-8").write("# animals\ndog\n\ncat  # pets\ndog\nknife\n")
        self.assertEqual(load_vocab(p), ["dog", "cat", "knife"])                                   # comments, blanks and duplicates are dropped
        with self.assertRaises(FileNotFoundError): load_vocab("nope.txt")
        empty = os.path.join(tempfile.mkdtemp(), "e.txt"); open(empty, "w").write("# nothing\n")
        with self.assertRaises(ValueError): load_vocab(empty)

    def test_the_vocab_command_lists_and_writes_an_editable_file(self):
        out = os.path.join(tempfile.mkdtemp(), "my_vocab.txt"); code, txt = cli(["vocab", "--out", out])
        self.assertEqual(code, 0)
        for s in ("животные (animal)", "оружие и острые предметы (weapon)", "knife", "нож", "danger", "покрытие (surface)", "words in the extended vocabulary"): self.assertIn(s, txt)
        words = load_vocab(out); self.assertEqual(set(words), set(EXTENDED_VOCAB)); self.assertEqual(len(words), len(EXTENDED_VOCAB)); self.assertIn("# животные (animal)", open(out, encoding="utf-8").read())
        code, one = cli(["vocab", "--category", "weapon"]); self.assertEqual(code, 0); self.assertIn("knife", one); self.assertNotIn("dog", one)
        code, bad = cli(["vocab", "--category", "nope"]); self.assertEqual(code, 2); self.assertIn("Choose from", bad)

    def test_replay_and_serve_explain_a_missing_vocabulary_file(self):
        d = tempfile.mkdtemp(); v = os.path.join(d, "v.mp4"); render_video(v, duration=2.0)
        code, txt = cli(["replay", v, "--detector", "world", "--vocab", os.path.join(d, "nope.txt")])
        self.assertEqual(code, 2); self.assertIn("Cannot run", txt); self.assertIn("not found", txt)


if __name__ == "__main__": unittest.main()
