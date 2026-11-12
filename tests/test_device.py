import contextlib, io, sys, types, unittest
import helpers
import scenelive.device as D


class FakeTorch:
    """A stand-in for torch: with or without a card, CPU-only or CUDA build."""
    def __init__(self, cuda=False, build="12.8", mps=False, name="NVIDIA GeForce RTX 3050 Laptop GPU", vram=4 * 1024 ** 3):
        cudnn = types.SimpleNamespace(benchmark=False)
        self.mod = types.ModuleType("torch"); m = self.mod
        m.__version__ = "2.11.0+cu128" if build else "2.14.1+cpu"; m.version = types.SimpleNamespace(cuda=build)
        m.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: name, get_device_properties=lambda i: types.SimpleNamespace(total_memory=vram))
        m.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps), cudnn=cudnn)

    def __enter__(self):
        self.old = sys.modules.get("torch"); sys.modules["torch"] = self.mod; return self.mod

    def __exit__(self, *a):
        if self.old is None: sys.modules.pop("torch", None)
        else: sys.modules["torch"] = self.old


@contextlib.contextmanager
def smi(present: bool):
    D.nvidia_present.cache_clear(); old = D.nvidia_present; D.nvidia_present = lambda: present
    try: yield
    finally: D.nvidia_present = old; D.nvidia_present.cache_clear()


class TestPick(unittest.TestCase):
    def test_auto_takes_the_card_when_there_is_one(self):
        with FakeTorch(cuda=True): self.assertEqual(D.pick_device("auto"), "cuda"); self.assertEqual(D.pick_device(None), "cuda")

    def test_auto_falls_back_to_the_processor(self):
        with FakeTorch(cuda=False): self.assertEqual(D.pick_device("auto"), "cpu")
        with FakeTorch(cuda=False, mps=True): self.assertEqual(D.pick_device("auto"), "mps")

    def test_asking_for_cuda_without_a_card_warns_and_uses_the_processor(self):
        msgs = []
        with FakeTorch(cuda=False, build=None), smi(True): self.assertEqual(D.pick_device("cuda", warn=msgs.append), "cpu")
        self.assertTrue(msgs); self.assertIn("setup_gpu.bat", msgs[0])

    def test_explicit_choices(self):
        with FakeTorch(cuda=True):
            self.assertEqual(D.pick_device("cpu"), "cpu"); self.assertEqual(D.pick_device("cuda:1"), "cuda:1"); self.assertEqual(D.pick_device("CUDA"), "cuda")
            msgs = []; self.assertEqual(D.pick_device("tpu", warn=msgs.append), "cuda"); self.assertIn("unknown", msgs[0])

    def test_without_pytorch_it_is_the_processor(self):
        old = sys.modules.get("torch"); sys.modules["torch"] = None                       # import fails
        try: self.assertEqual(D.pick_device("auto"), "cpu"); self.assertIn("not installed", D.describe("cpu"))
        finally:
            if old is None: sys.modules.pop("torch", None)
            else: sys.modules["torch"] = old

    def test_names_for_ultralytics_and_the_cudnn_switch(self):
        self.assertEqual((D.ultralytics_device("cuda"), D.ultralytics_device("cuda:1"), D.ultralytics_device("cpu"), D.ultralytics_device("mps")), ("0", "1", "cpu", "mps"))
        with FakeTorch(cuda=True) as t:
            D.configure("cpu"); self.assertFalse(t.backends.cudnn.benchmark); D.configure("cuda"); self.assertTrue(t.backends.cudnn.benchmark)


class TestDescribe(unittest.TestCase):
    def test_the_card_is_named(self):
        with FakeTorch(cuda=True), smi(True):
            s = D.describe("cuda"); self.assertIn("RTX 3050 Laptop", s); self.assertIn("4.0 GB", s); self.assertIn("CUDA 12.8", s)

    def test_a_cpu_only_build_next_to_a_card_is_pointed_out(self):
        with FakeTorch(cuda=False, build=None), smi(True):
            s = D.describe("cpu"); self.assertIn("2.14.1+cpu", s); self.assertIn("CPU-only build", s); self.assertIn("setup_gpu.bat", s)
        with FakeTorch(cuda=False, build=None), smi(False): self.assertNotIn("setup_gpu", D.describe("cpu"))                       # no card: nothing to fix

    def test_the_device_command(self):
        buf = io.StringIO()
        with FakeTorch(cuda=False, build=None), smi(True), contextlib.redirect_stdout(buf): D.main()
        out = buf.getvalue(); self.assertIn("CUDA build:       no (CPU-only build)", out); self.assertIn("CUDA available:   False", out); self.assertIn("setup_gpu.bat", out)
        buf = io.StringIO()
        with FakeTorch(cuda=True), smi(True), contextlib.redirect_stdout(buf): D.main()
        self.assertIn("scenelive uses:   GPU NVIDIA GeForce RTX 3050 Laptop GPU", buf.getvalue())


class TestModelsFollowTheChoice(unittest.TestCase):
    def test_easyocr_gets_the_card_only_when_there_is_one(self):
        import scenelive.ocr as O
        got = {}; old = O.EasyOcrEngine
        class Stub:
            def __init__(self, langs, gpu=False): got["gpu"] = gpu; got["langs"] = langs
        O.EasyOcrEngine = Stub
        try:
            with FakeTorch(cuda=True): O.make_ocr("easyocr", "ru,en", "auto"); self.assertEqual(got, {"gpu": True, "langs": ("ru", "en")})
            with FakeTorch(cuda=False): O.make_ocr("easyocr", "ru,en", "auto"); self.assertFalse(got["gpu"])
            with FakeTorch(cuda=True): O.make_ocr("easyocr", "ru", "cpu"); self.assertFalse(got["gpu"])
        finally: O.EasyOcrEngine = old

    def test_the_detectors_no_longer_force_the_processor(self):
        src = open(D.__file__.replace("device.py", "detect/yolo.py"), encoding="utf-8").read()
        self.assertNotIn('device or "cpu"', src); self.assertIn("pick_device(device)", src)

    def test_defaults_are_auto(self):
        from scenelive.server.config import ServerConfig
        self.assertEqual(ServerConfig().device, "auto")
        from scenelive import cli
        src = open(cli.__file__, encoding="utf-8").read(); self.assertNotIn('"--device", default=None', src); self.assertGreaterEqual(src.count('"--device", default="auto"'), 4)


if __name__ == "__main__": unittest.main()
