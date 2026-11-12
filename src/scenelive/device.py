"""One place that decides where the models run: the graphics card when there is one, the processor otherwise.

  auto (default) -> "cuda" if PyTorch sees an NVIDIA card, else "mps" (Apple), else "cpu"
  cuda / cuda:N  -> the card; if it is not available, falls back to "cpu" with a warning (never crashes)
  cpu            -> the processor

A CPU-only build of PyTorch (version ending in +cpu) never sees the card, even when it is there: then a hint says to run
windows\\setup_gpu.bat."""
from __future__ import annotations
import functools
import shutil
import subprocess
import sys


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def nvidia_present() -> bool:
    """Is there an NVIDIA card with a driver (nvidia-smi works)? Independent of PyTorch."""
    exe = shutil.which("nvidia-smi")
    if not exe: return False
    try: return subprocess.run([exe, "-L"], capture_output=True, timeout=5).returncode == 0
    except Exception: return False


def pick_device(requested: str | None = "auto", warn=None) -> str:
    warn = warn or (lambda m: print("WARNING:", m, file=sys.stderr, flush=True))
    req = (requested or "auto").strip().lower(); torch = _torch()
    cuda = bool(torch is not None and torch.cuda.is_available())
    if req == "cpu": return "cpu"
    if req.startswith("cuda"):
        if cuda: return req
        warn(f"device '{requested}' was asked for, but PyTorch does not see a CUDA card: using the processor. " + gpu_hint()); return "cpu"
    if req == "mps":
        if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
        warn("device 'mps' is not available: using the processor"); return "cpu"
    if req != "auto": warn(f"unknown device '{requested}': using auto")
    if cuda: return "cuda"
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def ultralytics_device(device: str) -> str:
    """ultralytics wants '0' / '1' for cards and 'cpu' / 'mps' otherwise."""
    if is_cuda(device): return device.split(":", 1)[1] if ":" in device else "0"
    return device


def configure(device: str) -> None:
    """Small speed-ups that are safe on a card: let cuDNN pick the fastest algorithms for our fixed input sizes."""
    torch = _torch()
    if torch is not None and is_cuda(device):
        try: torch.backends.cudnn.benchmark = True
        except Exception: pass


def gpu_hint() -> str:
    torch = _torch()
    if torch is None: return "PyTorch is not installed."
    if getattr(torch.version, "cuda", None) is None and nvidia_present():
        return f"PyTorch {torch.__version__} is a CPU-only build, but an NVIDIA card is present: run windows\\setup_gpu.bat to use it."
    if not nvidia_present(): return "No NVIDIA card or driver found (nvidia-smi does not work)."
    return "Check the NVIDIA driver (nvidia-smi) and that PyTorch matches its CUDA version."


def info() -> dict:
    torch = _torch()
    if torch is None: return {"torch": None, "cuda_build": None, "cuda_available": False, "gpu": None, "vram_gb": None, "nvidia_smi": nvidia_present()}
    d = {"torch": torch.__version__, "cuda_build": getattr(torch.version, "cuda", None), "cuda_available": bool(torch.cuda.is_available()), "gpu": None, "vram_gb": None,
         "nvidia_smi": nvidia_present()}
    if d["cuda_available"]:
        try: d["gpu"] = torch.cuda.get_device_name(0); d["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        except Exception: pass
    return d


def describe(device: str) -> str:
    i = info()
    if is_cuda(device): return f"GPU {i['gpu'] or '?'}" + (f" ({i['vram_gb']} GB)" if i["vram_gb"] else "") + f", PyTorch {i['torch']} (CUDA {i['cuda_build']})"
    if device == "mps": return f"Apple GPU (mps), PyTorch {i['torch']}"
    if i["torch"] is None: return "processor (PyTorch is not installed)"
    note = "" if i["cuda_available"] else ("; " + gpu_hint() if (i["cuda_build"] is None and i["nvidia_smi"]) else "")
    return f"processor, PyTorch {i['torch']}{note}"


def main() -> int:
    """python -m scenelive.device : what does this computer offer?"""
    i = info(); dev = pick_device("auto", warn=lambda m: None)
    print(f"PyTorch:          {i['torch'] or 'not installed'}")
    print(f"CUDA build:       {i['cuda_build'] or 'no (CPU-only build)'}")
    print(f"nvidia-smi:       {'works' if i['nvidia_smi'] else 'not found'}")
    print(f"CUDA available:   {i['cuda_available']}")
    if i["gpu"]: print(f"GPU:              {i['gpu']} ({i['vram_gb']} GB)")
    print(f"scenelive uses:   {describe(dev)}")
    if not i["cuda_available"]: print("hint:             " + gpu_hint())
    return 0


if __name__ == "__main__": raise SystemExit(main())
