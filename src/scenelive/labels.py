"""Street classes, canonical names, which classes move, and typical sizes (for distance without depth)."""
from __future__ import annotations

# classes an open-vocabulary detector is asked for (boxes only: names + direction need nothing more)
STREET_VOCAB = [
    "person", "car", "bus", "truck", "van", "bicycle", "motorcycle", "scooter", "dog",
    "traffic light", "stop sign", "street sign", "fire hydrant", "bench", "trash can", "bollard", "barrier", "fence", "gate",
    "pole", "lamp post", "tree", "wall", "building", "door", "garage door", "window", "stairs",
]

from .catalog import CATEGORY, MOVES, NAMES, RU_NAME, SEVERITY                               # noqa: E402  (re-exported for the rest of the package)
CORE_VOCAB = list(STREET_VOCAB)                                                            # the first, smaller vocabulary (28 words)
EXTENDED_VOCAB = list(dict.fromkeys(CORE_VOCAB + NAMES))                                   # + everything in the catalogue (core words kept first)

# raw detector name -> canonical name used everywhere else
CANONICAL = {
    "lamp post": "pole", "street lamp": "pole", "utility pole": "pole", "telephone pole": "pole",
    "handbag": "bag", "couch": "sofa", "potted plant": "plant", "dining table": "table", "cell phone": "phone",
    "taxi": "car", "suv": "car", "pedestrian": "person", "garage door": "door", "gate": "fence",
    "stop sign": "stop sign", "street sign": "sign", "traffic sign": "sign", "bike": "bicycle",
    "pistol": "gun", "rifle": "gun", "firearm": "gun", "handgun": "gun", "garbage": "trash", "rubbish": "trash", "trash bag": "garbage bag",
    "hole in the road": "pothole", "manhole": "manhole cover", "curtains": "curtain", "drapes": "curtain", "pavement": "sidewalk",
    "paving stones": "paving tiles", "tiles": "paving tiles", "cracked asphalt": "cracked road", "road surface": "road", "rocks": "rock",
    "branches": "branch", "dog on a leash": "dog", "bin": "trash can", "garbage can": "trash can", "wire": "cable", "tram rails": "tram tracks",
}

# classes of the COCO detector that make sense outdoors (the rest is indoor clutter)
COCO_STREET = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "boat", "traffic light", "fire hydrant",
               "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "bear", "backpack", "umbrella", "handbag",
               "suitcase", "chair", "couch", "bed", "potted plant", "skateboard", "knife", "scissors", "baseball bat", "bottle")

MOVABLE = {"person", "car", "bus", "truck", "van", "bicycle", "motorcycle", "scooter", "dog", "cat", "bird", "skateboard", "boat", "train"} | set(MOVES)

# name -> (which box side, typical size in metres, relative uncertainty). 'h' = box height, 'w' = box width.
RANGE_PRIORS = {
    "person": ("h", 1.70, 0.12), "car": ("h", 1.50, 0.12), "van": ("h", 1.90, 0.15), "bus": ("h", 3.00, 0.15),
    "truck": ("h", 3.00, 0.25), "door": ("h", 2.00, 0.07), "fire hydrant": ("h", 0.75, 0.15), "bench": ("w", 1.60, 0.20),
    "trash can": ("h", 1.00, 0.30), "traffic light": ("h", 0.90, 0.30), "bollard": ("h", 0.90, 0.20),
    "horse": ("h", 1.60, 0.15), "cow": ("h", 1.40, 0.15), "tram": ("h", 3.30, 0.15), "construction cone": ("h", 0.70, 0.20),
}


def load_vocab(spec: str) -> list[str]:
    """'core' | 'extended' | path to a text file (one word or phrase per line, # comments)."""
    if spec == "core": return list(CORE_VOCAB)
    if spec == "extended": return list(EXTENDED_VOCAB)
    import os
    if not os.path.isfile(spec): raise FileNotFoundError(f"vocabulary: use core, extended or a text file; not found: {spec}")
    words = [w.split("#")[0].strip() for w in open(spec, encoding="utf-8-sig").read().splitlines()]
    words = [w for w in words if w]
    if not words: raise ValueError(f"{spec}: the vocabulary file is empty")
    return list(dict.fromkeys(words))


def canonical(name: str) -> str:
    n = name.strip().lower()
    return CANONICAL.get(n, n)
