"""All tunable numbers in one place. Every dataclass is dumped into the run report, so a run is reproducible."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class CameraConfig:
    fov_long_deg: float = 69.0        # field of view across the LONG side of the frame; phone presets in camera.FOV_PRESETS


@dataclass
class FastConfig:
    imgsz: int = 640                  # long side of the frame the detector sees (smaller = faster)
    conf: float = 0.15                # detector confidence FLOOR: weak detections may continue a track (hysteresis) but never start one
    start_conf: float = 0.30          # a track may be started only by a detection at least this sure
    min_conf_publish: float = 0.30    # a track is shown in the list only if its name confidence reaches this
    publish_hits: int = 2             # ... and it was detected in at least this many frames ...
    publish_single_conf: float = 0.55 # ... unless one detection is already this sure (then the name appears at once)
    confirm_hits: int = 3             # detections needed before a track is "confirmed"
    confirm_conf: float = 0.40        # ... and the name confidence needed for it
    max_age_frames: int = 10          # a track is dropped after this many missed frames (kept longer: detectors flicker)
    publish_misses: int = 2           # an entry is shown as 'in view' for this many missed frames, then it lives on in the map as 'remembered'
    publish_hits_by_class: dict = field(default_factory=lambda: {"billboard": 4, "stroller": 4, "can": 4, "kiosk": 4, "mailbox": 4, "wheelchair": 4, "shopping cart": 4})   # classes that were seen to fire falsely need more sightings
    iou_match: float = 0.25           # minimum (class-weighted) IoU to continue a track
    min_box_px: int = 6               # ignore boxes smaller than this (either side)


@dataclass
class MapConfig:
    interval_s: float = 6.0           # the map loop runs this often (video time)
    az_merge_deg: float = 6.0         # tracks of the same class closer than this in world azimuth are one object
    min_obs: int = 2                  # tracks with fewer detections are not mapped (still shown by the fast loop)
    single_conf: float = 0.55         # ... unless one detection is already this sure (a glimpse of a car must not be forgotten)
    moving_gap_s: float = 3.0         # a moving class continues an object only after a gap shorter than this
    remembered_after_s: float = 1.0   # not seen for this long -> status "remembered" (out of view now)


@dataclass
class SimConfig:
    sample_fps: float = 3.0           # frames per second sent from the "phone"
    uplink_ms: float = 100.0          # network, phone -> server
    downlink_ms: float = 80.0         # network, server -> phone
    jpeg_quality: int = 70
    send_long_side: int = 640         # frames are resized to this before JPEG encoding
    latest_wins: bool = True          # when the server is behind, process only the newest waiting frame
    shared_cpu: bool = False          # True: the map loop competes with the fast loop for one processor
    compute_scale: float = 1.0        # multiplies measured fast-loop compute time ("what if 3x slower / 10x faster")
    map_compute_scale: float = 1.0    # same, for the map loop
    client_factor: float = 1.0        # multiplies phone-side (resize + encode) time: phones are slower than this laptop
    max_seconds: float | None = None  # stop after this much video


def to_dict(*cfgs) -> dict:
    return {type(c).__name__: asdict(c) for c in cfgs}
