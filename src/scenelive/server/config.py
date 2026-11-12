from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8443
    https: bool = True                  # phones allow the camera only on https (or localhost)
    cert_dir: str = "certs"
    token: str | None = None            # required as ?token=... for the WebSocket and /api/stats when set
    detector: str = "world"             # world | yolo | blobs
    weights: str | None = None
    device: str | None = "auto"        # auto: the graphics card if PyTorch sees one, otherwise the processor
    imgsz: int = 640
    conf: float = 0.30                  # a track may START at this confidence
    floor_conf: float = 0.15            # weaker detections may only continue a track
    map_interval_s: float = 1.0         # the map is cheap (milliseconds): rebuilt every second and whenever an object drops out of the live list
    max_sessions: int = 2
    max_frame_bytes: int = 400_000      # larger frames are refused
    idle_timeout_s: float = 180.0       # a connection that says nothing for this long is closed
    memory_minutes: float = 10.0        # tracks older than this are forgotten (keeps the map fast in long sessions)
    max_tracks: int = 800
    seg: str = "none"                   # segmentation of floor / walls / road for "where to go": none | segformer | colors
    seg_size: int = 0                   # model input, long side in pixels (0 = the model default 512x512)
    seg_every_s: float = 1.0
    seg_sync: bool = False              # tests only: run the model inline instead of in a thread
    camera_height_m: float = 1.4        # a hand-held phone; the distances to obstacles scale with it
    ocr: str = "none"                   # reading text (bus numbers, signs): none | easyocr
    ocr_langs: str = "ru,en"
    ocr_every_s: float = 1.0
    ocr_sync: bool = False              # tests only
    alerts: bool = True                 # spoken/visual warnings about danger, close vehicles, bad surface
    alert_cooldown_s: float = 15.0
    alert_range_m: float = 6.0
    ground: str = "none"                # surface under the feet: none | clip | colors
    ground_every_s: float = 1.0
    vocab: str = "extended"             # detector vocabulary: core | extended | path to a text file
    agent: str = "auto"                 # auto (Claude if a key exists, otherwise offline) | claude | offline
    model: str = "claude-sonnet-5-5"

    @staticmethod
    def from_env(**over) -> "ServerConfig":
        c = ServerConfig(token=os.environ.get("SCENELIVE_TOKEN") or None, model=os.environ.get("SCENELIVE_MODEL") or "claude-sonnet-5-5")
        for k, v in over.items():
            if v is not None: setattr(c, k, v)
        return c
