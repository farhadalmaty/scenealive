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
    device: str | None = None
    imgsz: int = 640
    conf: float = 0.30
    map_interval_s: float = 6.0
    max_sessions: int = 2
    max_frame_bytes: int = 400_000      # larger frames are refused
    idle_timeout_s: float = 180.0       # a connection that says nothing for this long is closed
    memory_minutes: float = 10.0        # tracks older than this are forgotten (keeps the map fast in long sessions)
    max_tracks: int = 800
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
