"""Pipeline configuration.

Load priority (highest → lowest):
    1. Environment variables (e.g. BATCH_SIZE, MAX_DRIFT_MS)
    2. JSON file at CONFIG_PATH
    3. Dataclass defaults
"""

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_CAMERAS = [
    {"camera_id": "cam_01", "fps": 15},
    {"camera_id": "cam_02", "fps": 15},
    {"camera_id": "cam_03", "fps": 10},
    {"camera_id": "cam_04", "fps": 10},
]


@dataclass
class CameraConfig:
    camera_id: str
    fps: int


@dataclass
class PipelineConfig:
    cameras: list[CameraConfig] = field(
        default_factory=lambda: [CameraConfig(**c) for c in DEFAULT_CAMERAS]
    )
    batch_size: int = 4
    max_drift_ms: float = 150.0
    frame_queue_maxsize: int = 100   # per-camera frame buffer; 0 = unbounded
    batch_queue_maxsize: int = 10    # assembled batches waiting for publish
    health_interval_s: float = 5.0  # watchdog check frequency
    camera_timeout_s: float = 10.0  # warn if a camera is silent for this long

    @classmethod
    def load(cls) -> "PipelineConfig":
        """Load from CONFIG_PATH JSON file, then apply env var overrides."""
        raw: dict = {}
        path = os.environ.get("CONFIG_PATH", "")
        if path:
            try:
                with open(path) as fh:
                    raw = json.load(fh)
                logger.info("Config loaded from %s", path)
            except OSError as exc:
                logger.warning("Cannot read CONFIG_PATH=%s: %s — using defaults", path, exc)

        cameras = [CameraConfig(**c) for c in raw.get("cameras", DEFAULT_CAMERAS)]

        return cls(
            cameras             = cameras,
            batch_size          = int(os.environ.get("BATCH_SIZE",           raw.get("batch_size",          4))),
            max_drift_ms        = float(os.environ.get("MAX_DRIFT_MS",       raw.get("max_drift_ms",        150.0))),
            frame_queue_maxsize = int(os.environ.get("FRAME_QUEUE_MAXSIZE",  raw.get("frame_queue_maxsize", 100))),
            batch_queue_maxsize = int(os.environ.get("BATCH_QUEUE_MAXSIZE",  raw.get("batch_queue_maxsize", 10))),
            health_interval_s   = float(os.environ.get("HEALTH_INTERVAL_S", raw.get("health_interval_s",   5.0))),
            camera_timeout_s    = float(os.environ.get("CAMERA_TIMEOUT_S",  raw.get("camera_timeout_s",    10.0))),
        )
