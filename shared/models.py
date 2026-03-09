"""Shared data models passed between pipeline services via Redis."""

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Frame:
    """A single captured camera frame with raw pixel data."""

    camera_id: str
    frame_number: int
    data: np.ndarray            # shape (H, W, C), dtype uint8
    timestamp: float = 0.0      # monotonic capture time (seconds)


@dataclass
class FrameMeta:
    """Per-frame metadata carried through the pipeline (no pixel data)."""

    camera_id: str
    frame_number: int
    captured_at: float       # monotonic time at capture (seconds)


@dataclass
class BatchMeta:
    """Batch metadata published to Redis after a batch is written to the shared volume."""

    batch_id: str
    file_path: str           # absolute path on the shared volume
    camera_ids: list[str]
    drift_ms: float
    timestamp: float         # monotonic capture time of the batch
    shape: list[int]         # [N, H, W, C]
    frames_meta: list[FrameMeta] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str | bytes) -> "BatchMeta":
        if isinstance(data, bytes):
            data = data.decode()
        d = json.loads(data)
        d["frames_meta"] = [FrameMeta(**fm) for fm in d.get("frames_meta", [])]
        return cls(**d)


@dataclass
class InferenceResult:
    """Inference output published to Redis by frame_processor."""

    batch_id: str
    camera_ids: list[str]
    frames_meta: list[FrameMeta] = field(default_factory=list)
    processed_at: float = field(default_factory=time.monotonic)
    processing_latency_ms: float = 0.0
    detections: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str | bytes) -> "InferenceResult":
        if isinstance(data, bytes):
            data = data.decode()
        d = json.loads(data)
        d["frames_meta"] = [FrameMeta(**fm) for fm in d.get("frames_meta", [])]
        return cls(**d)
