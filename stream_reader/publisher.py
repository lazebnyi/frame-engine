import logging
import os
import uuid

import numpy as np
import redis

from shared.models import BatchMeta, FrameMeta

from .batcher import Batch

log = logging.getLogger(__name__)

STREAM = "batches"


class Publisher:
    """Save batches to shared volume and publish metadata to Redis."""

    def __init__(self, r: redis.Redis, shared_path: str) -> None:
        self._r = r
        self._shared_path = shared_path

    def publish(self, batch: Batch) -> None:
        """Save batch data to shared volume and publish metadata to Redis."""
        batch_id = str(uuid.uuid4())
        file_path = os.path.join(self._shared_path, f"{batch_id}.npy")
        np.save(file_path, batch.data)

        frames_meta = [
            FrameMeta(
                camera_id=f.camera_id,
                frame_number=f.frame_number,
                captured_at=f.timestamp,
            )
            for f in batch.frames
        ]
        meta = BatchMeta(
            batch_id=batch_id,
            file_path=file_path,
            camera_ids=batch.camera_ids,
            drift_ms=batch.drift_ms,
            timestamp=batch.created_at,
            shape=list(batch.data.shape),
            frames_meta=frames_meta,
        )
        self._r.xadd(STREAM, {"data": meta.to_json()})
        cam_info = "  ".join(
            f"{f.camera_id}[frame={f.frame_number} {f.data.shape[1]}x{f.data.shape[0]}]"
            for f in batch.frames
        )
        log.info(
            "Published  batch=%s  drift=%.0fms  cameras=[%s]",
            batch_id[:8],
            batch.drift_ms,
            cam_info,
        )
