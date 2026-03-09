import logging
import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from shared.models import Frame

logger = logging.getLogger(__name__)

_BUFFER_MAXLEN = int(os.environ.get("BUFFER_MAXLEN", "10"))


@dataclass
class Batch:
    """A synchronised set of frames, one per camera, ready for GPU inference."""

    frames: list[Frame]
    data: np.ndarray
    drift_ms: float
    created_at: float = field(default_factory=time.monotonic)

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def camera_ids(self) -> list[str]:
        return [f.camera_id for f in self.frames]


class Batcher:
    """Collect frames from *in_queue* and emit time-aligned batches."""

    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        batch_size: int = 4,
        max_drift_ms: float = 150.0,
        timeout: float = 0.5,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.batch_size = batch_size
        self.max_drift_ms = max_drift_ms
        self.timeout = timeout
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="batcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Batcher started (batch_size=%d, max_drift_ms=%.0f)",
            self.batch_size,
            self.max_drift_ms,
        )

    def join(self) -> None:
        if self._thread:
            self._thread.join()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        buffers: dict[str, deque[Frame]] = defaultdict(lambda: deque(maxlen=_BUFFER_MAXLEN))

        while not self._stop.is_set():
            try:
                frame: Frame = self.in_queue.get(timeout=self.timeout)
                buffers[frame.camera_id].append(frame)
            except queue.Empty:
                partial = [buf[-1] for buf in buffers.values() if buf]
                if partial:
                    self._flush(partial)
                    buffers.clear()
                continue

            if len(buffers) < self.batch_size or not all(buffers.values()):
                continue

            self._try_sync(buffers)

        logger.info("Batcher stopped")

    def _try_sync(self, buffers: dict[str, deque[Frame]]) -> None:
        ref_ts = max(buf[0].timestamp for buf in buffers.values())

        selected = {
            cid: min(buf, key=lambda f: abs(f.timestamp - ref_ts)) for cid, buf in buffers.items()
        }

        timestamps = [f.timestamp for f in selected.values()]
        drift_ms = (max(timestamps) - min(timestamps)) * 1000

        if drift_ms <= self.max_drift_ms:
            for cid, frame in selected.items():
                buf = buffers[cid]
                while buf and buf[0].timestamp <= frame.timestamp:
                    buf.popleft()
            self._flush(list(selected.values()), drift_ms)
        else:
            oldest_cid = min(buffers, key=lambda cid: buffers[cid][0].timestamp)
            buffers[oldest_cid].popleft()
            logger.debug(
                "Sync: drift %.0fms > %.0fms, dropped oldest %s frame",
                drift_ms,
                self.max_drift_ms,
                oldest_cid,
            )

    def _flush(self, frames: list[Frame], drift_ms: float = 0.0) -> None:
        data = np.stack([f.data for f in frames])  # (N, H, W, C)
        batch = Batch(frames=frames, data=data, drift_ms=drift_ms)
        try:
            self.out_queue.put_nowait(batch)
            logger.debug(
                "Batch emitted  size=%d  drift=%.0fms  cameras=%s",
                len(frames),
                drift_ms,
                [f.camera_id for f in frames],
            )
        except queue.Full:
            logger.warning(
                "Batch queue full — dropping batch of %d frames (cameras=%s)",
                len(frames),
                [f.camera_id for f in frames],
            )
