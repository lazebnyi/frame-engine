import logging
import os
import queue
import signal
import threading
import time

import redis

from shared.config import PipelineConfig

from .batcher import Batch, Batcher
from .frame_reader import FrameReader
from .publisher import Publisher

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
SHARED_PATH = os.environ.get("SHARED_PATH", "/shared/batches")
STREAM = "batches"
MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT_BATCHES", "4"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stream_reader")


def _wait_for_capacity(r: redis.Redis, stop_event: threading.Event) -> None:
    """Block until in-flight batch count drops below MAX_INFLIGHT."""
    while not stop_event.is_set():
        try:
            inflight = 0
            for g in r.xinfo_groups(STREAM):
                name = g.get("name") or g.get(b"name")
                if name in ("frame_processor", b"frame_processor"):
                    inflight = (g.get("lag") or g.get(b"lag") or 0) + (
                        g.get("pending") or g.get(b"pending") or 0
                    )
                    break
        except redis.ResponseError:
            inflight = 0  # consumer group not yet created
        if inflight < MAX_INFLIGHT:
            return
        time.sleep(0.1)


def main() -> None:
    config = PipelineConfig.load()

    stop_event = threading.Event()

    def _shutdown(signum: int, _frame: object) -> None:
        log.info("Signal %d received — shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    os.makedirs(SHARED_PATH, exist_ok=True)

    r = redis.from_url(REDIS_URL)
    r.ping()
    log.info("Connected to Redis at %s", REDIS_URL)

    frame_q: queue.Queue = queue.Queue(maxsize=config.frame_queue_maxsize)
    batch_q: queue.Queue = queue.Queue(maxsize=config.batch_queue_maxsize)

    reader = FrameReader(
        camera_configs=[{"camera_id": c.camera_id, "fps": c.fps} for c in config.cameras],
        frame_queue=frame_q,
        stop_event=stop_event,
        health_interval_s=config.health_interval_s,
        camera_timeout_s=config.camera_timeout_s,
    )
    batcher = Batcher(
        in_queue=frame_q,
        out_queue=batch_q,
        batch_size=config.batch_size,
        max_drift_ms=config.max_drift_ms,
        stop_event=stop_event,
    )
    publisher = Publisher(r, SHARED_PATH)

    reader.start()
    batcher.start()
    log.info("Pipeline started (batch_size=%d)", config.batch_size)

    while not stop_event.is_set():
        try:
            batch: Batch = batch_q.get(timeout=1.0)
        except queue.Empty:
            continue

        _wait_for_capacity(r, stop_event)
        publisher.publish(batch)

    log.info("Draining remaining batches...")
    while True:
        try:
            batch = batch_q.get_nowait()
            publisher.publish(batch)
        except queue.Empty:
            break

    log.info("Stream reader stopped.")


if __name__ == "__main__":
    main()
