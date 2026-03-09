"""Frame Processor — Processing service entry point.

Pipeline per batch
------------------
1. Preprocess  — resize each frame 3840×2160 → 640×480, convert to grayscale
2. Inference   — time.sleep(0.1) simulating model latency
"""

import logging
import os
import signal
import threading
import time

import numpy as np
import redis

from shared.models import BatchMeta, InferenceResult

from .processor import Processor

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
BATCHES_STREAM = "batches"
RESULTS_STREAM = "results"
CONSUMER_GROUP = "frame_processor"
CONSUMER_NAME = "processor-1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frame_processor")


def main() -> None:
    stop_event = threading.Event()

    def _shutdown(signum: int, _frame: object) -> None:
        log.info("Signal %d received — shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    r = redis.from_url(REDIS_URL)
    r.ping()
    log.info("Connected to Redis at %s", REDIS_URL)

    try:
        r.xgroup_create(BATCHES_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("Consumer group '%s' created", CONSUMER_GROUP)
    except redis.ResponseError:
        log.info("Consumer group '%s' already exists", CONSUMER_GROUP)

    processor = Processor()
    log.info("Frame processor ready — consuming '%s'", BATCHES_STREAM)

    while not stop_event.is_set():
        messages = r.xreadgroup(
            CONSUMER_GROUP,
            CONSUMER_NAME,
            {BATCHES_STREAM: ">"},
            count=1,
            block=1000,
        )
        if not messages:
            continue

        for _stream, msgs in messages:
            for msg_id, fields in msgs:
                meta = BatchMeta.from_json(fields[b"data"])

                try:
                    raw = np.load(meta.file_path)
                    os.remove(meta.file_path)
                except FileNotFoundError:
                    log.warning(
                        "Batch file missing (stale message?) — skipping batch=%s", meta.batch_id[:8]
                    )
                    r.xack(BATCHES_STREAM, CONSUMER_GROUP, msg_id)
                    continue

                t0 = time.monotonic()
                processed = processor.preprocess(raw)
                preprocess_ms = (time.monotonic() - t0) * 1000

                detections = processor.run_inference(processed)
                latency_ms = (time.monotonic() - t0) * 1000

                result = InferenceResult(
                    batch_id=meta.batch_id,
                    camera_ids=meta.camera_ids,
                    frames_meta=meta.frames_meta,
                    processing_latency_ms=latency_ms,
                    detections=detections,
                )
                r.xadd(RESULTS_STREAM, {"data": result.to_json()})
                r.xack(BATCHES_STREAM, CONSUMER_GROUP, msg_id)

                log.info(
                    "Processed  batch=%s  preprocess=%.0fms  out_shape=%s  cameras=%s",
                    meta.batch_id[:8],
                    preprocess_ms,
                    processed.shape,
                    meta.camera_ids,
                )

    log.info("Frame processor stopped.")


if __name__ == "__main__":
    main()
