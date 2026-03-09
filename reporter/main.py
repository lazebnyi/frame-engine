"""Reporter — Final consumer service."""

import logging
import os
import signal
import threading

import redis

from shared.models import InferenceResult

from .reporting import Reporter

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
RESULTS_STREAM = "results"
CONSUMER_GROUP = "reporter"
CONSUMER_NAME = "reporter-1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("reporter")


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
        r.xgroup_create(RESULTS_STREAM, CONSUMER_GROUP, id="0", mkstream=True)
        log.info("Consumer group '%s' created", CONSUMER_GROUP)
    except redis.ResponseError:
        log.info("Consumer group '%s' already exists", CONSUMER_GROUP)

    reporter = Reporter()
    log.info("Reporter ready — consuming '%s'", RESULTS_STREAM)

    while not stop_event.is_set():
        messages = r.xreadgroup(
            CONSUMER_GROUP,
            CONSUMER_NAME,
            {RESULTS_STREAM: ">"},
            count=1,
            block=1000,
        )
        if not messages:
            continue

        for _stream, msgs in messages:
            for msg_id, fields in msgs:
                result = InferenceResult.from_json(fields[b"data"])
                reporter.report(result)
                r.xack(RESULTS_STREAM, CONSUMER_GROUP, msg_id)

    log.info("Reporter stopped.")


if __name__ == "__main__":
    main()
