import logging
import time

from shared.models import InferenceResult

log = logging.getLogger(__name__)


class Reporter:
    """Consume inference results and emit structured log lines."""

    def report(self, result: InferenceResult) -> None:
        """Log one structured line per frame in the batch."""
        now = time.monotonic()
        for fm in result.frames_meta:
            e2e_ms = (now - fm.captured_at) * 1000
            log.info(
                "frame  camera=%-8s  frame_no=%6d  captured_at=%.3f  "
                "proc_latency=%.0fms  e2e_latency=%.0fms  batch=%s",
                fm.camera_id,
                fm.frame_number,
                fm.captured_at,
                result.processing_latency_ms,
                e2e_ms,
                result.batch_id[:8],
            )
