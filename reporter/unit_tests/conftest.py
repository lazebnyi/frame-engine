import time

import pytest

from shared.models import FrameMeta, InferenceResult


@pytest.fixture
def make_result():
    """Factory: create an InferenceResult with n_frames cameras."""

    def _make(n_frames: int = 1) -> InferenceResult:
        now = time.monotonic()
        return InferenceResult(
            batch_id="test-batch-0000-1234",
            camera_ids=[f"cam_0{i + 1}" for i in range(n_frames)],
            frames_meta=[
                FrameMeta(
                    camera_id=f"cam_0{i + 1}",
                    frame_number=i + 1,
                    captured_at=now - 0.1,
                )
                for i in range(n_frames)
            ],
            processing_latency_ms=50.0,
        )

    return _make
