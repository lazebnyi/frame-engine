import numpy as np
import pytest

from shared.models import Frame
from stream_reader.batcher import Batch


@pytest.fixture
def fake_frame() -> np.ndarray:
    """Minimal 2×2 RGB frame for worker tests."""
    return np.zeros((2, 2, 3), dtype=np.uint8)


@pytest.fixture
def make_frame():
    """Factory: create a Frame with given camera_id and timestamp."""

    def _make(camera_id: str, ts: float) -> Frame:
        return Frame(
            camera_id=camera_id,
            frame_number=1,
            data=np.zeros((2, 2, 3), dtype=np.uint8),
            timestamp=ts,
        )

    return _make


@pytest.fixture
def make_batch():
    """Factory: create a Batch with n cameras, each contributing one frame."""

    def _make(n: int = 2, drift_ms: float = 10.0) -> Batch:
        frames = [
            Frame(
                camera_id=f"cam_0{i + 1}",
                frame_number=i + 1,
                data=np.zeros((4, 4, 3), dtype=np.uint8),
                timestamp=1.0 + i * 0.01,
            )
            for i in range(n)
        ]
        return Batch(
            frames=frames,
            data=np.stack([f.data for f in frames]),
            drift_ms=drift_ms,
        )

    return _make


@pytest.fixture
def mock_redis(mocker=None):
    """MagicMock standing in for a Redis client."""
    from unittest.mock import MagicMock

    return MagicMock()
