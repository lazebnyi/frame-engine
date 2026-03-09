from shared.models import FrameMeta, BatchMeta, InferenceResult


def test_batch_meta_round_trip():
    fm = FrameMeta(camera_id="cam_01", frame_number=42, captured_at=1.23)
    meta = BatchMeta(
        batch_id="abc-123",
        file_path="/tmp/abc.npy",
        camera_ids=["cam_01"],
        drift_ms=12.5,
        timestamp=1.0,
        shape=[1, 2160, 3840, 3],
        frames_meta=[fm],
    )
    restored = BatchMeta.from_json(meta.to_json())

    assert restored.batch_id == "abc-123"
    assert restored.drift_ms == 12.5
    assert restored.shape == [1, 2160, 3840, 3]
    assert restored.frames_meta[0].camera_id == "cam_01"
    assert restored.frames_meta[0].frame_number == 42
    assert restored.frames_meta[0].captured_at == 1.23


def test_batch_meta_from_json_accepts_bytes():
    meta = BatchMeta(
        batch_id="b1", file_path="/tmp/b1.npy", camera_ids=["cam_01"],
        drift_ms=0.0, timestamp=0.0, shape=[1, 2, 3, 4],
    )
    restored = BatchMeta.from_json(meta.to_json().encode())
    assert restored.batch_id == "b1"


def test_inference_result_round_trip():
    fm = FrameMeta(camera_id="cam_02", frame_number=7, captured_at=2.0)
    result = InferenceResult(
        batch_id="xyz-456",
        camera_ids=["cam_02"],
        frames_meta=[fm],
        processing_latency_ms=123.4,
        detections={"boxes": [1, 2, 3]},
    )
    restored = InferenceResult.from_json(result.to_json())

    assert restored.batch_id == "xyz-456"
    assert restored.processing_latency_ms == 123.4
    assert restored.frames_meta[0].frame_number == 7
    assert restored.detections == {"boxes": [1, 2, 3]}


def test_inference_result_from_json_accepts_bytes():
    result = InferenceResult(batch_id="r1", camera_ids=["cam_01"])
    restored = InferenceResult.from_json(result.to_json().encode())
    assert restored.batch_id == "r1"


def test_batch_meta_empty_frames_meta():
    meta = BatchMeta(
        batch_id="no-frames", file_path="/tmp/x.npy", camera_ids=[],
        drift_ms=0.0, timestamp=0.0, shape=[0, 2160, 3840, 3],
    )
    restored = BatchMeta.from_json(meta.to_json())
    assert restored.frames_meta == []
