import json

import numpy as np

from stream_reader.publisher import Publisher


def test_publish_saves_npy_file(make_batch, mock_redis, tmp_path):
    batch = make_batch()
    Publisher(mock_redis, str(tmp_path)).publish(batch)

    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".npy"
    assert np.load(files[0]).shape == batch.data.shape


def test_publish_calls_redis_xadd(make_batch, mock_redis, tmp_path):
    Publisher(mock_redis, str(tmp_path)).publish(make_batch())

    mock_redis.xadd.assert_called_once()
    stream, fields = mock_redis.xadd.call_args[0]
    assert stream == "batches"
    assert "data" in fields


def test_publish_meta_has_correct_camera_ids(make_batch, mock_redis, tmp_path):
    Publisher(mock_redis, str(tmp_path)).publish(make_batch(n=2))

    data = json.loads(mock_redis.xadd.call_args[0][1]["data"])
    assert set(data["camera_ids"]) == {"cam_01", "cam_02"}
    assert data["drift_ms"] == 10.0


def test_publish_meta_shape_matches_data(make_batch, mock_redis, tmp_path):
    batch = make_batch(n=3)
    Publisher(mock_redis, str(tmp_path)).publish(batch)

    data = json.loads(mock_redis.xadd.call_args[0][1]["data"])
    assert data["shape"] == list(batch.data.shape)


def test_publish_file_path_in_meta(make_batch, mock_redis, tmp_path):
    Publisher(mock_redis, str(tmp_path)).publish(make_batch())

    files = list(tmp_path.iterdir())
    data = json.loads(mock_redis.xadd.call_args[0][1]["data"])
    assert data["file_path"] == str(files[0])
