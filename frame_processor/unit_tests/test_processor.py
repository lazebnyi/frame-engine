import numpy as np

from frame_processor.processor import Processor


def test_preprocess_output_shape():
    data = np.zeros((3, 2160, 3840, 3), dtype=np.uint8)
    result = Processor().preprocess(data)
    assert result.shape == (3, 480, 640)


def test_preprocess_output_dtype():
    data = np.zeros((1, 100, 100, 3), dtype=np.uint8)
    result = Processor().preprocess(data)
    assert result.dtype == np.uint8


def test_preprocess_output_is_grayscale():
    # Output is (N, H, W) — no colour channel
    data = np.random.randint(0, 255, (2, 100, 100, 3), dtype=np.uint8)
    result = Processor().preprocess(data)
    assert result.ndim == 3


def test_preprocess_single_frame():
    data = np.zeros((1, 480, 640, 3), dtype=np.uint8)
    result = Processor().preprocess(data)
    assert result.shape == (1, 480, 640)


def test_preprocess_preserves_batch_size():
    processor = Processor()
    for n in (1, 2, 4):
        data = np.zeros((n, 100, 100, 3), dtype=np.uint8)
        result = processor.preprocess(data)
        assert result.shape[0] == n
