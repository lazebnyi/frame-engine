import queue
import threading
import time

from stream_reader.batcher import Batch, Batcher


def test_sync_emits_batch_when_all_cameras_ready(make_frame):
    in_q, out_q = queue.Queue(), queue.Queue()
    stop = threading.Event()
    batcher = Batcher(
        in_queue=in_q, out_queue=out_q, batch_size=2, max_drift_ms=200, stop_event=stop
    )
    batcher.start()

    now = time.monotonic()
    in_q.put(make_frame("cam_01", now))
    in_q.put(make_frame("cam_02", now + 0.05))

    batch = out_q.get(timeout=2.0)
    stop.set()

    assert isinstance(batch, Batch)
    assert len(batch.frames) == 2
    assert {f.camera_id for f in batch.frames} == {"cam_01", "cam_02"}
    assert batch.drift_ms < 200


def test_batch_drift_is_computed_correctly(make_frame):
    in_q, out_q = queue.Queue(), queue.Queue()
    stop = threading.Event()
    batcher = Batcher(
        in_queue=in_q, out_queue=out_q, batch_size=2, max_drift_ms=500, stop_event=stop
    )
    batcher.start()

    now = time.monotonic()
    in_q.put(make_frame("cam_01", now))
    in_q.put(make_frame("cam_02", now + 0.1))  # 100ms drift

    batch = out_q.get(timeout=2.0)
    stop.set()

    assert 80 < batch.drift_ms < 150


def test_partial_flush_on_timeout(make_frame):
    in_q, out_q = queue.Queue(), queue.Queue()
    stop = threading.Event()
    batcher = Batcher(
        in_queue=in_q, out_queue=out_q, batch_size=3, max_drift_ms=200, timeout=0.2, stop_event=stop
    )
    batcher.start()

    now = time.monotonic()
    in_q.put(make_frame("cam_01", now))
    in_q.put(make_frame("cam_02", now + 0.01))
    # cam_03 never arrives — batcher flushes partial after timeout

    batch = out_q.get(timeout=1.5)
    stop.set()

    assert len(batch.frames) == 2


def test_full_out_queue_drops_batches(make_frame):
    in_q = queue.Queue()
    out_q = queue.Queue(maxsize=1)
    stop = threading.Event()
    batcher = Batcher(
        in_queue=in_q, out_queue=out_q, batch_size=2, max_drift_ms=1000, stop_event=stop
    )
    batcher.start()

    now = time.monotonic()
    for i in range(6):
        ts = now + i * 0.01
        in_q.put(make_frame("cam_01", ts))
        in_q.put(make_frame("cam_02", ts))

    time.sleep(0.3)
    stop.set()

    assert out_q.qsize() <= 1


def test_stop_event_exits_batcher():
    in_q, out_q = queue.Queue(), queue.Queue()
    stop = threading.Event()
    batcher = Batcher(
        in_queue=in_q, out_queue=out_q, batch_size=2, max_drift_ms=100, stop_event=stop
    )
    batcher.start()
    assert batcher.is_alive()

    stop.set()
    batcher.join()
    assert not batcher.is_alive()
