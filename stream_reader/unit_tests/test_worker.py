import queue
import threading
import time
from unittest.mock import MagicMock, patch

from stream_reader.frame_reader.fake_camera import CameraError
from stream_reader.frame_reader.reader import FrameReader


def test_worker_stops_on_stop_event(fake_frame):
    out_q = queue.Queue()
    stop = threading.Event()

    with patch("stream_reader.frame_reader.reader.FakeCamera") as MockCam:
        mock = MagicMock()
        mock.read.return_value = fake_frame
        mock.frame_count = 1
        MockCam.return_value = mock

        t = threading.Thread(
            target=FrameReader._camera_worker, args=("cam_01", 15, out_q, stop, {})
        )
        t.start()
        time.sleep(0.05)
        stop.set()
        t.join(timeout=2.0)

    assert not t.is_alive()


def test_worker_puts_frames_in_queue(fake_frame):
    out_q = queue.Queue()
    stop = threading.Event()
    read_count = 0

    with patch("stream_reader.frame_reader.reader.FakeCamera") as MockCam:

        def fake_read():
            nonlocal read_count
            read_count += 1
            if read_count >= 3:
                stop.set()
            return fake_frame

        mock = MagicMock()
        mock.read.side_effect = fake_read
        mock.frame_count = 1
        MockCam.return_value = mock

        t = threading.Thread(
            target=FrameReader._camera_worker, args=("cam_01", 15, out_q, stop, {})
        )
        t.start()
        t.join(timeout=2.0)

    assert out_q.qsize() >= 1


def test_worker_drops_frames_when_queue_full(fake_frame):
    out_q = queue.Queue(maxsize=1)
    stop = threading.Event()
    read_count = 0

    with patch("stream_reader.frame_reader.reader.FakeCamera") as MockCam:

        def fake_read():
            nonlocal read_count
            read_count += 1
            if read_count >= 5:
                stop.set()
            return fake_frame

        mock = MagicMock()
        mock.read.side_effect = fake_read
        mock.frame_count = 1
        MockCam.return_value = mock

        t = threading.Thread(
            target=FrameReader._camera_worker, args=("cam_01", 15, out_q, stop, {})
        )
        t.start()
        t.join(timeout=2.0)

    assert out_q.qsize() == 1


def test_worker_updates_heartbeat(fake_frame):
    out_q = queue.Queue()
    stop = threading.Event()
    heartbeat: dict = {}
    read_count = 0

    with patch("stream_reader.frame_reader.reader.FakeCamera") as MockCam:

        def fake_read():
            nonlocal read_count
            read_count += 1
            if read_count >= 2:
                stop.set()
            return fake_frame

        mock = MagicMock()
        mock.read.side_effect = fake_read
        mock.frame_count = 1
        MockCam.return_value = mock

        t = threading.Thread(
            target=FrameReader._camera_worker, args=("cam_01", 15, out_q, stop, heartbeat)
        )
        t.start()
        t.join(timeout=2.0)

    assert "cam_01" in heartbeat
    assert isinstance(heartbeat["cam_01"], float)


def test_worker_reconnects_after_camera_error(fake_frame):
    out_q = queue.Queue()
    stop = threading.Event()
    connect_count = 0

    with patch("stream_reader.frame_reader.reader.FakeCamera") as MockCam:
        with patch("stream_reader.frame_reader.reader._BACKOFF_INITIAL", 0.05):

            def make_camera(*args, **kwargs):
                nonlocal connect_count
                connect_count += 1
                mock = MagicMock()
                mock.frame_count = 1
                if connect_count == 1:
                    mock.read.side_effect = CameraError("simulated disconnect")
                else:

                    def read_and_stop():
                        stop.set()
                        return fake_frame

                    mock.read.side_effect = read_and_stop
                return mock

            MockCam.side_effect = make_camera

            t = threading.Thread(
                target=FrameReader._camera_worker, args=("cam_01", 15, out_q, stop, {})
            )
            t.start()
            t.join(timeout=3.0)

    assert connect_count >= 2
