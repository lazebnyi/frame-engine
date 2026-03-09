import logging
import os
import queue
import threading
import time

from shared.models import Frame

from .fake_camera import CameraError, FakeCamera

logger = logging.getLogger(__name__)

_BACKOFF_INITIAL = float(os.environ.get("BACKOFF_INITIAL", "0.5"))
_BACKOFF_FACTOR = float(os.environ.get("BACKOFF_FACTOR", "2.0"))
_BACKOFF_MAX = float(os.environ.get("BACKOFF_MAX", "30.0"))
_STABLE_FRAMES = int(os.environ.get("STABLE_FRAMES", "5"))


class FrameReader:
    """Read frames from multiple cameras concurrently via threads."""

    def __init__(
        self,
        camera_configs: list[dict],
        frame_queue: queue.Queue | None = None,
        stop_event: threading.Event | None = None,
        health_interval_s: float = 5.0,
        camera_timeout_s: float = 10.0,
    ) -> None:
        self._configs = camera_configs
        self.queue: queue.Queue = frame_queue if frame_queue is not None else queue.Queue()
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._health_interval = health_interval_s
        self._camera_timeout = camera_timeout_s
        self._heartbeat: dict[str, float] = {}
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for cfg in self._configs:
            t = threading.Thread(
                target=FrameReader._camera_worker,
                args=(cfg["camera_id"], cfg["fps"], self.queue, self._stop, self._heartbeat),
                name=f"reader-{cfg['camera_id']}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            logger.info("Started thread for %s", cfg["camera_id"])

        monitor = threading.Thread(
            target=self._camera_monitor,
            name="camera-monitor",
            daemon=True,
        )
        monitor.start()
        self._threads.append(monitor)

    @staticmethod
    def _camera_worker(
        camera_id: str,
        fps: int,
        out_queue: queue.Queue,
        stop_event: threading.Event | None = None,
        heartbeat: dict | None = None,
    ) -> None:
        """Read frames from a single camera, reconnecting with exponential backoff."""
        _stop = stop_event if stop_event is not None else threading.Event()
        backoff = _BACKOFF_INITIAL
        logger.info("[%s] starting at %d fps", camera_id, fps)

        while not _stop.is_set():
            camera = FakeCamera(camera_id=camera_id, fps=fps)
            logger.info("[%s] connected", camera_id)
            stable = 0

            while not _stop.is_set():
                try:
                    data = camera.read()
                    frame = Frame(
                        camera_id=camera_id,
                        frame_number=camera.frame_count,
                        data=data,
                        timestamp=time.monotonic(),
                    )
                    if heartbeat is not None:
                        heartbeat[camera_id] = frame.timestamp

                    try:
                        out_queue.put_nowait(frame)
                    except queue.Full:
                        logger.warning(
                            "[%s] frame queue full — dropping frame %d",
                            camera_id,
                            frame.frame_number,
                        )

                    stable += 1
                    if stable >= _STABLE_FRAMES:
                        backoff = _BACKOFF_INITIAL

                except CameraError as exc:
                    logger.warning(
                        "[%s] disconnected: %s — retry in %.1fs",
                        camera_id,
                        exc,
                        backoff,
                    )
                    break

            # Interruptible sleep so SIGTERM is handled promptly.
            deadline = time.monotonic() + backoff
            while not _stop.is_set() and time.monotonic() < deadline:
                time.sleep(0.1)

            backoff = min(backoff * _BACKOFF_FACTOR, _BACKOFF_MAX)
            if not _stop.is_set():
                logger.info("[%s] reconnecting (next backoff %.1fs)", camera_id, backoff)

        logger.info("[%s] worker stopped", camera_id)

    def _camera_monitor(self) -> None:
        """Warn if any camera stops producing frames for longer than camera_timeout_s."""
        logger.info(
            "Camera monitor started (interval=%.1fs, timeout=%.1fs)",
            self._health_interval,
            self._camera_timeout,
        )
        started_at = time.monotonic()
        while not self._stop.wait(self._health_interval):
            now = time.monotonic()
            for cfg in self._configs:
                cid = cfg["camera_id"]
                last = self._heartbeat.get(cid)
                if last is None:
                    if now - started_at > self._camera_timeout:
                        logger.warning("[camera-monitor] %s — never produced a frame", cid)
                    continue
                lag = now - last
                if lag > self._camera_timeout:
                    logger.warning(
                        "[camera-monitor] %s — silent for %.1fs (threshold %.1fs) — may be stuck",
                        cid,
                        lag,
                        self._camera_timeout,
                    )
        logger.info("Camera monitor stopped")

    def join(self) -> None:
        for t in self._threads:
            t.join()

    def is_alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)
