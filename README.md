# frame-engine

On-premise real-time computer vision pipeline. Ingests camera streams, synchronises frames across cameras, runs CV/AI inference, and emits per-frame results to downstream consumers.

## Flow

First the cameras start capturing video streams, and each camera runs in its own thread. Next, the stream_reader service reads raw frames from each camera, aligns frames from different cameras based on the closest timestamps, and groups them into batches. These batches are saved to a shared volume, and their metadata is sent to a Redis stream called "batches". After that, the frame_processor service reads information about new batches from the "batches" stream and loads the frames from the shared volume, preprocesses them, and runs the inference model to detect objects. When processing is finished, it sends the results to another Redis stream called "results". Finally, the reporter service reads the results from the "results" stream and logs the detection results together with the latency for each processed frame.

All services run in separate Docker containers and communicate via two Redis Streams. Batch pixel data is exchanged through a RAM-backed tmpfs shared volume, keeping Redis free of large payloads.

## Services

### stream_reader

Reads frames from multiple cameras using one thread per camera. Synchronises frames into time-aligned batches and publishes them to the `batches` Redis stream.

**Backpressure** — before publishing each batch, stream_reader checks how many batches are currently in-flight using `xinfo_groups`. If the count reaches `MAX_INFLIGHT_BATCHES` it waits, preventing unbounded memory growth.

### frame_processor

Consumes the `batches` stream via a Redis consumer group. For each batch it:

1. Loads the `.npy` file from the shared volume and deletes it immediately after reading.
2. Preprocesses frames (resize, convert to grayscale).
3. Runs inference (currently simulated with `time.sleep`).
4. Publishes an `InferenceResult` to the `results` stream and acknowledges the message.

### reporter

Consumes the `results` stream and logs a line per frame with camera ID, frame number, processing latency, and end-to-end latency from capture to result.

## Frame synchronisation

The `Batcher` aligns cameras frames with a nearest-neighbour algorithm:

1. Each camera has a rolling frame buffer (FIFO, bounded by `BUFFER_MAXLEN`).
2. When all cameras have at least one buffered frame, a **reference time** is computed: the maximum of each camera's oldest frame timestamp. This is the first moment all cameras have coverage.
3. For each camera, the frame closest to the reference time is selected.
4. The **drift** (max timestamp − min timestamp across selected frames) is computed in milliseconds.
5. If `drift ≤ MAX_DRIFT_MS` the batch is emitted and consumed frames are discarded from the buffers.
6. Otherwise, the oldest frame of the most-behind camera is dropped and the algorithm retries on the next incoming frame.
7. If no new frame arrives within `timeout` seconds, whatever is buffered is flushed as a partial batch so downstream is never permanently starved.

## Camera health monitoring

`FrameReader` runs a dedicated `camera-monitor` thread alongside the per-camera capture threads. Every `health_interval_s` (5 sec - default) seconds it checks the last heartbeat time for each camera:

- If a camera has **never** produced a frame after `camera_timeout_s` seconds, a warning is logged.
- If a camera **stops** producing frames for more than `camera_timeout_s` seconds, a warning is logged indicating it may be stuck.

Camera capture threads also recover automatically from disconnects using **exponential backoff**: the retry delay doubles on each failure (`BACKOFF_FACTOR`) up to `BACKOFF_MAX` seconds. Once a camera produces `STABLE_FRAMES` consecutive successful frames, the backoff resets to `BACKOFF_INITIAL`.

## Configuration

All tuneable values live in `.env` (copy `.env.example` as a starting point):

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |
| `SHARED_PATH` | `/shared/batches` | Path for batch `.npy` files |
| `BATCH_SIZE` | `4` | Number of cameras per batch |
| `MAX_DRIFT_MS` | `150` | Maximum acceptable timestamp spread across cameras (ms) |
| `MAX_INFLIGHT_BATCHES` | `4` | Backpressure limit: batches waiting for processing |
| `BUFFER_MAXLEN` | `10` | Per-camera frame buffer depth |
| `BACKOFF_INITIAL` | `0.5` | Initial reconnect delay after camera disconnect (s) |
| `BACKOFF_FACTOR` | `2.0` | Backoff multiplier on each consecutive failure |
| `BACKOFF_MAX` | `30.0` | Maximum reconnect delay (s) |
| `STABLE_FRAMES` | `5` | Consecutive frames before backoff resets |
| `HEALTH_INTERVAL_S` | `5.0` | How often the camera monitor checks for silent cameras (s) |
| `CAMERA_TIMEOUT_S` | `10.0` | Silence duration before a camera is flagged as stuck (s) |
| `GPU_LATENCY_MS` | `20` | Simulated inference latency (ms) |

## Running

### Prerequisites

- [uv](https://github.com/astral-sh/uv) — Python package manager
- Docker

### Docker Compose (recommended)

Copy the example env file, then build and start all services:

```bash
cp .env.example .env
docker compose up --build
```

This starts `redis`, `stream_reader`, `frame_processor`, and `reporter` with a shared RAM-backed tmpfs volume for batch data.

### Unit tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
```
