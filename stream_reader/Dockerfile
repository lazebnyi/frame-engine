FROM python:3.12-slim

WORKDIR /app

RUN pip install numpy redis

# Service code (frame_reader/, batcher.py, main.py)
COPY stream_reader/ ./stream_reader/

# Shared models used by all services
COPY shared/ ./shared/

CMD ["python", "-m", "stream_reader.main"]
