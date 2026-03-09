FROM python:3.12-slim

WORKDIR /app

RUN pip install numpy redis

COPY stream_reader/ ./stream_reader/

COPY shared/ ./shared/

CMD ["python", "-m", "stream_reader.main"]
