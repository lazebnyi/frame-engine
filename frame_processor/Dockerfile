FROM python:3.12-slim

WORKDIR /app

RUN pip install numpy redis pillow

COPY shared/ ./shared/
COPY frame_processor/ ./frame_processor/

CMD ["python", "-m", "frame_processor.main"]
