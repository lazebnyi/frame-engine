FROM python:3.12-slim

WORKDIR /app

RUN pip install redis numpy

COPY shared/ ./shared/
COPY reporter/ ./reporter/

CMD ["python", "-m", "reporter.main"]
