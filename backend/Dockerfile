FROM python:3.8-slim

WORKDIR /app

COPY backend/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY shared /app/shared
COPY database /app/database

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"